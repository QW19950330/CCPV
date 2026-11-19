"""CPU reference input replacement; not a rollout or a causal-effect model.

State columns are (x_m, y_m, vx_m_s, vy_m_s). Only the selected player's
terminal velocity may change. No historical state, position, other entity,
model output, or persistent-control loop is modified by this module.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any

import torch


class InputContractError(ValueError):
    """An explicit mode, identity, tensor, unit or time contract was violated."""


class MotionConstraintError(ValueError):
    """The requested replacement was refused without changing the input."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("SET_VELOCITY refused: " + ", ".join(report["request_violations"]))


class ReferenceMode(str, Enum):
    NONE = "NONE"
    SET_VELOCITY = "SET_VELOCITY"


class EntityType(str, Enum):
    PLAYER = "PLAYER"
    BALL = "BALL"


@dataclass(frozen=True)
class MetricUnits:
    position: str
    velocity: str
    time: str
    acceleration: str
    turn_rate: str


@dataclass(frozen=True)
class MotionLimits:
    vmax_m_s: float
    amax_m_s2: float
    turnmax_rad_s: float


@dataclass(frozen=True)
class OperatorResult:
    state: torch.Tensor
    report: dict[str, Any]


def _enum(value: Any, kind: type[Enum], field: str) -> Enum:
    if isinstance(value, kind):
        return value
    if type(value) is not str:
        raise InputContractError(f"{field} requires its enum or an exact canonical string")
    try:
        return kind(value)
    except ValueError as exc:
        raise InputContractError(f"unknown {field}: {value!r}") from exc


def _number(value: Any, field: str, *, positive: bool) -> float:
    if type(value) not in (int, float):
        raise InputContractError(f"{field} must be a finite real Python scalar, not bool")
    try:
        number = float(value)
    except OverflowError as exc:
        raise InputContractError(f"{field} cannot be represented as a finite scalar") from exc
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise InputContractError(f"{field} is outside its finite allowed range")
    return number


def _id_key(value: Any, field: str) -> tuple[type, int | str]:
    if type(value) not in (int, str):
        raise InputContractError(f"{field} must be a strict int or string ID, not bool/float")
    if type(value) is str and (not value or value.strip() != value):
        raise InputContractError(f"{field} cannot be empty or silently whitespace-normalized")
    return type(value), value


def _tensor(value: Any, field: str, *, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise InputContractError(f"{field} must be an explicit torch.Tensor")
    if value.device.type != "cpu" or value.layout != torch.strided:
        raise InputContractError(f"{field} must be a strided CPU tensor; no implicit device copy")
    if value.dtype not in (torch.float32, torch.float64):
        raise InputContractError(f"{field} supports only float32/float64")
    if value.requires_grad:
        raise InputContractError(f"{field} must be detached; this reference has no autograd contract")
    if shape is not None and tuple(value.shape) != shape:
        raise InputContractError(f"{field} must have shape {shape}")
    if not bool(torch.isfinite(value).all().item()):
        raise InputContractError(f"{field} contains NaN or infinity")
    return value


def _velocity(value: torch.Tensor) -> tuple[float, float]:
    return float(value[0].item()), float(value[1].item())


def _motion(
    previous: tuple[float, float] | None,
    current: tuple[float, float],
    dt: float,
) -> dict[str, float | None]:
    speed = math.hypot(*current)
    acceleration = None
    turn_rate = None
    if previous is not None:
        previous_speed = math.hypot(*previous)
        acceleration = math.hypot(current[0] - previous[0], current[1] - previous[1]) / dt
        if previous_speed > 0 and speed > 0:
            # Normalize first so finite large components do not overflow dot/cross.
            ax, ay = previous[0] / previous_speed, previous[1] / previous_speed
            bx, by = current[0] / speed, current[1] / speed
            turn_rate = math.atan2(abs(ax * by - ay * bx), ax * bx + ay * by) / dt
    result = {"speed_m_s": speed, "acceleration_m_s2": acceleration, "turn_rate_rad_s": turn_rate}
    if any(x is not None and not math.isfinite(x) for x in result.values()):
        raise InputContractError("kinematic arithmetic overflow; no approximate replacement was applied")
    return result


def _violations(metrics: dict[str, float | None], limits: MotionLimits) -> list[str]:
    result = []
    if metrics["speed_m_s"] > limits.vmax_m_s:
        result.append("speed_exceeds_vmax")
    if metrics["acceleration_m_s2"] is not None and metrics["acceleration_m_s2"] > limits.amax_m_s2:
        result.append("acceleration_exceeds_amax")
    if metrics["turn_rate_rad_s"] is not None and metrics["turn_rate_rad_s"] > limits.turnmax_rad_s:
        result.append("turn_rate_exceeds_turnmax")
    return result


def apply_reference_operator(
    state: torch.Tensor,
    mode: ReferenceMode | str,
    *,
    entity_ids: Sequence[int | str],
    entity_types: Sequence[EntityType | str],
    ball_entity_id: int | str,
    target_id: int | str,
    absolute_velocity: torch.Tensor | None,
    dt: float,
    model_dt: float,
    units: MetricUnits,
    limits: MotionLimits,
) -> OperatorResult:
    """Validate and copy a single input tensor, without loading/calling a model.

    dt must exactly equal the explicitly supplied model_dt. SET bounds compare
    the old terminal velocity with the requested absolute velocity over dt;
    this is a declared command-interval check, not proof of an executed path.
    Zero-speed start/stop has no defined turn angle; acceleration is still
    checked. Limits are caller declarations, not estimated physiological truth.
    """
    parsed_mode = _enum(mode, ReferenceMode, "mode")
    state = _tensor(state, "state")
    if state.ndim != 3 or state.shape[0] < 1 or state.shape[1] < 2 or state.shape[2] != 4:
        raise InputContractError("state must have shape [H>=1, N>=2, 4]")
    for value, field in ((entity_ids, "entity_ids"), (entity_types, "entity_types")):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise InputContractError(f"{field} must be an explicit sequence")
        if len(value) != state.shape[1]:
            raise InputContractError(f"{field} length must equal the entity axis")
    keys = [_id_key(x, "entity ID") for x in entity_ids]
    if len(set(keys)) != len(keys):
        raise InputContractError("entity IDs must be unique without numeric/string coercion")
    kinds = [_enum(x, EntityType, "entity_type") for x in entity_types]
    ball_key = _id_key(ball_entity_id, "ball_entity_id")
    target_key = _id_key(target_id, "target_id")
    if ball_key not in keys or target_key not in keys:
        raise InputContractError("explicit ball and target IDs must each exist")
    ball_index, target_index = keys.index(ball_key), keys.index(target_key)
    if kinds[ball_index] is not EntityType.BALL or [i for i, x in enumerate(kinds) if x is EntityType.BALL] != [ball_index]:
        raise InputContractError("explicit ball ID and entity types disagree")
    if target_key == ball_key or kinds[target_index] is not EntityType.PLAYER:
        raise InputContractError("target must be an explicitly identified non-ball player")
    dt = _number(dt, "dt", positive=True)
    model_dt = _number(model_dt, "model_dt", positive=True)
    if dt != model_dt:
        raise InputContractError("dt must exactly match caller-declared model_dt")
    if not isinstance(units, MetricUnits) or asdict(units) != {
        "position": "m", "velocity": "m/s", "time": "s", "acceleration": "m/s^2", "turn_rate": "rad/s"
    }:
        raise InputContractError("explicit metric units are required; no implicit conversion")
    if not isinstance(limits, MotionLimits):
        raise InputContractError("explicit MotionLimits are required")
    limits = MotionLimits(
        _number(limits.vmax_m_s, "vmax_m_s", positive=True),
        _number(limits.amax_m_s2, "amax_m_s2", positive=False),
        _number(limits.turnmax_rad_s, "turnmax_rad_s", positive=False),
    )
    old = _velocity(state[-1, target_index, 2:4])
    previous = _velocity(state[-2, target_index, 2:4]) if state.shape[0] > 1 else None
    initial = _motion(previous, old, dt)
    report: dict[str, Any] = {
        "mode": parsed_mode.value,
        "scope": "SINGLE_TERMINAL_INPUT_REPLACEMENT_ONLY_NO_ROLLOUT_OR_CAUSAL_VALIDATION",
        "target_id": target_id,
        "target_index": target_index,
        "explicit_ball_entity_id": ball_entity_id,
        "dtype": str(state.dtype),
        "dt_s": dt,
        "model_dt_s": model_dt,
        "units": asdict(units),
        "caller_limits": asdict(limits),
        "initial_target_metrics": initial,
        "initial_target_violations": _violations(initial, limits),
        "initial_diagnostic_scope": "Selected target terminal speed; previous-to-terminal change if H>=2. Not an all-entity path audit.",
        "old_terminal_velocity_m_s": list(old),
        "requested_absolute_velocity_m_s": None,
        "request_metrics": None,
        "request_violations": [],
        "turn_at_zero_speed": "UNDEFINED_NOT_FILLED_WITH_ZERO; acceleration remains checked",
        "applied": False,
        "modified_slots": [],
        "original_tensor_modified": False,
        "historical_states_modified": False,
        "physical_unit_declaration_independently_verified": False,
        "model_or_rollout_called": False,
    }
    if parsed_mode is ReferenceMode.NONE:
        if absolute_velocity is not None:
            raise InputContractError("NONE requires absolute_velocity=None; no ignored command")
        return OperatorResult(state.clone(), report)
    command = _tensor(absolute_velocity, "absolute_velocity", shape=(2,))
    if command.dtype != state.dtype:
        raise InputContractError("absolute_velocity dtype must equal state dtype; no implicit rounding")
    requested = _velocity(command)
    metrics = _motion(old, requested, dt)
    report["requested_absolute_velocity_m_s"] = list(requested)
    report["request_metrics"] = metrics
    report["request_violations"] = _violations(metrics, limits)
    if report["request_violations"]:
        raise MotionConstraintError(report)
    output = state.clone()
    output[-1, target_index, 2:4] = command
    report["applied"] = True
    report["modified_slots"] = [[int(state.shape[0] - 1), target_index, 2], [int(state.shape[0] - 1), target_index, 3]]
    return OperatorResult(output, report)
