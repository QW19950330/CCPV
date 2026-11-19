"""Declarative allowlists and future stage gates; no payload-reading entrypoint."""
from pathlib import Path
import hashlib,json

HISTORY_MEMBERS={'states':'history_states','valid':'history_fullstate_valid',
 'position_valid':'position_valid','real_observation_valid':'history_real_observation_valid',
 'observation_age_s':'last_source_observation_age_s','teams':'teams','entity_ids':'entity_ids',
 'anchor_id':'anchor_id','fixed_team_A_index':'fixed_team_A_index','pitch_length_m':'pitch_length_m',
 'pitch_width_m':'pitch_width_m','native_to_model_sign':'native_to_model_sign',
 'supported':'supported','unsupported_reason':'unsupported_reason'}
PREFIX_MEMBERS=('X0','Z_cv','anchor_id','match_id','fixed_team_A','eligible_prefix','eligible_recorded10')
WORLD_FUNCTIONAL_MEMBERS=('state_dict','cell','seed','epoch','canonical_model')
HEAD_MEMBERS=('hidden_weight','hidden_bias','output_weight','output_bias',
 'scaler_mean','scaler_scale','scaler_completely_absent_training_feature',
 'raw_dimension','parameter_count','head','training_seed','output_targets')
ALPHA_MEMBERS=('alpha','head','seed','output_index','outer_held_match')
SCORE_MEMBERS=('match_id','anchor_id','eligible_recorded10','y_early2','y_late8','y_10')
STAGES=('trajectory_features','readout','score')
FORBIDDEN_GENERATION_INPUTS=('y_early2','y_late8','y_10','oracle_h2','X2_obs','future_xy','future_valid')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def root_stage_guard(authorization_path,implementation_lock_path,stage,prerequisite_record=None):
    """Future execution wrappers must call this before any actual payload read.

    Torch CP permission is explicitly whole-container deserialization; it is
    not represented as NPZ-style selective physical member access.
    """
    if stage not in STAGES:raise ValueError('Declared stage only')
    a=json.loads(Path(authorization_path).read_text(encoding='utf-8-sig'))
    locksha=sha(implementation_lock_path)
    if a.get('status')!='AUTHORIZED_FROZEN_WORLD_NATIVE_OUTCOME_TRANSFER' or a.get('implementation_lock_sha256')!=locksha or stage not in a.get('allowed_stages',[]):
        raise PermissionError('Separate root implementation-lock/stage authorization required')
    if stage=='trajectory_features' and a.get('whole_torch_checkpoint_container_deserialization') is not True:
        raise PermissionError('Whole torch container access was not authorized')
    if stage!='trajectory_features':
        if not prerequisite_record or sha(prerequisite_record['path'])!=prerequisite_record['sha256']:
            raise PermissionError('Exact preceding all-feature/readout freeze required')
        p=json.loads(Path(prerequisite_record['path']).read_text(encoding='utf-8-sig'))
        expected={'readout':'ALL11_TRAJECTORIES_AND_FEATURES_FROZEN_NO_Q_OR_LABELS',
                  'score':'ALL12_ARMS5SEEDS5344_READOUTS_FROZEN_BEFORE_LABELS'}[stage]
        if p.get('status')!=expected or p.get('implementation_lock_sha256')!=locksha:
            raise PermissionError('Wrong preceding freeze or lock lineage')
    return a
