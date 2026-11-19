# 最终选择器与层级边界

核对基准是 `CCPV_revision_v10_external9_20261118/content/supplementary_tables.md`，精确 SHA 见 `input_contract.json/current_reference`。旧 v8/v9 数字提取仅用于追溯作者流程，未作为生成输入或当前版本替代品。

## S3

原科学评分源是 `CCPV_mixture_control_pilot_20261112/evaluation/evaluate.py`；原报告源是同目录 `write_report.py`。本入口使用 `report01/seed_fixture_energy_comparisons.csv` 的 `left_value/reference_value`，逐种子先十二场等权，再五种子样本 SD（ddof=1）。逐比赛改善计数先该场五种子等权。窗口数来自保存的 `windows`，保持十二场共 191；M32 是 `evaluation/plan01.json` 已配置轨迹范围。

当前四行的精确选择器为 horizon_s=`2.0`：

| group | comparison | metric |
|---|---|---|
| all_known_history_players_including_targets | H_minus_M | position_energy_U |
| all_known_history_players_including_targets | H_minus_M | velocity_energy_U |
| ball | H_minus_D | position_energy_U |
| ball | H_minus_D | velocity_energy_U |

所有 56 个保存端点同时重聚合并与 `energy_endpoint_summary.csv` 高精度核对，没有按符号挑选种子或比赛。既有 v8 `build_v8.py` 从 `evidence_extract.csv` 的前四行显示上述选择，v10 保留这些单元格。本入口不靠该中间快照生成数字。

真正重建 H/M 轨迹另需 `run_inference.py`、`mixture_kernel.py` 的冻结 D/E 网络及既定历史输入；重算逐窗 Energy 另需原保存 M32 轨迹和真值/支持。这些均未由本入口运行。

## S4

源为 `CCPV_matched_value_objective_pilot_20261118/run_study.py` 的 `score_stage`，保存输入为 `score01/per_fixture.csv`。固定 target=`Y10`，四格 cell 为 OldTrained_Mean、OldTrained_MC、MCTrained_Mean、MCTrained_MC；每个 seed 先六场等权，再五个配对种子的均值与样本 SD。Mean 明确为 OldGPUmean，不替换 CPUmean bridge。

- A_match = MCTrained_MC − OldTrained_MC。
- B_replace = MCTrained_MC − OldTrained_Mean。
- 当前表的最后一行只读取 `unchanged_historical_CV.csv` 中 arm=`CV_Direct_WithEarlyBCE`, target=`q10` 的 `logloss_mean/logloss_seed_SD`。原 score_stage 从 `CCPV_risk_objective_factorial_20261030/evaluation/summary.csv` 原样复制此历史行，本次没有重算其训练或读出。

5,344 窗、235 个 Y10 阳性及剔除的零合格 J03WOH 由 per_fixture 与 `label_lineage.json` 交叉核对。A 的 Brier 反向和仅 3/6 改善比赛均值一并保存。完整模型生成级路线仍包括原 `fit_stage` 与 `readout_stage`，需要既定初值、fold scaler、特征/粒子输入和冻结头；本次不读这些数组。

## S5

源为 `CCPV_recorded_shot_dependence_20261118/analyze.py`。固定 phase=0/1/2/3、G/GM、seed=0…4：

- `phase_fixture_metrics.csv`：六场等权的 GM−G；`phase_coverage.csv`：按半场/比赛累加 candidate、eligible、positive、excluded，不把五个种子重复当五份数据。
- `event_support.csv`：保留所有 142 候选事件身份，113 有支持、29 无支持；`event_metrics.csv`：先每场对已支持事件等权，再六场等权，最后五种子汇总。无支持事件的空值不填零；29×5 个 contrast event-seed NA/指标被保留计数。该正例条件指标不是全体二分类风险或校准。
- 删除诊断从原 `CCPV_ballfree_recorded_shot_pilot_20261118/result01/fixture_metrics.csv` 重算：每次去掉一个评分比赛，对剩余五场分数等权；全部六种删除与原 `deletion_metrics.csv` 对照。不是重新留一训练，也不移除该场原训练历史影响。

重聚合结果另与原 phase/event/deletion 摘要核对。当前 v10 的 phase 3 不利 Brier、事件条件分数、全部删除范围与多事件窗口计数均保留。源预测、phase assignment 和 event links 仍是保存结果输入；不宣称本入口从 raw 重生成它们。

## S7：当前最终来源，不采用旧“二半场未验证”快照

| 当前行/事实 | 最终字段或保存表 |
|---|---|
| 首半场图像、共同键、坐标精确相同及 BAS 时钟 | `CCPV_authorized_gsr_qc_20261118/result01/summary.json` 的 counts、info、BAS |
| 25 个 unmatched 的 22 界外 / 3 内部 | 同目录 `frame_comparison.csv`：raw_finite_ids−matched_keys，按 g 是否位于 1…counts.images 分组 |
| 网格和 714,880/1,482,068 | `measurement_diagnostic/result.json` 的 on_hypothesized_grid_count_xy、zero_displacement_pairs、lag5_pairs；网格协议来自其 plan.json |
| 2,085 字面记录、13/12 SHOT、3/1 GOAL、无共组、899/895/39/4 | `CCPV_qc_event_contract_20261118/result01/summary.json` 与 literal_label_inventory.csv |
| 二半场 67,549 图像、1,355,004/1,484,520 固定零偏移失败 | `CCPV_authorized_gsr_qc_secondhalf_20261118/result01/summary.json` 的 counts.images、counts.above_serialization_tolerance_keys、counts.common_unique_finite_identity_keys |
| 原失败保留且 native clock 是不同检查 | `review/source_scope_decision.json` 的 scope_decision.legacy_raw_adapter.status=`FAIL_UNCHANGED` 与 released_GSR_and_BAS_clock |
| 两半场同发布绑定、合计 2,085 BAS、各 ≤20 ms | 两份 summary.json 的 BAS + `provenance_diagnostic/bas_native_clock_binding.json` 的 BAS_provenance.fixed_release_binding_matches_GSR |

v10 的 `authoring/update_supplement.py` 追加了二半场失败和两半场 native binding 行，并改写版本范围。本入口回到上述最终记录核对数值，不把旧 v9 的 `second_half_production_join=NOT_VERIFIED...` 当最终状态。27 帧关系仅作为明确未应用的 post hoc 诊断保留，不能使原失败变 PASS。S7 是保存 QC 结果的核对与显示，不是重新流式验证大文件。
