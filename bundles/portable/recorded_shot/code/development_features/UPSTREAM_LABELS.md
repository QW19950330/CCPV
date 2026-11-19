# 上游候选与标签的实际入口和闭合缺口

本次已实现的是 history 缓存到四组运动特征，以及已审 event_clocks 缓存到原 ledger 标签的回放。没有把原始事件到标签全链称为已复现。下面的历史入口是已存在、可追溯的源代码，不代表它们已在本交付目录中移植或重新运行。

## 候选格

实际入口是 CCPV_context_capture_audit_20261025/idsse/scan_ball_metadata.py 的 grid、scan 和 run；历史命令为 run --authorize-raw-ball-metadata-scan。它流式取得七场 positions XML 的 BALL 时间元数据，在每半场以 origin+24 开始，每75源帧取一次，直到该半场最大N，生成 ball_results/prefix_grid.csv.gz 共13,372行。这不是由未来预测分数选出的格，但采集范围来自BALL元数据。

CCPV_ballfree_recorded_shot_pilot_20261118/study.py 的 run 枚举该完整七场文件，保留原始全局 grid_index 后只取事先指定六场，得到11,542候选。它只取 match_id、period、anchor_source_N，丢弃旧资格、team/live/BALL时钟和旧标签字段。不能重新只生成六场格后从零编号；这会改变原始 grid_index。

## 记录射门和时钟

实际缓存生成器是 CCPV_shot_event_clock_audit_20261025/idsse/audit_events.py，历史命令 run --authorize-existing-event-audit。它的 discover_match 按原始 events XML 的文档修订顺序识别最终 ShotAtGoal、删除/改写、取消或多节点未知，并按 matchinformation roster 校验队伍与球员；不是把文本中的所有 ShotAtGoal 简单计作有效事件。

该生成器生成两种时钟情景。这里固定采用 AUTHOR3_PLUS_DIRECT_OTHER4：
- J03WMX、J03WN1、J03WPY 使用同一事件的作者标注帧时钟。
- 其他历史四场保留直接 CalculatedFrame/CalculatedTimestamp 精确源记录证明；最终开发六场剔除的第七场 J03WOH 不进入本标签子集。
- 作者情景不是物理触球时间真值，未知事件和已知40 ms冲突不得靠最近邻或事后偏移修补。

直接证明的实际上游是 CCPV_anchor_loss_calibration_sync_20261025/sync/audit_future_coverage.py 的 shot_mapping 和 run（run --authorize-readonly-feasibility），产出 coverage/shot_mapping.csv 和 window_coverage.csv。它需要原事件、metadata、positions中的BALL与actor时钟，以及其固定协议/父锁。旧完整性检查曾混合BALL有限性；当前事件审计明确保留该限制，不自动把其未知改成有效。

作者三场的实际上游是 CCPV_event_mapping_outcome_pilot_20261025/mapping/audit_author_crosswalk.py 的 strict_key、unique_link、period_origin 和 run（run --authorize-source-crosswalk）。需要六个ELASTIC parquet：benchmark/gt/ 与 benchmark/unsynced/ 下的 J03WMX、J03WN1、J03WPY；还需要三场原始metadata/events/positions、下载manifest、source manifest和旧shot_mapping。它通过半场、演员、射门子类型与unsynced UTC的唯一精确签名联系原EventId，再核验标注frame及源帧时间，生成 author_crosswalk/author_shots.csv。没有采用最近匹配或结果筛选。

audit_events.py 的 bindings/lineage/authors 继续依赖上述受哈希约束的 coverage、author_crosswalk、execution locks，以及 explicit_clock_case.json。精确文件名及哈希列于 upstream_input_inventory.json；该清单不复制其逐事件或逐球员内容。

## 最终11,542行标签

CCPV_ballfree_recorded_shot_pilot_20261118/study.py 的 anyshot 与 run 是最终规则：
- 从 event_clocks.csv 只选固定情景与六场。
- 同半场或未知半场存在任何未接受时钟/效度事件时，标签未知。
- 只有从当前N到N+250的全部PLAYER时间满足同半场40 ms时钟，且没有未知事件时，标签才是两队任一接受射门是否位于 (t0,t0+10s]。
- 窗口资格为原前13列支持完整且 y>=0；未知不是阴性。球员未来坐标不作输入或筛选。
- audit_labels.py 本次只重放上述 anyshot，复用 ledger 的 player_support 和 complete_recorded_clock10，并逐行核对 y、eligible、unknown_shot_ids。

## 精确尚未闭合的部分

1. 本包的 raw PLAYER positions/matchinformation 入口未在本轮实际扫描六份大XML；需用授权输入执行并通过同一8数组bit pin。它也不重建事件标签。
2. 原13,372行候选格生成器尚未移植成完全相对路径、可在新目录执行的发布入口；需要七场原BALL时间范围以维持全局grid_index，并保留当前六场筛选规则。
3. event_clocks生成器及direct/author两层证明链尚未移植和从原始文件重跑。仅给六场positions+metadata不足以生成作者三场标签；还缺原events、三对ELASTIC parquet、原source/download manifests、精确帧证明及固定协议。
4. 可搬迁发布版本需为历史写死ROOT、目标目录与授权锁提供保留来源哈希的独立适配；不得直接运行旧脚本覆盖其r/coverage/author_crosswalk目录，也不得绕过原时钟未知。
5. 完成上述技术重建仍不等于物理事件完整性、独立触球同步、真实跑位因果收益或完整CCPV有效性。许可、可再分发衍生记录范围及持久公开版本仍须作者确认。

本次没有下载、修改或打包这些原始文件，未重新拟合任何模型。
