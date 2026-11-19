当前状态：下文主体为封存前元数据审计；其后已收到root在协议seal后的明确授权，完成仅前缀提取，见文末阶段二与completion.json。没有模型前向。

# 原 SkillCorner 四场 source-test 路线元数据审计

结论：这条路线具备现成10Hz、23槽、球与来源球员ID的缓存基础，比重新适配外部25Hz来源直接；但四场已经曝光，不能称新的独立盲测。可实施一次预锁、冻结D五种子与CV的探索性持续性诊断。当前工作仅检查已有元数据、审计与代码，未打开任何NPZ成员、raw tracking、事件标签或模型。

|固定test ID|既有缓存窗数|已明确旧评分窗数|当前角色|
|---|---:|---:|---|
|1927964|1308|64|已曝光source-test|
|2006363|1261|64|已曝光source-test|
|2015213|1312|64|已曝光source-test|
|2017461|1111|64|已曝光source-test|

四场总4992窗；旧评分子集共256窗。不能由剩余4736窗未在这一次抽样中评分，推出它们或所属比赛从未被接触。五种子也不是五个新数据集。

## 曝光证据

1. skillcorner_split.json在2026-10-22下载/模型评估前冻结SHA256排序12 train /4 validation /4 test，其prior_model_exposure=false是该时间点的记录。
2. CCPV_confirmation_20261023/data/reserved_test_manifest.json明确四场各64窗在旧模型冻结后提取；其中test_metrics_viewed=false仅描述提取阶段。
3. 同研究runs/reserved_test_confirmation_v1/status.json已是SUMMARIZED、顶层test_metrics_viewed=true；2026-10-23 17:44:17至17:46:09 UTC完成测试，20个旧模型cell，10240评分记录。10240不是独立窗口数。本审计未打开旧结果CSV、误差数值或训练结果。
4. 后续1024开发窗manifest的test_metrics_read=false仅约束该轮开发操作；不能覆盖前一天已存在的测试曝光。其npz_caveat还说明整成员解压会触及未保留字节。对齐数据库审计亦声明test事件记录已被纳入来源审计数据库，不能说标签从来不曾准备。

## 已有数据与分窗契约

全20场skillcorner_10hz.npz存在（116411793字节）；旧reserved_test_256.npz存在（1183629字节）。4场tracking、match metadata、events与phases原文件均存在且stat尺寸与下载manifest可比。此次只stat，不重新读/哈希科学载荷；原SHA及逐文件状态见metadata_audit.json。

实际cache和prepare_skillcorner.py是非重叠30帧块、10历史+20未来、10Hz、同period内连续source frame；split早期stride10与实际stride30不一致，应以执行cache与代码为准并披露差异。实体为0–10 home、11–21 away、22 ball；球ID是显式ball_data语义键“ball”，不是唯一数字ID。球员ID为来源player ID；entity_ids、match_ids、clip_ids、source_period、source_frame_ids、frame_times_s与teams由原保存代码提供。ID需按每场roster独立核对，不能臆定攻守或控球方。

旧窗按完整30帧内ID并集分槽，并要求history与future各至少一个valid状态后才纳入，因此这是既有回顾性缓存人口；新prefix读取不会把它变成完全前缀采样人口。坐标按原场地长宽映射到105×68；valid要求当前与相邻前帧支持因果速度，position_valid是原provider detected位置支持。位置only时states被清零，不能直接把states.xy当完整position_valid真值。

## 必须保留的技术核对

原速度循环enumerate(homeIDs+awayIDs)与away物理offset11存在源代码级不一致条件：homeID数H<11时，第一历史帧的部分away槽可能以错误player ID取previous；循环末端部分away物理槽又可能未被访问，valid被压低。t>0对已经访问的槽仍按本槽位置差分，不能因此宣称所有H<11的全部有效数值均错。此次未解码test，所以影响窗数与实际valid支持未知。

已存position_truth_v2只对16个开发比赛恢复position-only xy，明确training_changed=false；它不是test速度修复。限定元数据/QA搜索未找到专门修复证据，不能把“未找到”写成“绝未修复”。后续应逐窗记录H、潜在错配槽、首历史帧这些槽的valid、未遍历槽的position_valid；保留全部4992行并明确技术unsupported，不静默修原cache。

## 一次性诊断的执行前提

- 冻结完整4992原test_indices顺序、四场、D的五个原epoch4/96-update final与单个CV；原数据/模型保持不变。
- 先seal协议和提取代码。复用full_intervention/data_prepare.selected仅将选定行前10帧转为数组；ZIP跳过仍可能inflate后续字节，须如实记录。来源索引、来源帧、ID/球队核验与每类排除分母完整保存。
- 继续原home/team0、t0 full-valid、字符串词典序前两位actor规则。新命令和缺失支持规则只用prefix；不以future真值、模型响应或表现选择行。
- 先对流式reader、registry和槽影响标记做手工合成检查；随后才允许新prefix抽取。未通过来源hash、ID、时间/shape或槽位检查者保留技术状态，不能替换好窗。
- 后续模型响应需原FP32/MATH、单线程、固定命令/算子与路径、不拟合；保持no-op、不可变输入和支持不增生校验。
- 若另外评分natural ADE/FDE，应在预测固定后独立恢复并核对provider-detected位置真值；不能用旧zeroed position-only值替代。四场等权，seed SD仅描述模型初始化变异；结论限于已曝光source-test人口的冻结模型响应，不是真实反事实或Goal效果。

## 新协议只读核对

已核对skillcorner_response_protocol.json（本次读取SHA见metadata_audit.json）：四场、4992窗、stride30说明、已曝光与回顾性人口、D epoch4/96更新及语义球键均符合来源证据。需按上述细分记录H<11影响，不将H<11本身直接等同全部有效输入错误。该协议仍由root负责seal；本审计未替代seal、未执行新实验。

机器证据及18份来源SHA：metadata_audit.json、evidence_SHA256SUMS.txt。无新训练、前向或标签读取。

## 阶段二：协议seal后的授权前缀提取

root封存协议SHA为7c6eead0eaafa99466eaa348e8261fa3ab5675941afeaac8d128127e514598f0。提取器SHA为1b2567a95cb1528bd5cc6fe0ce88454df4c966e75857c3466672635c74a98e09，8项合成检查通过后，执行全部4992原test_indices的前10帧流式提取与roster核对。输出位于父目录skillcorner_prepared/test_prefixes.npz（SHA a14904ed5a215805ce390fa0885b99e2e49752da096d47b119ba0019b0b8b656），manifest逐数组列shape/dtype/SHA。

四场actor支持分别为1303/1308、1249/1261、1302/1312、1091/1111；合计4945支持、47保留unsupported。所有窗home ID数为11，源代码条件H<11没有触发；technical_unsupported为4992位全false。该结论来自此次授权的ID/前缀支持检查，不是先前只读审计的推测。没有修补或替换原cache。

已核对完整source archive SHA和4场roster SHA；只将前10帧states/valid/position_valid、帧时间/ID及来源元数据转为数组。跳过ZIP字节仍可能inflate完整流，因此不声称未触及任何future字节；future数值未转为数组。未读raw tracking、事件标签，未加载模型或进行前向。原metadata_audit.json保留封存前阶段记录；最终结果以completion.json及skillcorner_prepared/manifest.json为准。
