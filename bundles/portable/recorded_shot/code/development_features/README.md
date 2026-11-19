# 开发集运动特征重建

本地待发布代码，从六场已保存的历史坐标缓存重建原 11,542 个候选窗的四组特征：native/grid × 200 ms/OLS1s。仅 9,483 个既定窗口保留原训练资格。代码不拟合模型，不改变标签、资格、采样、缺失规则或量化参数。

## 已执行的范围

本目录提供两个特征入口。history 模式输入原六个 history NPZ 和已审 all_anchor_ledger.csv；raw 模式输入同一 ledger 及六场 positions/matchinformation XML。所有输入采用固定字节哈希，输出必须是新目录，所有文件独占创建。raw 模式入口已提供，但本次没有重新扫描约 2.27 GB positions XML；不能把它称为本次完成的 raw 重放。

数学函数由已执行代码原文提取，片段和源文件哈希见 source_provenance.json。history 模式使用原六点支持和两个 motion variants 的原字段次序。各数组预期 dtype、shape、C-order 字节哈希见 input_contract.json；不允许用容差通过。保存 NPZ 使用原 np.savez_compressed 调用顺序。原 features.npz 字节 pin 保留，容器若不同必须显式报告，不会静默放松下游训练入口。

原数值环境：Python 3.11.5、NumPy 1.26.4、float64、单线程。XML 路径使用 lxml。其他平台/NumPy 的最后位行为不能预先保证；程序会实际检查并停止不匹配。

## 命令

取得明确获准使用的输入后：

    python -B build_features.py history --history-root /authorized/past_history --ledger /authorized/all_anchor_ledger.csv --reference-features /authorized/features.npz --output /private/new_features
    python -B verify_features.py --original /authorized/features.npz --rebuilt /private/new_features/features.npz --output /private/new_feature_comparison.json
    python -B audit_labels.py --ledger /authorized/all_anchor_ledger.csv --event-clocks /authorized/event_clocks.csv --output /private/new_label_check

reference-features 只用于核验，不用于生成 X；可省略，生成值仍逐数组校验固化的字节摘要。

已明确获准读取十二个原始 XML 时，可运行：

    python -B build_features.py raw --raw-root /authorized/idsse_raw --authorize-raw-xml --ledger /authorized/all_anchor_ledger.csv --reference-features /authorized/features.npz --output /private/new_raw_features

raw 模式用源 roster 的 PLAYER 身份，流式读取 PLAYER N/T，数值 XY 只取固定六个过去/当前时点。它核对原 PLAYER t0、当前支持和后续 251 帧时钟完整性，但 y 和 unknown 事件状态仍来自同一已审 ledger。只扫描六场已公开开发范围；不加载第七场或外九场，不访问网络。运行输出含受限逐窗数组/身份，必须留在用户授权的私有位置。

## 不变的定义

- 固定偏移 [-25,-20,-15,-10,-5,0]，25 Hz。PLAYER identity、半场、t0+offset×40,000,000 ns、frame clock 和有限 XY 同时满足才具有完整运动历史；重复/缺失不补值。
- 当前几何使用全部当前有效球员，不要求六点运动有效。200 ms 速度只在共同六点支持上由末两点计算；OLS 使用过去一秒六点，中心化时间和 0.7 分母保持原表达式。
- grid 为原点零、步长 (1.05,0.68) m、np.rint ties-to-even；对全部六点与当前几何量化，不裁剪。径向 d=0 保持缺失。
- 几何、数量、缺失指示及运动统计保持原 16 列和顺序；不按特征或结果重新筛窗。
- audit_labels 仅从已审 event_clocks.csv 回放原 anyshot 函数。它不验证原始事件时钟、物理事件完整性或映射真实性。

## 原始标签上游与缺口

详见 UPSTREAM_LABELS.md 和 upstream_input_inventory.json。上游需要固定候选格、原始 DFL events/metadata/positions、三场 ELASTIC gt/unsynced 标注及历史精确时钟交叉表。单靠 positions+metadata 无法重建原 AUTHOR3 标签。

本目录不包含逐窗缓存、标签、球员 ID 数据、权重、令牌或本机绝对路径。SHA256 和文件名不是数据再分发许可。本地代码尚未公开发布；作者代码许可与第三方输入/衍生物使用范围仍待落实。
