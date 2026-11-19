# 记录射门时钟：原始来源重建入口

本目录是本地待发布的代码与固定协议。它重建已经曝光的七场直接时钟证据和其中三场作者标注时钟证据；不是新的盲外验，不训练、不预测、不调参，也不代表完整 CCPV 或物理触球时间得到验证。代码许可和公开发布仍须作者确认。包中没有原始文件、逐事件记录、球员身份、权重或凭据。

## 输入与命令

使用 Python 3.11，依赖见 `requirements.txt`。所有数据必须由使用者自行取得授权；代码不下载数据。`--data-root` 是明确指定的数据根目录，不依赖当前工作目录或旧工作区位置。该根目录下需保持 `fixed_inputs.json` 的 27 个相对文件名、字节数和 SHA-256：七场各一份 events、matchinformation、positions XML，以及作者三场各一份 gt 与 unsynced parquet。

固定时钟冲突案例的选择器单独提供，不能放入公开代码包。它只含 `match_id`、`event_id`、`calculated_frame` 三字段，对应历史已指定案例；规范 JSON 的 SHA-256 必须匹配 `explicit_inputs.json`。此选择器只指定固定身份和帧，不提供坐标、原始时间或残差；这些证据全部从本次原始 XML 取得。不得换成其他事件或选择更好的偏移。

在本目录运行（变量由使用者设置；输出目录必须尚不存在）：

```powershell
python -B .\pipeline.py rebuild --data-root "$InputRoot" --explicit-selector "$PrivateSelector" --output "$NewOutput"
```

重建端不读取旧 shot_mapping、author_shots、event_clocks 或窗口 NPZ。历史 source/download manifest 的固定文件身份已投影到 `fixed_inputs.json`，保留原 manifest 哈希；不用原 manifest 内的本机绝对路径。小输入先验证 SHA，positions 在原解析循环中完整流式验证 SHA，parquet 同时验证原 Git blob SHA-1。

生成后可与另行授权的原始证据目录比较：

```powershell
python -B .\pipeline.py compare --reference-root "$EvidenceRoot" --run-dir "$NewOutput" --output "$NewComparisonJson"
```

比较原件只用于生成之后的核对，不参与重建或改进规则。原件相对路径与哈希也在配置中。所有输出 writer 使用独占新建；已有输出拒绝覆盖。建议把运行目录、日志和私有选择器放在代码包外。

## 输出及原逻辑边界

| 输出 | 含义 |
|---|---|
| `direct/shot_mapping.csv` | 七场最终原 ShotAtGoal 的原 CalculatedFrame/CalculatedTimestamp、BALL/actor 精确帧证据；保留原未映射项 |
| `author/author_shots.csv` | 三场所有作者 shot 的双向唯一精确签名和标注帧核验 |
| `author/raw_shots.csv` | 原始射门身份和作者联系分别保存 |
| `author/all_author_rows_inventory.csv` | 所有作者记录的原 inventory；不声称已完成非射门事件交叉匹配 |
| `author/all13_original_unmapped.csv` | 原全部未映射记录保留，作者时钟单列 |
| `explicit_clock_case.json` | 固定原 CalculatedFrame 及 ±2 帧上下文、actor 帧范围；只有 source_pin.path 改为相对路径 |
| `explicit_clock_residual.json` | 同一指定帧 raw T 减 literal CalculatedTimestamp 的带符号残差；邻帧不参与接受决定 |
| `opening.json` / `completion.json` | 实际 PID、版本、输入身份、完成范围和读取量 |

`direct_logic.py` 和 `author_logic.py` 保留原函数及运行片段文本，19 段对应关系见 `source_fragments.json`。直接映射只查询射门的 CalculatedFrame；旧窗口锚点的额外 actor 查询和旧窗口覆盖输出不影响这些查询，因此未带入 2009 窗口 NPZ。作者逐场逻辑与 positions 循环照原文保留，旧 `original_mapping` 诊断字段改读本次刚生成的直接映射。

固定冲突 observer 在直接解析器清理 XML 元素前只读记录，不改原解析循环，不增加第三次读取该场。它保留原调查脚本的固定帧 ±2、actor 范围计算；不是近邻匹配或标签修复。

直接循环完整扫描七场 positions，作者循环再扫描其中三场；这是保留两个原循环的代价。独立输入共约 2.63 GB，positions 累计读取 3,804,539,407 字节（7+3 次）。运行时间取决于磁盘和 CPU，实际时间以完成记录为准。输出为小型 CSV/JSON；私有逐事件输出不属于公开代码包。

本入口不生成旧 `window_coverage.csv`、最终 `event_clocks.csv` 或 11,542 行训练标签；这些属于下游事件有效性/情景聚合和 PLAYER 时钟覆盖入口。旧 direct 的 BALL/actor 有限 XY 条件、作者精确纳秒匹配、原 unknown 和已知时钟冲突均未改变。不能据此声称官方事件完全捕捉了物理事件、作者标注是视频真值、模型改善真实进球或跑位因果收益。

实际成功与否须读取独立验证目录的运行与逐字段比较记录；不能仅凭代码存在就声称原始链已经运行。
