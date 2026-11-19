# Table 9 / 10：冻结 q 到完整确定性 game

本包实际重算原自然参考与维持 v0 两套确定性 game，保留全部 family、world、seed、coalition 和分母。它只读取两份固定 `frozen_readouts.npz`，不读取事件标签、模型权重或世界轨迹，不拟合、不执行价值头或世界模型，也不使用 D1 后续随机结果替换旧表。

## 输入与安全输出

使用 Python 3.11、NumPy 1.26.4。`input_contract.json` 固定两份授权派生 readout 的相对路径、字节数与 SHA；总计 4,774,835 字节。使用者须自行合法取得它们；代码包不附 readout、逐场景身份或数值数据。其余 reference_outputs 只在生成后的核对阶段使用，不参与 game 计算。

**所有数值输出和核对 receipt 必须与授权输入源工程树分开，并置于代码包外。** `data-root` 可以是多个项目的公共父目录；不能因此把输出放进任何原源项目子目录。当前验证输出位于新的 `CCPV_neural_reproduction_delivery_20261119/validation/table9_10/run01/`，与两份旧 readout 及当前 v10 源工程分开。

推荐使用独立执行层保护入口，先检查或直接运行：

```powershell
python -B .\safe_run.py --data-root "$InputRoot" --output "$NewOutputDir" --check-only
python -B .\safe_run.py --data-root "$InputRoot" --output "$NewOutputDir"
```

`safe_run.py` 拒绝向任何绑定的旧输入/代码/结果/文稿项目树及公开代码包写输出，拒绝已有目录。它不导入 NumPy 或读取科学 payload；通过后调用已实跑的原 `pipeline.py run`，数学模块未因保护层而改动。路径保护单独做了拒绝验证，没有为该检查重跑全部数学。原低层入口保留以对应已执行版本：

```powershell
python -B .\pipeline.py run --data-root "$InputRoot" --output "$NewOutputDir"
python -B .\pipeline.py verify --reference-root "$OriginalEvidenceRoot" --run-dir "$NewOutputDir" --manuscript "$CurrentV10Markdown" --output "$NewVerificationReceipt"
```

核对 receipt 也必须位于独立新验证目录。所有 writer 都是独占新建，旧文件不覆盖。示例变量均由使用者显式设置；没有原 C 盘路径或旧工作区默认值。

## 完整计算与支持范围

- 自然参考：2 个价值 family × 5 seeds × 4 worlds × 2 durations × 768 场景，61,440 行完整 game。9 个路径的读出和全部预声明参考对比、actor-order、场景排序及 paired sensitivity 均保留。重新从原 supported 数组生成首步 683、持续 680、两者共同 680、全 9 路径 680 的掩码。
- 维持 v0：2 families × 5 seeds × 4 worlds × 768 场景，30,720 行 game，由四个真实 v0 coalition 读出计算。其 common mask 独立使用本组 supported/finite q 得到 680；新生成的自然 mask 仅用于覆盖交集核对，不强迫选择该 680 子集。
- 两套均保留 768 个计划场景、727 个 prefix-supported 场景、十二场比赛；不按 q 的符号、工程 guard 或收益筛选。共同数据按场内均值再十二场等权，五 seeds 的样本 SD 使用 ddof=1。CV 是一份世界轨迹、五份价值种子，不当五个独立数据集。
- Table 9 使用 D_coupled_ar 的固定自然/重复 v0 joint 对比；自然面板的重复 v0 仅是 joint 敏感性，未伪造混合 coalition。Table 10 来自真实四 coalition 的独立维持 v0 面板，显示均值与 SD ×10⁶；interaction 是完整二阶差分。

`game_math.py` 提取原 value_math/shared_value_math 的 exact_game、rank、actor 算术及原 exact_v0_game；两份原共享数学源字节相同。`natural_games.py` 与 `v0_games.py` 保留原完整数学循环，仅移除旧授权/隐式路径入口、改为显式输入和输出。`natural_report.py` 保留原 summarize_value_games 的全部报告计算。函数原文对应及适配范围见 `source_fragments.json`；不宣称整个原脚本字节一致。

## 本轮实跑与精确核对

实际计算 PID 32372，约 4.406 s。25 份原输出全部核对通过：21 CSV 共 2,950,715 格（含表头）逐字段精确且字节相同；两份 masks NPZ 的成员、dtype、shape、数组 bytes 完全相同；两份 summary JSON 仅排除 utc，其他字段的类型与值完全相同。当前 v10 Table 9/10 共 75 格（含表头）精确一致。

结果路径与完整来源哈希见独立 `validation/table9_10/completion.json` 和 `comparison01.json`；`run01/opening.json` 记录实际执行身份和代码哈希。代码尚未公开发布，许可及派生输入/输出分享范围待作者确认。

本次闭合的是 **冻结 q → 确定性 game → 当前表**。产生 q 的原头读出、特征、世界轨迹及训练不由本入口复现；也不证明生成状态校准、真实进球、跑位因果收益或物理球员信用。旧结果和全部不利/未定义值保持原样。
