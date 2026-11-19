# 当前稿 Table 8：保存世界状态起点的实际重算

此入口复现当前 v10 的 Table 8 全部 11 行。代码包是 `code/`、本说明与元数据；`input/` 是本机验证时另行保存的固定输入快照，约 1.10 GB，**不属于代码包**。移交代码时不要把输入快照当成源代码。输入来源及 SHA-256 清单在 `input/input_manifest.json`；在另一台机器运行时须显式提供同一清单对应的输入目录。

```powershell
python -B -X utf8 code/prepare_inputs.py --source-root PATH_TO_AUTHORIZED_ARCHIVE --output-dir NEW_INPUT_DIRECTORY
python -I -B -X utf8 code/reproduce_native.py --input-dir PATH_TO_FIXED_INPUTS --output-dir NEW_OUTPUT_DIRECTORY
```

小型 `prepare_inputs.py` 按 `metadata/input_manifest.json` 的原工作区相对路径、短目标路径、大小与 SHA-256 重组输入；随包保留 `portable_config.json` 和稿件显示核对元数据。它不硬编码盘符、不自动下载数据，原文件缺失或哈希不符即停止。可用 `--verify-only` 只检查源文件。此入口解决输入重组契约，**不代表原始数据或全部冻结模型已可公开获取**；来源权限与上游模型/数据供给仍是使用前提。

输出目录必须不存在。计算只依赖 NumPy 1.26.4，CPU 单线程；不需 GPU。15 个 `code/vendor/` 科学模块是原研究文件的逐字节副本。新入口替换历史绝对路径和运行许可元数据的 I/O 绑定，运行前核对全部源代码与输入哈希，原 E-OOF 的逐分片/前阶段完整性校验仍直接执行。

实际计算边界：

- 读取 11 个已保存的 E/D/CV 自然世界状态，以原每批 128 行路径重建 X2、Z、Early26、Direct36 特征。
- 用原 90 个冻结外折头和 120 个旧 D 域截距重新计算全部 native 读出及 16 臂评分，保留原负面比较。
- 用 450 个冻结内折头矩阵调用重建 E-OOF logits；用原 18 个训练标签胶囊、原权重/风险集规则，重新拟合 120 个标量截距，每个固定 100 次二分。
- 重新读出 E 的 Identity/OldDAlpha/EOOFAlpha 和冻结 D/CV 历史参考，再生成 Table 8 的 11 臂及全部 30 个固定配对评分。

本机隔离代码、独立工作目录、`python -I` 完整运行用时约 153 秒，证据目录为 `../validation/table8_native01/`。135 个 NPZ 的 1,885 个数组、30 份 100 步截距历史、10 份完整评分 CSV 均与原产物逐字节一致；当前稿 Table 8 的全部 11 行显示一致。`comparison.json`、`completion.json` 和 `read_access.json` 记录核对、范围与逐成员访问。

负面原结果未改变：E Chain OldDAlpha − D Chain OldDAlpha logloss = +0.0027980271866756058，0/5 seed 改善。后续 E-OOF 相对 E 自身旧校准的差 = −0.003193923167375401，5/5 seed 改善；这是另一个问题，不替代前一个负面结果。

范围缺口：自然世界状态作为固定输入，没有重新运行世界模型前向；90 外折头、450 内折头及原 120 截距作为冻结输入，没有重新训练这些 MLP 或重新拟合旧截距。仅后续 E-OOF 的 120 截距确实重新拟合。仍是六场已暴露开发数据，不构成独立盲验、真实 Goal 或足球因果价值验证。没有新 50 轮、外部九场或 WC2018 调参。

一次打包尝试因 Windows 路径长度失败，尚未进入数值计算；旧输入保留于 `bad01/`，摘要在 `../validation/table8_native_assembly_failure01.json`。新输入使用短文件名解决 I/O 问题，未改数学。异常运行会保留输出并写 `exception.json`，禁止覆盖或自动重启。
