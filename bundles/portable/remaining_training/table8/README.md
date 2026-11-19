# Qfirst 固定 60 头训练复现

此目录沿用任务分工名 `table8`；当前 v10 稿件中 Qfirst 实际位于 **Table 7 最后两行**。真正的 Table 8（native transfer / E-OOF）由邻接 `table8_native` 目录处理。

`code/vendor` 六个科学模块是 2026-10-30 原文件的逐字节副本。便携层只做显式路径重绑定、来源哈希、输出防覆盖、日志和严格核对。`input` 包含固定缓存特征、标签、Table 7 历史参考以及仅供完成后核对的原 Qfirst 产物；输入和代码不得作为输出目录。**代码包不含 `input` 快照**，代码交付仅含 `code/`、说明及元数据；验证输入须单独提供。

```powershell
python -B -X utf8 code/prepare_inputs.py --source-root PATH_TO_AUTHORIZED_ARCHIVE --output-dir NEW_INPUT_DIRECTORY
python -I -B -X utf8 code/reproduce_qfirst.py --input-dir input --output-dir NEW_OUTPUT_DIRECTORY
```

`metadata/input_manifest.json` 含每个原文件相对于授权旧工作区的路径、目标相对路径、SHA-256 与大小；`metadata/portable_config.json` 和显示核对元数据随代码交付。`prepare_inputs.py` 不猜测本机路径，不下载数据，先核对所有原文件再独占创建短路径输入。第二条命令的 `--input-dir` 应填前一步目录。也可加 `--verify-only` 只审计输入齐备性。公开发布数据和获取原始/冻结上游资产的授权仍需单独解决。

运行前目录必须不存在。全部 60 个头从固定随机种子初始化，每头 200 次 full-batch Adam 更新，float64、CPU 单线程。Table 7 参考 checkpoint 只允许读取 initial/scaler/行集合成员用于初始化审计；不读取其训练后参数进行训练。验证用原 Qfirst 权重只在训练、读出、评分全部结束后比较。

验证环境：Python 3.11、NumPy 1.26.4、PyTorch 2.7.0+cu126。当前同环境整套实跑约 44 秒，其中训练约 31 秒。隔离运行记录见 `../validation/table8_relocated/run01/`：60 头、12,000 更新，初始化、所有 checkpoint 数组、Adam 状态、梯度诊断、201 点训练历史、OOF 数组和全部评分 CSV 严格一致；当前稿 Table 7 六行九位小数显示一致。运行成功后 `comparison.json` 是机器可读核对结果，`read_access.json` 记录 NPZ 成员访问。

原负面比较完整保留：D Qfirst 相对 D DirectWith 的 Shot10 logloss 差为 +0.00025241642851409616，配对 seed SD 0.00003367542103757199，0/5 seed 改善。全部 54 比较均保留。这是同六场已暴露开发集的复现，不是新外部验证；未训练世界模型、未新增 50 轮训练、未使用 WC2018 调参。

输入来源目录及原文件 SHA-256 见 `input/input_manifest.json`；科学源代码 SHA-256 见 `code/source_manifest.json`。异常写入 `exception.json`，保留部分输出，禁止覆盖或自动重启。
