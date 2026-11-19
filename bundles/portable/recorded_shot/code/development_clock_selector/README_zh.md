# 固定时钟案例选择器：独立补充入口

本入口解决 `development_clock_sources` 原入口需要私有选择器的问题，不更改其已冻结 12 个文件，也不读取旧逐事件证据来生成选择器。

只需同一授权数据根目录下的固定 J03WOH events XML（635,747 字节）。代码校验原文件 SHA，然后保留每个 EventId 的文档最终版本，在固定比赛中精确匹配 literal `CalculatedFrame="158639"`、非 Delete、恰一个 ShotAtGoal。要求唯一命中，生成 match_id/event_id/calculated_frame 三字段，并核对既有规范选择器 SHA。该规则直接来自原 `investigate_unmapped.py` 已指定案例；不查找更佳案例、偏移或最近帧，不使用球员、结果、时间戳或残差决定选择，不推测半场。

使用 Python 3.11 与 lxml 4.9.3。先创建代码包外的私有输出目录，再运行：

```powershell
python -B .\prepare_selector.py --data-root "$InputRoot" --output "$PrivateSelector" --receipt "$NewReceipt"
```

需要核对历史已运行选择器时可额外传入 `--compare-selector "$ExistingPrivateSelector"`。它只在从原 XML 生成之后比较，不参与生成。输出均独占新建、拒绝覆盖。

随后把生成的 `$PrivateSelector` 传给原冻结入口：

```powershell
python -B "$ClockSourcesPackage\pipeline.py" rebuild --data-root "$InputRoot" --explicit-selector "$PrivateSelector" --output "$NewClockEvidenceOutput"
```

公开代码只包含固定 match/frame、文件身份与选择器哈希，不包含 event ID、player ID 或逐事件记录。生成的选择器和 receipt 留在私有验证目录。本补充无需读取 positions、不训练、不重跑完整 raw 时钟链，也不改变原 +40 ms 冲突。原始数据须自行取得授权；代码许可与发布仍待作者确认。
