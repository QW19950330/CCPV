CCPV 公共运行绑定候选（仅本地；未发布；许可仍待作者确认）

目的与证据
本目录将明确的公共衍生元数据绑定到原便携入口。原52份Python源码逐字节不变；15份代码侧JSON采用显式新SHA/路径绑定。原ZIP、原guard、450份history/control原件未改，不把新文件伪称旧SHA。
原65份元数据中44份被当前8组件契约消费、21份为当前入口不消费的来源记录。先前57份guard/catalog分类中实际41份被当前入口消费，16份不消费；其余8份provenance中3份被SHA guard消费。消费表示入口检查该文件，不表示每个字段是科学变量。逐项见metadata_consumer_matrix.json。

实际已完成
1. table5/table6/supplementary/table7/qfirst/native/s4/s5八入口全PASS。Table5核53文件；Table6核137文件、63case和334原冻结范围；Table7额外通过120历史replay_registry记录。T7/qfirst/native/S4只执行原函数至首次输出目录创建之前，未调用fit/readout；这一限定不认证整条训练复现。
2. S5实际保存统计重汇总：13CSV、数值summary和中文报告15文件与原冻结参考逐字节相同。supplementary的S3/S4/S5/S7等8输出与原代码+原输入基线逐字节相同。未重训、未产生新的研究预测。
3. 1200份明确引用NPZ：1110份安全非pickle对象保留原字节，单独在exact_replay兼容ZIP；90份仅old_DDirect_initial_checkpoint来源字符串改为公共相对身份，共4230字段的shape/dtype及其余4140数组的字节核验通过。三类原头的初始/最终共6组合成读出相同；12次技术predict函数调用不是新研究实验。这些完整引用含初始化、Adam、训练行/权重/标签，区别于推断-only权重包，不能相互冒充。
4. Table6所缺132条精确来源全存在且原SHA一致。新增独立输入ZIP含63份原字节保存轨迹NPZ、67份路径/依赖SHA衍生JSON及原CSV/代码；从两个清洁ZIP、无原工作区fallback，实际重建137文件并通过原metadata guard。
5. 原结果ZIP实际组合时发现65清单外的007_training_audit.csv，85行的folder/best_checkpoint/last_checkpoint三个来源列完成相对化；所有其他单元格、行序和字段保留。原失败部分包明确为LOCAL_ONLY_FAILED，不可上传。

实际ZIP与尚未生成的大包
clean_archive_map.json列3个已实际组合并逐成员重新核验的清洁ZIP：结果、支持材料、SkillCorner小型输入。另有exact_replay NPZ兼容ZIP和Table6保存输入ZIP。大derived中其余5包只完成原归档SHA及替换成员计划核验，本轮没有重复生成数GB的clean ZIP；compose_public_archive.py提供可执行组合路线。原dirty ZIP+overlay不构成clean ZIP，原dirty ZIP必须留本地。
每个组合ZIP的PUBLIC_CONTENT_MANIFEST.json是当前成员身份；包内原manifest保留为历史来源记录，不能拿其旧成员SHA验证新衍生文件。provider原通知保留。两份新的Table7 lock/freeze副本明示新SHA，不修改原控制载荷。

最小运行命令（Python3.11.5 / NumPy1.26.4；原guard依赖按各组件原README）
在本目录执行；RELEASE_ROOT是明确提供的公共归档布局根，不是原机器绝对路径。示例仅运行前置校验。
python -B compose_inputs.py --candidate . --archive-root RELEASE_ROOT --public-archive-map clean_archive_map.json --components table6 --output ../NEW_INPUTS
python -B check_entry.py --candidate . --inputs ../NEW_INPUTS --component table6 --output ../NEW_CHECK_TABLE6
其它组件选table5/supplementary/table7/qfirst/native/s4/s5；必须先具备其所有clean归档。输入、代码和输出分开；输出必须不存在。当前clean_archive_map只有3包，缺失的大包不会自动回退到dirty ZIP。

逐大包组合（作者本地的明确原归档，不下载、不训练）
python -B compose_public_archive.py --candidate . --archive-root LOCAL_SOURCE_ROOT --archive ORIGINAL_ARCHIVE_RELATIVE_ID
上一命令仅核验替换计划。追加 --output ../clean/NEW_NAME.zip 才生成新的候选。ORIGINAL_ARCHIVE_RELATIVE_ID逐项见runtime_catalog.json的metadata_derivatives.original_archive。所需空间按原包解压体积加64MiB核对；先产生BUILDING_LOCAL_ONLY，全部成员SHA验证成功后才取得候选文件名；已有输出拒绝覆盖。把原dirty包上传不能省略该步骤。
生成全部需要的clean包后，可用以下命令从显式成功receipt创建新公开映射，不修改原guard：
python -B build_clean_archive_map.py --archive-root RELEASE_ROOT --output ../NEW_CLEAN_MAP.json --receipts RECEIPT1 RECEIPT2
再向compose_inputs.py提供该新map。其余未变的合法legacy_controls等归档按runtime_catalog声明的相对布局放置；所有被消费的成员逐SHA/bytes核验。
--local-exact-source-root和--cache-root仅用于作者本地只读验证缓存，不是发布依赖；不把validation_inputs、local_only、失败中间件、旧models ZIP、审稿原文加入公共包。

边界
原作者许可/权属及实际仓库归属仍待确认；本候选不授予第三方或作者许可证。LICENSING_PROPOSAL.txt是提案，notices保留已有provider约束。没有发布、下载、重新训练或新数据采集。此次证明的是有限入口兼容、保存统计与单字段格式等价，不认证全论文raw-to-result复现、未执行的大包或跨平台数值行为。保留先前失败和NA，不新增科学性能主张。
