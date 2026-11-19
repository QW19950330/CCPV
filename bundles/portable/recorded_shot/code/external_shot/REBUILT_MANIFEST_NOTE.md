# 重建manifest的兼容核验补充

pipeline.py verify严格比较原manifest谱系，适合使用原head_freeze.json的重放。如果使用external_head_training重建或合法迁移的head manifest，即使数值完全一致，其manifest SHA256也不同；应使用本补充命令：

    python -B verify_compatible.py --prediction-root "<新预测目录>" --score-root "<新评分目录>" --freeze-sha256 "<新prediction_freeze.json的SHA256>" --reference-prediction-root "<原冻结预测目录>" --reference-score-root "<原冻结评分目录>" --out "<全新比较目录>"

此命令先验证新预测冻结、全部代码/config、15头推断参数指纹及score收据，并校验原参考文件SHA256；比较全部概率/先验/row key、标签和完整评分树。只排除评分树中已经分别核验的manifest_sha256身份字段，输出保留两份真实manifest哈希，不将不同谱系说成相同文件。所有817数值仍需通过1e-14绝对容差，并报告逐值/字节完全相等及最大差值。

推断参数指纹覆盖15头w、b、mean、scale、native_variant及固定先验。原加载器仍要求原格式的十五个head、训练索引结构及200步记录，但此核验不证明训练过程；它只允许推断数值等价的manifest迁移，不允许变更模型、再校准或挑种子。

compatibility_manifest.json为create-only补充封存：包含本目录全部先前文件和这两个新增文件的哈希，不修改原package_manifest.json、pipeline.py、fixed_inputs.json或五个原数值模块。包内仍无数据/索引/权重/凭据，不授予许可，不等于公开发布。
