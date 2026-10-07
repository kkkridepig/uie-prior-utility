# 固定视觉案例选择修补

规范：V2 §15 按 sample_id SHA256 取 8 例，再按固定候选 ΔPSNR 取最好/最差各 4 例及中位附近 4 例；并列按 ID，去重。

发现：冻结的 `uie_next/v2/delivery.py:visual_method_panels` 把 O 的质量行传给 `selected_cases`，因此原方法面板的质量排序依据是 O，而非候选。D0 诊断面板使用候选端点行，其依据正确。

影响范围仅是视觉案例挑选，不涉及模型、输入、输出、监督标签、逐图指标、校准、网络选点或科学闸门。没有用该视觉选择反馈调参。

修补：独立收尾脚本 `scripts/ssuie_v2_comparison_visuals.py` 从完整已评分名义表提取 `B1_producer` 行，按原固定规则生成 `candidate_ranked_<role>_selection.json`，重新生成同图 O 和冻结强对照双方的完整面板。检查候选行及已完成图像数量，校验全部权重与策略身份。原方法面板保留为辅助失败展示，不能称为规约规定的固定案例清单。

正式固定视觉案例以 `figures/candidate_ranked_utility_val_selection.json`、`comparison_utility_val_manifest.json` 及对应面板为准。封存未解锁时不生成封存案例。实际生成完成情况以这些文件和设备账本为准，本文不预先声称已运行。
