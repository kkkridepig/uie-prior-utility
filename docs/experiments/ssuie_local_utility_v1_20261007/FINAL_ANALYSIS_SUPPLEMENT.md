# 候选科学停止分析

本补充是冻结观察的CPU分析，不启动新训练、不改选择。主结果为 `STOP_PRIOR_UNUSED`，另一个候选空间条件也未通过。

## 为什么停止

主协议允许0更新参与检查点选择。两候选在所有非零验证点均低于该起点，故选择其零末层初始化。由结构可知，这时J1=clip(J0+0.25*tanh(0))=J0，r=0，a=c=b=v=U=0，active=false。任何固定alpha或oracle只能返回J0。

因此敏感性0不是随机波动；训练O会只得到inactive监督与零候选修正，不回答本轮H1/H2/H3。指南§12/§19要求退出当前候选配方，不允许换到未选最后权重、增宽网络、额外训练、强迫使用先验或换底座救场。

## 证据与局限

model_val上B1最后权重比起点低0.1188068 dB，B3低0.1805254 dB。两者只是固定开发验证的负面结果，不是新封存确认。候选checkpoint均有初始、25/50/75/100%及最新完整恢复状态，记录首尾相同数据/缺失随机流。

后验工程排查仅在model_fit输入上进行，没有GT评分：训练后的B1对先验有非零梯度，对字段干预有输出差异；选择后的0更新输出精确等于J0。该排查用于排除明显漏接，不能拿它重新决定科学候选。

SS-UIE model_val分数较高且候选训练轻量修正不提升，可能与公开预训练底座已经强、监督目标及MSE/逐图PSNR差别、有限配方优化有关。本轮没有运行额外实验区分原因，因此这些只是解释候选，不能写成已证实原因。不能以增加训练保证正结果。

## 实际结果文件

- 原始671对双策略：`metrics/baseline_policy_model_val.jsonl`。
- 136对候选全表：`metrics/candidate_diagnostics.jsonl`，含B0/B1/B3、五固定强度、三个oracle。
- 逐图完整身份：`metrics/candidate_per_image_with_lineage.csv`。
- 逐内容组：`metrics/candidate_per_group_with_lineage.csv`。
- 逐seed：`metrics/candidate_per_seed.csv`，seed仅20261007。
- 配对区间/尾部：`metrics/candidate_summary_with_paired_intervals.json`。
- 检查点选择：`checkpoints/B1/selection.json`、`checkpoints/B3/selection.json`。
- 工程排查：`tests/candidate_wiring_cpu_receipt.json`。
- 原始成本：`budget_ledger.jsonl`，完整矩阵预测：`budget_plan.json`。

每个所选方法相对J0的均值、中位数、5/10/90/95分位差和最差10%差均为0；严重退化/改善比例也为0。这不表示一个学到的非退化保证，而是选中输出恒等。

`primary_control`未校准冻结，因此不填虚构的对primary差值；逐图表明确标为not_frozen_after_S5_stop。所有oracle有uses_reference=true，不列入可部署方法创新排序。

## 未运行项

S6正式效用缓存/标签尺度、S7的B4/G0/G1/G2/F0/R0/O/四消融、S8校准、S9开发机制闸门、S10封存/全部压力/完整部署时延均not_run。不存在完成版normalization_stats、calibration_selection或selection_freeze_before_eval；临时工程夹具的同名统计不属于正式方法文件。

已实现这些模块，并有CPU公式、结构、损失/梯度测试与真实O小样本运行；没有真实完整控制矩阵效果回执，不写工程整体经过所有真实阶段证明，也不写C机制被充分证伪。

## 封存与哈希终检

177对sealed_eval未评分且没有J0缓存；calibration也未评分。审计可以读文件算哈希/重复，不等于解封评分。`closeout/terminal_integrity.json`独立记录缓存角色、无final freeze、厂商torch、历史记录和冻结源码身份。

同盘备份不是独立存储。未发现并核验授权独立挂载；只提供可下载包，independent_backup_verified=false。本轮不公开上传数据，不推送GitHub。
