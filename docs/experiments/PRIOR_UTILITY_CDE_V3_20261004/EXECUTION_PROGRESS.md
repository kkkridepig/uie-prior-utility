# 当前执行进度（非最终科学结论）

## 2026-10-05 最终收束

调度已正常结束，状态pilot_no_supported_gain，设备累计11.648062小时。C的完整pilot为utility_not_learned，随后一次D备选未通过；未启动额外两个seed或DATA_MATCHED。最终915图的冻结评估只有PARENT与BASE_CONT，没有C/D留出确认。11个权重CPU加载和4406项哈希核验通过；详细结果、导出与模板文字/LPIPS统计勘误见FINAL_REPORT.md及CLOSEOUT_REVIEW_20261005.md。以下内容为历史启动快照。

## 2026-10-05 恢复核验

以下旧启动快照保留供追溯。E现已完成864条subset、2730条full、48条legacy及内部状态敏感性记录。完整矩阵初次NCHW预测超预算，因此未启动C；该停止有独立记录。

训练图上的确定性channels_last推理核验和选择器PPU profile/恢复检查已通过，训练配方保持原NCHW。修正后的完整矩阵预测为C pilot 8.531小时、两个重复17.062小时、最终评估及导出9.990小时，均含安全余量，满足原包上限。62项CPU/MPA测试通过，选择器7+13对连续20步的参数和Adam状态差均为0。正式pilot前累计设备时间2.96688小时，实际新费用继续累计。

下一步由持久本地调度执行C固定5000步控制与后续科学闸门。这里的profile不计作正式pilot；尚无C有效性或创新成立结论。准确实时状态见LIVE_STATUS.md，历史停止及后续终态见FINAL_REPORT.md与保留快照。恢复分析见RESUME_PROFILE_REVIEW.md。

2026-10-05 06:21 UTC 已实际启动后台pipeline（PID 1416676；进程身份见pipeline_process.json），当前正式任务为20261004_BASE_CONT_V3_train。首个100步checkpoint已在CPU加载，父身份、seed和5000步终点匹配；Adam共672个状态张量均有限，源码仍匹配pilot前freeze。证据：runs/prior_utility_cde_v3_20261004/pilot_first_checkpoint_verification.json。当前loss有限，后续状态由调度持续更新。本段是启动核验，不能当作5000步完成或算法有效性结论。

## 旧启动快照

已完成现场审计、父checkpoint与原图哈希验证、协议/数据角色/预算冻结、CLI dry-run、CPU与PPU真实父模型梯度和null回退验收。

v3测试22项通过；连同旧MPA测试共58项通过。首次FFT常值奇数尺寸反例已保留并修正；CPU DepthAnything xformers无可用算子，改用其官方plain attention分支，仅限CPU验收，PPU配置不变。VGG16下载与SHA256核验完成，LPIPS已在E中实际运行；与训练VGG19区分。

E正在执行父85000步与V2 BASE_CONT5000步、24图、101/102/103三噪声的DDIM1/2/4/8/20及DDPM1000。已完成记录逐条写per_image.jsonl，进度写E/subset/progress.json。完整91图验证和无轨迹同步的效率表尚待执行，不引用部分样例作方法结论。

本地可恢复pipeline已启动，等待并接续当前E任务。之后顺序是E完整验证与机制诊断 → 100步PPU profile/恢复配对检查 → 完整控制矩阵成本判定 → C固定5000步pilot与oracle。只有正确且指标完整的C数值失败才触发一次D。当前C训练、D训练、三个微调种子确认、新留出评估均未执行完毕。

391张LSUI候选仅在现有日志和近重复筛查可审计范围内为未执行样本，DA/VGG上游未知。它们未被读取增强分数，须final_freeze后才可测试。旧UIEB/LSUI test保持历史暴露回归身份。

新账本上限72设备小时，其中12小时保留最终评估。旧V2目录与结果保持原样。严格状态以dispatch_state.json和budget.json为准，此文件是本轮启动快照；pipeline达到终态后生成FINAL_REPORT.md。
