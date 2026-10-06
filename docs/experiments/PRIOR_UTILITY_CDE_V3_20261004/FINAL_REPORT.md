# CDE V3 最终报告

2026-10-05收束复核：**`pilot_no_supported_gain`**。本轮只有微调种子20261004；C效用选择器未通过干净质量闸门，一次D备选也未通过。调度已正常结束，当前无活动设备任务。

| 结果 | 数值 | 结论 |
|---|---:|---|
| C oracle − 开发集最佳固定模式 | +0.364630 dB | 训练后bank有选择空间；不可部署 |
| C UTILITY − route_cal选择的BEST_FIXED | −0.002728 dB | 未通过+0.10 dB门槛 |
| C UTILITY − WINNER_CE | −0.109975 dB | 未支持效用回归增量 |
| C UTILITY − SHUFFLED | −0.066296 dB | 未支持输入可预测增量 |
| C UTILITY − ALL_ONLY | −0.086600 dB | 未超过结构强控制 |
| D SOFTPHASE − PHASE | −0.000251 dB | 未通过备选晋级 |

实际累计 **11.648062/72设备小时**，P0 0.155785、E 2.811094、C 5.116798、D 3.188144、FINAL 0.376242。达到科学停止条件后没有启动DATA_MATCHED或额外两个种子，剩余预算不构成无限续训授权。

最终915图、三个噪声的评估仅包含PARENT和BASE_CONT_V3。391张LSUI候选留出上的+0.761743 dB属于继续训练基线收益，不属于C/D；没有新方法留出确认。旧UIEB/LSUI test只作历史暴露回归。既有候选留出已产生基线分数，未来不可无条件称未触碰。

工程实现与科学证据分开：62项测试通过；11个权重CPU加载核验通过；七个增强器5000步公共RNG日志逐步配对；4406项原artifact哈希匹配。工程通过不表示创新成立。

详细数值、数学与归因解释、逐方向状态、成本、缺项、勘误及后续建议见 **[收束核验与结果解释](CLOSEOUT_REVIEW_20261005.md)**。

## 交付入口

- [E采样诊断](E_DIAGNOSTICS.md)、[C训练后oracle](C_BANK_ORACLE.md)、[D备选结果](D_FALLBACK.md)。
- [冻结后指标及勘误](FINAL_METRICS.md)、[留出审计](HOLDOUT_AUDIT.md)、[运行命令](RUN_COMMANDS.md)。
- 原始记录：`runs/prior_utility_cde_v3_20261004/`，包括budget、dispatch_state、protocol_frozen、final_freeze、lineage、数据角色/名单、测试日志、逐图JSONL、checkpoint和面板。
- 易读取导出：`runs/prior_utility_cde_v3_20261004/closeout/`：逐图/逐scene CSV、开发对照统计、效用regret/模式/null/污染/risk-coverage、权重大小与SHA256、完整性核验和额外证据清单。

原冻结JSON含通用“三seed”模板文字，实际seeds仅[20261004]；原LPIPS改善比例的方向错误已在独立补表修正。保留原件及其哈希，不用改历史记录掩盖问题。LPIPS原始分数和闸门结论不受影响。具体见收束报告的勘误章节。

原自动报告保留为FINAL_REPORT_AUTOMATED_SNAPSHOT.md；早期预算停止保留为FINAL_REPORT_PREFLIGHT_STOP.md。本报告覆盖其状态解释。服务器厂商PyTorch/PPU环境及旧V2结果保留。
