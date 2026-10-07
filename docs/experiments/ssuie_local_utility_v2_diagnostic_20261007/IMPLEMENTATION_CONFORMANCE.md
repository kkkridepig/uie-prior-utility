# V2 公式、源码与验收对应清单

主协议 SHA256：d8b5d2046ebf09ad12295052701ea0d12218983258aac775fd31bb9af0479034。以下保留实现前的对应清单，并依据实际测试回执更新工程状态。科学阶段完成情况以 state.json 和 FINAL_REPORT.md 为准。

| 规范／公式与输入输出 | 实际源码函数 | 数据角色 | 测试 | 结果文件 |
|---|---|---|---|---|
| 7通道颜色／对比度弱代理；字段干预后重算 | priors/heuristic.py:make_prior/interventions | I，不读取Y | T09 | checkpoint_prior_sensitivity.csv |
| J1=clip(J0+0.25tanh(delta))；r=J1-J0 | models/candidate.py:Candidate；math/utility.py:labels | model_fit训练，model_val选型 | T03/T11 | checkpoint_residual_diagnostics.csv |
| a,b,c,u,v,U=2b-a；RGB共享alpha | math/utility.py:geometry/labels/decision | utility_fit监督，开发Y只诊断 | T04/T10/T13 | 测试回执及标签尺度 |
| float64精确像素／32块／图连续／硬oracle | v2/diagnostics.py:exact_strategies | model_fit_probe/model_val/冻结后utility_val | T05/T06 | checkpoint_image_metrics.csv |
| standalone允许0；producer非零，粗／细资格、并列规则 | v2/selection.py:choose_producer | 仅model_val | T07 | producer_selection_before_utility_val.json |
| 检查点×网格选网络；CAL独立重新选参数 | v2/selection.py:choose_grid | utility_val选网络；calibration选参数 | T08/T15 | utility_checkpoint_grid_scores.csv |
| 共享正负网络、有符号投影；inactive安全 | models/utility.py；models/controls.py | utility_fit | T10/T11 | 11方法真实集成回执 |
| G0/G1/G2/F0/R0及四消融；缺U_hat为null | training.py:pair_objective；scientific_evaluation.py | utility_fit训练 | T12/T17 | 全方法真实集成指标表 |
| 每图先MSE后LOG、固定低LR完整恢复 | v2/training.py；checkpoint.py | model_fit；probe尺度 | T13/T14 | 条件救援日志／权重 |
| 图等权估计／5000组bootstrap | v2/statistics.py | 各自评估角色 | T16 | paired_intervals.json |
| 角色、身份、原子恢复、幂等、16小时继承 | records.py:RunContext；v2/context.py | 禁止提前CAL/封存评分 | T01/T02/T15/T18 | state/budget/events.jsonl |

实际阶段：RECOVERY_AUDIT → ENGINEERING_ACCEPTANCE → D0_MODEL_DIAGNOSTICS → MODEL_PRODUCER_FREEZE → D0_UTILITY_DIAGNOSTICS → 条件唯一救援 → PRODUCER_FROZEN → G1/R0/O → 其余8作业 → NETWORK_FREEZE → CALIBRATION → DEV_GATE → 仅DEV_PASS解封一次 → CLOSEOUT。

每阶段以依赖身份、数量与回执验证，not_run 必须注明原因。oracle 不可部署；任何测试通过都不是创新成立。

已核验的工程证据：pytest_cpu.xml 首轮71项；恢复/日志修补后最终75项、0失败/错误；11方法每个2次真实PPU临时更新；16个允许fit样本缓存与实时输出最大差0；真实PPU连续4更新与2+新进程恢复+2的参数/优化器差0，源图、视图、学习率和独立随机流一致。六个原始数据源码哈希与V1 source_snapshot一致，已Git跟踪，全新检出导入通过。

工程夹具不进入正式结果。D0精确oracle与真实可部署策略分开，step0不进入producer池；八点全量诊断已完成并通过唯一producer迁移检查；正式11方法各完成3000更新，网络选点和133对校准已完成。完整开发闸门执行失败，科学状态为STOP_CURRENT_RECIPE_NOT_SUPPORTED；177对封存集未评分。收尾分析脚本 scripts/ssuie_v2_finalize_evidence.py 只处理已完成CSV，不读取图像、不改变选择，另存源码哈希和CPU耗时。恢复包包含全部完整checkpoint及校验旁文件，源码包逐项核验六数据模块；服务器同盘包不视为异地备份。

额外恢复契约修补：旧候选仅恢复原四随机流；原4000模型与AdamW及下一批采样CPU回执通过；救援新检查点清单跨恢复保留；中断不伪造科学收尾；损失窗口、裁剪窗口和累计去重源图写入可恢复状态。纯数学另逐字复算§6.3的20/16.989700/23.010300dB反例，见 literal_protocol_counterexample.json。所有工程修订和保留的D0初始计算身份均在audit_history登记。

收尾的namespace兼容处理放在独立scripts/ssuie_v2_safe_closeout.py，不改74个冻结科学源码。官方net无__file__时先验证全部搜索路径，再清除该namespace缓存项；错源/混合namespace拒绝，重复strict加载逐参数完全相等。CPU访问导出保留汇总身份核验事件及逐图评分事件，未知事件报错；封存身份核验不计作模型评分。回执见tests/namespace_closeout_recovery.json、tests/table_export_self_check.json和audit_history/closeout_namespace_and_access_export/engineering_amendment.json。

规约日志字段存在已知缺项：每50更新记录真实损失与梯度窗口、学习率、源图/视图呈现及去重数，但没有独立逐行墙钟时间戳；设备总时间由budget_ledger实测记录。未事后猜补时间，不声称完全满足逐50更新测速字段。该缺项不改变标签、输出和选择；数学与所测协议接口通过不等于工程记录每一字段均无偏差，也不等于算法成立。
