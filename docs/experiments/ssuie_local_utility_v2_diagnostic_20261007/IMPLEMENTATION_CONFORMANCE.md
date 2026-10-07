# V2 公式、源码与验收对应清单

主协议 SHA256：d8b5d2046ebf09ad12295052701ea0d12218983258aac775fd31bb9af0479034。以下为实现前清单，结果尚未验证。

| 规范／公式与输入输出 | 源码函数（待新增标记） | 数据角色 | 测试 | 结果文件 |
|---|---|---|---|---|
| 7通道颜色／对比度弱代理；字段干预后重算 | priors/heuristic.py:make_prior/interventions | I，不读取Y | T09 | checkpoint_prior_sensitivity.csv |
| J1=clip(J0+0.25tanh(delta))；r=J1-J0 | models/candidate.py:Candidate；math/utility.py:labels | model_fit训练，model_val选型 | T03/T11 | checkpoint_residual_diagnostics.csv |
| a,b,c,u,v,U=2b-a；RGB共享alpha | math/utility.py:geometry/labels/decision | utility_fit监督，开发Y只诊断 | T04/T10/T13 | 测试回执及标签尺度 |
| float64精确像素／32块／图连续／硬oracle | v2/diagnostics.py:strategies（待新增） | model_fit_probe/model_val/冻结后utility_val | T05/T06 | checkpoint_image_metrics.csv |
| standalone允许0；producer非零，粗／细资格、并列规则 | v2/selection.py:choose_producer（待新增） | 仅model_val | T07 | producer_selection_before_utility_val.json |
| 检查点×网格选网络；CAL独立重新选参数 | v2/selection.py:choose_grid（待新增） | utility_val选网络；calibration选参数 | T08/T15 | utility_checkpoint_grid_scores.csv |
| 共享正负网络、有符号投影；inactive安全 | models/utility.py；models/controls.py | utility_fit | T10/T11 | 11方法真实集成回执 |
| G0/G1/G2/F0/R0及四消融；缺U_hat为null | training.py:pair_objective；scientific_evaluation.py | utility_fit训练 | T12/T17 | 全方法真实集成指标表 |
| 每图先MSE后LOG、固定低LR完整恢复 | v2/training.py（待新增）；checkpoint.py | model_fit；probe尺度 | T13/T14 | 条件救援日志／权重 |
| 图等权估计／5000组bootstrap | v2/statistics.py（待新增） | 各自评估角色 | T16 | paired_intervals.json |
| 角色、身份、原子恢复、幂等、16小时继承 | records.py:RunContext（待新增）；v2/context.py（待新增） | 禁止提前CAL/封存评分 | T01/T02/T15/T18 | state/budget/events.jsonl |

实际阶段：RECOVERY_AUDIT → ENGINEERING_ACCEPTANCE → D0_MODEL_DIAGNOSTICS → MODEL_PRODUCER_FREEZE → D0_UTILITY_DIAGNOSTICS → 条件唯一救援 → PRODUCER_FROZEN → G1/R0/O → 其余8作业 → NETWORK_FREEZE → CALIBRATION → DEV_GATE → 仅DEV_PASS解封一次 → CLOSEOUT。

每阶段以依赖身份、数量与回执验证，not_run 必须注明原因。oracle 不可部署；任何测试通过都不是创新成立。
