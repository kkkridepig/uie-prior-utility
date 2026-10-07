# Implementation conformance



| 条款与公式 | 文件／接口 | 输入→输出 | 允许参考图的角色 | 验收 |
|---|---|---|---|---|
| §3 严格加载与两个后处理 | backbones/ssuie.py | I→J0，float32 RGB 256×256 | model_val 仅用于一次后处理选择 | T26、T27、T37、至少20图 |
| §6 弱代理 d、tau、A、t、Q、q | priors/heuristic.py:make_prior/recompute | I→P(7通道)、V(1通道) | 推理不使用Y | T24、T25 |
| §6 七视图字段干预 | priors/heuristic.py:interventions | 固定I，修改字段并重算P/V | utility_fit 生成训练标签；utility_val 诊断 | T14、T24、T25 |
| §7 J1=clip(J0+0.25tanh(delta)) | models/candidate.py | I/J0/P/V(14通道)→J1 | model_fit梯度；model_val选检查点 | T05、零末层、115907参数 |
| §8 r=J1-J0，a=meanRGB(r²)，b=meanRGB((Y-J0)r)，U=2b-a | math/utility.py:labels | J0/J1/Y→标签 | utility_fit监督；其他角色仅各自评估 | T01、T02、T05 |
| §8 c=sqrt(a)，u=r/c，v=meanRGB(eu)，b=cv | math/utility.py:geometry/labels | r→a/c/u/active；Y仅定义真实v | 部署geometry不读取Y | T06、T09、T10 |
| §8 v_hat=(h(+u)-h(-u))/2，b_hat=cv_hat | models/utility.py | I/J0/u→单通道有符号局部作用 | utility_fit梯度 | T07、T08、T29、T30 |
| §8 alpha=clip((b_hat-tau)/(a+lambda),0,1) | math/utility.py:decision | alpha单通道、RGB共享→混合图 | 推理无Y | T11、T12、T20、T21 |
| §8 像素／先聚合a,b的32块／整图oracle | math/utility.py:oracle；诊断流程 | Y定义不可部署诊断 | utility_val；冻结后sealed_eval | T03、T04；uses_reference=true |
| §9 FIT-only固定尺度与投影／成对／部署损失 | normalization.py；losses.py；training.py:pair_objective | 同图两视图真实标签→loss | utility_fit | T14、T15、T28、T31 |
| §10 B4、G0/G1/G2、F0、R0、O及四消融 | models/controls.py；正式调度 | 同配方有效批量及冻结候选 | utility_fit；B4另允许model_fit | 参数数、梯度、冻结、同源图序列 |
| §11 命名随机流、精确LR、五个检查点 | training.py；checkpoint.py | 初始／恢复状态→继续更新 | model_val或utility_val选择 | T32、T33、T35 |
| §13 有限校准与§15闸门 | calibration.py；评测流程 | 冻结网络→部署参数／比较 | calibration；utility_val开发 | 守卫、约束、并列顺序 |
| §14 预算／§20恢复／§21备份 | budget.py；checkpoint.py；backup流程 | 计时／哈希／权重与恢复包 | 不改变数据访问 | T34、T35；累计16小时，保留3.5 |
| §15/16 指标、统计、服务测速与案例 | evaluation.py；部署测速／收尾流程 | 浮点RGB，统一逐图与逐组结果 | 仅允许角色；封存须DEV_PASS和freeze | T36；LPIPS真实VGG；测速禁缓存 |

上述映射不代表所有条款已验收；CPU核心实现和合成测试已存在，真实训练／校准／完整调度仍需补齐并执行。正式训练不得使用MockBackbone或随机底座。


Full dispatch: cli.py -> experiment.py -> integration.py/profiling.py/scientific_evaluation.py/timing.py/visuals.py/delivery.py.
Backbone identity: official public simplified implementation; no unpublished MCSS reconstruction.
Implemented is distinct from executed: state.json and per-job selection receipts identify the actual completed stages. B4 selects on model_val.

## 实际运行范围

60项CPU数学/协议/损失/参数/恢复测试通过；真实PPU验收回执在tests/real_integration，包括底座与43个BN冻结、候选/O各100更新、16样本J0/J1七视图与标签缓存、无参考部署及新进程恢复（误差0）。13作业profile均已测10预热+30更新。

B1/B3正式各4000更新及五点选点已运行，最佳均为0更新。S5敏感性与oracle空间未通过，故B4/G/F/R/O及消融正式训练、CAL/DEV/FINAL全部not_run；已实现代码不等于真实正式阶段验收。H1/H2/H3未得到有效候选条件下的完整实验，不宣称机制被证伪。

CPU接线排查见candidate_wiring_cpu_receipt.json，3个model_fit输入，不看Y、不改選点；训练后先验梯度非零。终止CLI再次resume没有新增训练或账本占用，见terminal_resume_receipt.json。冻结源码/历史文件/厂商torch终检见terminal_integrity.json。
