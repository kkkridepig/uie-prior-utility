# 权重补齐后的执行合同与任务

本文件在恢复实现前建立。沿用 `ssuie_local_utility_v1_20261007` 及同一预算账本，不重置已消耗的 83.58769 设备秒。科学主规范为服务器实际上传的完整指南；用户本条消息的底座身份和缺失权重处理更正优先适用。

## 底座限制与文档差异

本轮底座标注为 **SS-UIE 官方公开简化实现**，不是论文完整模型。用户确认上传权重来自作者 README 的 Google Drive `1YxyagMCbApON8dRdiQTaQG65g3tnZkPt`；本地 SHA256 为 `977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99`。来源记录区分用户声明和服务器独立下载，后者没有发生。

已实际读取 GitHub API 的作者 OWNER 回复：

- issuecomment-2894406180：“为了能在24g显存以下的单卡上运行，上传的预训练模型是参数简化版的。”
- issuecomment-2904097810：“多尺度的循环选择性扫描暂时不开源，我还有别的工作要用。”

最初读取的20261007文件标记1.0，没有§3.5；已保留该历史记录。随后现场发现用户新上传的20261008文件正文为1.1（制定/修订日期2026-10-07），内容与用户更正对应，现用该文件作为执行指南。两文件的科学公式、网络、阈值和预算一致；新文件补充简化实现身份与缺权重时依赖规则。用户文件未改写；身份补充记录在guide_v1_1_admission.json和protocol_amendments.jsonl。

## 公式、接口、角色与验收对应

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

## 执行任务

1. 已完成：完整读取实际上传指南；确认权重来源声明、作者简化与未开源限制、当前账本；保留历史。
2. 已完成：归档前次阻塞报告；核验来源／暴露、真实底座、调度实现和60项CPU测试。
3. 已完成：固定20图验收、671对model_val后处理选择、真实小样本/新进程恢复、13作业测速和成本预测。
4. 已完成：共同步数4000/3000冻结；两候选各4000更新；utility_val候选闸门未通过，STOP_PRIOR_UNUSED。
5. not_run：S5停止，七视图正式FIT尺度、11效用作业、校准与开发未解锁；不缩控制跑无效比较。
6. not_run：没有DEV_PASS；封存177对未评分、正式压力/服务测速未派发。
7. 已完成：最终报告、候选逐图／逐组／逐seed表、训练曲线和案例、可恢复权重包及哈希。独立备份未完成。
