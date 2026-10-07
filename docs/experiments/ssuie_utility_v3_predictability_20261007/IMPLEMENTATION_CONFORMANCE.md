# V3 协议—公式—源码—验收对应清单

协议 SHA256：181458b3b0d2acf8cb4d82069436dfd9bbe28e67436c03ff7aa0d76181624cc8。以下状态在最终报告收束；源码存在不代表实测通过。

|条款/公式|实现函数|测试|实际产物/字段|初始状态|
|---|---|---|---|---|
|§1/3/4 原权重、角色与继承预算|context.State.audit/Data.check/device_job|角色与写路径守卫；设备实测|recovery_audit/environment/budget/access_events|待实测|
|§5.1 实际裁剪后r，a/b，区域先聚合|moments.geometry/aggregate|quadratic_and_relative_identity/aggregate_before_ratio|D1/D6及sufficient_stats.A/B|已实现待测|
|§5.2 C=B−ref*A，相对中心决策|moments.relative_alpha/torch_relative|regularization_center_grid|D7_models.constants|已实现待测|
|§5.3 精确regret边界及带符号替换差|moments.regret；diagnostics.Diagnostics.scan|regret_boundary/negative_replacement|D1 reference_only_*|已实现待测|
|§6.1 D1 预测/正则/active分解|Diagnostics.scan|V2_real_regression，decomposition运行断言|D1_prediction_regret_per_image.csv|待实测|
|§6.2 D2 均值/能量/固定置乱|moments.spatial_variants|null_amplitude_and_alpha_shuffle|D2_spatial_controls.csv|待实测|
|§6.3 D3 3000点统一与各自合法策略|load_controller/scan|严格step/hash，实时回归|D3_common_policy_ablations.csv|待实测|
|§6.4 D4 七视图信息、有效秩|Diagnostics.scan|旧干预测试及字段缓存身份|D4_intervention_information.csv|待实测|
|§6.5 D5 三个可微loss与共同裁剪|Diagnostics.gradients|梯度和运行断言，不更新权重|D5_loss_gradient_components.jsonl|待实测|
|§6.6 D6 B3及两随机匹配方向|moments.matched_null/oracle_rows|幅度/合法范围/确定性测试|D6_matched_null_and_rgb_oracles.csv|待实测|
|§7.1 13→26/52可见统计|descriptors.descriptors|descriptor_and_nearest_contract|sufficient_stats.f_region/f_global|已实现待测|
|§7.2 训练池常量与10001细网格|ridge_probe.constants/fit|fit-only测试|D7_models各折常量|已实现待测|
|§7.3 五折内容组、岭系数.01|ridge_probe.run_probes/fit/predict|fit-only/截距不罚|D7_fold_manifest/oof/dev_per_image|已实现待测|
|§7.4 固定路由先区域再全局|ridge_probe.decide|route_priority_and_stop|stage1_decision.checks/route|已实现待测|
|§8 唯一5505参数共同step0与loss|models.MomentController/objective|step0/no-context/gradient/per-image-log|第二阶段回执|条件性待运行|
|§9/10 同步矩阵、profile、恢复|v3.training 待实现|T14及设备模拟角色|training_schedule/checkpoints|条件性未运行|
|§11/12 选点→冻结→CAL→开发|v3.evaluation 待实现|T15/T16|selection/metrics|条件性未运行|
|§13 封存一次|Data.check与条件冻结|权限拒绝|sealed仅DEV_PASS解锁|未运行|
|§14 压力/视觉/全流程计时|阶段允许时实现|右移核验/真实设备|figures/timing/stress|条件性未运行|
|§17/18 原子状态、交付与备份|State.complete与delivery|恢复身份/ZIP CRC成员SHA|state/delivery/四包|待收尾|

执行阶段：恢复审计→协议/角色冻结→CPU与真实8图验收→D1–D6→D7 OOF+DEV→固定路由；仅通过才实现并验收第二阶段正式调度、完整匹配矩阵、选点/CAL/开发；仅DEV_PASS才解封。停止时其余条目逐项not_run。

## 实际验收与停止后的边界

第一阶段已实现并实际运行：恢复审计、角色守卫、精确oracle/regret、D1–D6、D7两种五折OOF及全量重拟合、固定路由、补充SSIM/LPIPS、逐图/逐内容组与配对区间、固定失败视觉。CPU原75项与新增契约均运行，最新计数以tests/final_cpu.xml为准；真实8图底座/候选缓存差0，O/J0开发PSNR回归通过。D5关闭该诊断的TF32后原float32容差通过；旧模型没有更新。D1–D4/D6的初始源码、D5修补影响及最终源码分别封存，不伪称同一字节身份。

第二阶段唯一5505参数网络及损失公式已有合成测试；正式训练调度、profile、4对2+2真实训练恢复、全部条件匹配矩阵和正式CAL/DEV/sealed评估入口均**未实施/未运行**，原因是D7科学停止。不能把这些条件性条目写成工程验收完成。T14/T15及第二阶段真实设备全臂验收为not_run；CAL/封存拒绝守卫已测试。未创建虚假的训练检查点、CAL选择或DEV_PASS。

T18跨阶段上限、累计预算、孤儿费用恢复去重已有新CPU模拟验收；真实跨run共用设备锁。第一次仅查V1缓存的工程错误、两次D5数值失败及预热/补充评估/视觉/实时对照成本均记账；CPU统计时模型仍驻留的29.683683秒额外保守记入，未把CPU速度当PPU测速。

正式停止：STOP_NO_PREDICTABILITY_SIGNAL。source_snapshot.json记录初始阶段源码；audit_history/stage1_frozen_source保留改变文件的原字节，最终交付源码以final_source_snapshot.json为准。新文件增加不改变已冻结D7特征、岭系数、折、候选、尺度或阈值。旧V1/V2全文件哈希另验，打包成员哈希与全旧目录保护哈希是两种不同范围。
