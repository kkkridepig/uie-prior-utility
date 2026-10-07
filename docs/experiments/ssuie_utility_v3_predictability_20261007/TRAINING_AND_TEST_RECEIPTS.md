# 训练与测试回执

本轮新增正式神经训练更新为0。旧11方法各3000更新是V2事实；本轮仅读取，不改写旧停止状态。B1固定3000点不续训、不重新选型。

D7每折两探针，各只拟合一次；全443图每种探针各再拟合一次，共12个固定ridge拟合对象。属于诊断拟合，不计作新增神经种子或正式完整方法训练重复。五折内容组、每折源样本哈希、尺度、标准化、系数、截距均保存。

真实PPU验收：8图底座与7视图候选缓存回归；168图诊断；O/J0旧指标复现；固定8组两批真实可微梯度，0 optimizer.step；579图补充ridge质量；固定/最好/最差/中位规则36面板；额外8图两种ridge实时/缓存一致测试。每项设备事件均在预算。

CPU套件与失败修补记录：tests/cpu.xml、tests/final_cpu.xml、initial_descriptor_failure.log、budget_recovery.log、gradient_sum_failure.json（保留失败，不当最终状态）、audit_history/gradient_differentiation_fix/impact.json。D5最终梯度结果在diagnostics/D5_loss_gradient_components.jsonl，严格容差与未更新参数回执分别保留。

第二阶段训练恢复、选点×κ、CAL、DEV正式闸门及177封存均not_run：D7未解锁。数学测试不能替代这些正式验收，旧通用checkpoint的测试也不能冒充V3未运行的4对2+2训练恢复。
