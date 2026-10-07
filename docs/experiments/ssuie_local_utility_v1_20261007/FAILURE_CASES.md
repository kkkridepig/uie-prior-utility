# Failure cases and pressure scope

No registered intervention produces mean candidate MAE >=1e-4; CPU field/path tests passed.

选中0更新意味着J1=J0；六干预敏感性和oracle增量均0，不是先验接线缺失。最后训练权重的训练输入排查通过，见tests/candidate_wiring_cpu_receipt.json。没有改用最后权重救场。

已实际查看基线20图接触图、完整基线00、候选00和04面板、候选训练曲线。残差/U/alpha的恒零可视化与数学结果一致。候选best/worst全部并列，按稳定ID选择；不称真实改善/退化案例。O未训练，没有O失败例；所有未见压力族未解锁，不提供虚构鲁棒性结论。
Fixed visual records are in figures/baseline and, if reached, figures/utility_val or figures/sealed_eval.
No unrun pressure family or sealed score is synthesized. All registered families are reported only when their phase executes.
