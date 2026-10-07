# 底座图像核查

已生成并实际打开核查固定 model_val 前20个稳定ID的接触图 `figures/baseline/inspection_contact_20.png`，同时打开了完整 `00.png` 面板。接触图由已有浮点结果的显示PNG组成，没有重新评分或读封存集。

每例依次为输入、参考、clip01、逐图 official_minmax_float。20例显示的输入、参考和增强结果内容对应，未发现明显RGB通道错序、全黑输出或单通道崩坏；min–max 与 clip 的亮度差异可见。该人工检查只覆盖这些固定示例，不能保证所有图像质量。

原面板及清单保留在 `runs/ssuie_local_utility_v1_20261007/figures/baseline/`。模型输出的有限性、裸模型/封装一致性、单图/批量一致性与缓存误差由真实PPU回执单独记录，不能用视觉检查替代。

这些图来自公开底座来源相关的 LSUI model_val，因此图像接近参考不能视为独立泛化或新方法效果证据。SS-UIE仍标注为官方公开简化实现；本轮唯一后处理选择依据为671对 model_val 的规定PSNR规则。
