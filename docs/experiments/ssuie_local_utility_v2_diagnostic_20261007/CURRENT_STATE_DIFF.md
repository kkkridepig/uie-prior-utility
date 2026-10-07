# 现场状态与历史快照差异

接管HEAD为1f356d62c1ab1379546a3c6e1646be0e8a2953a8，工作区干净；旧V1 STOP_PRIOR_UNUSED保持不变。新V2分支research/ssuie-local-utility-v2-diagnostic独立运行，不恢复旧pipeline。六数据模块从服务器原件恢复跟踪，与source_snapshot.json逐项一致，无猜写替代。

厂商PPU-ZW810E单卡、约49GB，torch 2.0.0a0+nv2303原文件hash不变；活动任务接管时无训练。旧八非零候选与官方底座SHA完全一致，全部完整优化器/RNG存在。原UIEB/LSUI角色及图像hash全部核验；唯一跨角色内容组涉及排除项UIEB/917_img_，继续排除。

V1旧账本1503.1731119155884设备秒继承，非新开16小时。旧报告的step0 oracle无空间不代表非零候选空间已检验；本轮按新协议重新诊断。旧calibration与sealed未评分，且本轮尚未释放。
