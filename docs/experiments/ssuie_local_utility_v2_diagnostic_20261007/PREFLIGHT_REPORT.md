# V2 前置审计与执行回执

本文件汇总已经核验的机器证据，不记录未运行阶段的分数。主协议 SHA256：d8b5d2046ebf09ad12295052701ea0d12218983258aac775fd31bb9af0479034。

仓库 /mnt/workspace/uie-prior-utility，独立本地分支 research/ssuie-local-utility-v2-diagnostic。执行接管原提交 1f356d62c1ab1379546a3c6e1646be0e8a2953a8；没有 reset/clean 或远端推送。工程、源码恢复、诊断/调度及收尾补充均有本地提交，实际命令见 commands.jsonl。当前提交：71276f0e1dbbc447ca77384b289d6b02aad1a9c9。

源码：六个 data 模块全部从服务器原件恢复，与旧 source_snapshot.json 完全匹配，已跟踪；没有猜写替代。全新检出导入通过，路径回执见 tests/fresh_checkout_receipt.json。运行源码冻结在 source_snapshot.json；两次仅工程修订的影响范围、旧快照和已完成诊断谱系在 audit_history 保存。第二次修订只影响后续救援 RNG/训练日志/恢复清单和中断分类，不改变 D0 输入输出、先验、标签、oracle、数据角色或选型规则。

底座：SS-UIE 官方公开简化实现，非论文完整模型；上游 88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df，官方权重 977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99，strict=True 通过；八个旧非零检查点 SHA 与框架和旁文件一致。底座参数和 BatchNorm 冻结，候选在效用阶段冻结。不得以工程夹具作为正式权重。

环境：{'device': 'PPU-ZW810E', 'device_count': 1, 'device_memory': 51529121792, 'torch_version': '2.0.0a0+nv2303', 'vendor_torch_preserved': True}。厂商 torch 文件 hash 与 V1 一致，未覆盖 torch/torchvision，无 CPU 代替 PPU 测速。

数据：{'model_val': 671, 'model_fit': 3608, 'utility_fit': 443, 'utility_val': 136, 'calibration': 133, 'sealed_eval': 177, 'excluded_overlap': 1}。文件哈希和输入/参考配对逐项核验；内容代理组不跨有效角色，UIEB/917_img_ 排除。LSUI 上游训练逐图成员未知；UIEB nonoverlap 仅限作者公开 LSUI 来源与本地重叠审计。全部历史分析暴露继续保留，177 对是本轮封存而非全新盲测。未解锁前，只进行 calibration/sealed 文件身份审计，无模型评分、候选缓存或视觉输出。

验收：初始数学/协议套件71项通过，恢复日志修补后75项通过；全11方法各2次真实PPU临时更新、严格重载；16个fit图的缓存/实时配对最大差0；连续4更新与2+新进程2的参数/优化器差0且流/采样/LR一致。另核对原两头4000模型/AdamW/旧随机流的CPU恢复与下一批样本；逐字反例20/16.989700/23.010300 dB通过。工程夹具均明确标记，不进入正式表。

预算：沿用累计16设备小时，继承V1的1503.1731119155884秒及其旧账本hash，不另开额度。profile 全11正式作业各3000更新；完整矩阵预测 10905.86 秒，最终保护 15843.74 秒（大于3.5小时，按较高预测保护）。全部设备测试、缓存、重试、训练、评测和时延计账；CPU分析另记。唯一救援最多4小时，不自动延长。共享设备锁、dispatcher锁、原子结果/完整checkpoint、恢复身份拒绝与科学停止幂等均存在。

前置回执可证明工程通过所测契约，不保证候选资格、O收益或特殊机制增量。后续须依次完成全部D0、model_val冻结唯一producer、utility_val迁移检查、条件唯一救援、完整11矩阵、独立CAL和开发闸门；封存评分只在完整DEV_PASS后一次释放。同盘ZIP及本地Git不等于独立备份，independent_backup_verified=false。
