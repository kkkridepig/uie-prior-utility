# 单种子 A–G 接续实施记录（2026-10-03）

记录时间 UTC：2026-10-03T04:08:23Z

本文件是运行中证据快照，不能代替最终完成报告。

## 已核实状态

- 原 UIEB 训练在 225000 步安全保存；旧结果及厂商 PPU 环境保留。
- 固定共同父权重为历史 85000 步，SHA-256：`3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`。
- 仅按 91 张源验证图选父模型；父模型验证 PSNR 为 24.802888 dB，当前统一采样器为 DDIM20。
- 实测有效 batch4：micro1/2/4 平均更新耗时分别为 0.809/0.777/0.765 秒；选 micro4。缓存预热顺序是计时限制。
- 累计设备占用约 10.342/168 小时（含当前任务），最终 Benchmark 仍有独立 16 小时上限；探索上限 152 小时。
- 当前任务：`D_SOBEL_CONTROL_train_5000`。

## 本次读取的真实结果

下表仅为源验证。新增 5000 步的分支才可与同步数对照比较；24 张监测结果不与 91 张完整验证混比。

| 分支 | 状态 | 新增步数 | 完整验证 PSNR | 完整验证 SSIM |
|---|---|---:|---:|---:|
| BASE_CONT | completed_screen | 5000 | 24.660136 | 0.929550 |
| B_PROXY | completed_screen | 5000 | 24.614683 | 0.928754 |
| A_COUPLED | failed_implementation | 0 | 尚无当前步完整结果 | 尚无当前步完整结果 |
| A_SPLIT | completed_screen | 5000 | 24.608107 | 0.928964 |
| C_CONV | completed_screen | 5000 | 24.662445 | 0.929642 |
| C_FIXED | completed_screen | 5000 | 24.640466 | 0.929524 |
| C_ROUTED | completed_screen | 5000 | 24.646420 | 0.929559 |
| D_SOBEL_CONTROL | running | 4000 | 尚无当前步完整结果 | 尚无当前步完整结果 |
| D_PHASE | running | 4000 | 尚无当前步完整结果 | 尚无当前步完整结果 |

## 已实施修复与补充

1. A_COUPLED 首次启动以 SIGSEGV（exit −11）退出，目录未产生断点，原日志没有 Python traceback。不能推断具体算子根因。已保留事件，加入一次有边界重试；若再次失败则保留失败状态，不无限循环。
2. 新恢复任务等待现有设备锁，复用原父模型、训练源码、优化日程、账本和验证逻辑。不会重置已有分支或覆盖原测试冻结。A_SPLIT 的已有结果不是系数分离有效的证据，必须补齐 A_COUPLED。
3. 新增固定源验证扰动诊断：深度位移 2%/5%、16×16 噪声网格 sigma 0.05/0.10、缺失深度；直方图循环错配/缺失；高频先验副本噪声 sigma 0.01/0.03。C 分开全先验与新增路由输入两种作用范围。
4. 扰动诊断已实现且 3 项 CPU 测试通过；尚待独占 PPU 实验，不能称为已跑通或稳健性提升。预设登记晚于初轮干净验证，明确属于补充诊断，不用于改动已冻结测试选择。
5. 只读报告进程每 300 秒更新；主调度、A 对照恢复、补充扰动均有独立 tmux 会话与日志，实际设备任务通过同一锁串行执行。

## 资源与结论限制

- Atlantis 文件已取得并通过配对审计；新增官方证据显示上游生成器使用 UIEB，具体训练名单未公开。因此 B_ADAPT 的间接测试重叠未知；MiDaS 相对逆深度不是真实水下米制真值。
- DUO、RUOD 已重新核查官方/作者关联入口；数据许可和合格冻结水下检测器尚未闭环，G_FEATURE、G0/G1 不声称完成。
- B_PROXY 相对 BASE_CONT 的 5000 步均值变化约 −0.04545 dB；C_ROUTED 对 C_FIXED 约 +0.00595 dB，均未达到框架 +0.10 dB 质量候选门槛。这只描述当前父权重上的短程探索，不能判定方向永久无效。
- LPIPS/无参考指标、检测 AP、B_ADAPT 和外部公平方法参照仍有缺项；不能宣称完整 SOTA Benchmark。

## 可恢复命令与证据位置

在项目目录执行，三个计算调度入口只会取得同一把设备锁后使用 PPU；已有任务运行时不要重复手工启动训练脚本。

```bash
cd /mnt/workspace/uie-prior-utility
# 主任务恢复（仅在原调度器退出后）
PYTHONPATH=. .venv/bin/python scripts/explore_ag/dispatch.py
# 单次 A 对照恢复（有独立去重锁）
PYTHONPATH=. .venv/bin/python scripts/explore_ag/recover_missing_control.py
# 排队的补充扰动（等待 A 恢复终态）
PYTHONPATH=. .venv/bin/python scripts/explore_ag/run_supplemental_stress.py
# 只读刷新报告
PYTHONPATH=. .venv/bin/python scripts/explore_ag/report_watch.py --once
```

预算与状态：`runs/explore_ag_single_seed_v2_20261003/budget.json`、`dispatch_state.json`。
逐方向记录：`docs/experiments/AG_V2_*_RESULTS.md`；实时入口：`AG_V2_LIVE_STATUS.md`。
补充队列：`control_recovery.json`、`supplemental_stress_status.json`（开始执行后创建）；终态结果与缺项保留，不用占位指标填补。
