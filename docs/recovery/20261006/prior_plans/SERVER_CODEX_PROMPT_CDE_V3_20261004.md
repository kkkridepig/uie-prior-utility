# 给服务器 Codex 的完整提示词

方案编号 20261004，核验完成 2026-10-05。将本文件和同目录 `CODEX_TASK_FRAMEWORK_CDE_V3_20261004.md` 一起上传服务器。下面正文可以直接作为服务器 Codex 的任务输入；主框架已自包含必要公式、身份、对照和停止规则，不要求上传本地审计缓存。

---

请在我的 `uie-prior-utility` 服务器仓库执行随附 `CODEX_TASK_FRAMEWORK_CDE_V3_20261004.md`。目标是利用已有父权重，修正可验证的数学与实验契约，筛选并确认一个有可归因增量的水下图像增强方法。不要把“生成代码”“训练跑完”或“通过0.10dB门槛”直接称为论文创新成立。

已确定研究组合为 E、D、C。E先执行无训练采样诊断，C为唯一首选算法主线，只有正确完成且数值不支持的C pilot才触发一次D备选。不要把C说成旧实验中分数最好，也不要把E调用现成DDIM称为原创。F/A/B/G不在本轮默认训练范围内。框架对旧审计中的“偏离”作了更正，D浅层phase等是原初筛允许的简化，真正需要修正的是明确违反数学或协议的部分。

按下面顺序自主完成。常规代码、测试和已冻结预算内实验无需反复请求确认。

1. **现场核验。** 查明当前commit、工作区改动、已有进程、设备/环境、父checkpoint及哈希、data manifest、历史评估暴露和旧预算。保留原 `explore_ag_single_seed_v2_20261003` 目录及其冻结结果。对当前状态与文档快照的差异写 `CURRENT_STATE_DIFF.md`，不要覆盖用户改动或中断无关任务。默认历史85000步父模型，SHA256 `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`，若缺失先定位合法原权重，不用随机模型顶替。

2. **先冻结协议。** 新输出放 `runs/prior_utility_cde_v3_20261004/` 和对应docs目录。锁定数据角色、配置、全部阈值、种子、噪声键、主要对照和72设备小时总上限。若用户现行预算更小则用较小者。至少留12小时做最终评估与导出。历史剩余额度不能自动当新额度；不重训所有400000步父模型。记录完整对照组成本预测，再决定是否派发。

3. **数据与种子。** 按场景组将旧训练分为adapter_fit/route_fit/route_cal约70/20/10，保存名单及哈希；旧父模型已见过这些训练图的历史不能被新划分消除。已看过的UIEB/LSUI test标 `legacy_exposed_regression`；在可审计范围寻找真正未参与开发的新留出场景。没有新留出集不妨碍开发实验，但最终状态必须限制。实现真正的 `finetune_seed`，使用20261004/20261005/20261006。数据、训练t/noise、adapter初始化、模式/dropout各用独立可恢复随机流；eval键必须包括eval_noise_seed，不能只改全局manual_seed。主评估101/102/103、效用标签17/29，跨候选共享同图噪声。

4. **E先诊断。** 按框架核对x0/epsilon转换、t/index、t→0、真实NFE。DDIM1、2、4、8、20测试同权重与同噪声；特别记住1步最后调用index999、多步最后调用index0。先在固定24图作轨迹和时间/状态敏感性，再完整开发验证。分别计prior、sampling、模型流程和含I/O服务流程，显存统计从prior之前开始。不训练蒸馏学生；主方法比较仍固定DDIM20。

5. **C数学实现。** 实现 `C_ADD_BANK_SELECT_V3`，保留父模型旧先验通路，在1/4、1/8层加入真正QK^T softmax交叉注意力。主框架维度为64、4头、查询和各类key最多256token。物理20通道沿父physical_coordinate及色彩空间，κD=κB明确继承，不暗中加入A/B。histogram保留颜色bin语义，高频用父冻结特征。输出无bias且零初始化，全invalid安全置零，硬null直接跳过新模块，训练后仍精确恢复父模型。冻结父参数但保留穿过下游网络到adapter的梯度。

6. **C候选和效用。** 模式只有{null,physical,histogram,highfreq,all}，整幅图采样前选一次、全轨迹不改。先train-only均衡训练四个非空模式的bank至固定5000更新，再冻结bank；不要在零初始化上计算无意义的oracle闸门。所有监督utility标签来自route_fit，校准只用route_cal，使用真实最终PSNR差且stop-gradient。selector只看I和实际先验token摘要/有效统计，不看GT、真实分数、污染标签或数据集ID。其CNN+摘要输入116维按主框架实现，推理只执行选中模式的去噪，摘要成本计入。

7. **控制和闸门。** 先跑同seed的BASE_CONT、C_BANK、C_RGB_CONTROL；计算训练后有限bank的oracle相对best-fixed和强基线的空间。oracle先平均三个eval_noise，再逐图取最优模式，仅作不可部署诊断。空间不足按框架停止C。空间存在才补C_ALL_ONLY、同bank的BEST_FIXED、WINNER_CE、SHUFFLED、UTILITY，并用框架的干净质量/污染保护闸门判断。主要选择器表全部无额外阈值argmax；CE logit不是dB，不能套utility的dB拒绝阈值。UTILITY带校准门槛只作部署副表。C初步入围后补BASE_CONT_DATA_MATCHED，达到相应门槛才开始另外两个种子。

8. **多种子和备选。** 只对通过的一个方法补两个冻结协议重复，每个seed重训自己的bank/adapter、标签与选择器，并补齐全部必要控制。默认报告一个pilot+两个重复；若修改过配置，旧pilot不能算同配置三seed，按框架重跑或诚实缺项。不同微调seed不等于独立父预训练。C只有数值失败才能触发一次D；实现/指标/预算阻塞不能冒充科研失败。D按Sobel、hardphase、softphase三个相同adapter控制执行，softphase为Z/sqrt(|Z|²+tau²)、tau取非DC幅值median及floor，统一训练集冻结RMS，不使用每图RMS，不声称这是可靠性概率。D也先pilot过闸门再补seed，不与C叠加。

9. **三层验收。** 先数学/shape/mask/null/FFT/DDIM与标签角色检查，再CPU和PPU梯度/冻结/RNG配对检查，最后100步profile和可恢复训练检查。已有厂商torch与设备环境优先保留；不要静默用CPU速度冒充PPU速度。先验提取、标签生成、评估、失败重试都计设备预算。profile若预测完整控制矩阵超预算，先停止该包，不通过删除对照、缩图或减seed后仍写完成来凑数。

10. **最终评估和交付。** 将代码、配置、权重、sampler、阈值、数据名单和指标实现哈希写入final_freeze后再测新留出集。补齐PSNR/SSIM/LPIPS、逐图/逐scene/逐seed结果、计时与实际参数量；UCIQE/UIQM仅作辅助，缺权重指标明确缺失。保存统一输入/参考/父/强控制/方法面板、固定样例与最差失败例、效用regret和各污染组。旧test只放回归表，不用其结果循环调参。状态必须区分数学通过、pilot信号、同父多seed支持、新留出确认、未支持和资源阻塞。

执行过程中持续更新 `dispatch_state.json`、`budget.json`、`LIVE_STATUS.md`，记录完成项和真实阻塞；启动可恢复本地调度并避免重复派发。每个包达到文档最大更新数、预算或科学停止条件就停止，不用无限训练保证正结果。最终交付 `FINAL_REPORT.md`、运行命令、全部freeze/lineage/数据审计、测试日志、权重与指标/图像导出清单。

全部默认参数和数学细节以 `CODEX_TASK_FRAMEWORK_CDE_V3_20261004.md` 为准。若其中仍有不可执行冲突，先给出源码/数学依据、做不受阻塞的独立任务，别擅自换成更容易出正分的目标。任何扩大预算、丢弃不可恢复状态或改变核心研究目标的动作另行提出具体依据。

现在先完成现场审计、数据/预算计划和dry-run，再按冻结依赖顺序推进到可证实的最终状态。不要只回复计划，也不要把本提示词视为保证已有算法创新的承诺。
