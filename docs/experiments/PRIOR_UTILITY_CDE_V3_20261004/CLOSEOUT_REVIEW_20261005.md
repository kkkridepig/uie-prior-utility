# CDE V3 收束核验与结果解释（2026-10-05 UTC）

本轮已达到预注册科学停止条件，状态为 **`pilot_no_supported_gain`**。C为 `utility_not_learned`；D一次备选也未过闸门。没有胜出方法，没有启动20261005/20261006重复，没有确认算法创新。继续训练或更换目标需要另立研究协议，不能以尚有剩余预算为由无限续训。

## 实际执行范围

| 部分 | 实际完成 | 证据状态 |
|---|---|---|
| 数学与工程 | 数学、mask/null、FFT/DDIM、CPU/PPU梯度、RNG、恢复检查；62项测试 | 工程验收通过，不代表质量有效 |
| E | 双权重、24图轨迹/91图开发验证，DDIM1/2/4/8/20及子集DDPM；无需训练 | 工程诊断完成；现成DDIM不是原创 |
| C bank及控制 | BASE_CONT、BANK、RGB、ALL_ONLY，各5000更新 | 已实现、已跑通 |
| C选择器 | route_fit效用标签、route_cal校准；UTILITY/CE/SHUFFLED各2000更新；完整91图×3噪声×9干净/污染组 | oracle有空间，输入驱动效用收益未支持 |
| D | Sobel、hardphase、softphase各5000更新，完整噪声诊断 | 已跑通，未支持晋级 |
| 多种子/数据匹配控制 | 未启动 | 科学闸门未过；不是缺算力或漏派发 |
| 最终测试 | 391张LSUI候选留出、97张UIEB旧test、427张LSUI旧test，各3噪声 | 仅PARENT与BASE_CONT的冻结后评估，没有C/D方法确认 |

实际只有一个微调seed `20261004`，共享历史85000步父模型；所有新增分支独立从同一父权重开始。20261005、20261006仅为原计划，未执行。`BASE_CONT_DATA_MATCHED`的前置条件为C通过初筛，本轮不满足，因此未跑，不能宣称获得该控制支持。

## C：oracle互补性没有转化为可部署收益

源开发集91图，三噪声101/102/103先逐图平均，再跨图汇总；各模型DDIM20、float32、统一channels_last推理。BANK四种非空模式各1250次更新，总5000次。

| 模型/模式 | PSNR dB | SSIM | LPIPS↓ |
|---|---:|---:|---:|
| 父模型/BANK hard-null | 24.802890 | 0.929962 | 0.093624 |
| BASE_CONT_V3 | 24.637907 | 0.929084 | 0.089562 |
| BANK physical | 24.628951 | 0.929748 | 0.093223 |
| BANK histogram | 24.594158 | 0.929319 | 0.093039 |
| BANK highfreq | 24.601134 | 0.929608 | 0.093499 |
| BANK all / C_BEST_FIXED | 24.605860 | 0.929554 | 0.093292 |
| C_RGB_CONTROL | 24.636863 | 0.930015 | 0.092507 |
| C_ALL_ONLY | 24.689732 | 0.929878 | 0.090855 |
| C_WINNER_CE | 24.713107 | 0.929768 | 0.092757 |
| C_SHUFFLED | 24.669428 | 0.929656 | 0.093019 |
| C_UTILITY，主表无额外阈值 | 24.603132 | 0.929809 | 0.093242 |
| C_UTILITY_CALIBRATED，副表 | 24.608607 | 0.929794 | 0.093299 |

不可部署oracle为25.167520 dB，比DEV_ORACLE_FIXED（开发集最佳固定null）高0.364630 dB，比BASE_CONT高0.529613 dB，因此正确进入选择器阶段。部署C_BEST_FIXED则由route_cal选择为all，不能看过开发集后改成null来选模型。这两个“最佳固定”用途不同，报告必须区分。

UTILITY相对C_BEST_FIXED为 **−0.002728 dB**，相对CE为−0.109975，相对SHUFFLED为−0.066296，相对结构强控制ALL_ONLY为−0.086600，均未达到冻结门槛。SSIM/LPIPS保护及8个route-only污染组的保护通过，不能以此替代失败的干净质量门槛。

对BEST_FIXED的逐图中位差为0，改善44/91图（48.35%），去掉最高5个改善图后均值−0.031369 dB。按场景组1000次bootstrap的95%探索性区间为[−0.042241, +0.036780] dB。这不能证明一般效用路由无效，但不支持当前固定配方晋级。

实际选择模式：null 5、physical 55、histogram 18、highfreq 9、all 4。非null覆盖率94.51%；在86张接受新增分支的图中，44张相对null损失超过0.1 dB，条件比例51.16%（占全部91图48.35%）。四种非空效用预测的开发MSE为2.632153 dB²，平均oracle regret为0.564388 dB。选择器经常接受实际负效用，说明“存在互补性”和“输入能准确预测互补收益”之间的证据链没有成立。标签来自route_fit，开发统计只是诊断，不能回流训练或重新搜索阈值。

全部污染组的模式分布、null率、负效用接受率、效用误差和regret在`closeout/selector_diagnostics.csv`；固定阈值0/0.05/0.10/0.20的描述性risk-coverage表在`closeout/risk_coverage.csv`。这些为既有分数的CPU统计，不改变route_cal已选的0.05 dB部署副表门槛。

## D：一次合法备选未支持

C完整执行且指标齐全后的数值失败触发D，见`D_trigger.json`，不是把实现阻塞当成科研失败。

| 分支 | PSNR dB | SSIM | LPIPS↓ |
|---|---:|---:|---:|
| BASE_CONT_V3 | 24.637907 | 0.929084 | 0.089562 |
| D_SOBEL_STD | 24.728613 | 0.929349 | 0.093292 |
| D_PHASE_STD | 24.731435 | 0.929239 | 0.093242 |
| D_SOFTPHASE_STD | 24.731184 | 0.929288 | 0.093248 |

softphase相对phase为 **−0.000251 dB**，相对Sobel为+0.002571，相对BASE_CONT为+0.093277。相对最强phase控制没有+0.10 dB增量，故停止。softphase−phase的场景bootstrap区间[−0.005052,+0.004711] dB，不能解释为softphase稳健优越。

σ=0.005/0.02/0.05的噪声仅施加到新增条件，父条件保持干净；全部2457条记录包含τ、FFT虚部残差、标准化前后RMS、PSNR/SSIM/LPIPS和边缘误差，见`pilot/20261004/D_noise/`。冻结RMS未逐图重估。不把这张诊断表改成新的晋级目标。

## 最终测试究竟测了什么

没有候选晋级，最终冻结的模型列表只有PARENT、BASE_CONT_V3。以下增量全部属于继续训练控制，不属于C/D创新。

| 数据角色 | 图数 | PARENT PSNR | BASE_CONT PSNR | ΔPSNR | ΔSSIM | ΔLPIPS↓ |
|---|---:|---:|---:|---:|---:|---:|
| LSUI候选留出，仅基线评估 | 391 | 20.986554 | 21.748298 | +0.761743 | +0.006057 | −0.004553 |
| UIEB历史暴露回归 | 97 | 24.380012 | 24.921948 | +0.541936 | +0.006540 | −0.010796 |
| LSUI历史暴露回归 | 427 | 21.510031 | 22.278600 | +0.768568 | +0.007334 | −0.004139 |

391张图只在既有日志和近重复筛查可审计范围内曾未评分；不是新数据域，DA/VGG/LPIPS上游暴露未知。现已读取基线分数，今后不能再无条件称它们为未触碰的开发独立留出。未对C/D做留出测试，因此没有新方法确认结果。

最终共915图×3噪声=2745条记录，每条包含两个模型。保存了三个数据角色各12张固定样例、最差10图及相对父模型退化最大10图；合并去重后89张，生成267张完整/局部/差图面板。因为没有最终新方法，面板列为输入、参考、PARENT、PARENT、BASE_CONT；重复父列是基线导出模板，不能误解为有另一个强控制或C/D方法面板。全图输出没有全部保存，保留的是逐图指标及固定/失败样例图像。

## 预算、成本和参数量

| 阶段 | 实际设备小时 | 包上限 |
|---|---:|---:|
| P0 | 0.155785 | 4 |
| E，含安全分段续跑 | 2.811094 | 4 |
| C pilot | 5.116798 | 18 |
| D备选pilot | 3.188144 | 8 |
| FINAL，仅两个基线 | 0.376242 | 12 |
| 多种子重复 | 0 | 22 |
| 总计 | **11.648062** | **72** |

正式新增增强器更新7×5000=35000；三个选择器3×2000=6000更新，另有单列profile。BASE_CONT可训练5,995,762参数；C BANK/RGB/ALL_ONLY各149,248新增参数；各selector 37,524参数；D各10,112新增参数。参数迁移、父身份和每次实际成本分别在run_config、checkpoint identity、budget events中。父参数冻结不妨碍反向通过父网络到adapter。

72小时是上限，不是必须用满的目标。原最终预留12小时未被训练侵占；最终只需基线评估，故实际成本远低于完整三seed/九模型预测。这不是删对照以凑预算，而是预注册闸门没有选出方法。CPU收束核验额外设备耗时为0，CPU耗时另记`closeout/verification.json`。

## 收束核验与报告勘误

1. 原artifact_manifest的4406项全部核对哈希；final_freeze、训练源码、所有冻结模型/辅助文件仍匹配，旧V2未改。
2. CPU加载11个交付权重：父权重、7个5000步增强器checkpoint、3个2000步selector；参数及Adam张量有限。5000步结束状态完整。
3. 七个增强器全部5000步的数据ID、t、noise哈希、dropout种子逐步相同；模式使用独立流。不是只抽查前20步。
4. 关键逐图文件数量正确、复合键无重复、PSNR/SSIM/LPIPS有限；三噪声逐图汇总再按scene汇总。
5. 原`development_decision.json/final_freeze.json`的`seed_claim`、`final_summary.json`的`evidence_scope`含通用“三seed”模板文字，与`seeds=[20261004]`不符。实际只有pilot，两个重复未启动。保留冻结原件，通过本报告和`closeout/verification.json`明确更正；不改历史哈希。
6. 原`summary.json`中LPIPS的`improved_fraction`误用raw delta>0，方向相反；原始LPIPS值、均值、CI和科学闸门没有使用该字段，因此不改变闸门结论。正确改善比例为LSUI旧test 58.7822%、UIEB旧test 70.1031%、LSUI候选留出60.6138%；删去最高改善5图的方向也已在补表修正。以`closeout/final_statistics_corrected.json`为准。
7. 原artifact_manifest未列JSONL/log/XML；新增`closeout/evidence_inventory.json`覆盖这些证据及CSV的路径、大小与哈希，原manifest保留。

当前冻结训练/指标源码不改；CPU补充统计在独立`tools/cde_v3_closeout.py`，不会触发重新推理、选择或训练。自动生成的最终报告保留为`FINAL_REPORT_AUTOMATED_SNAPSHOT.md`、`FINAL_METRICS_AUTOMATED_SNAPSHOT.md`，原预算停止另见`FINAL_REPORT_PREFLIGHT_STOP.md`。

## 交付和后续边界

入口：`FINAL_REPORT.md`；原始运行证据根目录：`runs/prior_utility_cde_v3_20261004/`；方便表格软件读取的结果在`closeout/`。权重清单和SHA256见`closeout/weights.csv`，新增权重是delta，需要历史父权重与配置，不能当独立全模型加载。

完成了当前单seed停止路径所需的科学比较，但并非原计划的所有条件分支都执行。未执行项：DATA_MATCHED、两个重复、C/D最终Benchmark与面板；可选all-condition污染未做；无可靠权重的URanker未报告。UCIQE/UIQM没有加入本轮主表，不用无参考指标填补PSNR/SSIM/LPIPS证据。

后续建议属于新研究计划而非本轮自动追加：先基于现有route_fit/cal和开发记录调查效用标签分布、负效用识别及bank模式冗余；任何改架构/训练目标/阈值的实验都应重新冻结，并重新处理现有留出曝光。D当前微小差异不足以支持扩大训练。E可以作为工程收益保留，主算法比较仍DDIM20。现有结果不支持宣称论文创新成立或SOTA。

复核命令（不使用PPU）：

```bash
cd /mnt/workspace/uie-prior-utility
.venv/bin/python scripts/cde_v3/status.py --json
.venv/bin/python tools/cde_v3_closeout.py
```

精确历史训练/评估命令在budget.json events内。不要为了看到“running”而重新启动已达到科学终态的pipeline。
