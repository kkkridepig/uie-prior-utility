# MPA-Diff-Recon S0–S2

2026-09-28 新增的 `mpa_diff/` 是依据工作区 `CODE_RECONSTRUCTION_ARCHITECTURE.md` 独立实现的重建基线。原有 `uie/` 是不同算法，未覆盖。原项目的 Python>=3.10 元数据和 requirements-ppu.txt 不适用于本次 Python3.8/PPU 环境；本包从仓库根目录直接 `python -m` 运行，无需安装旧项目。

## 环境和启动

```bash
cd /mnt/workspace/uie-prior-utility
bash scripts/setup_mpa_ppu.sh
./scripts/mpa_python.sh -m pytest tests/mpa
./scripts/mpa_python.sh -m mpa_diff.cli.prepare --synthetic-fixture
./scripts/mpa_python.sh -m mpa_diff.cli.train --config configs/base/synthetic_smoke.yaml
```

`setup_mpa_ppu.sh` 继承系统厂商 torch，只按需在独立 `.venv` 安装固定版本 OpenCV，使用 `--no-deps`；不安装 torch/torchvision/numpy。CPU 长跑使用启动脚本固定 OMP/OpenBLAS 线程数。正式配置为 `configs/base/mpa_recon_v1.yaml`。`synthetic_smoke.yaml` 明示测试替身和禁用感知损失，仅作工程验证。

## 数据和六次正式训练

将 UIEB 原始/参考图分别放在 `data/uieb/raw-890` 和 `data/uieb/reference-890`；LSUI 配对放在 `data/lsui/input` 和 `data/lsui/GT`。支持在数据 YAML 中改路径。按文件 stem 配对，不能按目录列出顺序配对。输入/参考图尺寸必须一致。

```bash
./scripts/mpa_python.sh -m mpa_diff.cli.prepare --config configs/data/uieb_recon.yaml
./scripts/mpa_python.sh -m mpa_diff.cli.prepare --config configs/data/lsui_recon.yaml
./scripts/mpa_python.sh -m mpa_diff.cli.stages
./scripts/mpa_python.sh -m mpa_diff.cli.stages --execute
```

划分 seed=20260927；训练 seed=20260927、20260928、20260929。UIEB 702/91/97、LSUI 3423/429/427，为重建划分，非原作者名单。已知场景以 groups JSON 给出，分组划分必须另命名 `grouped`，不能为了凑数拆场景。图像16×16灰度缩略图仅用于近重复筛查，不保证发现所有同场景样本。未知场景元数据保持 null。

S2 启动前检查全部依赖，不使用示例数据冒充完整数据。每次正式训练400000步，默认336×336、Adam、全局batch4、DDPM1000步；验证每5000步，最佳验证PSNR（同分保留较早权重）。六次训练顺序运行；不自动调小预算。未完成训练的结果不能称 S2 完成。非参考指标/LPIPS在当前评测输出为null及原因，PSNR/SSIM为已独立验证的实现。

## 推理、恢复、缓存、评测

```bash
./scripts/mpa_python.sh -m mpa_diff.cli.train --config configs/base/s1_smallfit.yaml --resume runs/s1/real_prior_smallfit/last.pt
./scripts/mpa_python.sh -m mpa_diff.cli.enhance --checkpoint runs/s1/real_prior_smallfit/best.pt --input data/s1_author_examples/input/104.jpg --output outputs/mpa_example.png --device cuda
./scripts/mpa_python.sh -m mpa_diff.cli.evaluate --checkpoint runs/s1/real_prior_smallfit/best.pt --split test --output runs/s1/evaluation --device cuda
./scripts/mpa_python.sh -m mpa_diff.cli.cache --config configs/base/mpa_recon_v1.yaml
./scripts/mpa_python.sh -m mpa_diff.cli.benchmark --checkpoint runs/s1/real_prior_smallfit/best.pt --sizes 256 512 1080p --device cuda --output runs/s1/benchmark.json
```

路径支持中文/空格（shell中需加引号）。推理只有输入图像，不读取reference。缓存键含变换后图像内容、尺寸、深度权重/方向/预处理和完整静态配置；只缓存深度/背景/直方图/边缘/小波，不缓存可训练beta。恢复严格检查数据、配置和源代码哈希；保存optimizer/scheduler/全平台RNG/数据顺序，无AMP时scaler为null。仅对本工程生成的可信checkpoint使用torch.load。

## 对应规格和取舍

- §4：Haar、Sobel、分块RGBuv直方图、PIL量化Gaussian背景；冻结DA Small、原始逆深度minmax的基线兼容语义；保留方向修正的distance_proxy另字段；beta与主干联合训练。
- HIN U-Net显式skip栈，n_blocks=1/2均测试；中间块无时间嵌入；高频在浅层注入一次；极小/奇数图pad后裁回。
- §5：float64构造后float32存储噪声计划，数学t=0边界，x0/epsilon/v转换，DDPM fixed_small和DDIM。DPM/UniPC属于后续E方向，不冒充已实现。
- VGG19为官方权重，relu1_2/relu2_2/relu3_4、ImageNet归一化、增强分支有梯度。训练数据当前是全有效配对图，损失先裁去padding；任意空洞监督mask的VGG感受野腐蚀尚未实现，不支持该训练协议。
- PSNR逐图平均；SSIM RGB人口方差、11窗sigma1.5、valid边界，已与独立SciPy计算对照。PNG量化指标另列；完美PSNR以字符串`+inf`导出。学习式指标和未审计无参考指标留空。
- A–G的后续研究范围保留在上级架构文档。此实现专注S0–S2；未实现扩展会拒绝配置，不放空类伪装完成。SeaDiff审计配置也未伪装为可兼容训练配置。

## 来源

核心数学和网络代码独立实现，默认值来源标签见原架构第2–5节。DA官方代码单独放 `third_party/Depth-Anything-V2`，固定commit `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`，保留其Apache-2.0 LICENSE；Small权重Apache-2.0，哈希及获取URL在weights/*.receipt.json。VGG19官方torchvision权重另存来源和哈希。原工程LICENSE不会替代第三方权重或数据许可。

UIEB官方页明确学术用途、禁止再分发。本地仅供实验，不向仓库提交数据/权重。LSUI作者仓库示例用于S1工程验收，不作为S2训练/测试报告。下载/训练日志、失败记录及环境快照均在runs/。

## 2026-09-28 全数据接入更新

全实验下载清单见 [BENCHMARK_DOWNLOADS.md](BENCHMARK_DOWNLOADS.md)。上传包检查记录在 `runs/data_intake/`。正式训练需要压缩包完整性、图像/配对数量和划分审计全部通过；加密包缺密码或不完整ZIP不能启动S2。

为控制30GiB磁盘开销，缓存已改为按单张图无损压缩保存float32原始深度及有效mask（depth_v2）；其余静态先验重算，不改变模型。验证仅写逐图指标、汇总和最差样本名单；最终测试保存所有输出图，完整浮点诊断保留名单前10张（预先固定，不按分数挑选）。23项测试含缓存数值等价和断点恢复检查通过。旧S1检查点仍可推理，源代码哈希变化后不可直接续训，须使用新的运行目录。

## 当前正式运行与记录位置（数据补齐后）

新上传的LSUI_1.zip已校验，UIEB890/LSUI4279已齐备。由于原包内有重复/近重复图，采用`configs/data/uieb_grouped.yaml`和`lsui_grouped.yaml`，冻结名单`manifests/*_recon_grouped_v1.jsonl`；数量分别702/91/97、3423/429/427，名单不等于未分组随机划分。

- 记录总目录：`docs/experiments/`。每个种子独立Markdown：`s2_grouped_v1__<dataset>__seed_<seed>.md`；状态/训练观测每25步更新，正式测试完成后补结果及分析。
- 运行目录：`runs/s2_grouped_v1/`，总日志`queue.log`，队列状态`queue_status.json`，启动PID/命令`launcher.json`；源码/配置/清单快照`source_snapshot/`。
- 当前运行是完整预算的6次实验队列，尚未完成。S0/S1历史结果已各自补建独立记录，文件名以S0_/S1_开头。
- 状态查看：`tail -n 20 runs/s2_grouped_v1/queue.log`。
- 服务器重启或明确失败排查后恢复：`./scripts/mpa_python.sh scripts/launch_s2.py`。已有进程时该脚本拒绝重复启动，训练器从last.pt恢复；源代码/配置变化会拒绝续跑，以免混用实验身份。
- 准备检查：`./scripts/mpa_python.sh -m mpa_diff.cli.stages --profile grouped`。直接同步运行可加`--execute`，但运行中不要重复启动。
- 此前原始接入记录`docs/experiments/S0_S2_DATA_INTAKE.md`保留为历史；新状态见`S0_DATASETS_REUPLOAD_GROUPED_20260928.md`，不以旧阻塞状态代表当前情况。
