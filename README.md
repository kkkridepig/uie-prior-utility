# UIE Prior Utility

面向阿里云 DSW PPU 的水下图像增强实验项目：WWE 粗恢复 + 残差 Flow Matching / DDIM + 局部先验效用预测。

本项目要检验的问题是：**物理先验在不同区域可能有益或有害，能否预测固定恢复模型使用该先验后的实际收益，再控制先验注入？** 这是可运行的研究起点，尚未得到真实 benchmark 训练结果，不宣称 SOTA 或已证明有效。

- [数据下载与划分](docs/DATASETS.md)：UIEB、LSUI、UFO-120、C60、U45。
- [算法、消融与实验协议](docs/EXPERIMENTS.md)：效用标签、dense/global、FM/DDIM、跨数据集评估。
- [验证记录](docs/VALIDATION.md)：本地已通过的测试与尚未验证的 PPU 部分。

## 来源与改造

基于官方 [WWE-UIE](https://github.com/chingheng0808/WWE-UIE)（作者标注 WACV 2026，Apache-2.0），固定上游版本：

~~~text
fd0697558203f9f849800cc13b804116bb3f921c
~~~

原始 model.py 原样保留在 [uie/vendor/wwe.py](uie/vendor/wwe.py)，来源与 SHA-256 见 [UPSTREAM.json](UPSTREAM.json)，许可见 [LICENSE](LICENSE)、[NOTICE](NOTICE)。选择它是因为源码开放、网络较轻量、官方推荐 PyTorch 2.4.0，并覆盖本项目选择的数据集。

新增代码包括残差生成网络、先验分支、局部效用头、数据审计、三阶段训练、校准、评估和 PPU 安装脚本。**本项目使用了 WWE 主干，但没有复现 WWE 原论文的完整训练损失、评估实现和预训练权重。** 论文比较时应称“WWE backbone + our training recipe”；正式与 WWE 论文比较需另行按其官方配方复现。

## 在 PPU 服务器开始

以下为 Linux Bash 命令，使用实际服务器项目目录：

~~~text
/mnt/workspace/vads-vke/vlm/yjl/uie-vla/uie-prior-utility/
~~~

代码已经放在该目录时，直接进入目录执行首次环境安装：

~~~bash
cd "/mnt/workspace/vads-vke/vlm/yjl/uie-vla/uie-prior-utility"

# 先退出可能覆盖系统 torch 的旧项目虚拟环境。
# 若系统解释器不是 python3，请明确设置 UIE_SYSTEM_PYTHON。
UIE_SYSTEM_PYTHON=python3 bash scripts/bootstrap_ppu.sh
source environments/ppu/bin/activate

python -m uie smoke --output artifacts/ppu-fp32 --device cuda
python -m uie doctor --device cuda --precision bf16
python -m uie smoke --output artifacts/ppu-bf16 --device cuda --precision bf16
python -m uie smoke --output artifacts/ppu-ddim --device cuda --sampler ddim
~~~

每个 smoke 输出目录需要是新的。FP32 成功后再验证 BF16；若 BF16 失败，先使用默认 FP32。不要把 CPU smoke 的通过当作 PPU 通过。

安装策略针对提供的历史环境：Python 3.10、系统 PPU PyTorch 2.4.0、torchvision 0.19.0、torch.cuda 兼容接口。脚本创建带 system-site-packages 的虚拟环境，只安装 [requirements-ppu.txt](requirements-ppu.txt) 中的辅助包，并核对安装前后的 torch 路径完全相同。它不会安装 torch、torchvision、Triton、FlashAttention 或 CUDA 扩展。模型使用普通 PyTorch 算子，未启用 torch.compile、fused optimizer 或多卡 DDP。

环境安装完成后，每次打开新的服务器终端，先执行下面两行，再运行本文后续下载、训练、续训和评估命令：

~~~bash
cd "/mnt/workspace/vads-vke/vlm/yjl/uie-vla/uie-prior-utility"
source environments/ppu/bin/activate
~~~

所有相对路径均以该项目目录为起点。默认文件位置如下；脚本会自动定位项目根目录，无需把服务器绝对路径写进 Python 源码或 YAML 配置。

| 内容 | 实际服务器位置 |
|---|---|
| 虚拟环境 | /mnt/workspace/vads-vke/vlm/yjl/uie-vla/uie-prior-utility/environments/ppu/ |
| 下载的压缩包 | /mnt/workspace/vads-vke/vlm/yjl/uie-vla/uie-prior-utility/downloads/ |
| 数据集与划分清单 | /mnt/workspace/vads-vke/vlm/yjl/uie-vla/uie-prior-utility/data/ |
| 一键 UIEB 训练输出 | /mnt/workspace/vads-vke/vlm/yjl/uie-vla/uie-prior-utility/outputs/UIEB/seed42/flow-dense/ |
| 冒烟验证输出 | /mnt/workspace/vads-vke/vlm/yjl/uie-vla/uie-prior-utility/artifacts/ |

首次安装时的 python3 应能导入系统 PPU torch；如需其他解释器，通过 UIE_SYSTEM_PYTHON 指定。不要照搬旧项目的 NVIDIA CUDA wheel 安装命令。NumPy 固定为 1.26.4，以避开本地 torch 2.4 与 NumPy 2 的互操作问题。

## 下载代表性 benchmark

### 方案 A：先从官方 UIEB 数据跑通

~~~bash
python -m uie download uieb-raw --output downloads --extract-to data/uieb
python -m uie download uieb-reference --output downloads --extract-to data/uieb
bash scripts/prepare_uieb.sh
python -m uie audit data/manifests/UIEB.json

bash scripts/run_pipeline.sh UIEB 42
~~~

这套 UIEB 划分是固定 seed 的自定义划分，**不是官方 U90，也不是 MPA-Diff 的 U97**。测试约 10%、验证约 10%，再从剩余训练数据抽约 5% 做独立校准；实际数量取决于精确重复与 scene 分组，命令会打印数量并保存清单。

### 方案 B：扩展到五个数据集

WWE 作者提供了整理后的数据包，包含 UIEB、LSUI、UFO-120 以及无参考集合：

~~~bash
python -m uie download wwe-bundle --output downloads --extract-to data/wwe
bash scripts/prepare_wwe.sh
python -m uie audit data/manifests/UIEB.json

bash scripts/run_pipeline.sh UIEB 42
bash scripts/run_pipeline.sh LSUI 42
bash scripts/run_pipeline.sh UFO-120 42

# 使用 UIEB 权重做本域、跨域及先验干扰评估，不在目标测试集微调
bash scripts/evaluate_suite.sh UIEB 42
~~~

**A、B 选择一种划分来源用于同一轮实验。** 两者都写 data/manifests/UIEB.json，已有清单会拒绝覆盖。若要比较两套协议，使用独立工作目录或显式指定不同 manifest 与 output。方案 B 保留作者 test，从作者 train 导出 val/calibration；作者原 val 不参与本项目训练，避免未知上游划分产生混用。官方数据入口、配对结构、UFO 尺寸处理和手动导入方法见 [DATASETS.md](docs/DATASETS.md)。

Google Drive 在部分网络不可访问或会触发配额。实际本地 C60 下载发生连接超时，因此这里只确认来源页面和下载器逻辑，**不承诺当前网络可以下载完整数据包**。可在能访问官方页面的机器下载后，把 UnderWaterDataset.zip 上传到项目的 downloads/ 目录，再执行：

~~~bash
python -m uie extract downloads/UnderWaterDataset.zip --output data/wwe
bash scripts/prepare_wwe.sh
~~~

数据集遵循原作者许可。代码仓库不包含数据、模型权重或训练输出。

## 三阶段训练

| 阶段 | 更新参数 | 默认配置 | 检验内容 |
|---|---|---|---|
| baseline | WWE 粗恢复网络 | 20,000 optimizer steps，batch 8 | 基础颜色与细节恢复 |
| flow | 残差场 + patch 条件；粗恢复冻结 | 20,000 steps，batch 8 | FM 或 DDIM 是否提供额外收益 |
| utility | 仅效用头；整个恢复器冻结 | 5,000 steps，batch 4 × 累积 2 | 预测有/无先验的局部收益 |

这些是初始实验超参数，未在 PPU 或真实 UIEB 上调优。默认 256 × 256 随机裁剪训练，验证/默认评估统一 resize 为 256 × 256。每阶段按 val PSNR 选择 best.pt；温度只在 calibration split 拟合。test 不参与选模型。

一键脚本保存到 outputs/UIEB/seed42/flow-dense，顺序执行三阶段、校准、测试。重跑会从 last.pt 续训；完成的阶段会跳过计算。建议先保持默认 batch 8，不依据历史 96 GiB 显存直接放大 batch。

~~~bash
# BF16 通过 PPU smoke 后才这样启动
UIE_PRECISION=bf16 UIE_BATCH=8 UIE_WORKERS=4 \
  bash scripts/run_pipeline.sh UIEB 42

# 同结构的 DDIM 对照，输出独立保存
UIE_SAMPLER=ddim bash scripts/run_pipeline.sh UIEB 42

# Windows 或 DataLoader 排错时可设 workers=0
UIE_WORKERS=0 bash scripts/run_pipeline.sh UIEB 42
~~~

也可以分阶段执行，便于检查中间结果：

~~~bash
python -m uie train --config configs/baseline.yaml
python -m uie train --config configs/flow.yaml
python -m uie train --config configs/utility.yaml

python -m uie calibrate \
  --checkpoint outputs/UIEB/seed42/utility/best.pt \
  --manifest data/manifests/UIEB.json \
  --output outputs/UIEB/seed42/utility/calibrated.pt --device cuda

python -m uie evaluate \
  --checkpoint outputs/UIEB/seed42/utility/calibrated.pt \
  --manifest data/manifests/UIEB.json \
  --output outputs/UIEB/seed42/evaluation \
  --device cuda --steps 4 --diagnostics --save-images
~~~

分阶段配置与一键脚本使用不同输出目录，上述命令各自构成完整流程。续训必须沿用原配置、manifest 和 schedule_steps：

~~~bash
python -m uie train --config configs/flow.yaml \
  --resume outputs/UIEB/seed42/flow/last.pt
~~~

初次训练若只想试运行 100 步，可在独立 output 使用 --steps 100 --schedule-steps 20000；续训提高 steps，保持 schedule_steps 不变。只加载自己信任的 checkpoint，文件包含 optimizer/RNG 等 Python 状态。

## 评估与输出

~~~bash
# 只跑已经准备好的 UIEB、LSUI 和 U45，减少首轮耗时
UIE_TARGETS="UIEB LSUI U45" UIE_NFES="2 4" \
  bash scripts/evaluate_suite.sh UIEB 42

# 原始尺寸评估需要独立校准；这也会提高显存和运行时间
UIE_SIZE=0 UIE_NFES=4 bash scripts/evaluate_suite.sh UIEB 42
~~~

suite 为每个 NFE 分别在源数据 calibration split 校准温度；训练效用头的标签默认仍来自 NFE=4，因此其他 NFE 同时考察了效用预测的迁移。源 test 比较 none/all/fixed/utility；跨数据集保留权重；另测 clean/color/shift/missing/invert。缺失 manifest 会明确显示 Skipped。默认套件运行次数较多，可用上述环境变量缩小范围。

每个评估目录包含：

- summary.json：指标、NFE、精度、数据/权重 SHA-256、运行环境、batch=1 的 p50/p95 延迟和推理显存。
- per_image.csv：逐图 PSNR、SSIM、MAE、原图对照、uciqe_lab_v1；诊断模式增加相对无先验输出的退化率与差值。
- images/、gates/：使用 --save-images 时保存恢复图与门控热图。
- utility 诊断：Brier、10-bin ECE、可靠性分箱；这些是描述性统计，像素不能当作独立统计样本。

无参考集合只报告无参考诊断和可视化，不生成不存在的 PSNR/SSIM。uciqe_lab_v1 是明确写出的 CIELab 实现，不能与不同 UCIQE 实现的论文数字混排。暂未集成 LPIPS、UIQM、URanker、检测或 SLAM 指标。

## 当前实现的边界

- 默认先验是低对比度/红色缺失启发式、固定有效衰减率和环境光估计。它不是测量深度或完整多维物理模型；可以输入对齐的相对深度 NPY。
- dense 条件是小型 CNN patch-grid，global 对照对同一网格做全局平均。没有接入 DINO、CLS token、Patch Policy 控制器或 Wan。
- 效用头预测固定求解器使用先验后的局部收益，在初始状态预测一次 gate；它不是时间世界模型或每步预测未来轨迹的 Foresight。
- 二元收益标签加 soft gate 不保证最优插值，也不保证所有图像改进。必须报告相对关闭先验的退化率，比较普通重建监督 gate，并检查有益样本是否足够。
- FM/DDIM 共享网络容量，但训练目标、路径和噪声尺度不同；等 NFE 不是等质量或等训练成本。
- 本地 CPU 测试证明流程能运行，不证明 PPU 算子完整兼容、数据下载可达或算法有效。上线训练前执行上面的 PPU smoke。

## 本地开发验证

仅在普通 CPU 开发环境安装 CPU torch；**以下 torch 安装命令不要用于 PPU 环境**：

~~~bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ppu.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest
python -m uie doctor --device cpu
python -m uie smoke --output artifacts/dev-flow --device cpu
python -m uie smoke --output artifacts/dev-ddim --device cpu --sampler ddim
~~~

GitHub Actions 在 Python 3.10 + PyTorch 2.4 CPU 上执行测试和 smoke。数据/outputs/checkpoints/虚拟环境已被 .gitignore 排除。
