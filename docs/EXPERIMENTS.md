# 算法与实验协议

## 首个可证伪问题

固定物理候选不一定可靠。我们检验：在相同输入、相同恢复器和相同初始噪声下，预测“注入先验是否降低局部恢复误差”，能否减少错误先验导致的退化，并保留有益先验的收益？

先检验收益是否存在，再讨论复杂 Foresight 或外部大模型。若 all 与 none 输出几乎一样，效用标签全为零，首先检查恢复器是否学会利用先验，不能靠增加门控头复杂度解释实验失败。

## 架构与训练

~~~mermaid
flowchart LR
    I[退化图像 I] --> B[WWE 粗恢复 b]
    I --> P[固定先验 p]
    I --> C[空间条件 dense / global / none]
    I --> U[效用头]
    B --> U
    P --> U
    N[初始噪声] --> U
    U --> G[局部 gate]
    G --> F[残差场 FM / DDIM]
    P --> F
    C --> F
    B --> F
    N --> F
    F --> J[最终恢复图]
~~~

1. baseline：训练 b=B(I)，损失为 Charbonnier + 0.1 × 一阶梯度 L1。
2. flow：冻结 B，训练残差生成器。先验 dropout 包含全图和局部平滑 mask，防止恢复器只适应 gate=1。条件分支是 CNN stride-4 网格；global 对同一网格平均，none 置零。
3. utility：固定整个恢复器，以配对参考图构造最终恢复收益标签，仅训练门控头。

### 残差 Flow Matching

~~~text
b = clip(B(I), 0, 1)
r1 = J - b
r0 = sigma * epsilon, epsilon ~ N(0, I)
rt = (1-t) * r0 + t * r1
v* = r1 - r0
L = MSE(v_theta(rt,t,I,b,g*p,c), v*)
    + 0.1 * reconstruction(b + rt + (1-t)*v_theta, J)
~~~

推理以 Euler 从 t=0 积分至 1，输出 clip(b+r1_hat,0,1)。默认 sigma=0.05。没有用目标图或真实深度作为推理时必须输入。

### DDIM 对照

使用完全相同的残差网络容量，学习 clean residual x0；cosine alpha_bar、eta=0、标准高斯初始噪声，时间从 1 向 0。训练损失为 x0 MSE + 同权重重建损失。未做分类器指导、预训练扩散模型蒸馏或大规模 diffusion 调参。

它是本项目的条件残差 DDIM 基线，不等同于任意既有 UIE diffusion 论文。FM/DDIM 的噪声尺度与目标参数化不同，结论首先只能说明“这两套具体配置”的质量/速度差别。建议后续对 FM sigma 做 0 / 0.05 / 1 敏感性实验、检查 DDIM 的 x0/epsilon 参数化，再归纳生成范式的差别。

### 效用标签

对同一个样本和同一个初始噪声，运行固定 NFE 的完整求解器：

~~~text
J_off = Solver(I, b, p, gate=0, epsilon)
J_on  = Solver(I, b, p, gate=1, epsilon)
u = AvgPool_9( mean_RGB |clip(J_off)-J| - mean_RGB |clip(J_on)-J| )
y = 1[u > delta], delta=0.001
L_utility = BCEWithLogits(U(I,b,epsilon,p), y)
g = sigmoid(logit / T)
~~~

使用实际终点输出，且所有生成器参数 stop-gradient。标签是在当前固定恢复器下的模型内对照，不等同于真实世界的物理因果效应。

T 在独立 calibration split 从固定候选集合选择；不使用 test。BCE 未加类别权重，以免改变概率含义。收益正负分布应从诊断中的 reliability 分箱与实际样本核查；Brier/ECE 很低也可能仅因标签几乎全零。

gate 只在初始状态预测一次，不逐步更新；它预测“使用先验有益”的概率，不直接预测最优残差速度。因此不把这份实现命名为已验证的世界模型，也没有使用 Wan2.2 监督。

全开/全关的二元标签不足以证明空间 soft gate 最优：网络存在跨像素作用，局部 gate 会改变全局输出。必须同时看最终 PSNR/SSIM 与 harm_vs_off，不能只报效用分类准确率。后续可探索多强度反事实标签或小集合 gate 搜索，但应先完成当前对照。

## 先运行这组实验

| 比较 | 固定条件 | 目的 |
|---|---|---|
| WWE backbone vs FM/all | 同 train/val、分辨率、基础配方 | 残差生成器有无收益 |
| FM/none vs all vs fixed=0.5 vs utility | 同一效用 checkpoint、同 seed/噪声/NFE | 效用 gate 是否优于固定注入 |
| reconstruction-gate vs counterfactual-utility | 同骨干、训练集、参数规模，记录训练成本差异 | 额外收益是否来自标签而非仅增加参数 |
| dense vs global vs none | 同容量网络、数据划分与训练步数 | 局部空间条件价值；不直接复述 Patch Policy 的 40% |
| Flow Matching vs DDIM | 同容量、NFE、分辨率、硬件；各自完整训练 | 具体路径/目标的质量-延迟权衡 |
| clean/color/shift/missing/invert | 同 checkpoint；相同噪声 | 先验失效时是否能减少损害 |
| UIEB → LSUI / UFO-120 | 冻结源权重，审计目标 test 重叠 | 跨域稳定性 |
| C60、U45 | 不调参、不训练 | 无参考困难域的颜色/细节检查 |

完整实验至少三个训练 seed；数据 manifest 固定不随训练 seed 改变。测试时随机初始噪声 seed 随训练 run 记录，相同 run 内 gate 消融保持同噪声。

~~~bash
for seed in 42 43 44; do
  bash scripts/run_pipeline.sh UIEB "$seed"
  bash scripts/evaluate_suite.sh UIEB "$seed"
done

# 空间信息消融（各自输出目录独立）
UIE_PATCH=global bash scripts/run_pipeline.sh UIEB 42
UIE_PATCH=none bash scripts/run_pipeline.sh UIEB 42
UIE_PATCH=global bash scripts/evaluate_suite.sh UIEB 42
UIE_PATCH=none bash scripts/evaluate_suite.sh UIEB 42

# 同网络 DDIM 路线
UIE_SAMPLER=ddim bash scripts/run_pipeline.sh UIEB 42
UIE_SAMPLER=ddim bash scripts/evaluate_suite.sh UIEB 42
~~~

一键消融会重复训练各自粗恢复网络，方便保持流程独立；要节约算力，可以显式复用同一 seed 的 baseline checkpoint，flow 初始化只提取粗恢复权重。data manifest 必须一致。

普通 reconstruction-gate 对照：

~~~bash
python -m uie train --config configs/flow.yaml \
  --init outputs/UIEB/seed42/flow-dense/baseline/best.pt \
  --flow-gate learned --output outputs/UIEB/seed42/reconstruction-gate
python -m uie evaluate \
  --checkpoint outputs/UIEB/seed42/reconstruction-gate/best.pt \
  --manifest data/manifests/UIEB.json \
  --output outputs/UIEB/seed42/reconstruction-gate-test \
  --gate learned --steps 4 --diagnostics --save-images
~~~

learned gate 与恢复场联合接受重建/场回归监督；没有“是否降低恢复误差”的监督，所以不报告其 Brier/ECE 为效用概率校准。它与三阶段路线的训练计算量不相等，应补充成本匹配实验，不能把差异全归因于反事实标签。

## 扰动与部署测量

默认训练扰动抽样为 [clean, clean, color, shift, missing]，大致 40% clean、其余各 20%。invert 只作为未见过的扰动测试。color 改 RGB 候选；shift 对整个先验平移；missing 清零中央区域。这些是受控应力测试，不代表真实水体退化分布。

为研究温度校准，必须使用匹配的数据域、NFE、resize 尺寸、精度与 clean prior。代码对不匹配发出提示。套件针对每个 NFE 校准 T；跨域评估保留源域温度但不宣称其仍然校准；扰动测试使用未校准 checkpoint。效用头默认在 NFE=4 训练，跨 NFE 校准不能替代为新 NFE 重训练效用预测器。

时间统计 batch=1，包含粗恢复、先验、CNN 条件、gate、全部求解器步骤和输出裁剪；排除数据读取、H2D 与 CPU 指标，额外诊断 rollout 不计入时间/推理峰值显存。该定义适合网络比较，**不是机器人相机到控制输出的端到端延迟**。另应测预处理、深度估计、传输与下游耗时。

评估支持 --size 0 原分辨率，网络内部 pad 后裁回。默认方形 resize 是统一开发协议，可能改变纵横比；正式论文优先补原分辨率结果并保持所有方法处理相同。UFO 按 DATASETS.md 的独立增强协议报告。

## 结果判读与停止条件

报告每张图的差值、各 seed 均值/标准差、分域结果、p50/p95 延迟、NFE、显存，以及 harm_vs_off。若图像来自同一航次或视频，用场景/序列级 bootstrap；不要把 patch 或像素当作独立样本制造显著性。当前没有自动输出置信区间。

若 utility 在 clean 域损害超过它在错误先验下的收益，不能用无参考指标上升掩盖；优先调整标签阈值、先验有效性、gate 训练域或退回固定/关闭先验。若 dense 相对 global 无收益，应检查是否已有空间卷积路径足以传递信息，而不是直接引入更大的视觉模型。

在稳定完成上述检验后，才值得加冻结 DINO patch tokens、测量/可靠深度、动态逐步效用预测或视频模型教师。每增加一种模块，记录前向开销与独立消融；视频方向需要时间一致性数据与指标，不能只在单图 UIEB 得分上宣称世界模型优势。
