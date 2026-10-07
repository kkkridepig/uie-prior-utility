# SS-UIE V3：可预测性诊断与条件训练——服务器 Codex 执行框架

版本：2026-10-07 / V3.0。建议 run_id：`ssuie_utility_v3_predictability_20261007`。

这是一份**待在服务器实现和执行的实验协议**，不是已经完成训练的报告。请同时使用配套文件《SSUIE_V3_服务器Codex提示词_20261007.md》。框架是参数、公式、分支和结论边界的唯一主文档，提示词只负责启动任务。

## 0. 本轮究竟要回答什么

核心问题：**固定候选确实存在参考图辅助的改进空间，但这种空间能否由部署时可见的信息预测，并超过充分校准的固定修正与普通门控？**

本轮分两个阶段：

1. **第一阶段：已有权重诊断与低容量可预测性探针。** 不更新旧模型；分开检验预测错误、正则收缩、空间位置、干预信息和损失梯度。增加两个预先固定的 CPU 岭回归探针，检验整图和 32×32 区域的决策能否泛化。探针属于诊断拟合，不能说本阶段“完全不训练任何参数”。
2. **第二阶段：仅在诊断通过时开展一次条件训练。** 根据固定规则进入整图路线或区域路线，两者择一。区域路线中保留整图方法作为必要对照，并非同时搜索两条主线。完成全部匹配控制、选点、校准、开发判定；仅开发全闸门通过，才一次评分 177 对封存数据。

默认不继续训练旧 O，不训练候选 B1/B3，不从零训练 SS-UIE，不扩到 50000 更新，不重启其他 A–G 路线。第一阶段没有信号或第二阶段没有超过强控制，就完成报告并停止。未使用的预算保留。

本轮检验的具体机制是：**能否从输入、底座输出及实际候选修正的统计信息，预测正确的采用尺度；显式误差矩监督与整图上下文是否提供可归因增量。** 坐标变换、普通门控、log-MSE 和二次融合公式本身均不构成原创贡献。本协议不保证正收益或论文新颖性；其作用是使正、负、不确定结果都能回答明确问题。

## 1. 已核对的起点与不得改写的历史

### 1.1 身份

| 项目 | 固定身份 |
|---|---|
| 用户仓库 | `https://github.com/kkkridepig/uie-prior-utility` |
| V2 分支 | `research/ssuie-local-utility-v2-diagnostic` |
| 本地已审阅 V2 提交 | `0521bfc1442cf3c69630bd9199a4aa360740a815` |
| V2 run | `ssuie_local_utility_v2_diagnostic_20261007` |
| 官方 SS-UIE 提交 | `88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df` |
| 官方权重 SHA256 | `977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99` |
| 固定 producer | `B1_003000`，旧候选头 3000 更新 |
| producer 权重 SHA256 | `c1af746056f0419eeb5787283a5c5e28d15d1ceec95d1dd6cedd4ed7592879af` |
| V2 角色文件 SHA256 | `e6176357cbccc2424c268c245ef7e8c3b7dd36c4ef7f2fbbd2fa9b4070b520ce` |

底座是**官方公开简化版**，不能称为完整复现 SS-UIE 论文。producer 指提供修正方向的冻结候选，不是独立输出的最佳模型。B1/B3 的独立端点最佳仍为零更新，不得改写这个历史。

如果服务器分支已出现后续提交，先核对冻结源码、改动和旧结果身份；不可仅因分支名相同就认为内容一致。不要强制 reset 或覆盖其他未提交工作。以独立 V3 分支或工作区实施，保留 V1/V2 原件。

### 1.2 本轮设计所依据的实测结果

以下来自 V2 的 136 对 `utility_val`，不是本轮新结果：

| 策略 | PSNR / dB | 相对正确底座 J0 / dB |
|---|---:|---:|
| J0，clip01 | 24.355885 | 0 |
| B1_003000 完整候选 | 24.019366 | −0.336519 |
| O，V2 冻结部署策略 | 24.369342 | +0.013457 |
| G2，普通门控控制 | 24.372761 | +0.016875 |
| 每图连续 oracle | 24.765962 | +0.410077 |
| 32×32 区域连续 oracle | 24.868134 | +0.512249 |
| 像素连续 oracle | 25.085275 | +0.729389 |

`oracle` 是使用参考图得出的理想决策，只是这个候选空间中的上界。O 的改善区间跨零，主要机制消融近乎重合，所以 V2 的 `STOP_CURRENT_RECIPE_NOT_SUPPORTED` 保留。

V2 已实际训练 11 个方法各 3000 更新；不能说 O 没有运行。候选续训和候选 MSE/LOG 损失对照未运行；不能说更长训练已证明无效。旧 CAL 上细搜得到 alpha=0.2325 的分析只是事后诊断，不把它直接指定成本轮训练常数或新胜出结果。

### 1.3 必须先读的旧文件

从服务器原 run 或恢复包读取：`FINAL_REPORT.md`、`NEXT_DECISION.md`、`budget.json`、`budget_ledger.jsonl`、`source_snapshot.json`、`roles.jsonl`、`split_freeze.json`、`method_registry.json`、`selection/producer_freeze.json`、`selection/calibration_selection.json`、相关 checkpoint 和 CPU/设备验收回执。

补读用户本地独立审阅：`SSUIE_V2_独立审阅与下一轮建议_20261007.md`。未上传时可以从已冻结原始报告与记录重建必要事实，不要求用户因缺少这篇解读而重新准备整个实验。

旧源码重点：`uie_next/math/utility.py`、`uie_next/losses.py`、`uie_next/training.py`、`uie_next/v2/{training,evaluation,selection,diagnostics}.py`。**V2 的实际评估入口优先于仓库里未被该 run 调用的同名旧函数。**

## 2. 术语与符号

| 符号或名词 | 含义 |
|---|---|
| I / Y | 水下输入 / 配对参考图；Y 只用于允许的训练与评估 |
| J0 / J1 | 冻结底座输出 / 固定候选输出，均在 [0,1] |
| r | 实际修正 `J1−J0`，使用候选裁剪之后的输出计算 |
| alpha | 采用强度，属于 [0,1]；同一像素或区域的 RGB 三通道共享 |
| R | 区域；本轮只能是整图或不重叠 32×32 块 |
| A、B、C | 修正能量、误差与修正的内积、相对固定参考强度的内积 |
| oracle regret | 实际结果比同一候选、同一决策空间的理想结果多出的 MSE |
| matched control | 数据、训练量、输入、参数规模等关键因素相匹配的控制 |
| ablation / 消融 | 删除指定机制的控制，其他设置尽量不变 |
| OOF | out-of-fold，某内容组的预测来自没有用该组训练的探针 |
| 岭回归 | 带 L2 参数惩罚的线性回归；本轮用作低容量可预测性诊断 |
| 内容组 | 旧近重复审计形成的代理分组，不是人工核实的采集场景 |
| PSNR / SSIM / LPIPS | 像素重建质量 / 结构相似度 / 学习感知距离；前两项越高越好，LPIPS 越低越好 |
| 设备小时 | 占用计算设备的时间乘设备数量；本轮最多同时一张设备 |

## 3. 预算、授权范围与运行环境

### 3.1 累计预算，不重置

已核对旧账本：

```yaml
maximum_device_seconds: 57600.0
inherited_used_device_seconds: 15623.031541585922
inherited_hours_approx: 4.3397309838
remaining_hours_approx: 11.6602690162
stage1_new_device_seconds_cap: 3600.0
stage2_new_device_seconds_cap: 14400.0
protected_closeout_device_seconds_min: 12600.0
max_concurrent_devices: 1
formal_seed: 20261007
```

旧 `used_device_seconds` 已含 V1 的 1503.1731119155884 秒，**不得再次相加**。若服务器存在更晚、可核验的真实设备事件，纳入去重后的累计成本，不能向下归零。旧 60 秒保守估计仍标估计。

| 额度 | 上限或保护额 | 覆盖范围 |
|---|---:|---|
| 第一阶段 | 新增最多 1 小时 | 身份/设备验收、诊断前向、旧模型梯度诊断、缓存、必要重试 |
| 第二阶段 | 新增最多 4 小时 | 本阶段集成测试、profile、全部匹配训练及其检查点验证 |
| 评估与收尾 | 至少保护 3.5 小时 | 最终 CAL、开发指标、条件封存、压力、测速、视觉、恢复验收 |
| 按上述额度后的余量 | 约 3.160269 小时 | 默认不分配给新实验，不因剩余而启动候选救援或更多种子 |

纯 CPU 分析另记墙钟时间；若 CPU 正在计算而设备已真正释放，不把这段时间伪记成设备占用。共享设备锁必须跨 run 生效。所有真实设备测试、失败、预热和缓存都记账。

先真实测速，再以 **1.3 安全系数**预测完整矩阵和剩余评估。收尾预测超过 3.5 小时时，保护额取更大预测并计入总账；不能挪走所需收尾额度来完成训练。第一/第二阶段各自上限不得自动借用余量突破。预算不足输出 `INCONCLUSIVE_BUDGET`，不削弱控制或只保留较好方法。

### 3.2 环境与恢复

保留原厂商 PyTorch/PPU 运行环境；不要为普通 CUDA 教程重装覆盖。检测实际设备及同步接口，缺失时明确阻塞。恢复包包含底座和实验权重，不应再次从零重训。

对实际读取的权重检查 SHA256、结构和严格加载；记录模型参数、dtype、设备与数值后端。先用 `utility_fit` 的固定 8 张核对 J0/J1 缓存与实时推理，float32 默认 `atol=1e-5, rtol=1e-4`；精确身份不匹配时禁止偷偷复用缓存。

缺失文件时先搜索现有工作区、用户上传目录及恢复包；有源码无权重时先完成不依赖权重的实现和合成测试，正式流程标 `BLOCKED_RECOVERY`。不以随机底座或 mock 权重代替正式模型。

## 4. 数据边界与历史暴露

| 原角色 | 对数 / 内容组数 | 本轮用途 |
|---|---:|---|
| LSUI model_fit | 3608 / 3330 | 旧候选来源；本轮不新增候选训练 |
| LSUI model_val | 671 / 588 | 旧选型事实；本轮不重新选择 producer |
| UIEB utility_fit | 443 / 440 | 诊断探针、尺度估计、OOF、第二阶段正式训练 |
| UIEB utility_val | 136 / 132 | 已暴露开发集，诊断、检查点选择、开发闸门 |
| UIEB calibration | 133 / 132 | 已暴露 CAL，仅网络冻结后的固定规则参数选择 |
| UIEB sealed_eval | 177 / 176 | 本轮仍未评分，开发全闸门通过后一次确认 |
| excluded_overlap | 1 | 继续排除 |

读取原 manifest 和 group_id，不按上述数量重新随机拼一个划分。输入/参考同规则 RGB float [0,1]、256×256，插值、预处理和指标沿用 V2 冻结实现。训练不做新增裁剪、翻转、颜色增强或归一化；这是本轮控制复杂度的预登记选择。

`utility_val`、CAL 已用于历史研究，不能重新称“未见验证/独立校准”。新的 OOF 划分只能减少新探针的拟合泄漏，不能洗掉父模型、旧方法和研究者的历史暴露。177 对称“历史暴露条件下、本轮尚未评分的封存集”，不称完全独立盲测。底座作者逐图训练名单未知的事实保留。

代码必须实现访问守卫：

- 第一阶段：只评分 `utility_fit/utility_val`；CAL/sealed 的身份和重复性核验可读哈希，不做前向、参考标签、视觉评分或尺度估计。
- 第二阶段训练：只用 `utility_fit` 标签反传；检查点只在 `utility_val` 选。
- CAL：仅全部正式网络冻结后开放；只按本协议网格选部署参数，不反传、不重选网络、不改配方。
- sealed：只有本轮 `DEV_PASS` 与完整冻结单存在才开放；所有访问事件写入账本。

固定诊断探针：在 utility_fit 的 440 组中，按 `SHA256("v3-fit-probe-20261007|"+group_id)` 排序取前 32 组，组内图全部保留。探针不是全训练集。前 8 组用于固定梯度批次，前 4 组用于临时工程样例；不从最好/最差案例挑训练探针。

## 5. 必须实现和测试的数学契约

### 5.1 实际二次误差

逐像素先对 RGB 求均值：

\[
r=J_1-J_0,\quad e=Y-J_0,\quad
a=\operatorname{mean}_{RGB}r^2,\quad b=\operatorname{mean}_{RGB}(er).
\]

当 \(J(\alpha)=J_0+\alpha r\) 时：

\[
\ell(\alpha)=\ell(0)+a\alpha^2-2b\alpha.
\]

因为 J0/J1 均在 [0,1] 且 alpha 在 [0,1]，输出是凸组合，不需要一个额外非线性裁剪改变这条恒等式。可做数值保护，但必须验证其没有改变有效样本。不得用候选裁剪前的残差制作标签。

区域控制先聚合：

\[
A_R=\operatorname{mean}_{p\in R}a_p,\quad
B_R=\operatorname{mean}_{p\in R}b_p,\quad
g_R(\alpha)=2B_R\alpha-A_R\alpha^2.
\]

整图为一个区域，256×256 的 32×32 块为 8×8=64 个区域。先聚合 A/B 再求比值，**不是平均像素 alpha**。输出采用 nearest/repeat 扩展为块常数，不用双线性插值破坏区域定义。

\[
\alpha_R^*=\operatorname{clip}(B_R/A_R,0,1)\ (A_R>0);
\quad A_R=0\Rightarrow\alpha_R^*=0.
\]

精确 oracle 使用 float64，A>0 时求比值；另报与旧 active 阈值兼容的版本，不能混为一条曲线。

### 5.2 相对固定参考强度

令 alpha_ref 是只用当前训练池确定、冻结的固定参考强度，t=alpha−alpha_ref：

\[
C_R=B_R-\alpha_{ref}A_R,
\quad g_R(\alpha)-g_R(\alpha_{ref})=2C_Rt-A_Rt^2.
\]

预测 C 后，本轮唯一连续决策定义为：

\[
\alpha_R^{raw}=\operatorname{clip}\left(
\alpha_{ref}+\frac{\widehat C_R}{A_R+\lambda_0},0,1\right).
\]

它对应惩罚 \(\lambda_0(\alpha-\alpha_{ref})^2\) 的相对决策。这里的正则把决策拉回 **alpha_ref**，与 V2 把 alpha 拉向 0 的公式不同，不可沿用旧含义。

若使用真实 C，上式等价于 \((B_R+\lambda_0\alpha_{ref})/(A_R+\lambda_0)\) 再裁剪。C 与 B 的变化是已知 A 的坐标平移，**不产生新信息**。不建立一个与之代数等价的“绝对效用消融”来假装发现机制差异。

### 5.3 regret 与诊断分解

对同一决策空间的精确最优 alpha*：

\[
\ell(\alpha)-\ell(\alpha^*)
=A(\alpha-\alpha^*)^2+
2(A\alpha^*-B)(\alpha-\alpha^*).
\]

最优点在 0/1 边界时第二项一般不为零。不能全体使用仅适用于内点的平方距离公式。

诊断 V2 时，令 alpha_hat 为真实部署输出；alpha_breg 为代入真实 b、使用**相同 lambda/tau/active 规则**所得结果：

\[
L(\alpha_{hat})-L(\alpha^*)=
[L(\alpha_{hat})-L(\alpha_{breg})]+[L(\alpha_{breg})-L(\alpha^*)].
\]

第一项是带符号的预测替换差，可能为负；第二项是正则/阈值/active 限制造成的非负空间损失。不要将两项强制当非负“归因百分比”。MSE 上核验恒等式，dB 差另报，不把 dB 的差当 MSE 分解。

### 5.4 主目标

主质量统计量为每图 PSNR 的算术均值：

\[
\overline{PSNR}=\frac1N\sum_i-10\log_{10}(MSE_i).
\]

不是先平均 MSE 再换算 PSNR。第二阶段图像损失先求每图完整 RGB/空间 MSE 再取 log；禁止改成逐像素 log，也不以所有区域作为独立统计样本。

## 6. 第一阶段 D1–D6：有限诊断清单

### 6.1 D1：预测、正则与决策误差

在固定训练探针和全部 utility_val，用已选 O_3000、固定 B1 producer：

1. 复算真实 a/b/U，O 的 v_hat/b_hat/U_hat，V2 正式 alpha 与实际输出。
2. 复算 `lambda=1e-4,tau=0` 和 `lambda=1e-3,tau=0` 两个既有设置；正式部署结果仍标后者。
3. 每个设置分别用真实 b 替换 b_hat，其余部署规则不动；与像素、32 块和图级精确 oracle 分表比较。
4. 输出 v、b、U 的 RMSE、MAE、每图空间 Pearson 相关、符号一致率；常数数组相关记 null，注明原因。不能仅凭 U 相关低判定部分采用决策无效。
5. 输出 alpha 的均值、分位数、取 0/1 比例、alpha>0.05 覆盖率、采用后的真实 MSE 收益、精确 regret 和上述两项分解。

所有标签字段单独命名 `reference_only_*`；预测字段禁止用真值补空。保存逐图结果和按图等权汇总，不把数百万像素作为独立置信样本。

### 6.2 D2：空间位置究竟有无作用

对同一部署 alpha，不再训练或重新校准，比较：

- 原 alpha；
- 每图空间算术均值 alpha；
- 每图能量加权均值 `sum(a*alpha)/sum(a)`，分母为零取 0；
- 每图一次固定空间置乱 alpha，保持其值分布。

置乱只依赖 UTF-8 字符串 `SHA256("v3-alpha-shuffle-20261007|"+sample_id)`，种子取摘要前 8 字节按 big-endian 转成的无符号整数，NumPy PCG64 生成排列；不得读取 Y 决定排列。V2 像素图打乱 H×W 个位置；新区域模型后续诊断打乱 64 个块。只执行这一固定排列，不试多个取最有利者。

空间均值/置乱的差异是特定诊断证据，不是随机化因果证明。原 alpha 若与其置乱几乎等效，不足以支持“学对了局部位置”。

### 6.3 D3：同参数消融

O、O-NP、O-NI、O-ND 均取 3000 更新的已有权重，不再按诊断结果选点。比较两张表：

1. 四者统一 `tau=0,lambda=1e-3`；
2. 四者各自 V2 已合法冻结的部署策略。

同时记录预测差、alpha 差、输出差和逐图质量差。这样才区分训练机制与校准参数的影响。保留 O-NS 旧结果用于背景，但它同时改变奇性约束和幅度输入，不能作为只改一个因素的因果消融。

### 6.4 D4：先验干预的有效信息量

仅固定训练探针；同一 B1 producer 上使用 V2 `interventions` 的名义与六个既有干预，字段全部重算。输出：

- 实际 r_k 的 RMS 幅度、与名义 r_0 的全图及区域方向余弦；
- 各视图 U、B、区域最优 alpha 的差及符号；
- 六行 `flatten(r_k-r_0)` 构成矩阵的奇异值；令 p_j=s_j²/sum(s²)，有效秩为 `exp(-sum(p_j log p_j))`，矩阵全零取 0；
- U 差相对原配对损失尺度 `s_U` 的大小，以及有非微小目标差的样本比例，原值和分位数均保存。

区域像素数不增加源图样本数。有效秩高不等于质量有益；方向接近也不必然表示干预完全无用。此项提供解释，不据某个任意秩阈值自动启动更复杂干预。

### 6.5 D5：分项梯度，而不是只看总 loss

对 O_3000 和固定探针前 8 组，按 sample_id 排序、batch=4，固定视图对为 `(nominal,tau_075)`。只求梯度，不 optimizer.step，不改变权重、BN、优化器或 RNG 恢复状态。

计算真实可微投影项、配对项、决策项；旧 `parts` 中 detach 的值只能记录，不能据它们求梯度。使用 V2 原尺度和权重 1、0.25、0.1。分别记录：

- 原始和加权 loss；各项对同一参数向量的梯度范数；两两夹角余弦；
- 总梯度范数与 `norm(sum(g_j))/sum(norm(g_j))`；
- 原 `clip_norm=1` 的共同裁剪系数。对分项使用这个**共同系数**解释裁剪后贡献，不分别裁剪分项后冒充实际总梯度。

不根据诊断自动改旧梯度阈值，不将高裁剪率单独诊断为欠训练。本轮不自动加入小样本过拟合训练；如果工程梯度不能启动，以测试失败处理。

### 6.6 D6：oracle 负控制与候选专属性

B3 只使用旧同步数 `B3_003000`，从原 inventory 查出真实权重哈希；固定训练探针和 utility_val 上重用或复算整图/块/像素 oracle。不得事后挑 B3 最弱点。

另做两个固定随机方向负控制，种子分别来自 `v3-null-1|sample_id` 和 `v3-null-2|sample_id` 的 SHA256，摘要到整数的转换与 D2 相同，使用 PCG64，均与 Y 无关。逐像素生成三维标准高斯 q，归一化至 `mean_RGB(q²)=1`；c=sqrt(a)。先计算最大 eta∈[0,1]，使 `J0+eta*c*q` 各通道都在 [0,1]。对随机方向和原方向都使用同一幅度 eta*c：

`r_null=eta*c*q`，`r_matched=eta*r`。

比较这两个**匹配缩幅**方向各自的 oracle；另列原始 r oracle。eta=0 合法，报告保留的能量比例和零幅度比例。不要先裁剪随机图、改变幅度后仍称精确幅度匹配。两个负控制逐一报告及等权均值，不选最佳一个。

随机方向 oracle 大并不证明方法无效；它说明使用 Y 选择本身可产生空间。B1 接近 B3 也不等于物理代理必然无用，但不能据此声称先验专属收益。

## 7. 第一阶段 D7：整图与区域的低容量预测探针

该项是进入第二阶段的必要筛选。仅使用 `utility_fit` 训练新探针；旧 O 的训练探针表现不能代替 OOF。

### 7.1 部署可见特征，固定唯一版本

逐像素 13 通道：

`Z = concat(I[3], J0[3], r[3], abs(r)[3], a[1])`。

对每个 32 块计算每通道均值与总体标准差，得到 `f_R∈R^26`；对整图同样计算 `f_img∈R^26`。std 使用总体定义 `sqrt(max(E[Z²]-E[Z]²,0))`，不使用无偏样本估计。不得把区域 std 的平均当整图 std。

两个探针：

- `RIDGE_GLOBAL`：输入 f_img，预测整图 C；
- `RIDGE_REGION`：输入 concat(f_R,f_img)，共享一个回归器预测各区域 C。

不输入文件名、数据角色、参考统计、旧指标、预测目标或 sample_id。sample_id 仅作关联和确定性分组。特征是一个有限可行性检验，失败不能证明所有网络均不可学习。

### 7.2 每个训练子集的固定常量

在该训练子集，使用整图 A_i/B_i/MSE0_i 的二次式，以 10001 个固定值 `{k/10000:k=0..10000}` 最大化平均逐图 PSNR，求 `alpha_fit_star`。并列误差 1e-12 dB 内选较小 alpha。

`alpha_ref=clip(alpha_fit_star,0.05,0.95)`，仅为了所有新网络的共同内点初始化；未裁剪的 `alpha_fit_star` 仍是必须报告的强固定控制。不得把这个参考强度等同于真实最优或擅自用历史 0.2325 替代。

从该训练子集计算：

`s_A=max(mean_i mean_R A_iR,1e-8)`；`lambda0=0.05*s_A`。

`s_C=max(sqrt(mean_i mean_R C_iR²),1e-6)`，全局和区域分别有自己的 s_C；lambda0 共用图像等权的修正能量。所有常量与源样本哈希绑定。

标准化特征：训练子集的均值、总体 std，std 下限 1e-6；区域统计按每图均权。不能使用验证折或 utility_val 来算尺度。

### 7.3 岭回归与交叉划分

固定五折：对 440 个 group_id，按 `SHA256("v3-oof-20261007|"+group_id)` 排序，第 j 组分配到 `j mod 5`。同组不拆，折清单在评分前冻结。

每折在另四折独立计算 alpha_ref、尺度和标准化，最小化：

\[
\frac1{N}\sum_i\frac1{n_R}\sum_R
(w^T\tilde f_{iR}+d-C_{iR}/s_C)^2+0.01\|w\|_2^2.
\]

截距 d 不正则化；用稳定线性求解器，固定唯一 ridge 系数 0.01，不搜索特征、核或惩罚网格。验证折输出 `C_hat=s_C*(w^T f+d)`，依 §5.2 决策，κ=1，不在验证折调强度。

每个 OOF 样本同时保存：J0、仅由训练折选择的未裁剪 alpha_fit_star 固定控制、两种 ridge 输出、对应 oracle。聚合全部 OOF 预测，不平均五个均值来改变图像权重。

随后两个探针各只在全部 443 图重拟合一次，常量全从这 443 图重算，在 136 图 utility_val 上各评估一次。两者的全量拟合对象保留为后续正式强控制，不能丢掉表现较好的 ridge。

### 7.4 进入第二阶段的固定规则

下列是小试的**可行性筛选线**，不是降低正式 +0.10/+0.03 dB 闸门，也不是显著性声明。

对一个 probe，`predictable(probe)` 要求：

1. OOF 和 utility_val 两者上，相对各自训练来源的 `FIXED_FIT` 平均逐图 PSNR 均 ≥+0.03 dB；相对 J0 也均 ≥+0.03 dB。
2. 两者上相对 J0 下降超过 0.10 dB 的比例，不超过对应 FIXED_FIT 的比例加 **0.05 绝对比例**（5 个百分点）。
3. 数值、角色、缓存、身份检查全部通过；完整两种 probe 已报告。

区域资格额外要求：在 OOF 和 utility_val 上，RIDGE_REGION−RIDGE_GLOBAL 均 ≥+0.02 dB；且 utility_val 的 32 块 oracle−图级 oracle ≥+0.05 dB。

分派顺序固定：

```text
若 REGION 满足 predictable 且满足区域额外条件 → TRAIN_REGION
否则若 GLOBAL 满足 predictable → TRAIN_GLOBAL
否则 → STOP_NO_PREDICTABILITY_SIGNAL，收尾，不进入第二阶段
```

区域能预测而全局不能时仍可进入 TRAIN_REGION，不能用“全局失败”抹掉真实区域信号。反之，仅 oracle 有空间、O 的训练集拟合好、或补看了某种视觉案例，都不授权跳过此门槛。

输出 `stage1_decision.json`，包含全部输入文件哈希、逐项布尔值、未过原因、固定 route。不得试完失败后改 ridge 参数、强度网格、折、阈值或候选。失败表示这一有限信息表示和探针没有足够信号，不表示不可学习定理。

## 8. 第二阶段：唯一网络与学习问题

### 8.1 新方法的名字与范围

本轮用描述性编号 `MOM` 表示预测误差矩 C 的控制器，`DIRECT` 表示直接预测采用强度的门控。MOM 是内部实验编号，不是已经确立的新算法名称。

底座、B1 producer、先验构造、图像尺寸全部冻结。新控制器只使用名义视图，不添加 V2 的配对干预损失、奇性约束或额外正负网络。本轮是在完整实现**一个新且受限的假设**，不是声称完成原 A–G 全方案。阶段一已对旧干预配方作诊断，不能再把删除旧机制后的收益归给旧机制。

### 8.2 网络结构与张量

沿用 §7.1 的 26 维区域/全图描述子；不调用参考图特征网络。全量 utility_fit 上预先计算本轮标准化参数，并与所有对应新臂共享：区域描述子用区域统计的标准化，全局描述子用全局统计的标准化。

以下结构图中的 f_region/f_global 表示各自标准化后的描述子；GLOBAL 将同一已标准化 f_global 复制两份，NC 将同一已标准化 f_region 复制两份，不按输入槽位偷偷换尺度。描述子缓存为 float32；用于拟合均值、标准差、s_A/s_C 和二次式审计的汇总在 CPU float64 完成，不要求厂商设备支持 float64。

```text
I, J0, J1                    [B,3,256,256]
r = J1-J0                    [B,3,256,256]
Z = [I,J0,r,abs(r),a]         [B,13,256,256]
f_region = [mean32(Z),std32]  [B,26,8,8]
f_global = [meanHW(Z),stdHW]  [B,26,1,1]

region input: [f_region, broadcast(f_global)] → [B,52,8,8]
global input: [f_global, f_global]            → [B,52,1,1]
no-context input: [f_region, f_region]        → [B,52,8,8]

Conv1x1(52,64,bias=True) → SiLU
Conv1x1(64,32,bias=True) → SiLU
Conv1x1(32,1,bias=True) → z
```

这里的逐点网络等价于共享小型多层感知器，不使用 BatchNorm、Dropout、注意力、额外卷积、坐标编码或隐含预训练特征。三层总可训练参数 **5505**。首两层用相同确定性初始化，末层 weight/bias 全零。

复制描述子使参数数目相同，但不使各臂信息内容或有效特征秩相同；全局/局部信息差异正是被干预因素。不能仅以参数数目相同声称所有表示自由度相同。

MOM 输出 `C_hat=s_C*z`，决策严格使用 §5.2。全局模型使用整图 A/C/s_C；区域模型使用区域 A/C/s_C。`MOM_NC` 是 no-context，在输入中以局部描述子替代全局描述子，不对重复特征重新计算一套独有尺度。

DIRECT 使用同样的输入和网络：

\[
\alpha^{raw}=\operatorname{sigmoid}(\operatorname{logit}(\alpha_{ref})+z).
\]

因此所有臂的 step0 输出都精确对应同一个 alpha_ref 固定修正，不能让新方法从有利输出开始、控制从 J0 或随机输出开始。不要在 MOM 额外加入未规定的拒绝头或直接回归整张 Y。

若 A=0，MOM 设置 alpha_raw=alpha_ref，输出仍为 J0，相关监督权重为零；DIRECT 的 alpha 数值保留但实际残差为零，不虚构质量增量。alpha_ref∈[0.05,0.95] 避免 DIRECT 的 logit 无穷。

### 8.3 唯一训练目标

在 utility_fit 上按每图等权计算 §7.2 的 alpha_ref、s_A、lambda0、s_C；所有正式模型共享数据来源和参考强度。区域/全局 s_C 不同的原因是目标聚合尺度不同，必须分别存档。

定义 `J_ref=J0+alpha_ref*r`，`m_i=mean_RGB,HW((J_i−Y_i)^2)`，`m_ref_i` 同理。两者都先整图平均。图像损失：

\[
L_{dec}=\frac1B\sum_i\log\frac{m_i+10^{-6}}{\operatorname{stopgrad}(m_{ref,i})+10^{-6}}.
\]

分母只减去训练参考常数，不改变参数梯度。负 loss 合法。由于加了 1e-6，它与无 epsilon 的平均 PSNR 仅近似对应；正式指标仍按冻结指标实现计算，不把两者数值完全等同。

误差矩监督：

\[
L_{mom}=\frac1B\sum_i\frac1{n_R}\sum_R
\operatorname{Huber}_{\delta=1}\left(\frac{\widehat C_{iR}-C_{iR}}{s_C}\right).
\]

这里所有区域等面积。A=0 的项置零但仍按全部 n_R 求均值，避免不同方法重新按 active 数改变图像权重。标签和候选全部 detach；Huber 是小误差平方、大误差线性惩罚的鲁棒损失。

完整 MOM 使用：`L = L_dec + 0.1*L_mom`。`MOM_NM` 仅去掉矩监督，使用相同网络、输入、决策和 L_dec。DIRECT 仅用 L_dec。**不改变 loss 权重、lambda0 或学习率来追逐第一批结果。**

额外矩监督是否有用，由 MOM 对 MOM_NM 检验。相对坐标的写法不是单独待宣传的机制；解析决策是否值得保留，由相同容量 DIRECT 与完整 MOM 比较。公式正确不保证未知 C 可准确预测。

## 9. 第二阶段完整匹配矩阵

### 9.1 TRAIN_REGION：五个正式训练臂

| method_id | 输入/输出尺度 | 损失 | 身份 |
|---|---|---|---|
| MOM_R | 区域+全局 → 64 块 | L_dec+0.1 L_mom | 唯一主候选 |
| MOM_G | 全局 → 整图 | L_dec+0.1 L_mom | 粒度控制 |
| MOM_R_NC | 区域+区域 → 64 块 | L_dec+0.1 L_mom | 删除全局上下文 |
| MOM_R_NM | 区域+全局 → 64 块 | L_dec | 删除矩监督 |
| DIRECT_R | 区域+全局 → 64 块 | L_dec | 参数匹配普通门控 |

MOM_G/NC/NM 是机制控制；DIRECT_R 是外部强控制。不能只完成 MOM_R 和最弱一臂就宣称匹配实验完成。区域主线失败后，不在同一个协议内改称 MOM_G 成功主线并解封。

### 9.2 TRAIN_GLOBAL：三个正式训练臂

| method_id | 输入/输出尺度 | 损失 | 身份 |
|---|---|---|---|
| MOM_G | 全局 → 整图 | L_dec+0.1 L_mom | 唯一主候选 |
| MOM_G_NM | 全局 → 整图 | L_dec | 删除矩监督 |
| DIRECT_G | 全局 → 整图 | L_dec | 参数匹配普通门控 |

该路线不支持“局部空间机制”的结论。即便有效，主张只能是给定全局统计时的采用决策；是否新颖需要针对最终方法另作文献核验。

### 9.3 不新增训练的强控制

正式比较至少包括：

- J0、B1_003000 完整输出、B3_003000 完整输出；
- `FIXED_FIT`，443 图选择的未裁剪固定强度；`FIXED_REF`，用于初始化的 alpha_ref；
- `FIXED_CAL_B1`、`FIXED_CAL_B3`，各自候选在 CAL 上细网格选择的固定强度；
- D7 两个全量拟合 ridge，后续只允许与新网络相同的 κ 网格校准；
- V2 所有正式已选可部署条目及其旧合法策略，至少包括 G0/G1/G2/F0/R0/B4/O 和 O 消融。别名可复用计算，角色必须保留。

旧方法无需重训，其结果只是历史强参考；**本轮新机制的训练公平性由 §9.1/9.2 同步矩阵承担**，不能把新 12000 更新对旧 3000 更新的差独自当机制证据。

旧官方 `minmax_float` 可保留作背景，不参与新增模块贡献计算，也不作为必须击败的主要强控制。主底座始终是 `B0_clip01`。

## 10. 训练量、优化、选点与恢复

### 10.1 步数只由预先测速决定

先完成全部新臂的真实短 profile，临时权重不可进入正式候选。候选训练步数只允许 `{6000,12000}`，选 **能在第二阶段 4 小时内完成整套矩阵及其测试/验证的最大值**；安全系数 1.3。6000 仍不可完成则停止，不删臂、不降低图像数量、不改 batch 伪装同协议。

每个新臂以正式 batch 和 loss 做 2 次预热、20 次同步计时更新；临时模型和 RNG 与正式训练隔离。另实测检查点评分、最终指标和输出缓存的成本，不能只用最便宜的一次网络 forward 外推全部实验。profile 本身计入第二阶段上限。

同时检查累计总上限和受保护收尾额。记录 `training_schedule_freeze.json` 后才训练。正式运行开始后不能从 6000 追加到 12000，或者从 12000 裁成 6000；中断未完成记不完整，不能与完整控制等同。

### 10.2 共同训练配方

```yaml
seed: 20261007
batch_source_images: 8
source_pool: utility_fit
sampling: image_uniform_with_replacement
views_per_source: 1
view: nominal
augmentation: none
optimizer: AdamW
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay_all_parameters: 1.0e-4
peak_lr: 1.0e-3
warmup_updates: 200
final_lr: 1.0e-5
clip_global_grad_norm: 1.0
precision: float32
checkpoint_fractions: [0, 0.25, 0.5, 0.75, 1.0]
recovery_interval_updates: 250
logging_interval_updates: 50
```

学习率：第 t 次更新（t 从 1 开始）在前 200 步为 `1e-3*t/200`；之后为 `1e-5+0.5*(1e-3−1e-5)*(1+cos(pi*(t−200)/(N−200)))`。在 optimizer.step 前设置，恢复时由真实 global_step 继续，不重开 warmup。

V2 使用内容组均匀采样；本轮明确改为**图像均匀**以对齐主评测图像权重，全部新臂相同。按内容组等权统计另作敏感性分析，不混入主表。

各方法使用相同源图顺序：独立 NumPy PCG64(seed=20261007) 从按 sample_id 排序的 443 图中有放回取 `[N,8]` 索引，正式前保存序列和哈希。模型初始化用独立 Torch RNG，不能被数据、profile 或诊断消耗。各相同形状层初始权重逐项相同，最后层为零。设备非确定算子必须登记；不能声称字节可复现而未验证。

正式训练所有臂都跑到 N，再统一选点。禁止用“某一臂暂时领先”删除控制。保存 step0、N/4、N/2、3N/4、N 和 latest，包含 optimizer、学习率规则、CPU/设备 RNG、独立数据序列位置、完整身份和未完成日志窗口。

### 10.3 训练日志必须能解释失败

每 50 更新记录源图 ID、源图呈现数、去重数、各 loss、学习率、总梯度范数、裁剪比例、输出 alpha 的边界占比和非有限数检查。源图呈现数是 `8*N`，不是独立场景数。

每个正式检查点在固定训练探针记录 L_dec/L_mom、预测 C 误差、实际质量；使用同样的固定批次额外记录分项梯度及共同裁剪系数。验证集不反传。

出现 NaN、角色泄漏、解码错配、训练参数不更新等可定位工程问题：保存现场，先复现再修复；重跑**所有受影响**臂与下游结果，计入失败成本。没有工程证据时，不把负结果自动归为 bug。

## 11. 检查点、CAL 与方法身份冻结

### 11.1 唯一部署缩放自由度

新模型与 ridge 均使用：

\[
\alpha^{deploy}=\kappa\alpha^{raw},\quad
\kappa\in\{0,0.25,0.5,0.75,1\}.
\]

κ=0 是回到底座，不能称为机制正收益。alpha_ref、s_A、s_C、lambda0、特征标准化和网络参数在 CAL 前全部冻结；CAL 不重新估计这些量。

对每个新训练臂，在 utility_val 上穷举 **五检查点×五 κ**，保存完整逐图表。以平均 PSNR 选网络；并列 1e-12 dB 内选更早检查点，再选较小 κ。此时 κ 仅辅助选网络，不是最终部署值。

全部网络选定后冻结哈希，再在 CAL 上给每个新臂和两个 ridge **重新**从同一五值 κ 网格选最终 κ；不得从 utility_val/CAL 中择优保留较好参数。

### 11.2 固定强度必须充分校准

`FIXED_CAL_B1` 和 `FIXED_CAL_B3` 各使用全体 CAL 的整图二次式，在 10001 值 alpha 网格上选一个全局常数；不是逐图取最佳。这里选择平均逐图 PSNR 最优值，不能只为通过尾部门槛挑一条较差固定控制。

细网格可先用 CPU 精确二次式计算 PSNR；最终冻结常数的 SSIM/LPIPS/视觉必须用真实 float 输出计算。并列选较小 alpha。额外绘制该网格的质量—退化比例曲线，只作描述，不据此更换主对照或反向训练。

### 11.3 冻结主对照

CAL 上从所有**非本轮主候选、非本轮机制消融**的可部署强控制选平均 PSNR 最高者为 `primary_control`。包括新 DIRECT、两个 ridge、细固定控制、正确 J0 与全部有效历史参考；旧 O 消融属于历史控制，不能因为它较强排除。并列按 method_id 字典序。

TRAIN_REGION 的机制集合为 `{MOM_G,MOM_R_NC,MOM_R_NM}`；在 CAL 上选其中最强为 `primary_mechanism_ablation`，三者仍分别报告。TRAIN_GLOBAL 的机制集合仅 `{MOM_G_NM}`。

区域机制控制不参与外部 primary 的选取，但正式闸门另外要求主候选对它们的全部差值，不能隐去某个更强消融。这是预登记角色区分，不是评分后排除强方法。

任何旧控制缺少必要权重、指标实现不一致或不能复现时，先尝试恢复；无法恢复不得宣称完整强对照通过。每个条目保存 `method_id / source_run / checkpoint_sha / candidate_sha / policy / input_role / oracle_space / can_be_primary / requires_reference`。

## 12. 正式开发闸门与停止规则

在最终部署策略冻结后重新汇总 utility_val。它已用于选点，因此是开发判定，置信区间仅作探索性说明，不是独立确认。

唯一主候选 S 必须同时满足：

| 闸门 | 明确条件 |
|---|---|
| 底座收益 | S−J0 平均逐图 PSNR ≥+0.10 dB |
| 强对照收益 | S−冻结 primary_control ≥+0.10 dB，且高于所有其他已登记外部强控制 |
| 机制收益 | S−primary_mechanism_ablation ≥+0.03 dB，且高于所有本轮机制控制 |
| 区域专属要求 | TRAIN_REGION 时，S−MOM_G、S−MOM_R_NC、S−MOM_R_NM **各自** ≥+0.03 dB |
| 区域位置归因 | TRAIN_REGION 时，同一最终策略的 S 分别超过自身 alpha 的空间算术均值、能量加权均值、确定性块置乱输出 ≥+0.03 dB；这些变体不再校准 |
| 结构/感知 | 对 primary_control：平均 SSIM 下降 ≤0.001，平均 LPIPS 上升 ≤0.002 |
| 退化比例 | 相对 J0 下降超过 0.10 dB 的图像比例，不高于 primary_control 的对应比例 |
| 完整性 | 全部本路线训练臂、控制、身份、配对、工程验收与指标完整通过 |

primary_control 若为 J0，其退化比例为 0，这条线确实严格，不能临时放宽。+0.10/+0.03 是沿用的研究操作线，不是领域统一的发表定义。尾部同时报告最差 10% 的平均损失和全分位数；只通过比例线不代表无风险。

状态：

- 全通过：`DEV_PASS`，允许下一节的一次封存。
- 外部质量闸门通过而机制闸门未通过：`QUALITY_ONLY_NO_MECHANISM_EVIDENCE`，不解封，不换名主张机制成功。
- 软件/数据正常，正式闸门失败：`STOP_CURRENT_RECIPE_NOT_SUPPORTED`。
- 不足以完成完整矩阵或存在未解决数值/数据问题：`INCONCLUSIVE_BUDGET` 或具体工程阻塞，不伪装科学否定。

训练充分性单列：若某必要臂的最佳固定网格开发分数位于 N，且最后两次检查点提升都 >0.05 dB，标 `optimization_trend_unresolved=true`；闸门失败时使用 `INCONCLUSIVE_OPTIMIZATION`。未触发不等于证明收敛。该标志不授权自动续训，仍封存数据并收尾。全部闸门通过时亦在报告保留趋势，不能声称穷尽优化。

失败后不自动改阈值、改采样、换候选、添加 O_RGB、启动 B1/B3×MSE/LOG 或扩到 50000。候选目标实验是另一研究问题，只能列为未来方案，不能悄悄作为本轮第三阶段。

## 13. 一次封存确认与统计口径

### 13.1 解锁前的完整冻结

仅 `DEV_PASS` 创建 `selection_freeze_before_sealed.json`，固定：协议和源码哈希、底座/候选、所有最终网络和策略、CAL 选择、主对照/主要消融、原角色清单、全部指标、bootstrap 种子、压力与视觉规则、预算和完整性回执。

一次对 177 图计算所有已冻结条目的名义指标，并生成同源配对记录。中断允许在**相同身份**下补缺失样本，已完成结果按哈希复用；不得改模型后再评分。读过结果后修改科学实现，旧 sealed 即标已暴露，不能静默删除旧记录再称第一次。

### 13.2 统计

主估计量按图像等权。bootstrap 是对内容组有放回抽样，每次取所抽组的全部图像并保留方法配对；重复组按次数计入，再按图像数求均值。不得先平均组再称图像等权。

固定 5000 次、seed=20261017；报告 2.5%/97.5% 分位区间。所有比较共用同一组抽样索引。另列组等权敏感性，不能择优使用。相关性、像素量和区域量的误差条也以源图/组为单位，不把区域当独立训练重复。

封存确认要求保留 §12 全部均值/指标/退化条件，并要求：S−primary_control 与 S−primary_mechanism_ablation 的组配对 95% 区间下界均 >0。区域路线另外要求三个机制差值均为正，三项 ≥+0.03 dB 的均值条件照旧，不因选了主要消融就隐藏其他消融。

通过：`CONFIRMATION_PASS_SINGLE_SEED_HISTORICAL_EXPOSURE`。均值方向合理但区间或机制不足：`INCONCLUSIVE_CONFIRMATION`。负收益或控制胜出：`CONFIRMATION_FAIL`。这仍只有一个新增模块种子，不覆盖底座预训练波动；不能直接写“原创算法已获独立证实”。

sealed 上如报告 oracle/regret，仅限预先固定候选与空间、冻结方法后的诊断；oracle 不进入主排行榜或选择程序。

## 14. 压力、视觉、时延与归因

### 14.1 固定压力

新方法冻结后，在 utility_val 运行 V2 实际 `v2/evaluation.py:stress_field` 中的四族：`tau_050`、`tau_150`、`ambient_green`、`shift_right_4`。检查确为**右移 4 像素且不环绕**；不能误用其他模块的 `shift_down_4` 或六族旧入口。

重新构造先验、候选及实际 r/A/可见描述子；训练尺度、alpha_ref、lambda0、网络、κ 不变。对同一 producer 的控制使用同样受扰候选；RGB/B0 不注入并不存在的先验通路。

报告名义/压力下的逐图 PSNR、SSIM、LPIPS、退化比例、alpha 覆盖率、最差 10% 损失。regret 只能相对同一方法可访问候选空间；区域模型主报块 oracle regret，并可另报到像素 oracle 的 gap。跨候选不能混称 regret。

与 V2 一样，某族 S−J0 或 S−primary_control 平均 PSNR <−0.10 dB 时标 `ROBUSTNESS_CLAIM_FAIL`。未触线只表示四种固定压力没有触发此操作线，不证明普遍鲁棒。压力不改变名义闸门，也不补救名义失败。sealed 的同四族仅在已解锁后按原冻结方案一次补充，不重选参数。

### 14.2 空间归因和视觉

TRAIN_REGION 对最终 MOM_R 追加 §6.2 的块置乱、算术均值和能量加权均值 alpha。使用原策略，不重新校准；这些结果在 DEV_GATE 前完成，用于 §12 的区域位置归因条件。若只有保守缩幅有用、空间位置几乎无增量，就不晋级区域机制，即使部分质量均值为正。它们是冻结策略的诊断变体，不进入新一轮参数搜索；sealed 若解锁，同样按冻结规则计算一次。

每个已允许评测集合的视觉案例：按 `SHA256("v3-visual|"+sample_id)` 前 8 例；再按 S−primary 的 PSNR 差取最好 4、最差 4、中位附近 4，去重，保留 ID 与规则。阶段一无 S 时按 O−J0。sealed 未解锁时没有 sealed 面板。

面板含 I/Y/J0/J1、细固定控制、主候选、primary、主要消融、alpha、实际误差差图；预测 C/真实 C 分开标识。所有比较同一色标，零图不自动放大成伪纹理。PNG 仅用于展示，指标来自 float 输出。保存失败例，不能只看新增方法最好案例。

### 14.3 全流程测速

至少比较 J0、最终 S、对应 DIRECT、FIXED_CAL_B1 与最强历史可部署控制；区域路线另比较 MOM_G。batch1、256×256、相同实际设备和精度，20 预热、100 同步计时，报告均值、中位、p95、峰值显存。

禁用训练时 J0/J1/描述子缓存；包含底座、先验、候选、特征统计与控制器。各方法独立进程测显存，不能把多模型同时驻留峰值当单方法值。另用 fit/val 中预定 20 图测读取→传输→推理→保存的服务耗时，注明磁盘缓存状态和首次编译成本。所有设备耗时计账。

## 15. 实现架构与旧代码隔离

本节是**建议新增的接口**，不是声称仓库已经存在这些文件。优先复用已核验的底座、候选、指标、恢复、角色与预算基础设施；为 V3 增加独立入口与配置。不复制整个仓库另造一套不同公式。

```text
uie_next/v3/
  context.py       # V3 RunContext、身份、路径、状态及角色守卫
  moments.py       # 区域充分统计、相对标签、决策与精确regret
  descriptors.py   # 13→26/52维输入、fit-only标准化
  diagnostics.py   # D1-D6，旧模型只读
  ridge_probe.py   # 五折OOF、两个固定ridge、路由条件
  models.py        # 唯一5505参数网络、MOM/DIRECT变体
  training.py      # 同源图序列、分项日志、完整恢复
  evaluation.py    # 统一注册表、选点、CAL、开发、一次封存
  delivery.py      # 证据、报告、源码与权重可恢复交付
configs/ssuie_utility_v3.yaml
scripts/ssuie_v3_run.py
tests/ssuie_v3/
docs/experiments/ssuie_utility_v3_predictability_20261007/
runs/ssuie_utility_v3_predictability_20261007/
```

实际文件可因仓库风格适度合并，但数学契约、数据边界、状态机和输出不能删减。不得直接把 `V2State` 的停止状态改为通过。新增代码不要向旧 run 写新 selection 或覆盖原报告。

建立 `IMPLEMENTATION_CONFORMANCE.md`，每条列出：本协议条款→公式→函数和文件→测试→实际结果字段。对没有实现、没有运行或无法验证的条款分别标注，不能都写 complete。

部署函数签名只接受图像和冻结模型/配置；Y、sample_id 派生质量、角色、oracle 不得作为预测输入。诊断函数与部署函数分开，实时/缓存/导出/测速均从统一 method_registry 分派，不能再发生“评测一版、部署另一版”。

缓存键至少包含输入哈希、预处理、底座提交/权重/policy、候选权重、先验/干预版本、dtype/后端、描述子版本。标签缓存另外包含 Y 哈希、角色、公式版本；模型输出缓存再含网络及策略哈希。全图/区域尺度文件和不同折缓存必须隔离。

## 16. 训练前的验收清单

测试验证实际风险，不用测试数量代替正确性。保留并运行与复用模块相关的 V2 测试；新功能需以下验收：

| 编号 | 必须覆盖的行为 |
|---|---|
| T01 | 裁剪后 r、RGB共享alpha、凸组合范围、step0相同输出、零残差输出 |
| T02 | float64二次恒等式、相对增益恒等式、区域先聚合后求比值，误差≤1e-10 |
| T03 | oracle内点/边界/零A、精确regret边界项；像素≤块≤图级最小MSE |
| T04 | 真实C正则解与独立数值网格一致；明确收缩中心alpha_ref；预测替换项可为负的反例 |
| T05 | 描述子通道/尺寸/std、整图std不等于平均块std的反例、32块nearest映射 |
| T06 | 五折group不交叉；每折alpha_ref/尺度/标准化不读验证折；改变验证Y不改变拟合对象 |
| T07 | 岭回归图像等权、截距不罚、唯一系数、折级固定控制无泄漏、路由正负与并列样例 |
| T08 | 5505参数计数、共同初始化、DIRECT/MOM相同step0、无上下文输入确实不含全局信息 |
| T09 | 目标C正负均可启动末层；第二次更新前层可收到梯度；底座/候选无梯度及统计变化 |
| T10 | L_dec是每图完整MSE后log；标签detach、epsilon、区域均权和三臂/五臂损失唯一差异 |
| T11 | 分项梯度求和等于总梯度；使用共同裁剪系数；detach日志不能充当可微loss |
| T12 | 配对身份、方法全覆盖、缺效用字段为null、旧基线实时/缓存一致 |
| T13 | CAL/sealed权限拒绝、DEV未过不可解锁、旧状态/结果不可被V3覆盖 |
| T14 | 连续4更新与2+新进程恢复+2：参数/优化器/样本/LR/RNG一致；正式N日程一致 |
| T15 | checkpoint×κ完整表、网络先冻CAL后选参数、强控制/机制角色及并列固定 |
| T16 | 不等组大小的图像等权和组bootstrap反例；同图方法成对；所有指标行数/ID一致 |
| T17 | 随机负控制与匹配原方向幅度一致、合法范围、随机种子不读取Y；置乱保留alpha直方图 |
| T18 | 跨run设备锁、设备事件不重复、恢复不重记旧成本、阶段/总上限与收尾保护 |
| T19 | 科学停止和预算中断均生成真实报告；未执行项为not_run/null，不生成伪分数 |

第一阶段开始前通过其依赖的 CPU 数学/角色测试和真实 8 图旧模型恢复；第二阶段正式训练前通过全部适用测试。

设备集成只使用 fit 允许样本和合成角色：全部本路线臂各至少 2 次临时更新，完整走训练→检查点接口→模拟CAL→指标→压力→导出→新进程加载。模拟CAL/sealed使用合成角色，不借真实保留图。至少一个完整 4 对 2+2 的恢复比较。临时权重标 `engineering_fixture_only`。

GPU/PPU 数值测试先用 float32 `atol=1e-5,rtol=1e-4`；恢复若同环境可逐项一致则记录精确一致，否则解释算子与差值，不为通过放宽容差。实际预训练回归至少复算 J0 与 V2 O 的 utility_val 平均 PSNR，允许与旧 float 指标差≤1e-4 dB；不一致先定位数据/后处理/设备，而非改基准。

## 17. 状态机与执行顺序

```text
RECOVERY_AUDIT
 → PROTOCOL_AND_DATA_FREEZE
 → STAGE1_ACCEPTANCE
 → D1_D6_DIAGNOSTICS
 → D7_OOF_AND_DEV_PROBES
 → STAGE1_DECISION
     ├─ STOP_NO_PREDICTABILITY_SIGNAL → CLOSEOUT
     └─ TRAIN_GLOBAL / TRAIN_REGION
          → STAGE2_ACCEPTANCE_AND_PROFILE
          → TRAINING_SCHEDULE_FREEZE
          → COMPLETE_MATCHED_MATRIX
          → NETWORK_FREEZE
          → CALIBRATION
          → DEV_GATE
               ├─ fail / inconclusive → CLOSEOUT
               └─ DEV_PASS → SELECTION_FREEZE
                            → SEALED_ONCE → CLOSEOUT
```

压力、视觉、时延为相应已允许阶段的附属任务，不能越过角色权限。各阶段恢复必须同时检查：状态、协议/源文件哈希、应有样本数、完整产物、预算事件。不能仅以某个文件存在就跳过。

只读诊断不修改原 checkpoint。每次正式 checkpoint 原子保存并写校验旁文件，状态文件原子替换，事件追加。发现结果影响型 bug 时记录影响依赖图，重算双方所有受影响结果，不选择性保留好数值。

预登记科学配方变化不是普通工程修补：若改变公式、监督、采样、阈值、分支或候选，记录为新协议，当前轮不把它当同配方恢复；先完成当前轮证据和停止说明。可按本文件已授权分支自动推进，无需每阶段重复询问是否训练。

## 18. 必须交付的证据和恢复材料

无论走到哪一步，都生成中文 `FINAL_REPORT.md` 与 `NEXT_DECISION.md`，以及以下适用产物。未执行的阶段在清单内写 not_run 和具体原因，不创建空数字冒充测量。

```text
protocol_source.md
protocol_resolved.yaml
protocol_sha256.txt
IMPLEMENTATION_CONFORMANCE.md
source_snapshot.json
recovery_audit.json
environment.json
data_exposure_ledger.json
roles.jsonl / split_freeze.json
diagnostics/probe_manifest.json
diagnostics/D1_prediction_regret_per_image.csv
diagnostics/D2_spatial_controls.csv
diagnostics/D3_common_policy_ablations.csv
diagnostics/D4_intervention_information.csv
diagnostics/D5_loss_gradient_components.jsonl
diagnostics/D6_matched_null_and_rgb_oracles.csv
diagnostics/D7_fold_manifest.json
diagnostics/D7_oof_per_image.csv
diagnostics/D7_dev_per_image.csv
diagnostics/stage1_summary.md
stage1_decision.json
training_schedule_freeze.json                 # 条件训练时
normalization_stats.json                      # 条件训练时
method_registry.json
checkpoints/<method>/training.jsonl           # 条件训练时
selection/checkpoint_kappa_scores.csv         # 条件训练时
selection/network_freeze.json
selection/calibration_all_scores.csv
selection/calibration_selection.json
selection/development_gate.json
selection/selection_freeze_before_sealed.json # 仅解封时
metrics/development_per_image.csv
metrics/sealed_per_image.csv                  # 仅解封时
metrics/paired_intervals.json
metrics/stress_per_image.csv
timing/deployment.json
figures/manifest.json
tests/acceptance_receipts.json
budget.json / budget_ledger.jsonl
state.json / events.jsonl
FINAL_REPORT.md / NEXT_DECISION.md
delivery/manifest_sha256.json
delivery/recovery_instructions.md
```

报告必须回答：

1. D1 中预测替换和正则/active 的损失分别多少？有无带符号项被误解？
2. 空间均值/置乱后质量改变多少？是否真正利用位置？
3. 旧干预/配对目标为何可能弱，分项梯度证据是什么？哪些仍未知？
4. OOF 与 utility_val 是否都有预测信号？为何选择该路线或停止？
5. 实际训练哪些臂、多少更新、每臂所选检查点和策略是什么？
6. 是否超过细固定强度、ridge、普通门控与历史强控制？
7. 矩监督、整图上下文、空间粒度各自有无增量？有没有仅仅更保守？
8. 数据暴露和 sealed 状态是什么？哪些结论仍只适用于单种子？
9. 实际预算、失败成本、剩余额度和未运行项是什么？
10. 应停止、重新定义研究问题、还是申请后续确认？不能只写“建议加训练”。

生成四类可下载包：轻量审阅包（报告/源码差异/表格/日志）、完整源码协议包、视觉包、权重恢复包。权重恢复必须包含本轮所有正式点及继续实验所需旧底座/producer/旧控制；可以引用另一个同时提供且已核验的继承包，不能依赖服务器私有绝对路径。大包可分卷，提供完整依赖关系。

源码包必须包含 `uie_next/data/` 原六文件、V3 文件、配置、测试和第三方版本身份；检查 `.gitignore` 没有再次遗漏。建立新环境恢复说明，读取并校验所有包的 ZIP CRC 与成员 SHA256，再测试至少所选方法的新进程严格加载。哈希完整不等于算法正确，二者分开报告。

不把服务器同盘 ZIP 当独立备份。有用户已授权的独立存储位置则复制并重验；否则生成下载包，标 `independent_backup_verified=false`，明确需要下载校验。Git 源码备份不替代训练权重。不要自动把图像/权重公开推送到 GitHub。

## 19. 成功以后才考虑的后续内容

本轮达到确认条件后，才能将其作为“值得追加资源的候选”。下一份协议再设计：至少三个同配方新增模块种子、同构 RGB producer 对照、未用于研究决策的新数据/真实场景划分、第二底座和完整成本比较。

三个头部种子共享同一底座，只估计头部训练波动，不代表独立预训练重复。MOM 对 DIRECT 的优势、对矩监督/上下文消融的优势、对 RGB producer 的优势是不同论断，不能互相替代。

本轮仅使用固定 B1 producer，即使成功也不能证明“物理先验不可替代”。即使同时超过所有控制，也需要对最终具体机制做新颖性检索；二次式、log-MSE、门控、正则或改变量名均不是新的理论。

如果失败，保留三类不同结论：有限可见特征缺乏预测信号；新训练配方未转化信号；质量有信号但特殊机制没增量。它们指向不同后续问题，不统称“模型不够复杂/训练不够”。

## 20. 来源、核验范围与交付前检查

### 20.1 事实与参考来源

- V2 固定代码：<https://github.com/kkkridepig/uie-prior-utility/tree/0521bfc1442cf3c69630bd9199a4aa360740a815>。
- 旧数学实现：<https://github.com/kkkridepig/uie-prior-utility/blob/0521bfc1442cf3c69630bd9199a4aa360740a815/uie_next/math/utility.py>。
- SS-UIE 官方：<https://github.com/LintaoPeng/SS-UIE>；论文 *Adaptive Dual-domain Learning for Underwater Image Enhancement*：<https://ojs.aaai.org/index.php/AAAI/article/view/32692>。底座身份与简化实现限制继承已核验 V2 记录。
- *Noise2Noise: Learning Image Restoration without Clean Data*，ICML 2018：<https://proceedings.mlr.press/v80/lehtinen18a.html>。作为图像恢复目标与统计估计的背景，不表示本轮采用无干净参考训练。
- *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*，ICML 2019：<https://proceedings.mlr.press/v97/geifman19a.html>。说明选择/拒绝已有相关研究，不把保守采用直接当原创。

上述两个 PMLR 页面及固定 GitHub 数学文件在本次制订时联网读取成功；远程数学文件与本地源码包字节相同。这里没有声称完成整个领域的系统新颖性检索。

二次恒等式、区域最优解、相对坐标变换和 regret 直接由平方展开推导，正确性不依赖论文权威背书。新网络、损失权重、探针阈值和阶段规则是**本轮预登记设计选择**，不是既有论文保证有效的参数。

### 20.2 本地文档数学核验

制订时使用 NumPy float64 的独立随机数组验证了以下性质：

| 核验 | 结果 |
|---|---|
| 实际二次误差恒等式 | 最大绝对误差约 2.75e-16 |
| 含边界项的 regret | 最大绝对误差约 1.12e-16；覆盖 alpha*=0 和 1 |
| 相对固定强度的增益式 | 最大绝对误差约 1.12e-16 |
| 相对中心正则解的等价写法 | 最大绝对误差约 1.12e-16 |
| 像素/块/图级最优 MSE 包含关系 | 通过 |
| 52→64→32→1 网络参数计数 | 5505 |
| 16 小时预算继承和阶段额度核算 | 一致；计划后未分配约 3.160269 小时 |

这些是**文档公式与预算验证**，不是服务器 V3 代码已实现、真实设备测试通过或产生了新实验收益。服务器仍必须完成 §16 的数学、数据、恢复与真实集成验收，才能开始正式训练。
