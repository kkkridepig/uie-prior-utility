# SS-UIE 非零候选诊断与局部效用验证：V2 完整执行框架

版本：1.0；制定日期：2026-10-07。建议实验身份：`ssuie_local_utility_v2_diagnostic_20261007`。

本文是交给服务器 Codex 的执行规约，覆盖恢复、代码修补、测试、诊断、条件训练、对照、校准、条件确认和交付。它是对 V1 的显式协议修订，不改变 V1 的结果，不承诺正收益，也不把跑完流程等同于创新成立。

## 0. 执行摘要与授权边界

按以下顺序自动执行，无须在每个阶段重新询问是否继续：

1. 找回缺失的原始数据模块源码，核验代码、数据、官方权重和旧候选身份。
2. 修复已确认的下游评测缺陷，完成所有方法的小规模端到端验收。
3. 仅推理诊断 B1/B3 各 1000/2000/3000/4000 更新的八个非零检查点。
4. 把直接部署的 `standalone_best` 与局部控制使用的 `producer` 分开选择，禁止将零候选作为唯一局部空间诊断对象。
5. 若旧检查点提供合格空间，直接固定 producer；否则最多执行本文规定的一条候选救援路线。
6. 固定 producer 后训练 O/G1/R0；它们直接计入完整矩阵，随后补齐其余八个作业。不以某个默认部署参数下 O 的暂时分数设置新的早停门槛。
7. 全部检查点冻结后，统一校准与开发判定。只有完整开发闸门通过，才一次评分 177 对本轮封存数据。
8. 任一合法停止都执行收尾，输出机器记录、中文分析、可恢复权重与完整源码包。

本轮仍为一个新增模块训练种子 `20261007`。累计设备预算上限沿用 16 小时，包含旧轮约 0.417548 小时；不是另外增加 16 小时。不自动训练 50000 步、不重训 SS-UIE、不全量微调底座、不重启 A/B/D/F/G 独立路线，不引入新模型或随机搜索。

本文允许在现有仓库中创建新实验目录、补充诊断与调度代码、修复确定的实现错误及运行所规定的实验。保留厂商 PyTorch，不做环境整体升级。不得改写 V1 科学记录、删除旧权重或覆盖用户已有工作。本提示不授权上传数据、公开权重、发布代码或向第三方发送消息；本地 Git 提交和可下载备份可以完成，远端推送按用户已有明确授权执行。

## 1. 本轮要回答的问题

### 1.1 已知事实

V1 的 B1 是先验候选头，B3 是相同参数量的 RGB 对照头。两者各完成 4000 更新、32000 张次采样；不是只处理了 4000 张图片。底座为作者公开简化版 SS-UIE，冻结训练。

| 更新 | B1：model_val PSNR | B3：model_val PSNR |
|---:|---:|---:|
| 0 | 27.5233317253 | 27.5233317253 |
| 1000 | 27.4924796114 | 27.4717629538 |
| 2000 | 27.4528014760 | 27.4100068722 |
| 3000 | 27.3835339822 | 27.3147362041 |
| 4000 | 27.4045249374 | 27.3428062845 |

原规则按完整候选 PSNR 选中零初始化输出，所以所选候选满足 `J1=J0`。由此得到 oracle 增量 0、先验敏感性 0；它们不是对八个非零检查点的诊断结果。O 及正式控制、消融没有运行。

4000 更新的 B1 在三个训练输入上的先验梯度与干预响应非零。这排除了明显漏接，但不证明先验有益。旧状态 `STOP_PRIOR_UNUSED` 保持原样，本轮用“所选候选恒等，机制未正式检验”解释其边界。

### 1.2 问题和可接受答案

| 问题 | 本轮对应证据 |
|---|---|
| 完整输出筛选是否漏掉局部有用的修正？ | 八个非零检查点的端点、固定强度、块/像素 oracle |
| 负收益是否伴随 MSE 与平均 PSNR 的权衡？ | 同图逐图 MSE、log-MSE、PSNR、配对区间 |
| 4000 步后的有限续训是否有帮助？ | 仅必要时执行的配对低学习率续训 |
| 改变训练目标是否有帮助？ | 仅满足分支条件时执行的 B1/B3 × 两种损失对照 |
| 无参考图的网络能否利用真实局部空间？ | 固定候选后的 O 和强控制 |
| 收益是否属于特殊机制？ | 数据匹配控制、门控、误差预测融合及主要消融 |

允许结论为正、负、预算不足或原因未决。不能要求系统一定找到正结果。

### 1.3 名词与结果身份

- **底座 J0**：公开预训练 SS-UIE 的固定输出；本轮只训练外接模块。
- **候选 J1**：在 J0 上增加小网络修正后的图像。
- **残差 r**：实际裁剪后的 `J1-J0`。
- **producer**：给局部控制器提供修正的固定非零候选网络；其完整输出不必先超过 J0。
- **standalone_best**：只按完整图像质量选择的候选，允许是零更新。
- **oracle**：使用参考图计算的理想选择，用于开发诊断，不能部署。
- **控制/消融**：前者比较其他解释，后者去除特定机制检验其增量。
- **校准**：只选择固定有限部署参数，不训练网络。
- **内容组**：现有重复/相似内容形成的代理组，不冒充人工核验的采集场景。
- **设备小时**：占用设备的时间乘设备数量；本轮限单卡。

## 2. 与 V1 的显式差异

1. 分离 standalone 与 producer 检查点。producer 不参与零更新竞争。
2. 对全部八个旧非零检查点做诊断，而非仅诊断旧 selected。
3. 候选先在 model_val 选型并落盘冻结，再对 utility_val 做开发检查；不得依据 utility_val 在失败后换第二名反复尝试。
4. 块 oracle 是粗粒度筛选，不是像素方法的严格上界。允许预登记的“仅细粒度空间”分支，见 §8。
5. 效用网络检查点在 utility_val 上用本文固定的“检查点 × 部署网格”选择；随后独立 calibration 只选参数。不同部署默认值不能成为误淘汰检查点的原因。
6. B4 的检查点也在 utility_val 上选择，而非旧实现硬编码的 model_val。这是为匹配目标场景的显式协议变更。
7. 不再将 `convergence_unresolved=false` 解释为收敛证明；保存固定探针与完整逐图曲线。
8. 补齐真实全方法评测验收，门控不具备的预测效用字段记为 null，不伪造。

上述选择发生在开发数据上，不构成测试泄漏，但增加了开发搜索自由度，必须报告所有候选和网格数量。旧结果不重命名为本轮成功。

## 3. 恢复身份与代码完整性

### 3.1 自动发现路径

服务器优先检查：

```text
/mnt/workspace/uie-prior-utility/
/mnt/workspace/TEMP-FILE-STATION/
/mnt/workspace/ssuie_local_utility_v1_20261007_official_resume_code_protocol.zip
```

旧实验相对目录：`runs/ssuie_local_utility_v1_20261007/`。如果路径不同，仅调整路径配置，不修改样本角色、权重、科学参数。ZIP 解压前校验成员路径，防止绝对路径和 `..` 越界。

### 3.2 固定参考身份

| 对象 | 身份 |
|---|---|
| 本轮审计的用户仓库分支 | `research/ssuie-local-utility` |
| 本轮审计的用户仓库提交 | `1f356d62c1ab1379546a3c6e1646be0e8a2953a8` |
| 官方 SS-UIE 上游提交 | `88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df` |
| 官方权重 SHA256 | `977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99` |
| 原源码协议包 SHA256（记录值） | `ed1bc4501047f3d6485cde6f8852e55204d954309f76839f8c5028cc78a759d5` |

若服务器已有更新提交，先比较科学相关差异与旧 `source_snapshot.json`，不能强行 reset 或静默使用不同实现。50 个可获取源码文件与运行快照一致，6 个缺失文件需恢复。以上 SHA 是文件身份，不是第三方对科学结论的认证。

### 3.3 必须恢复的六个源码文件

```text
uie_next/data/__init__.py
uie_next/data/audit.py
uie_next/data/cache.py
uie_next/data/manifest.py
uie_next/data/roles.py
uie_next/data/runtime.py
```

先取服务器仍在的原文件，其次从原 `code_protocol.zip` 恢复，逐一比对旧 `source_snapshot.json`。不能根据接口猜写文件后称作原实现。若原件不可恢复，输出 `BLOCKED_SOURCE_RECOVERY`，仍交付审计报告和缺失清单；不启动科学训练。

修复忽略规则时仅加入精确例外：

```gitignore
!uie_next/data/
!uie_next/data/*.py
```

检查六文件均被 Git 跟踪；保留真正数据集、权重、缓存的忽略规则。不得把图片误加入仓库。将恢复及工程修补提交到本地独立分支并记录提交，不要求重建整个项目。

用全新检出在仓库外启动导入和测试；清除指向旧工作区的 PYTHONPATH 等路径污染。记录实际导入文件的绝对路径。使用现有厂商解释器和已安装依赖即可，不必更换 torch。

### 3.4 权重与环境

严格加载官方模型，禁止 `strict=False` 掩盖不匹配。不安装新版 torch 覆盖厂商环境。记录 Python、torch、设备、驱动、依赖、官方模型参数、权重哈希和实际严格加载结果。

八个诊断检查点：B1/B3 × `step_001000.pt`、`step_002000.pt`、`step_003000.pt`、`step_004000.pt`。另保留各自 step0 和完整 latest，用 sidecar/清单核对。不能将 latest 或旧 selection 指向的 step0 当作指定非零文件。

| 检查点 | SHA256 |
|---|---|
| B1 1000 | `c78961b97dfd7302c6fb1e4f01f4f20f6edfe19786fee5586245434cc3dcb1ea` |
| B1 2000 | `f0074ee1def232f1790a309f1e899b109d7066bfa2c336d1e2d303f9642c08d7` |
| B1 3000 | `c1af746056f0419eeb5787283a5c5e28d15d1ceec95d1dd6cedd4ed7592879af` |
| B1 4000 | `29c77e1a7def6c7ba1a596a919b2e945f127b9349f47b07e8658c51e1193e47e` |
| B3 1000 | `bf973a9a97636c2b5d97e5c61c1633d46b82772eab8a88bad546725e07f7a28f` |
| B3 2000 | `17972e27de546dc6222fcf9cd1903c3b2a653f208cc4d3ecea3d6fbd634ffd37` |
| B3 3000 | `75178a852c8f7078a50a7c8c592427c7ccea608f210b41dcd1a1084d293c4901` |
| B3 4000 | `52183ff21e71a3e301b8cb29264a03c54ddb3c094bab76d15ac3843526eecea1` |

机器执行以旧已验证 sidecar/原始文件重新算出的哈希为权威；文档表格若不一致必须定位并勘误，不能放宽身份校验。不得反序列化来源不明的其他权重。

### 3.5 已知下游缺陷

审计提交的 `uie_next/scientific_evaluation.py` 在汇总预测效用时，对所有模型无条件取 `p['U_hat']`，但 G0/G1/G2 没有该输出，会触发 `KeyError`。修复为按能力显式分派：

- G0/G1/G2 仍完整报告质量、alpha、采用覆盖率、真实效用、oracle regret。
- 它们不提供预测效用时，预测效用误差/相关性为 null，并附 `not_provided_by_method`。
- 不得用真实 U、随机值或零值冒充模型预测。
- 不得用捕获所有异常并跳过该方法的方式“修好”评测。

此缺陷在 V1 尚未运行的后续路径，不能解释 B1/B3 已发生的负收益。修补与全矩阵验收是本轮前置条件。

### 3.6 运行路径与冻结检查的迁移

审计提交的 `records.py` 硬编码旧RUN/DOC/GUIDE，`schema.py` 运行时解析旧Markdown中的配置，部分备份文件名也固定为旧run。必须使用显式V2 `RunContext` 提供run/doc/cache/backup目录、协议和身份，新schema_version承认本轮12k续训与部署网格选点；不能只修改config.run_dir。

运行时以归档的V2结构化配置为依据，不依赖外部旧指南的章节编号、文件名或可能被后来编辑的内容。协议Markdown、解析后的配置和schema均保存哈希。修补后用写路径哨兵测试全部写入均位于新实验允许根目录；旧run、旧docs、原权重与selection在验收前后哈希不变。新增源码工作区修改由Git记录，不与只读旧运行快照混淆。

解封前统一重核源码、配置、所有检查点、候选、角色、训练身份、指标与校准文件，不能只检查calibration/roles两项。身份变化必须回到其依赖阶段重做，不能继续沿用已经冻结的结论。

## 4. 数据角色、预处理与访问权限

复用旧角色文件，不重新随机划分，不以新 run_id 清除暴露历史。

| 角色 | 对数 | 用途 |
|---|---:|---|
| LSUI model_fit | 3608 | 候选训练；固定训练探针 |
| LSUI model_val | 671 | 候选检查点开发选型 |
| UIEB utility_fit | 443 | 效用网络、B4 的新增监督；尺度统计 |
| UIEB utility_val | 136 | 候选开发检查、效用检查点选择、开发诊断 |
| UIEB calibration | 133 | 所有网络检查点冻结后，仅选有限部署参数 |
| UIEB sealed_eval | 177 | 完整开发闸门通过后，一次确认评测 |
| UIEB excluded_overlap | 1 | 继续排除 |

这些是本次预期计数。必须核对原角色、输入与参考配对、文件哈希、内容组不跨角色；不能通过删除异常样本凑数。未知差异归为数据阻塞。原数据曾历史分析，177 对只是本轮未评分，不能称新盲测。

所有正式输入维持 RGB、float32、[0,1]、256×256、OpenCV `INTER_LINEAR`，无随机图像增广。底座主输出固定 `clip01`，不再重新选择后处理。官方 `minmax_float` 只保留为报告中的基线对照，不用于新候选输入。

固定训练探针：对 model_fit 的 group_id 按 `SHA256('20261008|' + group_id)` 的十六进制字符串升序选前 128 组，包含这些组的全部图像。先冻结 ID 和哈希，再读取质量结果。名称为“128 组探针”，报告实际图像数，不能简写为 128 张。

诊断阶段不得给 calibration/sealed 生成 J0/J1 缓存、评分、视觉图或用于尺度估计。只读文件身份/重叠审计不等于评分，但必须另记权限事件。工程验收用 model_fit/utility_fit 和独立合成夹具，不能借用真实封存图。

## 5. 候选与先验：本轮保持不变的科学对象

### 5.1 先验

令 I 为输入，blur 是 9×9 replicate 边界均值滤波。下标 c 为 RGB 通道：

\[
d=\operatorname{clip}(0.5+blur_G-blur_R-\tfrac13\sum_c|I_c-blur_c|,0,1),
\]
\[
k=(2.4,1.2,0.8),\quad \tau_c=k_cd,\quad t_c=\max(e^{-\tau_c},0.05),
\]
\[
A_c=\operatorname{clip}(\max_x blur_c(x),0.05,0.95),\quad
Q_c^{raw}=\frac{I_c-A_c(1-t_c)}{t_c},\quad Q=\operatorname{clip}(Q^{raw},0,1),
\]
\[
q=\tfrac13\sum_c\mathbf1[Q_c^{raw}\notin[0,1]],\quad P=concat(Q,t,q),\quad V=1.
\]

d 是颜色/对比度代理，不是真实深度。P 为 7 通道，V 为 1 通道有效标记。不得添加深度网络、估计器训练或声称完成 A/B 全物理模型。

七视图固定：名义、tau×0.75、tau×1.25、A 加 `[0.04,0,-0.04]`、A 加 `[-0.04,0,0.04]`、d 右移 2 像素、d 左移 2 像素。A 仍裁到原范围；平移不环绕，越界设无效并令 P=0,V=0。干预字段后重算 t/Q/q；不得仅扰动缓存 P 而不更新其物理计算关系。具体边界核对经哈希验证的 `priors/heuristic.py` 与 `priors/interventions.py` 及其真实调用链，不只检查同名接口；差异须先回归测试。

### 5.2 候选网络

`concat(I,J0,P,V)` 共 14 通道；3×3 卷积 14→32 + SiLU；6 个宽32残差块；3×3卷积32→3、末层零初始化。每个残差块为 `x + 0.1*conv2(SiLU(conv1(x)))`，卷积 stride1/padding1、带偏置。无 BatchNorm 和内部随机 Dropout。

\[
J_1=\operatorname{clip}(J_0+0.25\tanh\delta_\theta,0,1).
\]

参数数应为 115907。RGB 对照同结构，`P_rgb=concat(I,ones_like(I),zeros_like(I[:,:1]))`，V=1。训练时两类都以 0.2 概率整图令 P=0,V=0，使用相同随机流。推理不随机丢先验。

## 6. 数学与 oracle 合同

### 6.1 基础恒等式

对每个像素，RGB 取均值、空间不平均：

\[
r=J_1-J_0,\quad e=Y-J_0,\quad a=mean_{RGB}(r^2),\quad b=mean_{RGB}(er),\quad U=2b-a.
\]

\[
J_\alpha=J_0+\alpha r,\quad 0\le\alpha\le1,\quad
\ell(J_\alpha,Y)-\ell(J_0,Y)=a\alpha^2-2b\alpha.
\]

U>0 表示完整修正有利；b>0 仅表示某个正的小强度可能有利，两者不可互换。

对效用网络：`c=sqrt(a)`、`active=(c>1e-6)`，active 上 u=r/c，其余 u=0；v=mean_RGB(e*u)、b=c*v（仅在 active 上精确），预测 inactive 作用设0。先构造安全分母，不允许除零后用 where 掩盖 NaN。

### 6.2 必算策略

- B0：J0。
- 完整端点：J1。
- 固定强度：alpha∈{0,0.25,0.5,0.75,1}，每个值全图共享且全数据固定。
- 图级硬 oracle：每图用 Y 比较 J0/J1 的 MSE 选其一。
- 图级连续 oracle：每图先对 a/b 空间求均值，再取 clip(B/A,0,1)。这是额外诊断，不能部署。
- 32 块 oracle：256×256 划分为不重叠32×32块；先在块内分别平均 a/b，再求比值，块内强度相同。
- 像素 oracle：每像素 `clip(b/a,0,1)`。

oracle 不使用部署 tau/lambda。精确 oracle 在 float64 下由 float32 的实际输出计算，a>0 处求比值、a=0处取0。另输出采用 active 阈值的 `oracle_pixel_active`，用于检查与实际方法定义的差距；二者不得混同。块 oracle 对零块同样取0。

精确像素 oracle 是 RGB 共享、[0,1]凸融合类的上界；块 oracle 不是像素控制器的上界。理论排序（用MSE，允许浮点容差）：像素oracle ≤ 块oracle ≤ 图级连续oracle ≤ 任一全局固定强度；图级连续oracle ≤ 图级硬oracle ≤ 两端点中逐图较好者（最后二者相等）。验证不等式并记录最大数值误差。

任何“当前数据集上最好的固定强度”均为参考辅助开发诊断；部署强度只能从规定校准阶段得到。禁止逐测试图挑强度/检查点。

### 6.3 选择错配单测

两区域 RGB 同值：Y=(0.5,0.5)，J0=(0.4,0.4)，J1=(0.5,0.3)。端点 PSNR 为20与16.989700；局部采用(1,0)为23.010300；任何全局固定alpha的MSE为0.01(1+alpha²)。必须通过这个测试：完整端点不佳时，非零producer仍能进入局部空间诊断。

## 7. D0：旧非零检查点的完整诊断

### 7.1 顺序

1. 冻结诊断协议、八检查点清单、探针、代码和预算。
2. 验证 step0 与 J0 一致作为数值基准；step0不参与producer排名。
3. 对 model_fit_probe 和 model_val 完成八检查点的名义输出与 §6.2 所有策略。
4. model_val 上对 B1 全部四点执行六干预敏感性；同图名义与干预都使用相同候选哈希。
5. 只根据 model_val 按 §8 冻结 producer 或“没有合格producer”的记录，并确定唯一救援分支。
6. 再对 utility_val 报告八检查点全部策略及 B1 干预敏感性。已冻结的选择不得改名次；这里是开发验证，不是盲测。
7. 根据状态转移表继续，不手工挑好图或跳过坏检查点。

### 7.2 每图指标

主指标为每图 RGB MSE 的 PSNR，再对图算术平均；最大像素值1：`PSNR_i=-10*log10(MSE_i)`。保存 MSE、ln(MSE)、PSNR、SSIM。零 MSE 保存 +inf PSNR 和对应标记；不得静默用 epsilon 制造有限分数；出现不可定义差值时停止自动选择，输出 `BLOCKED_NONFINITE_METRIC` 并检查参考泄漏/重复。

`log_mse_eps=ln(mse+1e-6)` 可额外用于损失诊断，但不能替代正式PSNR。平均logMSE与平均PSNR只是线性换算，不能算两份独立有效性证据。

JSON严格禁止NaN/Infinity裸数值；不可定义数值用null并附metric_status，确切无穷可另存字符串`+inf`与原MSE。CSV同样显式标识。不得因序列化失败而丢样本，或将无穷截断成任意“优秀”分数。

SSIM 使用原锁定的 RGB、11×11高斯、sigma1.5、population covariance、valid border实现。D0全部名义端点计算LPIPS；其他oracle/固定强度的LPIPS可不计算，但必须统一标为 `not_scheduled_in_D0`，不以缺失填0。后续正式可部署策略全部报告LPIPS，沿用相同VGG权重及输入归一化。

报告平均值、中位数、5/10/90/95分位、最差ceil(0.1N)张的平均PSNR差、改善>0.10dB比例、退化<−0.10dB比例。差均以同图J0为基准。对每图记录a/b/U分布、残差MAE/RMS、active比例、alpha均值和零/一比例、输出边界像素比例。

### 7.3 逐图记录格式

`checkpoint_image_metrics.csv` 一行=检查点×角色×图像×策略：

```text
run_id,protocol_sha256,checkpoint_id,checkpoint_sha256,training_step,
role,sample_id,group_id,input_sha256,reference_sha256,
prediction_kind,alpha_fixed,uses_reference,n_pixels,
mse,log_mse,log_mse_eps,psnr,ssim,lpips,lpips_status,
baseline_mse,baseline_psnr,delta_mse,delta_psnr_db,delta_ssim,finite_pass
```

另输出 `checkpoint_residual_diagnostics.csv`、`checkpoint_prior_sensitivity.csv`、`checkpoint_summary.csv`、`checkpoint_comparisons.csv`。保存每个干预的名称、视图哈希和逐图响应。比较表至少含同一步B1−B3、各步−0、同方法4000−2000及4000−3000。不得只保留汇总平均数。

所有分位数使用`numpy.quantile(method="linear")`或旧NumPy的等价`interpolation="linear"`。先验响应逐图计算全RGB像素的`mean(abs(J1_view-J1_nominal))`，再跨图算术平均。敏感性表至少含checkpoint_id/checkpoint_sha256/role/sample_id/group_id/intervention_id/view_spec_hash/mae_vs_nominal。残差、标签、alpha的像素分位数是每图内部统计；跨图分位数用不同列名。每个汇总与区间附n_images/n_groups/aggregation="image_weighted"。

### 7.4 统计

主估计量保持图像等权平均。bootstrap 以内容组为单位、有放回抽G组、同一抽样序列配对所有方法，共5000次，seed=20261017，取2.5/97.5百分位。每次抽中组保留全部图片及重复次数：

\[
\Delta^{(s)}=\frac{\sum_g k_g^{(s)}\sum_{i\in g}\Delta_i}{\sum_g k_g^{(s)}n_g}.
\]

不能把“每组均值等权”偷换为主指标；可另报组等权辅助结果。最差10%和退化比例在每次重采样后重算。区间不包含训练种子波动；经过开发选型后的区间是探索性，不能声称多重搜索后的确认显著性。

## 8. producer 选择：冻结后不反复换候选

### 8.1 分开的三类身份

- `B1_standalone_best`、`B3_standalone_best`：各自在允许池内按完整名义平均PSNR选，包含原step0，1e-8dB内并列选较早点。
- `B1_producer`：非零候选，为O和所有基于该候选的控制提供同一修正。
- `B3_oracle_diagnostic_best`：按相同空间排名输出的RGB诊断结果，仅供解释，不自动替换主方法的producer。

另外报告与所选B1同配方、同步数的B3端点，不能只比较各自选择的零点。

每次选择前保存 `allowed_checkpoint_pool.json`。无救援时standalone池各为旧0/1000/2000/3000/4000；低学习率续训时加入6000/8000/10000/12000；损失比较时每配方各为0/1000/2000/4000/8000/12000。损失比较选定producer配方后，匹配的B1/B3 standalone均限该配方池；另外每个已完成配方也单独冻结standalone并注册为额外控制。B4只从匹配B3 standalone初始化。standalone并列在1e-8dB内按较早步、固定配置ID字典序。

### 8.2 确定性资格与排名

所有量均在model_val上计算。对每个B1非零检查点定义：

```text
S = 六个规定干预中最大的一项跨图平均候选MAE
H32 = 块oracle平均PSNR - B0平均PSNR
G32 = 块oracle平均PSNR - 同一候选五值网格最佳固定强度PSNR
Hp = 精确像素oracle平均PSNR - B0平均PSNR
Gp = 精确像素oracle平均PSNR - 同一候选五值网格最佳固定强度PSNR
```

必须 `S>=1e-4` 且残差/标签有限。

- **粗粒度资格**：H32>=0.15dB、G32>=0.05dB，同时 Gp>=0.10dB。先在合格者中按块oracle平均PSNR最大选；并列按像素oracle、较早步、固定配置ID字典序。
- **细粒度资格**：没有粗粒度合格者时，允许 Hp>=0.15dB 且 Gp>=0.10dB 的检查点进入；按像素oracle、块oracle、较早步、固定配置ID排序。状态标为 `FINE_ONLY_HEADROOM`，不能声称已经通过原粗块闸门。
- 两类均无合格者，记 `NO_QUALIFIED_PRODUCER_ON_MODEL_VAL`。

这是预先规定的资源筛选规则，不是学术界统一阈值。细粒度分支防止把粗块能力误作像素方法上界。Gp筛选提供相对简单缩强度的空间要求，但不保证学习器能达到，也不是对最终校准控制的严格优势证明。

producer所有PSNR排序项同样以1e-8dB为并列容差；资格阈值用原始float64汇总比较，不用四舍五入后的展示值。

### 8.3 utility_val 开发检查

model_val选中后写 `producer_selection_before_utility_val.json`，包括全部排名和哈希。对这个唯一候选在utility_val按同一资格规则检查：通过粗粒度或细粒度任一资格则继续。

若model_val有合格候选但utility_val不合格，状态 `STOP_PRODUCER_TRANSFER_GATE`；报告所有已诊断检查点但不换第二名、不执行基于LSUI的盲目续训救场。它表示当前选择与迁移规则未通过，不表示所有局部算法不可能，也不把数据集PSNR差直接当成纯域差。

若model_val无合格者，允许 §9 唯一救援；utility_val八点的结果仍完整保存，但不能用于选择救援类型或损失超参数。

## 9. 条件候选救援：两条路线只能执行一条

仅在旧非零检查点无model_val合格producer时执行；预算上限4设备小时，包含训练、所有验证、缓存和profile。若预计不足，不能删减匹配臂，输出预算不足。

### 9.1 在旧D0结果上一次决定路线

只使用旧B1的4000步model_val：

```text
relative_mse_change = (mean_mse_B1_4000 - mean_mse_B0) / mean_mse_B0
delta_psnr = mean_psnr_B1_4000 - mean_psnr_B0
if relative_mse_change <= -0.005 and delta_psnr <= -0.02:
    rescue = LOSS_COMPARISON
else:
    rescue = LOW_LR_CONTINUATION
```

写 `rescue_choice.json` 后不可更改。0.5%与0.02dB是操作阈值，不是因果确诊；区间和固定探针证据另报。即便趋势不明确，默认续训也只是一次有限的“更多低学习率更新是否有用”诊断，不称已证欠训练。

### 9.2 LOW_LR_CONTINUATION

- 从B1/B3各自step4000的完整模型、AdamW状态、RNG、组采样和缺失先验随机流恢复。
- 明确用新固定学习率阶段替换旧调度器：总步4001～12000恒为1e-5；不重新计算12000余弦、不再预热、不重置AdamW。
- 两者各额外8000更新，有效batch8，其余科学设置不变。
- 保存/验证总步4000、6000、8000、10000、12000；固定探针和model_val所有规定诊断同步执行。
- 中途不因一方落后减少其更新，不先完成B1后依据结果取消B3。
- 原调度器保留为source_scheduler_state，新继续训练配置入身份；恢复检查精确验证下一步学习率和采样序列。

完成后选型池为旧1000/2000/3000/4000及新6000/8000/10000/12000，各头8个非零点。仍按§8只在model_val选择，然后仅对所选新producer执行utility_val资格检查；不再执行第二条救援。

新检查点没有全部做utility_val扫描：未测范围必须写清，不能从所选点失败推断所有新点都无法迁移。对应RGB同配方同步数端点为正式控制，后续可在规定阶段评分，但不用于再次挑producer。

### 9.3 LOSS_COMPARISON

从原step0模型参数和原始初始随机流重建四个作业：`B1_MSE`、`B3_MSE`、`B1_LOG`、`B3_LOG`。从零初始化新增头，不重训SS-UIE。四者同样本顺序、先验缺失流、12000更新、batch8，AdamW参数沿用原值，5%预热后余弦1e-4→1e-5。保存0/1000/2000/4000/8000/12000点。

令 m_i 是每图先平均RGB和全部像素的MSE。MSE臂使用mean_i(m_i)。LOG臂使用：

\[
L_{LOG}=s_{ref}\,mean_i\left[\ln\frac{m_i+10^{-6}}{s_{ref}+10^{-6}}\right],
\]

其中 `s_ref=max(mean_probe(MSE(J0,Y)),1e-4)`，仅用预冻结model_fit_probe的J0计算一次，两LOG臂共用。正的常数缩放为控制数值量级，不改变该目标最优解；不能称两损失梯度完全匹配。记录未缩放log、s_ref和梯度裁剪比例。负loss合法。不是逐像素取log，不允许给PSNR也偷偷加同样epsilon。

组均匀采样仍与图等权评测有权重差异；LOG只是更接近平均逐图PSNR的目标，不声称完全一致。后续a/b/U和效用损失仍定义在原MSE上，不改成log效用。

完成后新producer只在本次两种B1配方的非零点中按§8选择，旧点作为报告参考不重新与新配方混合择优。B3_standalone_best取与最终B1相同损失配方的对应池；同时完整报告另外配方，不能隐藏对照。B4继承这一普通RGB配方的损失；LOG所需s_ref保持原训练探针值。

两种已训练损失下的各自B1/B3 standalone端点及固定强度策略都进入后续共同校准、开发和条件封存，参与primary_control；不能只在诊断里提到另一配方却从正式强对照中移除。无需为另一配方额外训练第二个B4。

该矩阵可比较本次匹配条件下损失效应，但与V1的差别还包括学习率日程和训练长度，不把V1→新结果的全部差归因于损失。

### 9.4 救援终点

救援后仍无合格空间：`STOP_NO_USABLE_PRODUCER`；只有pixel空间却无先验敏感性：另标 `PRIOR_SPECIFIC_CLAIM_UNSUPPORTED`；不自动改成RGB主方法。迁移检查失败仍按§8.3收束。

本轮不自动超过12000总更新。报告固定探针和验证趋势，供下一轮决定是否20k/50k。不能因为预算有余额就增加模型宽度、增强、第二次损失搜索或新先验。

## 10. 效用学习数学合同

固定producer与底座参数、BatchNorm统计，全部缓存绑定候选哈希。候选和标签不接收梯度。

共享h输入 `concat(I,J0,u)` 9通道，stem宽32，4个§5.2残差块，输出1通道3×3卷积无偏置、零初始化：

\[
\hat v=\frac{h(I,J_0,u)-h(I,J_0,-u)}2,\quad \hat b=c\hat v,
\quad \alpha=clip\left(\frac{\hat b-\tau}{a+\lambda},0,1\right).
\]

inactive位置b_hat/alpha为0。h不读取Y、干预类别、原先验P、group_id、幅度c或r；幅度在外部乘回。预测v有正有负。正负前向共享参数。

奇性和正齐次仅针对整张残差共同翻号/正比例缩放、active不变、未额外裁剪的条件；不保证逐像素任意翻号等变，不保证最终alpha尺度不变，也不自动保证预测误差下不退化。

每次更新4张utility_fit源图，每图名义+从六干预均匀选一个，共8个视图。按组均匀采样再在组内选图。所有方法用配对随机流；O-NI也消耗相同视图随机数后将第二视图替换为名义，避免后续序列分叉。

尺度仅从utility_fit七视图算：

\[
s_v=\max(\sqrt{E[v^2|active]},10^{-3}),\quad
s_U=\max(\sqrt{E[U^2]},10^{-4}),\quad
s_{e2}=\max(E[mean_{RGB}(e^2)],10^{-4}).
\]

每图每视图先各自平均再聚合；v忽略无active视图；s_e2只算每张J0误差一次，不再次平方。固定同一尺度文件供全部方法。

Huber rho(z)：|z|≤1为z²/2，否则|z|−1/2。

\[
L_{proj}=mean_{image,view,active}\rho((\hat v-v)/s_v),
\]
\[
\hat U=2\hat b-a,\quad
L_{pair}=mean_{image,active\ union}\rho(((\hat U_0-\hat U_m)-(U_0-U_m))/s_U),
\]
\[
L_{dec}=mean((J_\alpha-Y)^2)/s_{e2},\qquad
L_O=L_{proj}+0.25L_{pair}+0.1L_{dec}.
\]

投影项先对每视图有效像素平均，再对同源图的有效视图平均，最后对有效源图平均；不能平铺所有有效视图，使只有一个有效视图的源图权重减半。成对项先对每图union掩码平均，再对有效源图平均。无有效视图/图只从对应项的分母排除并记数量；无有效项返回能参与反向的零项。尺度统计同样明确源图等权，v²先每有效视图平均、再每图平均、再有效图平均。全数据零有效标签属于工程/候选阻塞。训练决策固定tau=0,lambda=1e-4。部署与训练共用实际决策函数。

## 11. 完整控制矩阵：共11个训练作业

不把候选救援训练次数算成这些方法的独立种子。

| ID | 输入/作用 | 训练目标 |
|---|---|---|
| B4 | 从匹配B3_standalone_best继续训练；每步4 model_fit+4 utility_fit | 普通RGB候选损失，MSE或选中配方LOG |
| G0 | concat(I,J0,r)→1×1线性卷积→sigmoid门控 | L_dec |
| G1 | 同输入，宽32、4残差块→logit | L_dec |
| G2 | 同输入，宽45、4残差块→logit | L_dec |
| F0 | concat(I,J0,J1)，预测误差二阶矩后融合 | 二阶矩Huber+0.25pair+0.1decision |
| R0 | concat(I,J0)，预测基线误差e_hat后投影 | 误差Huber+0.25pair+0.1decision |
| O | §10完整方法 | projection+0.25pair+0.1decision |
| O-NI | 两名义视图，无干预变化 | 同O，pair自然为0 |
| O-NP | 同O | 去pair |
| O-NS | concat(I,J0,u,c)，单网络直接预测v | 同O，去强制奇性/齐次结构 |
| O-ND | 同O | 去decision |

G1/G2/F0/R0/O-NS输出卷积带偏置；O及其保留共享结构的消融不带输出偏置。除F0特殊偏置外，最后一层均零初始化。G0/G1/G2初始alpha=0.5是规定行为，不要求它们初始等于J0。G2仅近似匹配计算量，实际参数/MAC/时延需报告。

R0的e_hat为3通道、无界、零初始化。每张图只前向一次，两个视图共享预测；`v_hat=meanRGB(e_hat*u)`。误差项为mean Huber((e_hat-e)/sqrt(s_e2))。

F0宽32、4残差块，输出z0/z1/zrho：

\[
\hat C_{00}=s_{e2}softplus(z_0)+10^{-8},\quad
\hat C_{11}=s_{e2}softplus(z_1)+10^{-8},\quad
\hat C_{01}=0.999\tanh(z_\rho)\sqrt{\hat C_{00}\hat C_{11}},
\]
\[
\alpha_F=\operatorname{clip}\left(\frac{\hat C_{00}-\hat C_{01}-\tau}{\hat C_{00}+\hat C_{11}-2\hat C_{01}+\lambda},0,1\right),\quad
\hat U_F=\hat C_{00}-\hat C_{11}.
\]

clip范围[0,1]。C的标签分别是meanRGB((J0−Y)²)、meanRGB((J1−Y)²)、meanRGB((J0−Y)(J1−Y))，不是减均值后的统计协方差。C预测误差除以s_e2后对三通道Huber取平均。F0末层偏置为log(expm1(1))、log(expm1(1))、atanh(0.9/0.999)。它是已有误差融合思想的适配控制，不称完整复现对应文献。

参数计数：B1/B3/B4=115907；G0=10；G1=76897；G2=150256；F0=77475；R0=76611；O=76896；O-NS=77185，其余O消融=76896。若不符需解释结构差异，不能悄悄替换模型。

B4是新增监督匹配控制，不等于排除全底座微调。O-NS同时改变方向奇性与幅度输入，只能支持这个组合约束的增量，不能单凭它声称分别证明每个约束独立必要。

B4继承LOG时，仍明确记录它与效用控制的MSE部署项不是相同目标；“数据匹配”不等于所有损失、源图数、计算量相同。日志同时保存标注源图呈现数、源图去重数、视图数和设备时间，不用统一“样本数”掩盖差异。

## 12. 效用训练与检查点选择

### 12.1 固定训练配方

11作业各3000更新，float32，AdamW lr峰值1e-4，betas(.9,.999)，eps1e-8，weight_decay1e-4，梯度范数裁剪1；150更新线性预热后余弦到1e-5。所有方法独立从规定初始化启动，同训练种子和源图/视图流。B4初始化为匹配B3 standalone，优化器新建。

训练顺序：G1、R0、O，然后B4、G0、G2、F0、O-NI、O-NP、O-NS、O-ND。前三个直接保留作完整矩阵，不重复训练。只要producer合格、工程正常且预算允许，就继续剩余方法；不添加“默认tau/lambda下O先赢”的中间性能闸门。

检查点0/750/1500/2250/3000。每250更新恢复点；每50更新记录最近50批的均值/最小/最大损失以及当批值，两者不同字段；记录实际样本数、梯度范数、裁剪比例、学习率、时间。不得以稀疏当批loss代表训练均值。

### 12.2 先在开发集选网络，再独立校准

每个检查点都在utility_val名义条件下比较§13网格。先过滤SSIM/LPIPS越界策略，然后取该点的最佳策略分数；再跨五检查点选择最大平均PSNR。

每个带10策略的方法最多5×10个开发组合；B4为5×5。必须保存全表，不让O额外搜索。若分数1e-8dB内并列，先选择return_base，再较早检查点，再§13策略ID顺序。未校准效用默认结果仍另报。

固定强度网格的alpha=0统一规范化为`policy_kind=return_base`，即使没有独立return_base条目，也按相同并列逻辑处理。B4角色守卫必须显式允许utility_val的只读选点，禁止梯度更新。

这里得到的网格值是 `development_policy_hint`，不是最终部署参数；只冻结网络检查点。在独立calibration上重新从同一固定网格选一个最终部署策略。校准前不许再根据utility_val更换producer、损失、训练总步数或网络。

开发网格存`development_policy_hint`和`utility_checkpoint_grid_scores`，不得提前写入正式`calibration_selection`。所有方法若最终选中全回退，应报告没有可部署增量；不是“安全收益成立”。

旧代码按默认参数选择检查点的路径需要显式替换为V2路径；不能沿用旧 `selection.json exists → return` 而跳过V2选择。

## 13. 部署网格、基线与共同校准

所有网络检查点、候选、尺度、方法列表及源码冻结后，首次评分133对calibration。

- O/R0/F0及O消融：tau∈{0,1e-5,1e-4} × lambda∈{1e-6,1e-4,1e-3}，另加return_base，共10策略。
- G0/G1/G2：temperature∈{0.5,1,2} × margin∈{0,0.25,0.5}；alpha=clip((sigmoid(logit/temperature)−margin)/(1−margin),0,1)，另加return_base，共10策略。
- 固定融合：alpha∈{0,0.25,0.5,0.75,1}。

策略先过滤相对主J0平均SSIM下降>0.001或LPIPS上升>0.002的组合；可行者按平均逐图PSNR最大。return_base总可行。

并列顺序：return_base优先；O类tau降序、lambda降序；门控margin降序、temperature按[1,2,0.5]；固定alpha升序。门控温度排序只是确定性规则，不解释为严格风险排序。

完整可部署基线表至少包括：

1. B0 clip01 与官方minmax_float，两种身份分别报告。
2. B1_producer完整输出、B1_standalone_best完整输出。
3. B2：B1_producer的校准固定强度。
4. B3_standalone_best、与producer同配方同步数的B3端点。
5. B2_RGB：对B3_standalone_best校准固定强度；对同步数B3也加同网格，防止只保留RGB零点削弱对照。
6. B4校准后的固定强度输出。
7. G0/G1/G2/F0/R0、O、四个O消融。

损失比较分支另将所有实际训练配方的B1/B3各自standalone端点与固定强度策略加入上述表和primary竞争。权重完全相同的别名可去重计算，但保留角色映射和完整候选来源。

某两方法输出恒等时保留别名与成本，不冒充独立增益。oracle始终单独表，不参与primary_control。

在calibration上从所有可部署非O/非O消融方法选最强 `primary_control`；并列按预先固定method_id字典序，避免尚未测时却用虚构时延破同分。主要消融 `primary_mechanism_ablation` 从O-NI/O-NP/O-NS按calibration PSNR选；O-ND另报。

所有方法及网格搜索数完整公开；不能只拿原生底座或弱控制作主比较。

## 14. 开发判定与一次封存确认

### 14.1 明确开发判定数据

先完成§13校准并固定部署策略，随后在utility_val用固定策略评估开发闸门。utility_val已经用于检查点选择，因此这里是探索性开发判定；区间不得当独立检验。calibration只确定部署策略与主对照身份，不用于反向训练。

O须同时满足：

1. 对主J0平均逐图PSNR≥+0.10dB。
2. 对固定primary_control≥+0.10dB，并且严格超过所有已登记关键控制的平均PSNR。
3. 对primary_control平均SSIM下降≤0.001、LPIPS上升≤0.002。
4. 对primary_mechanism_ablation≥+0.03dB；对O-NI/O-NP/O-NS均为正。O-ND单独解释。
5. 相对J0退化超过0.10dB的图像比例不高于primary_control的对应比例。若primary_control就是J0，要求这种退化比例为0；这是严格操作线，不能声称存在天然安全保证。
6. 所有关键对照完成、数值与数据审计通过，未发现未解决的工程问题。记录收敛风险，不能将软件flag为false当收敛证据。

结果：全部通过为 `DEV_PASS`；仅质量但机制不通过为 `QUALITY_ONLY_NO_MECHANISM_EVIDENCE`；当前配方未通过且工程完整为 `STOP_CURRENT_RECIPE_NOT_SUPPORTED`。若关键方法最后检查点最好且最后两次提升>0.05dB，或存在明显未解决训练异常，未通过时改记 `INCONCLUSIVE_OPTIMIZATION`，不自动延长。

这些是本轮资源与结论操作线，不是领域统一发表门槛。失败不能泛化为所有局部效用机制无效。

### 14.2 一次性确认

只有DEV_PASS才生成完整 `selection_freeze_before_eval.json`，固定全部网络、producer、部署策略、代码、主对照、主要消融、数据与指标身份，并开启177对sealed_eval。

一次生成所有已选可部署方法的名义结果，之后不可改参数再评分。运行中断允许按同身份恢复缺失样本，不视为新调参机会；已完成样本通过哈希复用。不得逐图重试不同seed求好结果。

确认要求：O相对J0、primary_control均≥+0.10dB；相对primary_control的组配对95%区间下界>0；没有其他关键控制均值超过O；SSIM/LPIPS/退化比例满足上述边界；对冻结主要消融≥+0.03dB且区间下界>0，其他主要消融方向均为正。

通过为 `CONFIRMATION_PASS_SINGLE_SEED_HISTORICAL_EXPOSURE`；区间/机制不足为 `INCONCLUSIVE_CONFIRMATION`；负收益或强控制胜出为 `CONFIRMATION_FAIL`。这些单种子结果不覆盖独立底座预训练或微调种子波动，不自动开启更多种子。

## 15. 压力、视觉与时延

开发完整评估时在utility_val做固定压力测试；若封存解锁，封存压力在名义一次结果完成后用同一固定策略评估。压力不用来重新选模型。固定四族：tau×0.5、tau×1.5、A加[0,0.06,0]后原范围裁剪、d右移4像素且不环绕。字段重算，真实标签只用于评测；名义正收益不等于一般错误先验鲁棒性。

各族报告PSNR/SSIM/LPIPS、损害比例、采用覆盖率和以MSE定义的oracle regret。不能用压力改善替代名义失败。门控没有预测U时相关诊断标null。

regret只能相对该方法所属的同一候选融合空间计算：O/G/F0/R0及B1_producer/B2用该producer的oracle；不同B1 standalone、B3/B4及其融合用各自候选的oracle。另一底座后处理等不属于该线段的方法记regret=NA；如另报与B1 oracle的差值，命名`gap_to_B1_oracle`且允许为负，不能称regret。报告oracle是精确版本还是active阈值版本。

每族单独生成结论：O相对J0平均PSNR<−0.10dB，或相对冻结primary_control<−0.10dB，记 `ROBUSTNESS_CLAIM_FAIL`；其余记录差值与尾部风险，不仅凭未触线就称普遍鲁棒。该标志不覆盖名义质量状态。使用同一受扰候选比较局部控制方法，独立RGB端点/底座继续按其实际输入运行，不向它们注入不存在的先验通路。

视觉固定采用：每个阶段按sample_id哈希取8例、按候选ΔPSNR排序取最好/最差各4例及中位附近4例，去重并保存规则和ID；同分按ID。面板标注真实检查点与是否oracle，包含I/Y/J0/J1/部署输出、实际残差、真实U、预测U（如有）、alpha、误差差图。图像仅作展示，指标从float输出计算；零残差不能自动拉伸成伪纹理。

时延至少比较J0、完整O、primary_control、G1/G2/R0及B4；若无producer，仅报告完成的可部署方法。禁用J0/J1缓存，包含实际底座、先验、候选、控制器。batch1、256×256、相同设备/精度；20次预热、100次计时，显式设备同步，报告均值/中位/p95/峰值显存。另用固定20个允许角色输入测端到端读取/传输/推理/写PNG，说明文件缓存状态。编译首次成本单列，不把训练profile冒充部署加速。

单方法峰值显存必须逐方法独立进程或确实卸载其他方法后测量，不能沿用旧“全部方法同时驻留”的峰值冒充单方法显存。若只取得共享驻留峰值，明确标为共享值并记录单方法数值缺失，不编造差分。测速缓存绑定完整冻结身份，代码/预处理/策略改变即失效。

## 16. 预算与资源守卫

精确旧成本以核验后的原账本为准；本地审计值如下：

```yaml
maximum_device_seconds: 57600.0
inherited_used_device_seconds: 1503.1731119155884
remaining_device_seconds: 56096.82688808441
protected_final_device_seconds: 12600.0
max_devices: 1
```

若服务器旧账本记录更高有效成本，取真实值；不得向下重置。旧60秒保守额度保留其估计身份，不称实测。

| 阶段 | 新增设备小时计划上限 |
|---|---:|
| 工程设备验收、profile与D0 | 1.0 |
| 唯一候选救援，仅必要时 | 4.0 |
| G1/R0/O | 1.5 |
| 其余8作业及训练阶段验证 | 5.0 |
| 校准、开发、条件封存、收尾 | 3.5保护额度 |
| 未分配缓冲 | 约0.582452 |

上限不是耗时承诺。profile使用临时模型/独立RNG，开销计入；不能污染正式训练。未用阶段预算可以转移，但设备总上限和最终保护额度不能突破。保护额度包含最终评估预测，二者不能重复相加。

每个阶段先真实测时，估计全部必要前向、六干预、指标网络、磁盘读写、验证及缓存，乘1.3安全系数。启动前检查当前累计+本阶段预测+剩余必要训练预测+最终保留≤57600秒。若最终评估预计超3.5小时，使用更高真实预测并相应减少前置可用额度。

11效用作业启动前必须确认完整矩阵3000更新能完成；不足则 `INCONCLUSIVE_BUDGET`，不临时缩到不同更新数、不删除强控制。所有设备运行、失败重试、恢复验收、预热计账；CPU文件分析另记墙钟时间。不要求花满预算。

设备锁位于整个仓库/设备共享位置，不能各run各持一把锁后同时占用同一设备。验收两个不同run_id的进程互斥。新账本保留旧成本来源哈希，每个新设备事件只记一次，不把旧1503秒重复累加。

## 17. 代码修改与状态机

在现有uie_next中复用数学、候选、先验、控制和指标模块；给V2增加独立入口/配置/状态文件。允许的新增实现包括非零检查点全扫描、双重身份选型、有限部署网格检查点评分、救援分派、固定探针、逐图统计与预算守卫。不得复制旧run后修改其selection冒充新实验。

建议新运行目录 `runs/ssuie_local_utility_v2_diagnostic_20261007/`，文档目录 `docs/experiments/ssuie_local_utility_v2_diagnostic_20261007/`。

建立唯一 `method_registry.json`：每个method_id绑定底座policy、候选类型/配方/权重SHA、控制器SHA（可空）、部署网格、oracle所属空间、是否可参与primary、是否用参考图。评测/校准/视觉/实时推理/时延/导出均依此分派，替换旧硬编码METHODS/CONTROLS/get_output。新增B1_standalone或B2_RGB不能误走B1_producer分支，也不能尝试加载不存在的控制器。每类ID在16个允许样本上验证缓存质量路径与实时部署路径输出一致。

```text
RECOVERY_AUDIT → ENGINEERING_ACCEPTANCE → D0_MODEL_DIAGNOSTICS
 → MODEL_PRODUCER_FREEZE → D0_UTILITY_DIAGNOSTICS
 → [旧producer合格] PRODUCER_FROZEN
 → [model无producer] ONE_RESCUE → RESCUE_MODEL_FREEZE → RESCUE_UTILITY_GATE
 → PRODUCER_FROZEN → UTILITY_CORE → UTILITY_REMAINING
 → NETWORK_FREEZE → CALIBRATION → DEV_GATE
 → [DEV_PASS] SEALED_ONCE → CLOSEOUT
```

其他分支均进入CLOSEOUT。恢复时只有“阶段标记+文件存在+协议/哈希一致+数量完整”同时满足才能跳过。禁止仅检查selection文件存在。状态写入原子替换，追加事件日志，记录工程修订和科学修订的区别。

旧run与原包只读引用。缓存身份至少包含输入RGB哈希、预处理、底座commit/weight/policy、候选hash、先验与干预版本、dtype及数值后端；标签缓存另含参考哈希、标签公式和角色清单。不同检查点不能共享同名候选缓存。可复用合格J0缓存，先16例验证实时差≤1e-5。

## 18. 训练前必须通过的测试

采用测试驱动修复：先复现失败，再修补，再回归。测试服务实际数学/权限/恢复风险，不以测试数量当质量证明。

| 测试 | 必须验证 |
|---|---|
| T01 全新检出 | 六数据源码被跟踪；导入路径无旧工作区污染 |
| T02 身份 | 错误权重/角色/协议/候选缓存必须拒绝 |
| T03 残差 | 用裁剪后J1−J0，零点精确回J0 |
| T04 二次式 | 随机样本MSE恒等式float64误差≤1e-10 |
| T05 oracle | 像素/块解析解与数值网格最小值一致；排序正确 |
| T06 反例 | §6.3完整候选差仍有局部空间；不能被standalone选择删除 |
| T07 选型 | synthetic指标覆盖粗资格、细资格、无资格、并列、迁移失败 |
| T08 网格检查点 | 默认参数落后、另预定参数有益的检查点能被正确保留 |
| T09 干预 | 字段重算，平移不环绕，标签依据新实际候选，不凭名称造标签 |
| T10 O结构 | 共享参数、整体翻号/正缩放限定性质、inactive处理、有限梯度 |
| T11 梯度 | 正负投影标签均能在零末层启动；冻结底座/候选无梯度与统计变化 |
| T12 控制 | F0矩阵非负定、R0投影、门控不伪造U_hat、消融各只改规定因素 |
| T13 损失 | 每图聚合、尺度量纲、LOG每图先MSE再log、负loss合法 |
| T14 恢复 | 连续4更新与2+恢复+2的参数/优化器/RNG/样本一致；学习率无跳变 |
| T15 数据守卫 | 诊断访问calibration/sealed评分被拒绝；开发未过不能解封 |
| T16 统计 | 不等组大小toy检验按图主估计量与组bootstrap实现 |
| T17 全方法评测 | G0/G1/G2缺U_hat不崩；所有方法均有质量行、NA原因及身份 |
| T18 幂等 | 终止后resume不重复训练、记账、校准或解封；残缺阶段能安全续跑 |

真实设备集成：仅从model_fit/utility_fit取固定各8张，使用官方底座和一个非零旧候选；全部11方法各至少2次临时更新，走“训练→五点接口的合成检查点记录→网格选点→模拟校准→完整指标表→压力→导出→新进程加载”。校准/封存状态机用独立合成角色模拟，不读取真实保留集。

临时权重、尺度和指标标记 `engineering_fixture_only`，不得混入正式结果。设备数值容差预设float32 atol1e-5/rtol1e-4；数学纯float64用更严格容差。遇到偏差先定位原因，不能为通过随意放宽。

## 19. 必须交付的文件

无论在哪一阶段终止，均生成以下适用文件；未执行项在状态表写not_run与原因，不创建虚构分数。

```text
protocol_source.md
protocol_resolved.yaml
protocol_changes_from_v1.md
source_snapshot.json
source_recovery_receipt.json
environment.json
preflight_report.md
tests/acceptance_receipts.json
diagnostics/checkpoint_inventory.json
diagnostics/probe_manifest.json
diagnostics/checkpoint_image_metrics.csv
diagnostics/checkpoint_residual_diagnostics.csv
diagnostics/checkpoint_prior_sensitivity.csv
diagnostics/checkpoint_summary.csv
diagnostics/checkpoint_comparisons.csv
diagnostics/paired_intervals.json
selection/standalone_selection.json
selection/producer_selection_before_utility_val.json
selection/producer_transfer_gate.json
selection/rescue_choice.json                    # 仅需要时
selection/producer_freeze.json                  # 合格时
selection/utility_checkpoint_grid_scores.csv   # 执行效用训练时
selection/network_freeze_before_calibration.json
selection/calibration_selection.json
selection/selection_freeze_before_eval.json     # 仅实际解封时
metrics/development_per_image.csv
metrics/sealed_per_image.csv                    # 仅实际解封时
metrics/method_summary.csv
metrics/stress_summary.csv
timing/deployment.json
budget.json
budget_ledger.jsonl
state.json
events.jsonl
FINAL_REPORT.md
NEXT_DECISION.md
delivery/manifest_sha256.json
delivery/recovery_instructions.md
```

保存所有正式实际训练检查点、优化器/RNG恢复点、初始/最后/所选身份。每阶段将源码和权重分别生成可恢复包；源码包必须包含uie_next/data六文件，包内所有源码与快照逐一比对。权重与视觉不放审阅轻量包，LPIPS权重/第三方依赖身份另列。ZIP做CRC校验和成员哈希核验。

若已有授权的独立存储路径，实际复制并重新核哈希；没有则明确 `independent_backup_verified=false` 并提供可下载包。服务器同盘ZIP不是独立备份，源码Git备份也不替代被忽略的权重备份。

## 20. 最终报告必须回答的十个问题

1. 八个非零检查点是否全部诊断？每个端点和oracle数字是什么？
2. 旧零点选择是否确实漏掉空间，还是新诊断仍无足够空间？
3. 先验通路是否响应？响应与质量改善是否分别验证？
4. 有没有MSE降低而平均PSNR下降的同图证据，区间和权重口径是什么？
5. 是否执行救援、是哪一条、为何选择？续训/损失对照的具体结论是什么？
6. producer与standalone分别来自哪个配方、步数和哈希？oracle选型绝不混为部署收益。
7. O是否超过固定缩强度、普通门控、误差预测、额外监督控制？消融支持哪些具体机制？
8. 哪些阶段/数据集从未评分，177封存是否真的解锁，暴露历史如何表述？
9. 实际花费多少设备小时，多少是实测、保守估计或未知CPU时间？
10. 当前应停止、修改候选、研究优化，还是进入多种子/新数据/第二底座确认？

禁止用“代码完整”“测试通过”“oracle有空间”“训练跑完”中的任何一项代替算法有效性。候选涨点不等于O成立；同一固定底座上的多个新增模块种子也不代表独立底座预训练波动。

本轮矩阵检验固定先验候选下的局部效用机制，尚无同一个O套在匹配RGB非零producer上的正式训练臂。因此即便成功，也不能单凭本轮声称“物理先验不可替代”；后续可用O_RGB等同构控制检验先验专属性。

## 21. 结论状态与用语

| 状态 | 允许表述 |
|---|---|
| BLOCKED_SOURCE_RECOVERY / BLOCKED_ENGINEERING / BLOCKED_DATA | 存在具体工程或证据阻塞，未检验科学假设 |
| STOP_PRODUCER_TRANSFER_GATE | 固定选型规则下的候选未通过目标开发集空间筛选 |
| STOP_NO_USABLE_PRODUCER | 在本轮有限池与规则中无合格候选，不代表一般机制不可能 |
| INCONCLUSIVE_BUDGET / INCONCLUSIVE_OPTIMIZATION | 当前资源或优化证据不足 |
| QUALITY_ONLY_NO_MECHANISM_EVIDENCE | 有质量信号，特殊机制增量未获支持 |
| STOP_CURRENT_RECIPE_NOT_SUPPORTED | 当前完整配方未通过开发操作线 |
| CONFIRMATION_PASS_SINGLE_SEED_HISTORICAL_EXPOSURE | 单种子、历史已暴露、本轮封存条件下支持进一步确认 |
| INCONCLUSIVE_CONFIRMATION / CONFIRMATION_FAIL | 完整报告确认的不确定或负面结果，不继续调封存参数 |

## 22. 数学依据和来源

本轮是对现有机制的实验诊断，未新增声称原创的平方损失公式。二次恒等式、凸融合解析解均直接由展开平方得到；正确性由§18数值验证。

- 原执行指南：`SSUIE_C局部效用学习_本轮完整执行指南_20261007.md`，其§11.2/§12解释V1为什么选中零点。本V2在冲突处覆盖旧指南，未改模块沿用原数学定义。
- 旧证据：`FINAL_REPORT.md`、`FINAL_ANALYSIS_SUPPLEMENT.md`、旧run的`source_snapshot.json`、`checkpoints/*/selection.json`、`tests/candidate_wiring_cpu_receipt.json`及预算账本。
- 用户代码固定身份：<https://github.com/kkkridepig/uie-prior-utility/tree/1f356d62c1ab1379546a3c6e1646be0e8a2953a8>。
- SS-UIE官方：<https://github.com/LintaoPeng/SS-UIE>；论文 *Adaptive Dual-domain Learning for Underwater Image Enhancement*，AAAI 2025，<https://ojs.aaai.org/index.php/AAAI/article/view/32692>。
- 官方简化版权重说明：<https://github.com/LintaoPeng/SS-UIE/issues/1#issuecomment-2894406180>；扫描模块未公开说明：<https://github.com/LintaoPeng/SS-UIE/issues/1#issuecomment-2904097810>。因此本轮不得称完整复现论文SS-UIE。
- F0的相关思想背景：Choi、Elgendy、Chan，*Optimal Combination of Image Denoisers*，IEEE TIP 2019，<https://doi.org/10.1109/TIP.2019.2903321>。本地F0是明确指定的控制，不声称复现该文全部设置。

新机制是否有文献上的原创性，仍需在有效结果出现后针对最终方法做独立检索。本规约只能保证实验问题、对照和证据边界可审查，不能保证创新或正收益。

## 23. 本文件交付前核验

本地已对本文件进行独立科学流程、工程与统计复核，并完成以下数值/文件检查：

- UTF-8严格解码、Markdown代码围栏、数学块配对、章节引用检查通过。
- 文档列出的8个非零候选SHA256逐一与用户提供的实际权重文件一致。
- 16小时累计预算与旧1503.1731119155884秒继承成本核算一致。
- 随机float64样本的局部MSE二次恒等式最大绝对误差约1.11e-16。
- 像素、块、整图连续oracle的MSE排序验证通过。
- 两区域选择反例复算为20.000000、16.989700、23.010300dB；MSE/PSNR目标权衡反例与缩放LOG梯度检查通过。

这些是文档与数学验证，不是服务器V2代码已经实现、设备测试已通过或实验已有收益。服务器仍必须执行§18真实验收，随后按状态机运行。
