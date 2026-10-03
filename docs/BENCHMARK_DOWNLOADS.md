# 架构全实验的数据下载清单

依据 `/mnt/workspace/CODE_RECONSTRUCTION_ARCHITECTURE.md` §6、§8、§12–15，2026-09-28核对。数据集、指标权重、算法基线是不同依赖；下面先列数据。S0–S2只需要前两项，后续方向按条件使用其他数据。

| 数据 | 角色 | 官方入口/下载 | 下载内容与边界 |
|---|---|---|---|
| UIEB890 | S2配对基线训练/验证/测试 | https://li-chongyi.github.io/proj_benchmark.html | raw-890 + reference-890，各890张；重建划分702/91/97 |
| LSUI4279 | S2独立配对基线 | https://github.com/LintaoPeng/U-shape_Transformer ; https://drive.google.com/file/d/10gD4s12uJxCHcuFdX9Khkv37zzBwNFbL/view ; https://pan.baidu.com/s/1dqB_k6agorQBVVqCda0vjA （lsui） | 完整4279对，保留input/GT；重建划分3423/429/427，不用仓库示例替代 |
| UIEB Challenging-60 | 无参考/跨域测试；常称C60/U60，最终需核对论文身份 | https://drive.google.com/file/d/1Ew_r83nXzVk0hlkfuomWqsAIxuq6kaN4/view ; https://pan.baidu.com/s/1nujbTEC8mMPDmft_XUlpCg | 60张原始图，无配对GT；不用于有监督调参 |
| U45 | 无参考/跨域测试 | https://github.com/IPNUISTlegal/underwater-test-dataset-U45- ; https://github.com/IPNUISTlegal/underwater-test-dataset-U45-/archive/refs/heads/master.zip | 原始45张输入；仓库也有其他算法输出，不能误当GT |
| RUIE | UIQS质量/UCCS色偏/任务测试；F/G | https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark ; https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark/archive/refs/heads/master.zip | 建议完整仓库ZIP。2026-09-28 API核验根目录UIQS、UIQS.rar、UCCS、UTTS；论文任务集称UHTS，需保留命名溯源。UCCS与UIQS重叠，不累加成独立样本。检测须核实框标注和训练来源 |
| Atlantis CVPR2024 | B深度适配/合成场景验证 | https://github.com/zkawfanx/Atlantis ; https://www.kaggle.com/datasets/zkawfanx/atlantis/data | 图像/深度/mask/元数据/原场景ID/划分信息；Kaggle可能需要登录。仓库现同时介绍Atlantis++，本架构使用2024版，必须记录下载版本。生成条件或伪深度不能冒充真实米制测量 |
| RUOD（建议补充，G检测） | 补足带框检测训练/测试与第二数据集验证 | https://github.com/dlut-dimt/RUOD ; http://pan.dlut.edu.cn/share?id=ynp3v2tkkiiq | 作者README已核验；图像/框/类别/划分全部保留。不是架构原文指定的固定集合 |
| DIODE（按需） | 自行重新生成Atlantis或补源场景几何 | https://diode-dataset.org/ | RGB、深度、有效mask、场景身份；不需要重新生成且Atlantis数据已齐时可暂不下载 |

## 架构尚未指定具体数据集的部分

- B真实水下米制验证：需要标定过的真实RGB+深度/range+mask+相机标定及场景/序列ID。架构没有指定可直接下载的固定集合；不能宣称下载Atlantis就覆盖此项。无数据时只研究relative_proxy。
- G检测训练：RUIE论文提到300任务图及1800浅水训练图，但仓库实际标注和训练来源必须核实。官方RUIE树中已确认UTTS/pic_A/Annotations等XML标注目录，但训练来源/配对/划分仍需审计。建议补充RUOD，冻结版本、类别、框格式和划分；不能只用300测试图训练/调参。不同年份URPC不能混为一个固定数据集，本架构没有选定年份，暂不提供未经核验的网盘ID。
- 可选分割：建议SUIM（不是架构点名必需项），https://github.com/xahidbuffon/SUIM ，下载图像及像素类别mask，保留作者train/test。不能把语义mask直接称检测框标注。
- SLAM/VO：架构当前只保留扩展接口。需要真实序列、时间戳、相机标定、轨迹GT；无这些数据时不报告SLAM精度。

## 不需要重复下载的内容

DA-V2 Small和VGG19已在服务器weights/并校验哈希。A的受控物理合成样例由本项目生成；C/D/E主要复用上述训练数据和已训练教师，不各自新增必需公开集。E的256/512/1080p是性能协议，不是数据集名称。

LPIPS、URanker属于指标实现/预训练权重（非benchmark图像包），需另外锁定版本和权重；UCIQE/UIQM属于公式实现核验。目前实现只对PSNR/SSIM给出已核验数值，其他项缺测不能更名替代。EUVP、UFO-120虽常见，但不是本架构必需下载项。

上传至 `/mnt/workspace/TEMP-FILE-STATION/`，保留原压缩包、文件名、数据许可和下载说明。UIEB仅学术用途、禁止再分发。建议本地先测试解压并保存SHA256，上传后逐包比对。服务器当前工作区磁盘约30GiB，Atlantis/DIODE大包宜先确认大小；数据和静态缓存需预算空间。
