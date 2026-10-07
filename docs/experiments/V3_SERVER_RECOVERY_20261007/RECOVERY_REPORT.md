# V3 服务器代码与环境恢复记录

本轮已恢复 GitHub 的 V3 工程、历史文本记录、厂商 PPU 兼容环境和 UIEB 数据，并完成 CPU/PPU 运行检查。**没有启动正式训练或旧 V3/V4 调度。历史自训权重仍然缺失，LSUI 上传包损坏，不能声称原实验状态已经全部恢复。**

目录名称采用本次恢复任务日期 `20261007`；检查的实际 UTC 时间另记于 `runs/server_recovery_20261007/environment_final.json`，不修改服务器时钟。

## 1. 恢复来源与范围

| 项目 | 本轮核验结果 |
| --- | --- |
| 仓库 | `https://github.com/kkkridepig/uie-prior-utility.git` |
| 本地路径 | `/mnt/workspace/uie-prior-utility` |
| Git HEAD | `f6921cfd97c969bc29ec77f1aa0fe7285992de13` |
| 分支 | `main` |
| AGENTS.md | 工作区和仓库中未发现 |
| V3 研究源码 | 已恢复；本轮没有修改 `scripts/cde_v3/` 或 `mpa_diff/` |
| 历史运行记录 | 从三个已核验归档中的两个 runs 包恢复 1,252 个文件 |
| 原父权重及 V3 delta/selector | 备份不包含，当前全部缺失 |
| V4 服务器实现 | 未恢复；仓库中只有方案文档，不能当成实现 |
| 正式训练/调度进程 | 本轮未启动 |

历史记录中 `cpu_load_verified=True`、预算、进度、模型成绩、LIVE_STATUS 等字段属于断电前的证据。它们不是这台新服务器重新加载权重或重新执行实验的结果。

恢复结束再次验证：归档 SHA256/CRC 检查通过，已展开的 1,252 个记录与归档完全一致，待恢复数为 0。103 个受保护源码、配置、数据清单及 V3 文档与当前 Git HEAD 的逐文件 SHA256 相同。

## 2. 厂商环境与依赖

使用 `.venv`，创建方式为 `python3 -m venv --system-site-packages .venv`。不使用旧 Flow bootstrap 安装整个依赖列表，也不以普通 CUDA wheel 替换厂商运行时。

| 组件 | 当前版本/状态 |
| --- | --- |
| Python | 3.8.10 |
| PyTorch | `2.0.0a0+nv2303`，仍来自 `/usr/local/lib/python3.8/site-packages/torch` |
| torchvision | `0.15.1a0`，仍来自系统 site-packages |
| PPU | `PPU-ZW810E`，1 卡，约 48 GiB |
| torch CUDA 兼容接口 | 12.1，`torch.cuda.is_available() == True` |
| NumPy / SciPy | 系统 `1.23.5 / 1.9.3`，未替换 |
| Pillow | venv 中 `10.4.0`，提供旧工程使用的 Resampling/Transpose API |
| OpenCV | `opencv-python-headless==4.8.1.78` |
| LPIPS / libarchive-c | `0.1.4 / 5.1` |
| scikit-image | `0.21.0`，兼容 Python 3.8 |
| imageio / PyWavelets | `2.35.1 / 1.4.1` |

新增 `requirements-recovery-ppu.txt` 和 `scripts/bootstrap_v3_recovery_ppu.sh`。后者只通过 `--no-deps` 安装恢复补充依赖，并比较安装前后 torch、torchvision、NumPy、SciPy 的版本和文件路径，最后执行 PPU FP32 doctor。该脚本不启动实验。

`pip check` 剩余一条系统预装问题：`resampy 0.2.2 requires numba, which is not installed`。系统 Python 与项目 venv 均存在同一问题；这是音频依赖，不在本轮图像恢复调用链中。本轮没有为消除该提示改动厂商系统环境。所有实际图像调用和测试结果见下文，不能把 `pip check` 写成全通过。

## 3. 数据恢复与划分

### UIEB：已完整核验

使用用户之前提供的解压密码完成两个 RAR5 包的完整性测试与解压，退出码均为 0。解压工具来自 RARLAB 官方 Linux 包，保存在项目的 `downloads/rar/`，没有安装到系统路径。

| 内容 | 结果 |
| --- | --- |
| 输入目录 | `data/uieb/raw-890/`，890 张 |
| 参考目录 | `data/uieb/reference-890/`，890 张 |
| 冻结 manifest | `manifests/uieb_recon_grouped_v1.jsonl` |
| 清单 SHA256 | `2f2ade9c77a4022d85f3df870c4ed4c8386f73933be8d087bf444a43e02b12fe` |
| train / val / test | `702 / 91 / 97`，沿用原清单，没有重新随机划分 |
| 图片核验 | 全部 input/reference 文件 SHA256 与原清单一致；配对尺寸一致 |
| 分组核验 | 原分组隔离检查通过；16x16 灰度近重复启发式检查未报跨划分疑点 |

启发式近重复检查并不证明所有真实场景独立；本轮没有把历史已评分数据改称新盲测。

### LSUI：上传包损坏，尚未恢复

上传文件 `/mnt/workspace/TEMP-FILE-STATION/LSUI.zip` 为 492,692,469 bytes，SHA256：

`9cd5e18549bd40486804f90db1c43f71732b6be371988c2a31251e5d48b676d0`

Python ZIP 检查和 unzip 均不能读取其中央目录。对整个文件的 local-header/CRC 诊断仅识别到 GT 相关条目，没有可识别的 input 条目；部分 GT 条目也存在 CRC 不符或 deflate 截断。故这不是密码未填写的问题，不能仅补建 ZIP 目录就作为完整配对数据集使用。

未将部分 GT 抽取成正式 LSUI 数据，也没有用它替代缺失的 input。原 `manifests/lsui_recon_grouped_v1.jsonl` 及其 4,279 对数据身份仍保留不变。请在本地先完整测试解压，再重新上传完整压缩包，建议新文件名 `LSUI-complete.zip`；收到完整包后应按原清单核验图片哈希，不另造划分。

诊断记录为 `runs/server_recovery_20261007/lsui_local_entry_scan.json` 和 `lsui_archive_diagnostic.json`。

## 4. 公开权重与作者源码

已重新下载三个公开依赖并匹配历史完整 SHA256。这些是深度/感知依赖，不是原 UIE 父模型、bank 或 selector 的替代品。

| 文件 | SHA256 |
| --- | --- |
| `weights/depth_anything_v2_vits.pth` | `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378` |
| `weights/vgg19-dcbb9e9d.pth` | `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44` |
| `weights/cde_v3/vgg16-397923af.pth` | `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0` |

Depth Anything V2 源码位于 `third_party/Depth-Anything-V2`，固定到原文档列出的 commit `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`，保留上游 LICENSE。LPIPS 的校准权重来自 `lpips==0.1.4` 包，VGG16 backbone 严格加载已核验的官方权重。

历史自训权重缺失清单见 [MISSING_WEIGHTS.md](MISSING_WEIGHTS.md)，机器可读清单见 `runs/server_recovery_20261007/historical_weights_status.json`。父权重哈希、种子、配置及日志不能反推出丢失的模型参数；公开下载也不能找回自己训练的 checkpoint。

## 5. 必要兼容修复

初次全仓库测试为 75 通过、9 失败，失败原因保留在 `tests_all.log`。解决项为：

- `uie/data.py` 与 `uie/download.py`：将 Python 3.9 才提供的 `Path.is_relative_to()` 换成等价的路径包含判断，保留数据/解压路径守卫。
- `uie/engine.py`：将 `str.removeprefix()` 换成受 `startswith()` 约束的前缀切片，支持 Python 3.8。
- venv 中补齐支持 Resampling/Transpose 的 Pillow；修正 imageio 的版本约束。
- 按原生成器恢复 16 对合成测试图片，生成清单另存恢复目录，与已有合成清单 SHA256 完全一致，未覆盖已有清单。

修改只涉及旧 `uie/` 兼容点。V3 数学、训练、采样源码与冻结配置保持原字节内容。本轮修改尚未提交或推送 GitHub。

## 6. 实际运行检查

| 检查 | 真实结果与边界 |
| --- | --- |
| 全仓库 pytest | **84 passed**，CPU 单元/协议/恢复测试；不是原权重重跑 |
| PPU FP32 doctor | 前向/反向通过，设备为真实 PPU |
| V3 CPU 冒烟 | 通过；随机内存测试 fixture，32x32 |
| V3 PPU 冒烟 | 通过；相同拓扑检查，真实 PPU FP32，32x32 |
| 采样契约 | DDIM 1/2/4/20 实测 NFE 与网格正确；同 seed 可重现、不同 seed 噪声不同 |
| C 通路 | 零初始化各模式精确等于测试父模型；硬 null 轨迹一致；adapter 输出层有梯度，冻结父参数无梯度且未改变 |
| selector/FFT | 116 维结构及输出 shape、Sobel/phase/softphase 的有限值和常数图检查通过 |
| 真实公开依赖 PPU 检查 | 使用原 UIEB train 输入、336x336、真实 DA-V2 与 VGG19；去噪前向/反向、DDIM2、LPIPS 均通过 |
| LPIPS/图像指标自检 | 同图自比为 SSIM=1、LPIPS=0；可控扰动后距离为正。未用随机模型成绩充当实验指标 |
| 语法/差异检查 | compileall 与 git diff --check 通过 |

两个新增冒烟工具都不构造 optimizer、不执行更新、不写模型 checkpoint。随机恢复网络只存在内存中，回执明确标注 `historical_parent_loaded=false` 和 `scientific_result=false`。真实输入检查不读取参考图，不对历史 test 做实验评分。

pytest 自身会在临时目录进行少量合成更新和 checkpoint 写入，以验证断点恢复和冻结协议；这不是正式数据训练，也没有写回原模型路径。

原 V3 Metrics 构造器会写旧 metric identity。本轮依赖检查只在恢复包装器中将回执路径重定向至新目录，随后验证旧 identity SHA256 未变，没有改动原 Metrics 源码。

## 7. 证据位置与当前可用范围

全部本轮运行日志/JSON/XML 位于：

`/mnt/workspace/uie-prior-utility/runs/server_recovery_20261007/`

主要回执：`environment_final.json`、`environment_freeze.txt`、`data_audit.json`、`tests_all_after_compatibility.xml`、`v3_cpu_smoke.json`、`v3_ppu_smoke.json`、`public_ppu/dependency_smoke.json`、`protected_source_hashes.json`、`archive_records_verification.json`。

实际执行命令与安全复查方式见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。使用环境：

```bash
cd /mnt/workspace/uie-prior-utility
source .venv/bin/activate
python -m uie doctor --device cuda --precision fp32
```

当前可以运行代码测试、PPU 数学链路和公开依赖检查。**仍不能从原 85,000 步父模型继续训练，也不能复现原 V3 权重推理。**下一步先补齐完整 LSUI；自训权重只能从用户实际保存的本地文件、云盘快照、OSS 或其他真实备份找回。本轮没有启动新的父模型长训、没有创建随机替代 checkpoint、没有自动恢复旧 pipeline。
