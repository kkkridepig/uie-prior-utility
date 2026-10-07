# 上传 SS-UIE 权重核验

## 结论与边界

上传文件与锁定的官方 SS-UIE 架构完整匹配，严格加载及真实 PPU 前向通过。在技术上可以补齐此前缺失的 SS-UIE 底座 checkpoint。

**官方下载来源尚待登记。** 官方源码兼容性不等于文件来源证明；锁定 README 未提供可独立比对的 checkpoint SHA256。本次未从官方入口重新下载并比对文件，未将上传文件直接标为来源已核验。已向用户请求 Google Drive、百度官方入口或其他下载来源说明。

未启动正式训练，未读取参考图或封存图，未计算增强质量指标，未选择底座后处理。此前 FINAL_REPORT.md 和缺失权重回执保留为历史快照，当前增量核验以本文件及下述 JSON 为准。正式流程仍需完成来源登记、20 图基线验收、model_val 后处理选择、预算 profile 和缺失调度实现。

## 文件身份

| 项目 | 实际记录 |
|---|---|
| 上传路径 | `/mnt/workspace/TEMP-FILE-STATION/SS_UIE.pth` |
| 字节数 | 83,131,898，约 79.3 MiB |
| SHA256 | `977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99` |
| 文件结构 | OrderedDict，1262 个张量条目，统一 `module.` 前缀 |
| 官方源码 | `https://github.com/LintaoPeng/SS-UIE` |
| 锁定 commit | `88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df` |
| 构造 | `SS_UIE_model(in_channels=3, channels=16, num_resblock=4, num_memblock=4, H=256, W=256)` |
| 参数量 | 20,633,645 |

仅删除统一的一层 `module.` 前缀，随后 `strict=True` 加载；没有缺失键、多余键或尺寸差异，没有随机参数补缺。所有 checkpoint 张量有限。原上传文件未移动、覆盖或修改；尚未写入正式 protocol.yaml，避免绕过协议身份和来源守卫。

## 真实设备验证

厂商环境保持 `torch 2.0.0a0+nv2303`，路径 `/usr/local/lib/python3.8/site-packages/torch/__init__.py`。实际设备 `PPU-ZW810E`，单卡；没有 CPU 推理替代或安装替换 torch。

固定使用已审计的 `model_fit` 输入 LSUI/0、LSUI/1，OpenCV INTER_LINEAR、RGB、float32、256×256、除以 255。没有读取它们的参考图。

| 检查 | 结果 |
|---|---|
| 输出形状／dtype | `[2,3,256,256]`／float32 |
| 原始输出有限 | 通过，范围约 `[-0.067953,1.117831]` |
| 裸网络与封装原始输出最大误差 | 0 |
| 封装与 raw.clamp(0,1) 最大误差 | 0 |
| batch1 与 batch2 原始输出最大误差 | 0 |
| batch1 与 batch2 的 clip01／official_minmax_float 最大误差 | 均为 0 |
| 封装 train() 后内部模型仍 eval | 通过 |
| 底座参数全部冻结 | 通过 |
| 前后全部 state_dict 条目，包括 43 个 BatchNorm 模块状态 | 完全不变 |
| 上传文件前后 SHA256 | 不变 |

这些是小样本加载和推理验收，不能替代完整 T26–T28 训练集成、论文分数复现、20 图质量表或完整部署测速。`clip01` 仅用于本次封装回归，不构成本轮最终后处理选择。

设备阶段计入同一累计预算：本次约 4.55424 设备秒，账本累计 83.58769 秒，约 0.02322 设备小时；16 小时总上限和 3.5 小时保留额度未改变。记录的 4.40033 秒集成流程时间不是部署 benchmark。峰值 allocated 显存 227,522,560 字节仅覆盖这次集成测试。

## 命令与证据

在 `/mnt/workspace/uie-prior-utility` 实际执行成功：

```bash
.venv/bin/python -B tools/verify_uploaded_ssuie_weight.py \
  --checkpoint /mnt/workspace/TEMP-FILE-STATION/SS_UIE.pth \
  --sha256 977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99 \
  --output runs/ssuie_local_utility_v1_20261007/backbone_upload_verification_20261007/receipt_retry.json
```

成功退出码 0，回执状态 `STRICT_MATCH_GPU_PASS_SOURCE_PENDING`。核验工具拒绝覆盖既有回执；复验需要新的 `--output` 路径。

证据路径：

- `runs/ssuie_local_utility_v1_20261007/backbone_upload_verification_20261007/receipt_retry.json`：成功核验的逐项回执。
- `runs/ssuie_local_utility_v1_20261007/backbone_upload_verification_20261007/receipt.json`：第一次工具执行失败的原始回执，退出码 1。
- `runs/ssuie_local_utility_v1_20261007/backbone_upload_verification_20261007/generated_import_bytecode/`：保留首次直接加载生成的两个未跟踪 Python 3.8 字节码文件。它们触发了上游目录守卫，源码本身没有修改；移出上游并禁用字节码写入后同一权重通过。
- `runs/ssuie_local_utility_v1_20261007/budget.json` 和 `budget_ledger.jsonl`：累计设备成本。

未覆盖旧运行或旧报告，没有自动推送 GitHub。没有生成独立存储备份；当前上传文件仍位于服务器原路径。
