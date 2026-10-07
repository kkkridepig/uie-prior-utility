# 下载、校验与恢复

本文件说明交付包用途，不表示客户端下载已经完成。实际包的大小、SHA256、CRC 和成员核验结果以 `runs/ssuie_local_utility_v2_diagnostic_20261007/delivery/archive_receipts.json` 为准。

建议下载顺序：

1. 本目录的 `FINAL_REPORT.md`、`NEXT_DECISION.md`。
2. `/mnt/workspace/ssuie_local_utility_v2_diagnostic_20261007_review.zip`：报告、审计、测试、逐图表、状态与预算；不含权重、数据图片或候选缓存。
3. `/mnt/workspace/ssuie_local_utility_v2_diagnostic_20261007_source_protocol.zip`：完整代码、协议和数据角色名单。必须包含六个原始 `uie_next/data` 模块。
4. `/mnt/workspace/ssuie_local_utility_v2_diagnostic_20261007_weights_recovery.zip`：官方简化底座、原 B1/B3 和 V2 实际训练恢复点、优化器和随机流、校验旁文件。不能用只含 `model_state` 的导出替代此恢复包。
5. `/mnt/workspace/ssuie_local_utility_v2_diagnostic_20261007_visuals.zip`：固定规则的实际诊断和方法面板；参考图只用于展示与诊断。
6. `/mnt/workspace/ssuie_local_utility_v2_diagnostic_20261007_SHA256SUMS.txt`：包的外部校验清单。

还应保存 `ssuie_local_utility_v2_diagnostic_20261007_local_history.bundle` 和单独的 `ssuie_local_utility_v2_diagnostic_20261007_upstream_ssuie.bundle`。前者保留本地项目提交，后者恢复严格底座加载所核验的官方 Git 身份。官方 bundle 含作者仓库自己的公开样例图片，与不含图片的源码/审阅 ZIP 分开。

度量网络权重与厂商扩展轮子是单独的恢复依赖，不是方法权重：`ssuie_local_utility_v1_20261007_metric_weights.zip`、`ssuie_local_utility_v1_20261007_official_resume_ppu_runtime_wheels.zip`。它们是否实际存在、大小与校验值见 `delivery/recovery_dependencies.json`。保留厂商 torch/torchvision；不要用 NVIDIA torch 安装包覆盖厂商环境。

在本地保存下载包与校验清单后，Linux/macOS 可使用：

```bash
sha256sum -c ssuie_local_utility_v2_diagnostic_20261007_SHA256SUMS.txt
```

只下载其中一部分时，对已下载文件逐项运行 `sha256sum 文件名` 并与清单比对，清单中其他文件缺失不代表已下载包损坏。Windows PowerShell 可使用 `Get-FileHash 文件名 -Algorithm SHA256`。客户端校验完成前，`independent_backup_verified` 维持 `false`。

源码 ZIP 中以下六项必须逐一存在，并与包内成员清单及原 V1 源码快照一致：

```text
uie_next/data/__init__.py
uie_next/data/audit.py
uie_next/data/cache.py
uie_next/data/manifest.py
uie_next/data/roles.py
uie_next/data/runtime.py
```

复原时先在新空目录检查 ZIP 成员，不向现有工作区覆盖解压。源码、权重包均以项目根目录为相对路径；运行名单的原始绝对数据路径需恢复到相同位置并通过全部文件身份检查。数据图片不在本轮这些包中，需要另保留用户原始 LSUI/UIEB 压缩包及其校验值。Git bundle 也不包含被忽略的实验权重。

原运行的源码快照、协议、角色、权重和校准身份必须全部匹配；不匹配时停止恢复并定位，不能以 `strict=False`、重新划分数据或删除状态文件绕过。终止科学状态下的 `uie_next.v2.cli run` 只复用收尾证据，不会合法地重新训练或解封数据。实际执行过的命令见 `commands.jsonl` 和 `RUN_COMMANDS.md`；本文件中的客户端下载与恢复操作是说明，没有声称已在用户本地执行。

服务器同盘 ZIP 和本地 Git 提交均不构成异地备份。没有可核实的独立存储复制或客户端下载回执时，不登记备份成功。
