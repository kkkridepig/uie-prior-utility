# LSUI 重新上传校验与恢复

本次新上传的文件实际为 `/mnt/workspace/TEMP-FILE-STATION/LSUI_1.zip`。旧文件 `LSUI.zip` 仍为此前损坏的上传包，未删除、替换或解压。

新包通过完整校验后，已解压并核验到项目配置要求的目录：

- `/mnt/workspace/uie-prior-utility/data/lsui/input/`：4,279 张输入图片。
- `/mnt/workspace/uie-prior-utility/data/lsui/GT/`：4,279 张参考图片。

## 校验结果

| 检查 | 结果 |
| --- | --- |
| ZIP 文件大小 | 492,658,225 bytes |
| ZIP SHA256 | `e15c4a2203f4ddf7ecf4af4327f2fa65a4636eeead9ee7185ab5fd2146eb4d3d` |
| 中央目录与文件 CRC | 全部正常，8,558 个文件通过 |
| 图片 SHA256 | 全部与原冻结 manifest 一致 |
| 图片格式与配对尺寸 | 全部通过 |
| 清单 | `manifests/lsui_recon_grouped_v1.jsonl`，未修改 |
| 清单 SHA256 | `b91dad8f23c32aa3c07a9f9fae34b1544c1ec8febc13f91511b2192ea3ecc5fc` |
| train / val / test | `3423 / 429 / 427`，沿用原划分 |
| 场景分组检查 | 原分组隔离检查通过 |
| 近重复检查 | 原 16x16 灰度启发式检查未报跨划分疑点；不证明所有场景独立 |
| 加密 | 新 ZIP 未加密，无需密码 |

既有同分组、同划分的重复文件按原清单保留，完整列表在 JSON 回执中；没有删除图片、重新划分或把历史测试集改称新盲测。

执行时先在 ZIP 内逐文件验证，再解压到独立临时目录并运行原数据审计，全部通过后将临时目录整体改名为 `data/lsui`。本次没有覆盖旧数据目录、修改厂商环境或启动训练。历史自训权重缺失状态未改变。

实际执行命令：

```bash
cd /mnt/workspace/uie-prior-utility
.venv/bin/python tools/recover_lsui_archive.py \
  --archive /mnt/workspace/TEMP-FILE-STATION/LSUI_1.zip \
  --receipt runs/server_recovery_lsui_20261007/lsui_restore_receipt.json
```

该命令拒绝覆盖已有数据目录和回执。机器可读证据：`runs/server_recovery_lsui_20261007/lsui_restore_receipt.json`。先前 `RECOVERY_REPORT.md` 和 `runs/server_recovery_20261007/status.json` 中的 LSUI blocked 为重新上传之前的历史状态；本记录说明 LSUI 数据现已恢复。
