# Benchmark 获取与初步审计（2026-09-28）

用户已授权从公开来源补齐所需benchmark。S2正式训练只使用UIEB和LSUI。以下下载未接入训练或用于调参。

## 已完成

- U45：作者GitHub仓库ZIP通过CRC/SHA256记录。仓库共450张图包含算法结果和分组重复副本；实际原始输入取upload/U45/U45中的45张，清单manifests/u45_public_v1.jsonl。
- RUIE：作者GitHub仓库ZIP通过CRC/SHA256记录。目录为UIQS/UCCS/UTTS。UIQS与UCCS内容可能重叠，后续不能当作完全独立数据相加。任务集有300张图，299份XML；0316.JPG缺少XML，不推断其为空背景。

原包、解压目录：data/benchmarks/；URL/SHA256：runs/data_intake/public_benchmarks.json。

## 未完成或按阶段获取

- C60：官方Google Drive在服务器连接超时，证据runs/data_intake/c60_download_probe.log。不会从未授权再分发包替代官方UIEB数据。它不阻塞当前S2。
- Atlantis及DIODE：B方向适配/几何数据依赖，当前S2无需使用；待进入该方向按固定版本获取，并核算30GiB工作区空间。GitHub登录不代表已有Kaggle权限。
- RUOD/SUIM：G检测/可选分割候选，不在当前S2中混入。公开下载入口已在docs/BENCHMARK_DOWNLOADS.md核验记录。

## 观测与分析

下载成功不等于完成评测。RUIE标注缺失和子集重叠需要独立任务协议；U45无配对参考，不计算PSNR/SSIM。后续跨域评测前还需检查与源训练图的重叠。

```json
{
  "u45_original_images": 45,
  "u45_manifest": "manifests/u45_public_v1.jsonl",
  "ruie_tasks": {
    "pic_A": {
      "images": 60,
      "xml": 59
    },
    "pic_B": {
      "images": 60,
      "xml": 60
    },
    "pic_C": {
      "images": 60,
      "xml": 60
    },
    "pic_D": {
      "images": 60,
      "xml": 60
    },
    "pic_E": {
      "images": 60,
      "xml": 60
    }
  },
  "annotation_missing_or_invalid": [
    "RUIE/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark-master/UTTS/pic_A/JPEGImages/0316.JPG"
  ],
  "annotation_object_counts": {
    "urchin": 1060,
    "holothurian": 282,
    "scallop": 19
  },
  "training_use": "none; S2 uses UIEB/LSUI only",
  "limitations": "RUIE task partitions/overlap require dedicated audit before G; no mAP claimed"
}
```
