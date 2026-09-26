# 数据下载与 benchmark 协议

选择 UIEB、LSUI、UFO-120 作为配对训练/评估数据，C60、U45 作为无参考外部检查。选取依据是覆盖真实色偏与散射、较大训练规模、不同数据来源，以及困难图像；不把同一数据集换个划分当作独立域。

| 数据集 | 本项目用途 | 入口与限制 |
|---|---|---|
| UIEB | 第一轮训练、同域测试 | [官方](https://li-chongyi.github.io/proj_benchmark.html)：890 对及独立 60 张困难图；参考图是增强参考，不是真实无水同场景测量 |
| LSUI | 独立训练与跨域测试 | [作者页面](https://lintaopeng.github.io/code/)，或 WWE 整理包；实际 split 数量由 manifest 报告 |
| UFO-120 | 独立训练、外部配对测试 | [官方](https://irvlab.cs.umn.edu/resources/ufo-120-dataset)、[作者代码](https://github.com/xahidbuffon/Deep-SESR)；原任务含增强/超分辨率，必须说明尺寸处理 |
| Challenging-60 / C60 | UIEB 训练后的困难域测试 | UIEB 官方 60 张无参考图；不参与训练 |
| U45 | 无参考外部检查 | WWE 作者整理包；不参与训练，不把它当作可计算 PSNR 的配对集 |

## 可直接运行的下载命令

~~~bash
python -m uie download --list

# 官方 UIEB
python -m uie download uieb-raw --output downloads --extract-to data/uieb
python -m uie download uieb-reference --output downloads --extract-to data/uieb
python -m uie download uieb-challenging --output downloads --extract-to data/c60

# WWE 作者提供的多个数据集与划分
python -m uie download wwe-bundle --output downloads --extract-to data/wwe
~~~

核对日期：2026-09-26。代码中的 Google Drive ID 与 UIEB 官方页面、WWE 固定版本 README 对应。LSUI/UFO 官方网页访问可能超时/403，没有把未核实的 URL 或百度提取码写进下载器。WWE 链接属于模型作者提供的整理包，不是各数据集自己的官方发布源。

下载后生成本地 SHA-256 receipt。作者未提供可核实的官方压缩包 checksum，因此本地 receipt 仅用于追踪本次文件，不能声称完成官方 checksum 校验。本地网络对 C60 的下载超时，未验证完整远程压缩包和实际解压后的规模。

Google Drive 失败时：

1. 打开来源页面，在能够访问它的环境下载合法数据，再传到 CPFS。
2. 已有 ZIP/TAR 可以导入，无需重新下载。
3. 若作者更新合法 Google Drive 链接，可用 --url 显式覆盖来源，并自行保存新来源说明。

~~~bash
python -m uie extract /mnt/workspace/raw-890.zip --output data/uieb
python -m uie extract /mnt/workspace/reference-890.zip --output data/uieb
python -m uie extract /mnt/workspace/UnderWaterDataset.zip --output data/wwe

# 仅在确实拿到作者更新后的链接时使用：
# python -m uie download wwe-bundle --url "作者提供的新链接" --output downloads
~~~

解压器拒绝路径穿越、符号链接以及覆盖已有文件；中断后建议选择新的解压目录。下载器保留 .part 文件以便 gdown 续传。

UIEB 官方页面限定学术/非商业用途，并禁止再分发数据。公开仓库只提供代码与作者链接；其他集合也应遵守各自条款。

## 两种固定协议

### 官方 UIEB 数据 + 自定义分组划分

~~~bash
bash scripts/prepare_uieb.sh data/uieb
~~~

脚本寻找 raw-890 和 reference-890 目录。若压缩包层级变化，可显式指定：

~~~bash
python -m uie prepare --layout pairs --dataset UIEB \
  --root data/uieb \
  --input data/uieb/raw-890 --target data/uieb/reference-890 \
  --output data/manifests/UIEB.json \
  --seed 42 --val-fraction 0.1 --test-fraction 0.1 --calibration-fraction 0.05
~~~

目录需按实际解压结果调整。文件按相对路径去扩展名后配对，例如 input/a/001.png 对 reference/a/001.jpg；禁止简单排序后 zip。同名歧义、缺配对会报错。

默认先划出 val/test，再从剩余 train 抽独立 calibration。相同解码像素及提供的 scene 分组先合并，因此重复图不跨 split。该协议不是 U90/U97；不能把其 PSNR 与不同 test list 的论文 PSNR 排名比较。论文应附 manifest、seed、划分数量和图像 ID。

### WWE 作者 test 划分 + train 内验证/校准

~~~bash
bash scripts/prepare_wwe.sh data/wwe
~~~

脚本基于作者 README 所列结构：

~~~text
UnderWaterDataset/
  UIEB/train/input/    UIEB/train/GT/
  UIEB/val/input/      UIEB/val/GT/
  UIEB/test/input/     UIEB/test/GT/
  LSUI/...            UFO-120/...
  Challenging-60/test/*.png
  U45/test/*.png
~~~

test 保留。脚本指定 --validation-from-train，从 train 导出 val/calibration，不使用作者原 val。如果你已独立核查作者 val/test 没有重叠，CLI 支持去掉该参数后采用原 val；这会改变实验协议，必须另存 manifest。

对于 UFO-120，脚本显式启用 --allow-target-resize：当目标分辨率不同，先把目标用 bicubic 对齐原输入尺寸，再做相同训练裁剪或评估缩放。这是**恢复至输入分辨率的增强协议**，不等同于原论文的超分辨率指标。默认其他集合尺寸不匹配会报错。

## C60 / U45 无参考清单

方案 B 已自动准备。若使用独立官方 C60 压缩包：

~~~bash
# --input 可指向只含困难图的解压根，扫描支持多层子目录
python -m uie prepare --layout nonref --dataset Challenging-60 \
  --root data/c60 --input data/c60 \
  --output data/manifests/Challenging-60.json
~~~

不要向 nonref 的 input 放恢复结果、拼图或 GT。所有扫描到的图像都会作为测试输入。

## 防泄漏、深度与搬迁

~~~bash
python -m uie audit data/manifests/UIEB.json
python -m uie audit data/manifests/UIEB.json \
  --target-manifest data/manifests/LSUI.json
~~~

清单记录文件哈希、解码像素哈希、split、可选 scene、深度哈希。跨域审计检查源 train/val/calibration 与目标 test 的精确像素重叠。若发现泄漏，先确定合法去重方案并另存清单，不要关闭检查后继续宣称跨域泛化。

**精确哈希不能识别裁剪、重新压缩、近重复帧或同一航次。** 有场景信息时提供 --groups scenes.json，格式为相对文件 stem → 场景 ID，例如：

~~~json
{"scene01/001": "dive_A", "scene01/002": "dive_A", "scene02/001": "dive_B"}
~~~

WWE 布局中 groups 的 key 相对于各个 input 子目录；用于深度文件的 manifest ID 则带 train/、test/ 等前缀。深度是远处更大的二维 NPY，必须与原图空间对齐、数值有限，作为固定相对深度输入；不是米制深度。用 --depth 指定目录，NPY 路径镜像 manifest ID。代码同步裁剪、翻转图像/参考/深度，并对相对深度做逐图 min-max 归一化。外部深度估计器及其权重没有内置。

manifest 路径相对于数据 root，root 相对于 manifest 位置，因此保持相对目录结构即可在机器间搬迁。不要编辑已开始训练的清单；续训与阶段初始化检查清单 SHA-256。

训练只读取 train，挑选 checkpoint 只读取 val，温度拟合只读取 calibration，最终分数只读取 test。不要把 C60/U45 或目标 test 用来选择 gate、NFE、超参数或最佳 checkpoint。
