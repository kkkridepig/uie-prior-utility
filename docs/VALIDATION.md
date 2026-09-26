# 验证记录

日期：2026-09-26。

## 已实际运行

本地 Windows 11、Python 3.12.0、PyTorch 2.4.0+cpu、NumPy 1.26.4；与服务器 PPU 环境不同。

- 20 项 pytest 测试通过：FM/DDIM 的三种空间条件和奇数尺寸前后向；先验边界；效用标签符号与 stop-gradient；相同初始噪声；按文件名配对与缺配对拒绝；跨 split 精确重复拒绝；压缩包路径检查；深度哈希与同步增强；已知 PSNR/SSIM；拒绝使用未训练门控；恢复器冻结；CPU 续训与不中断训练逐参数完全一致。
- Flow Matching 完整 CPU smoke 通过：baseline 训练与续训 → 残差训练 → 效用训练 → 独立校准 → 两张合成 test 图评估与图片导出。
- DDIM 完整 CPU smoke 同样通过。
- CPU doctor 的完整模型前向/反向通过。
- editable 安装通过，使用 --no-deps --no-build-isolation。
- 在 Git Bash 隔离目录实际执行 prepare_wwe.sh、run_pipeline.sh（含第二次续跑跳过）、evaluate_suite.sh，合成数据的训练、校准、gate/NFE/扰动评估全部完成。独立保存合成 manifest，未占用正式 data/manifests。
- 所有 Bash 脚本语法与 LF 换行检查通过；文档相对链接、UTF-8、上游代码与 LICENSE 的 SHA-256 检查通过。

初次测试发现 NumPy 2 与本地 torch 2.4 的互操作错误，已固定 NumPy 1.26.4 并重新验证。后续测试覆盖了未训练 gate 被误用、深度变更未审计、缺参考路径等问题，相关修复已纳入。

smoke 使用程序生成的 12 张小图、极少训练步数，仅验证软件链路；其数值不属于 UIEB/LSUI/UFO-120 科学实验结果。本地报告保存在被 Git 忽略的 artifacts/smoke_cpu_final 和 artifacts/smoke_ddim。

## 尚未验证

- 没有访问用户 PPU 服务器；尚未运行 PPU FP32/BF16 smoke、正式训练或吞吐测量。
- 没有下载完整 benchmark 后训练；C60 官方下载在本地访问 Google Drive 时连接超时。来源链接已核对，但数据包当前可达性、内部具体数量和作者划分不作已验证声明。
- 没有 SOTA 分数、相对提升或成功率结论。
- 未实现近重复/航次自动分组、下游检测/SLAM评估、LPIPS/UIQM/URanker、视频一致性。

服务器验收：

~~~bash
python -m uie doctor --device cuda --precision fp32
python -m uie smoke --output artifacts/server-flow-fp32 --device cuda
python -m uie doctor --device cuda --precision bf16
python -m uie smoke --output artifacts/server-flow-bf16 --device cuda --precision bf16
python -m uie smoke --output artifacts/server-ddim --device cuda --sampler ddim
python -m pytest
~~~

CPU CI 与上述本地检查均不能替代 PPU 上的算子和真实数据验证。
