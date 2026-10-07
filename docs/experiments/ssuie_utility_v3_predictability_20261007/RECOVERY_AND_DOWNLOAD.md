# 下载、恢复与核验

本轮已在服务器临时新目录解压源码/权重包，利用包内上游Git bundle离线重建官方提交，并在新进程用当前厂商Python严格加载底座、固定producer、B3、O与全部旧registry条目；16个必要权重身份逐项通过。这个验收是新路径/新进程CPU严格加载，不是新装服务器的PPU环境验收，不是用户本地下载。

优先下载FINAL_REPORT.md、NEXT_DECISION.md、review.zip、source_protocol.zip、weights_recovery.zip，再下载visuals.zip、SHA256SUMS与ssuie_v3_verify_downloads.py。四包都下载后在本地执行：

```bash
python ssuie_v3_verify_downloads.py --directory . --independent-copy --receipt V3_CLIENT_COPY_VERIFICATION.json
```

该命令未在用户本地执行。服务器回执independent_backup_verified=false。同盘ZIP不能抵御服务器磁盘丢失。

恢复依赖：源码协议包包含Git历史/官方上游bundle、完整源码、六个原data模块、V3配置/测试和证据；权重包包含完整继承训练点、底座与感知指标权重，D7拟合对象与尺度也随包保存。数据本体另存，按原roles哈希核验；厂商torch/驱动/SDK需单独恢复，environment.json与environment_requirements.txt记录身份，不能用普通CUDA wheel覆盖。PPU专用mamba/causal_conv1d轮子另在既有official_resume_ppu_runtime_wheels.zip；它不包含完整厂商torch/驱动。

恢复到原/mnt/workspace/uie-prior-utility路径可保留原数据路径身份。若改根目录，权重恢复脚本已使用相对路径，数据角色清单不得为迁移随意改写；需显式路径映射并核验原图/参考哈希。旧大型J0/候选缓存未打包，实时部署不依赖缓存；本次验收不声称全缓存或厂商环境已异地重建。

最终核验范围：包SHA256、ZIP CRC、所有载荷成员SHA256、六模块原哈希、最终源码快照；另有全旧目录保护哈希。哈希一致不等于算法有效。
