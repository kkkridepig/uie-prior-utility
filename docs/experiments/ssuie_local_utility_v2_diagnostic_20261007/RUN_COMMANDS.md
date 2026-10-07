# V2 实际入口、收尾与恢复命令

固定工作目录 `/mnt/workspace/uie-prior-utility`，使用本项目 `.venv` 和厂商 PyTorch/PPU；无需更换 torch。原实验实际派发、训练和子进程命令保存于新 run 的 `commands.jsonl`，安全中断/恢复/科学停止历史见 `events.jsonl` 与 `logs/`。下面提供可复现接口；客户端下载步骤没有在用户本地执行，不以文档中的命令列表代替运行回执。

```bash
cd /mnt/workspace/uie-prior-utility
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m uie_next.v2.cli status
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m uie_next.v2.cli run
```

本轮已经 `STOP_CURRENT_RECIPE_NOT_SUPPORTED`。终止状态下 run 只复用收尾，不解锁新训练或177对；不要删除状态来重新派发。V1和更早MPA的README一键命令不用于本轮。官方源码net是namespace包，原冻结loader同进程重复加载有兼容问题；收尾包装器保留科学源码并验证namespace身份：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.ssuie_v2_safe_closeout
```

该入口已实际通过终止幂等测试：状态/预算/访问/producer/校准哈希未变，见 `tests/terminal_closeout_idempotence.json`；namespace重复strict加载和错源拒绝见 `tests/namespace_closeout_recovery.json`。

科学调度停止后的收尾接口按如下顺序运行，不训练、不改变选择；GPU视觉与测速计入原累计账本：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.ssuie_v2_export_tables --self-check
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.ssuie_v2_export_tables
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.ssuie_v2_finalize_evidence --analysis-only
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.ssuie_v2_full_pipeline_timing
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.ssuie_v2_comparison_visuals --role utility_val
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.ssuie_v2_export_tables
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.ssuie_v2_final_audit
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.ssuie_v2_render_final_report
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.ssuie_v2_finalize_evidence
python scripts/ssuie_v2_verify_downloads.py --directory /mnt/workspace --receipt runs/ssuie_local_utility_v2_diagnostic_20261007/delivery/SERVER_PACKAGE_VERIFICATION.json
```

后一次表导出包含追加的允许开发视觉访问。恢复/审计只处理同身份已完成样本；数据、标签或输出若变，必须重新生成全部受影响的配对侧。本次额外修补只涉及收尾导出、访问事件解析和设备锁，不改这些科学对象。

最终源码/ZIP包含完整恢复点及checksum旁文件。恢复需要原UIEB/LSUI图片、指定官方Git身份及度量权重/PPU扩展依赖；路径、配置、来源和哈希须通过身份审计。新空目录恢复，不覆盖现有工作区，不用strict=False绕过。

客户端下载四ZIP与SHA256SUMS、验证脚本后，在独立存储目录运行（标准库，无需torch）：

```bash
python ssuie_v2_verify_downloads.py --directory . --independent-copy --receipt CLIENT_COPY_VERIFICATION.json
```

服务器执行没有 `--independent-copy`，仅检验包完整性。客户端副本和回执未取得前，`independent_backup_verified=false`。没有自动推送GitHub。
