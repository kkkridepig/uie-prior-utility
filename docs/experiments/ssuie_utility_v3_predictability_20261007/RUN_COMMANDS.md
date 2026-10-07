# 实际命令与恢复

从仓库根执行；厂商torch保留，使用项目.venv。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.ssuie_v3_run audit
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/ssuie_v3 tests/uie_next
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.ssuie_v3_run stage1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.ssuie_v3_run closeout
```

实际失败与成功日志及events保留在新run；不调用旧pipeline。新环境先恢复源码、厂商环境、数据，再解压weights_recovery到仓库根，逐项核对recovery_audit中的hash。角色原路径如变更需显式身份迁移，不能只按数量重划。数据本体不在交付包，SS-UIE上游源码固定commit，原Git对象另包/源码身份恢复要求见恢复说明。

服务器真实恢复命令（已经执行）：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.ssuie_v3_portable_verify
```

包CRC/成员SHA命令见RECOVERY_AND_DOWNLOAD；客户端命令明确尚未在用户本地执行。

已执行的完整阶段恢复验收：

```bash
.venv/bin/python -m scripts.ssuie_v3_run verify
```
