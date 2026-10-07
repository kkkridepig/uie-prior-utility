# 本次官方权重接续的实际命令

工作目录：`/mnt/workspace/uie-prior-utility`。使用已有 `.venv`，没有替换厂商 torch/torchvision。

本次已执行的入口命令：

```bash
.venv/bin/python -B -m pytest tests/uie_next -q --junitxml=runs/ssuie_local_utility_v1_20261007/tests/pytest_official_dispatch_final.xml
.venv/bin/python -B -m uie_next.cli verify-backbone --config configs/uie_next/protocol.yaml
.venv/bin/python -B -m uie_next.cli --help
nohup .venv/bin/python -B -m uie_next.cli run --config configs/uie_next/protocol.yaml --resume > runs/ssuie_local_utility_v1_20261007/logs/official_resume_dispatch.log 2>&1
```

pytest 实际 stdout 记录在 `logs/tests_official_dispatch_final.log`；核验 stdout 在 `logs/verify_official_admission.log`。调度入口的 PID、argv 与阶段回执见 `state.json`、`commands.jsonl`、`dispatch_events.jsonl`。正式工作是否派发及实际完成情况以这些记录为准；调用 run 本身不代表训练已完成。

此前同一预算中，基线策略选择、真实集成和临时测速是直接调用 `Experiment.baseline()`、`accept_real(e)`、`profile_and_freeze(e, profile_only=True)` 完成的。真实恢复三个新进程的完整 argv 与退出码保存在 `tests/real_integration/receipt.json`。

主规范使用实际新上传的 `SSUIE_C局部效用学习_本轮完整执行指南_20261008.md`（正文版本1.1，修订日期2026-10-07）。它补充简化底座身份与 §3.5；旧20261007文件和原阻塞快照保留，不改写用户原文件。算法配方、阈值和16设备小时预算未改变。

结束后还实际执行相同run --resume一次，终止状态幂等核验通过，没有GPU训练派发。实际CPU分析/排查命令：

```bash
.venv/bin/python -B runs/ssuie_local_utility_v1_20261007/tests/candidate_wiring_cpu.py
.venv/bin/python -B runs/ssuie_local_utility_v1_20261007/tests/export_training_curves.py
.venv/bin/python -B runs/ssuie_local_utility_v1_20261007/tests/finalize_observed_statistics.py
.venv/bin/python -B runs/ssuie_local_utility_v1_20261007/tests/verify_terminal_closeout.py
```

绘图首次缺importlib_resources失败，安装CPU依赖后同身份重试通过；首次失败日志保留。数学/科学源码冻结后没有修改，新增的是runs中的离线证据分析脚本。最后只重打包已生成证据，不再次训练或评分。
