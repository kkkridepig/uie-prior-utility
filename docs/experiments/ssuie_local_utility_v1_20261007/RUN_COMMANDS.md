# 实际执行命令与范围

工作目录 `/mnt/workspace/uie-prior-utility`，解释器 `.venv/bin/python`。以下均为本轮实际调用；失败也记录。不是未来训练命令清单。

| 实际命令/调用 | 状态与回执 |
|---|---|
| `python -m gdown 1YxyagMCbApON8dRdiQTaQG65g3tnZkPt -O downloads/ssuie/official_checkpoint.download --continue` | 网络失败；logs/official_weight_download.log；没有权重文件 |
| `curl` 官方百度分享链接、Google view/usercontent、官方 GitHub releases | 百度 errno117/链接不存在；Google 超时；GitHub releases为空；downloads/ssuie |
| `git clone --branch v1.1.1 --depth 1` mamba/causal-conv1d官方仓库 | 固定源码commit；未替换SS-UIE模型 |
| `MAMBA_FORCE_BUILD=TRUE MAX_JOBS=2 python -m pip wheel --no-deps --no-build-isolation third_party/mamba_1_1_1 --wheel-dir downloads/ssuie` | 首次限时600秒、恢复1800秒/打包依赖失败、再次600秒限时；编译出的完整.so保留并验收，日志均保留 |
| `CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=2 python -m pip wheel --no-deps --no-build-isolation third_party/causal_conv1d_1_1_1 --wheel-dir downloads/ssuie` | 成功；logs/causal_conv1d_build.log |
| `cp -p build/lib.linux-x86_64-3.8/selective_scan_cuda.cpython-38-x86_64-linux-gnu.so build/lib.linux-x86_64-cpython-38/selective_scan_cuda.cpython-38-x86_64-linux-gnu.so` | 复制已经实际编译并回归的二进制到升级后的setuptools产物目录；没有编辑kernel源码 |
| `MAMBA_FORCE_BUILD=TRUE python setup.py bdist_wheel --skip-build --dist-dir ../../downloads/ssuie` | 在third_party/mamba_1_1_1执行，成功；logs/mamba_verified_binary_packaging.log |
| `python -m pip install --no-deps` 本机mamba/causal wheels及requirements-ssuie-ppu中辅助依赖 | venv内适配；厂商torch/torchvision保留；系统transformers等未卸载 |
| `python -m pip install --no-deps --no-build-isolation -e .` | 初次旧setuptools产生UNKNOWN包，移除本轮该错误安装，补齐本地打包依赖后安装uie-prior-utility成功 |
| `python -m uie_next.cli inspect --config configs/uie_next/protocol.yaml` | 成功；logs/inspect.log、inspect_final.log |
| `python -m uie_next.cli audit-data --config configs/uie_next/protocol.yaml` | 全量审计/完整哈希重验成功；logs/audit_reverify.log |
| `python -m uie_next.cli run --config configs/uie_next/protocol.yaml --resume` | exit2/BLOCKED_BACKBONE；logs/run_resume.log；没有派发正式训练 |
| `python -m uie_next.cli plan-budget --config configs/uie_next/protocol.yaml` | 容量/矩阵计划生成；正式速度和步数尚未实测冻结 |
| `python -m uie_next.cli --help` | 成功；T38另在仓库外目录调用并核对模块路径 |
| `python -c '...local_head_smoke(RUN)...'` | PPU合成13作业各一次更新；logs/ppu_local_heads.log、ppu_local_heads_final.log；不是正式训练/profile |
| `python -c '...recovery(RUN)...'`、`recovery_ppu(RUN)` | 合成连续20与10+新进程10；各CPU/PPU日志和tests/recovery_*回执 |
| `python -c '...runtime_probe.probe(RUN)...'` | 实际PPU kernel数值检查；首次导入依赖失败、补齐transformers后通过，再从安装后的wheel通过；logs/runtime_probe*.log |
| `python -m pytest tests -q --junitxml=runs/ssuie_local_utility_v1_20261007/tests/pytest_delivery.xml` | 以最终XML与logs/tests_delivery.log为准；旧失败日志保留 |
| `python -m compileall -q uie_next` | 成功 |
| `python -m uie_next.cli closeout --config configs/uie_next/protocol.yaml` | 已执行成功；实际输出为报告和5个ZIP，/tmp/ssuie_closeout_cli.log；最终包将按最新审计与测试记录重新生成 |

依赖回执、预算、上游身份、源码快照和包哈希在同名runs目录；`run --resume`目前只实现入场依赖推进，S3–S10完整正式调度仍未完成。没有“训练已经完成”的虚构命令。
# 官方权重接续说明

本文件保留之前缺权重阶段的实际命令历史。最新实际执行、主规范1.1身份和真实调度命令见 `RUN_COMMANDS_OFFICIAL_RESUME.md`，当前科学状态为 `STOP_PRIOR_UNUSED`，不再是BLOCKED_BACKBONE。
