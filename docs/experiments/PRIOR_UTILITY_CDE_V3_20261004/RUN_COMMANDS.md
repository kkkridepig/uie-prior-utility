# V3 运行与查看命令

工作目录 `/mnt/workspace/uie-prior-utility`，新分支 `experiment/prior-utility-cde-v3-20261004`。

2026-10-05：本轮已达到科学停止条件pilot_no_supported_gain，无活动设备任务。日常查看用status；收束CPU复核和CSV导出用`.venv/bin/python tools/cde_v3_closeout.py`。下列launch属于历史恢复入口，当前无需再次启动，不应用剩余预算无条件增加训练。

```bash
# 只读状态，不启动加速器
.venv/bin/python scripts/cde_v3/status.py --json
# 完整记录
cat runs/prior_utility_cde_v3_20261004/budget.json
cat runs/prior_utility_cde_v3_20261004/dispatch_state.json
# 当前调度日志
tail -f runs/prior_utility_cde_v3_20261004/pipeline.log
# 启动后台可恢复调度；重复执行只报告已有进程，不重置预算
PYTHONPATH=. scripts/mpa_python.sh scripts/cde_v3/launch.py --resume
# 测试与不派发计划
PYTHONPATH=. scripts/mpa_python.sh -m pytest tests/cde_v3 tests/mpa
PYTHONPATH=. scripts/mpa_python.sh scripts/cde_v3/dispatch.py --dry-run
```

协议参数与哈希在protocol_frozen.yaml；数据名单在data_roles.json；每个完整调度命令会写入budget.json的events。正在执行的任务会写入active，含PID、截止时间和心跳。

E固定24图、完整开发验证与机制诊断已完成。首次成本预测导致的安全停止记录已保留；经训练图上的确定性channels_last推理验证，完整矩阵的新预测通过原预算闸门。正式调度按BASE_CONT_V3、C_BANK、C_RGB_CONTROL、训练后oracle及后续冻结依赖执行。若预算预测或依赖失败，真实终态会写FINAL_REPORT.md，不把实验未执行写成数值阴性。

推理提速证据见RESUME_PROFILE_REVIEW.md。训练仍用原NCHW配方；预算跨恢复累计。implementation_freeze_resume_before_pilot.json记录正式pilot前的源码和测试身份；改变源码会使后续派发停止，需要审查原因。
