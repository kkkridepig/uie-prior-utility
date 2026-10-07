# 继承预算与成本计划

```json
{
  "status": "PROFILED",
  "D0_seconds_per_image_including_six_views_metrics": 0.24842009450003388,
  "D0_predicted_seconds": 3058.299783389917,
  "full_matrix_training_and_validation_predicted_seconds": 10905.864640387772,
  "cache_predicted_seconds": 373.97161026035104,
  "rescue_predicted_seconds": 6418.000162734785,
  "protected_final_seconds": 15843.736787023161,
  "used_device_seconds_at_plan": 1631.0984604358673,
  "safety_factor": 1.3,
  "all_11_methods_3000_updates": true,
  "complete_without_rescue_predicted_seconds": 31812.97128149707,
  "maximum_device_seconds": 57600,
  "all_profile_is_temporary_and_counted": true,
  "CPU_bootstrap_not_device_runtime": true
}
```

所有估计来自PPU真实前向／训练／IO／指标profiling，乘1.3安全系数。完整11方法固定3000更新，不删控制。最终保护取12600秒与实际预测最大值；当前约15843.74秒（4.4小时）。旧60秒保守预算仍保持估计身份，不变成实测。CPU bootstrap与压缩另记CPU墙钟；同盘压缩不是独立备份。
