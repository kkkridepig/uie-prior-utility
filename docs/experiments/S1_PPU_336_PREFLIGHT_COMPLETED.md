# S1_PPU_336_PREFLIGHT_COMPLETED

336×336、正式网络宽度、Adam/global batch4，110步；使用作者示例小集测吞吐，不计S2训练种子。

原始产物：`runs/s1/ppu_336_preflight`。本文件据已有日志补建，不重新执行或改写历史结果。

```json
{
  "logged_steps": 110,
  "first": {
    "step": 1,
    "loss": 1.0065350234508514,
    "pixel_mse": 0.32228248193860054,
    "lr": 0.0001,
    "prediction_outside_fraction": 0.6572952419519424,
    "seconds": 12.656483432045206
  },
  "last": {
    "step": 110,
    "loss": 0.23217758908867836,
    "pixel_mse": 0.022500003455206752,
    "lr": 0.0001,
    "prediction_outside_fraction": 0.004052402267234356,
    "seconds": 0.640569661045447
  },
  "timing.json": {
    "stable_steps": 100,
    "mean_step_seconds": 0.6508924246788956,
    "estimated_400000_hours": 72.3213805198773,
    "excludes": "validation/depth cache warmup/evaluation/other seeds"
  }
}
```

分析：训练误差下降仅说明优化链路能够学习；性能数字不能直接当作正式全数据集质量结果。时间估算不含长验证/最终测试等开销。
