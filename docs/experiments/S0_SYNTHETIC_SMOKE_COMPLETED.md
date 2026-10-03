# S0_SYNTHETIC_SMOKE_COMPLETED

合成fixture、测试深度替身、无VGG；只验证实现可学习，不是benchmark。

原始产物：`runs/s0/synthetic_smoke`。本文件据已有日志补建，不重新执行或改写历史结果。

```json
{
  "logged_steps": 150,
  "first": {
    "step": 1,
    "loss": 0.503681480884552,
    "pixel_mse": 0.503681480884552,
    "lr": 0.001,
    "prediction_outside_fraction": 0.4021809995174408,
    "seconds": 0.12533535691909492
  },
  "last": {
    "step": 150,
    "loss": 0.003870380111038685,
    "pixel_mse": 0.003870380111038685,
    "lr": 0.0001,
    "prediction_outside_fraction": 0.0013020833721384406,
    "seconds": 0.04310374008491635
  },
  "timing.json": {
    "stable_steps": 100,
    "mean_step_seconds": 0.049059499339200556,
    "estimated_400000_hours": 5.451055482133395,
    "excludes": "validation/depth cache warmup/evaluation/other seeds"
  }
}
```

分析：训练误差下降仅说明优化链路能够学习；性能数字不能直接当作正式全数据集质量结果。时间估算不含长验证/最终测试等开销。
