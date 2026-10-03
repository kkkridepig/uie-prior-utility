# S1_REAL_PRIOR_SMALLFIT_COMPLETED

12张作者示例训练、2验证、2测试，64×64，真实DA-V2/VGG19，200步；工程小集拟合，不代表泛化。

原始产物：`runs/s1/real_prior_smallfit`。本文件据已有日志补建，不重新执行或改写历史结果。

```json
{
  "logged_steps": 200,
  "first": {
    "step": 1,
    "loss": 1.4164185523986816,
    "pixel_mse": 0.3188854642212391,
    "lr": 0.0001,
    "prediction_outside_fraction": 0.6578979641199112,
    "seconds": 7.016196164069697
  },
  "last": {
    "step": 200,
    "loss": 0.33660484105348587,
    "pixel_mse": 0.020163299748674035,
    "lr": 1e-06,
    "prediction_outside_fraction": 0.0036010742969665444,
    "seconds": 0.4627995060291141
  },
  "test/summary.json": {
    "count": 2,
    "mean_psnr": 16.388325624816606,
    "mean_ssim": 0.7708137308195966,
    "perfect_reconstructions": 0,
    "missing_metrics": {
      "lpips": "pretrained metric not provisioned",
      "uciqe": "not yet independently audited",
      "uiqm": "not yet independently audited",
      "uranker": "weights unavailable"
    },
    "test_fixture": false,
    "protocol": "RGB float clipped once; SSIM valid 11x11 sigma1.5; PNG metrics separate"
  },
  "timing.json": {
    "stable_steps": 100,
    "mean_step_seconds": 0.462970902195666,
    "estimated_400000_hours": 51.441211355073996,
    "excludes": "validation/depth cache warmup/evaluation/other seeds"
  }
}
```

分析：训练误差下降仅说明优化链路能够学习；性能数字不能直接当作正式全数据集质量结果。时间估算不含长验证/最终测试等开销。
