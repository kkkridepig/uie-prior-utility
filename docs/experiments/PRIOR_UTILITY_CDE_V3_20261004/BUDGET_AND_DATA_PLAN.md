# 预算与数据计划

新72设备小时上限，最终保留12小时，原阶段上限不变。NCHW旧预测保留在cost_prediction_v1_nchw.json，因预测超限停过一次；正式pilot尚未启动时完成训练集上的channels_last数值/计时核验。训练配方、DDIM20、图数、噪声键、seed与所有对照未变。

新预测按各分支真实PPU p95、每次输出额外0.10秒I/O/冷先验余量、全矩阵1.25倍安全系数估算；最终额外预留0.5小时导出。不假定删除或缓存任何命名对照。所有实际设备时间继续累计到同一账本。

```json
{
  "profiles": {
    "BASE_CONT_V3": {
      "status": "complete",
      "step": 100,
      "target": 100,
      "elapsed_seconds": 75.74229741096497,
      "median_step_seconds": 0.7463824576698244,
      "p95_step_seconds": 0.901199839473702,
      "probe_output_rms_change": 0.03331134095788002,
      "mode_counts": {
        "null": 100
      },
      "last": {
        "step": 100,
        "loss": 0.021594170480966568,
        "seconds": 0.8286206899210811,
        "lr": 9.821800000000001e-06,
        "mode": "null"
      },
      "checkpoint_sha256": "f9b5a0ada0994c191ab0941ccb96d17b101a7bd309fe0885090ddf01d096313f"
    },
    "C_BANK": {
      "status": "complete",
      "step": 100,
      "target": 100,
      "elapsed_seconds": 68.99142122268677,
      "median_step_seconds": 0.6717222512234002,
      "p95_step_seconds": 0.8329734912840648,
      "probe_output_rms_change": 0.0014268204104155302,
      "mode_counts": {
        "highfreq": 25,
        "histogram": 25,
        "physical": 25,
        "all": 25
      },
      "last": {
        "step": 100,
        "loss": 0.02355702966451645,
        "seconds": 0.763749263714999,
        "lr": 9.821800000000001e-05,
        "mode": "physical"
      },
      "checkpoint_sha256": "10219a4c7574b3fa47b244850c0d57e89e14e7a7e2a24d436976f435d9d559a5"
    },
    "C_RGB_CONTROL": {
      "status": "complete",
      "step": 100,
      "target": 100,
      "elapsed_seconds": 68.3285710811615,
      "median_step_seconds": 0.6546600700821728,
      "p95_step_seconds": 0.8584710034541786,
      "probe_output_rms_change": 0.001421610708348453,
      "mode_counts": {
        "highfreq": 25,
        "histogram": 25,
        "physical": 25,
        "all": 25
      },
      "last": {
        "step": 100,
        "loss": 0.023579753935337067,
        "seconds": 0.8727285000495613,
        "lr": 9.821800000000001e-05,
        "mode": "physical"
      },
      "checkpoint_sha256": "51f90fe43ddf2fe8f0e4b5c340634932151721dfeb2b870cac49413d9dafb15e"
    },
    "C_ALL_ONLY": {
      "status": "complete",
      "step": 100,
      "target": 100,
      "elapsed_seconds": 69.30327367782593,
      "median_step_seconds": 0.6696882443502545,
      "p95_step_seconds": 0.8470079476945102,
      "probe_output_rms_change": 0.004371143411844969,
      "mode_counts": {
        "all": 100
      },
      "last": {
        "step": 100,
        "loss": 0.02348214201629162,
        "seconds": 0.7442007381469011,
        "lr": 9.821800000000001e-05,
        "mode": "all"
      },
      "checkpoint_sha256": "873c122e3c62459e0d0f6aa5d29145b21fef801c3273245d3af48a5410f36b74"
    }
  },
  "training_seconds_5000": {
    "BASE_CONT_V3": 3731.912288349122,
    "C_BANK": 3358.611256117001,
    "C_RGB_CONTROL": 3273.3003504108638,
    "C_ALL_ONLY": 3348.4412217512727
  },
  "evaluation_seconds_per_image_conservative": 0.3888058299431577,
  "evaluation_profile_seconds": [
    1.0927460049279034,
    1.083793830126524,
    1.173178629949689,
    1.198461547959596,
    1.3257438177242875
  ],
  "selector_label_denoiser_calls_per_seed": 42400,
  "initial_C_package_hours": 3.0809448134369837,
  "full_C_pilot_hours_with_controls": 6.82475340297628,
  "two_repeats_hours": 13.64950680595256,
  "safety_multiplier": 1.25,
  "C_pilot_predicted_hours_with_safety": 8.53094175372035,
  "repeats_predicted_hours_with_safety": 17.0618835074407,
  "remaining_steps_expression": "sum(5000 * measured_seconds_per_step) + modes*noise_keys*images*measured_inference + gates",
  "final_reserved_hours": 12,
  "required_disk_gib_estimate": 2.5,
  "dispatch_allowed": true,
  "prediction_note": "NCHW training unchanged; verified channels-last inference; per-branch PPU p95 plus 0.10s/output I/O/cold-prior headroom; 1.25x margin; full controls",
  "final_3seed_full_controls_hours_with_safety": 9.989786983268278,
  "prediction_version": 2,
  "stage_seconds_per_seed": {
    "training_all_five_enhancers": 17444.17740497738,
    "development_all_five_enhancers": 924.1868654060411,
    "route_fit_and_cal_labels": 824.2683594794944,
    "selector_all_clean_and_corruption_groups": 4776.479620851693,
    "three_selector_trainings": 300.0,
    "startup_and_export_headroom": 300
  },
  "inference_seconds_by_branch_with_IO_headroom": {
    "BASE_CONT_V3": 0.3557293913559988,
    "C_BANK": 0.3888058299431577,
    "C_RGB_CONTROL": 0.3653602629434317,
    "C_ALL_ONLY": 0.36445167791098354,
    "PARENT": 0.3223385628312826
  },
  "IO_cold_prior_headroom_seconds_per_output": 0.1,
  "final_image_count": 915,
  "final_named_models_per_seed": 9,
  "selector_step_seconds_bound": 0.05,
  "previous_prediction_sha256": "7e01434829c9831a23ff02e9f8647ef1568e0e052f1bfce587276fb2d5f1874d",
  "layout_validation_sha256": "890f97cdcdd7be44e971848f39873734b0c37454061343f84e1760208f6027cc",
  "current_prediction_code_sha256": "f834914a8f57b8f46e608791b008171f24ace6e6116745bde78626350d84913e"
}
```
