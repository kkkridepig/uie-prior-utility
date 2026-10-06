# CDE V3 最终状态报告

状态：`pilot_no_supported_gain`。

累计设备时间 11.64806 / 72 h，最终保留12 h；旧V2结果原样保留。

数学/shape/null/FFT/DDIM、CPU/PPU证据、独立RNG与恢复测试分别见对应JSON及日志。测试通过不代表算法有效；闸门只决定研究投资，不代表论文创新。

```json
{
  "status": "pilot_no_supported_gain",
  "method": null,
  "seeds": [
    20261004
  ],
  "models": {
    "20261004": {
      "BASE_CONT_V3": "/mnt/workspace/uie-prior-utility/runs/prior_utility_cde_v3_20261004/pilot/20261004/BASE_CONT_V3/step_05000.pt"
    }
  },
  "checkpoints": [
    "/mnt/workspace/uie-prior-utility/runs/explore_ag_single_seed_v2_20261003/parent_0.pt",
    "/mnt/workspace/uie-prior-utility/runs/prior_utility_cde_v3_20261004/pilot/20261004/BASE_CONT_V3/step_05000.pt"
  ],
  "strong_control": "BASE_CONT_V3",
  "parent_initializations": 1,
  "seed_claim": "one pilot plus two same-protocol fine-tuning repeats; not independent pretraining",
  "threshold_is_not_innovation": true
}
```

E详情见E_DIAGNOSTICS.md和E目录；C见C_BANK_ORACLE.md、pilot/<seed>/oracle.json、gate_result.json及全部逐图记录；D见D_FALLBACK.md。新留出数据身份见HOLDOUT_AUDIT.md，旧test均为legacy_exposed_regression。缺少输出意味着未执行，不填虚构指标。

复现/恢复：`PYTHONPATH=. scripts/mpa_python.sh scripts/cde_v3/pipeline.py --resume`；只读计划：`PYTHONPATH=. scripts/mpa_python.sh scripts/cde_v3/dispatch.py --dry-run`。恢复不重置账本或重复完成ID。权重为delta，必须结合lineage中的原父checkpoint；artifact_manifest.json列出产物与哈希。

## 推理与预算证据

channels_last仅用于推理，训练仍为原NCHW；完整DDIM20 hard-null逐位通过。训练图布局数值核验最大PSNR漂移 0.000855156 dB。E效率表来自原NCHW配置，不与新布局的单点时间混算加速比。旧超预算预测和停止快照均保留。

预测C / 2个重复 / 最终全对照设备小时：8.531 / 17.062 / 9.990（含1.25倍余量）；这些是预测，实际成本以budget.json为准。


## 实际训练与方向状态

```json
[
  {
    "finetune_seed": "20261004",
    "branch": "BASE_CONT_V3",
    "step": 5000,
    "target": 5000,
    "status": "complete"
  },
  {
    "finetune_seed": "20261004",
    "branch": "C_ALL_ONLY",
    "step": 5000,
    "target": 5000,
    "status": "complete"
  },
  {
    "finetune_seed": "20261004",
    "branch": "C_BANK",
    "step": 5000,
    "target": 5000,
    "status": "complete"
  },
  {
    "finetune_seed": "20261004",
    "branch": "C_RGB_CONTROL",
    "step": 5000,
    "target": 5000,
    "status": "complete"
  },
  {
    "finetune_seed": "20261004",
    "branch": "D_PHASE_STD",
    "step": 5000,
    "target": 5000,
    "status": "complete"
  },
  {
    "finetune_seed": "20261004",
    "branch": "D_SOBEL_STD",
    "step": 5000,
    "target": 5000,
    "status": "complete"
  },
  {
    "finetune_seed": "20261004",
    "branch": "D_SOFTPHASE_STD",
    "step": 5000,
    "target": 5000,
    "status": "complete"
  },
  {
    "finetune_seed": "20261004",
    "branch": "D_noise",
    "step": null,
    "target": null,
    "status": "complete"
  },
  {
    "finetune_seed": "20261004",
    "branch": "selector_evaluation",
    "step": null,
    "target": null,
    "status": "complete"
  }
]
```

profiles/目录的100步及恢复验证属于工程验收，不算正式5000步pilot或算法正证据。完整科学比较见各seed的oracle.json、gate_result.json、data_matched_gate.json、d_gate_result.json（仅实际存在的文件有效）。

D触发记录：D_trigger.json存在，原因见该文件；是否跑完以实际D控制记录为准。

逐图导出：E/*/per_image.jsonl、pilot/<seed>/<branch>/eval/per_image.jsonl、selector_evaluation/per_image.jsonl、final/<seed>/<role>/per_image.jsonl。缺失者为未执行，不能宣称三seed或新留出确认。最终权重/指标/图像哈希见artifact_manifest.json，统一面板与失败例名单仅在实际final导出完成后存在。
