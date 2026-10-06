# V3 实时状态

状态：pilot_no_supported_gain；已计设备时间 11.64806 / 72 h，最终预留12 h。

活动：None

完成任务：['P0_PPU_contracts', 'E_subset', 'E_full', 'E_legacy', 'E_interior_sensitivity', 'P0_metrics_and_capacity', 'P0_statistics', 'P0_profile_resume', 'P0_inference_cost_diagnostic_v1', 'P0_inference_layout_validation_v1', 'P0_selector_profile_resume_v1', '20261004_BASE_CONT_V3_train', '20261004_BASE_CONT_V3_eval', '20261004_C_BANK_train', '20261004_C_BANK_eval', '20261004_C_RGB_CONTROL_train', '20261004_C_RGB_CONTROL_eval', '20261004_C_ALL_ONLY_train', '20261004_C_ALL_ONLY_eval', '20261004_labels_route_fit', '20261004_labels_route_cal', '20261004_gate_UTILITY', '20261004_gate_WINNER_CE', '20261004_gate_SHUFFLED', '20261004_selector_eval', 'D_preflight_profile', '20261004_D_SOBEL_STD_train', '20261004_D_SOBEL_STD_eval', '20261004_D_PHASE_STD_train', '20261004_D_PHASE_STD_eval', '20261004_D_SOFTPHASE_STD_train', '20261004_D_SOFTPHASE_STD_eval', '20261004_D_noise', 'FINAL_frozen_test']

当前阻塞原因：C完整pilot为utility_not_learned；一次D备选也未通过。科学停止，无额外seed/新训练。最终只有PARENT/BASE_CONT基线评估。

历史失败/安全分段停止（resolved_by_completion表示后续已完成）：{'E_subset': {'id': 'E_subset', 'package': 'E', 'elapsed_seconds': 7087.820166110992, 'exit_code': 75, 'command': ['.venv/bin/python', 'scripts/cde_v3/diagnose_e.py', '--stage', 'subset'], 'log': '/mnt/workspace/uie-prior-utility/runs/prior_utility_cde_v3_20261004/logs/E_subset_attempt1.log', 'resolved_by_completion': True}}

此页是状态，不是方法有效性证明。
