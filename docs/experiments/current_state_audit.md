# Single-seed exploration: current state audit

Audit time: 2026-10-02 17:43-17:49 UTC. The 2026-09-28 ETA document is historical only.

## Live queue and protection

- Project: `/mnt/workspace/uie-prior-utility`; active implementation: `mpa_diff/`.
- Live PID 1423731, parent PID 1, command `.venv/bin/python -u -m mpa_diff.cli.stages --execute --profile grouped --output runs/s2_grouped_v1`; working directory is the project root. The `queue.lock` is held by this queue.
- `runs/s2_grouped_v1/queue_status.json` reports UIEB seed 20260927, completed runs 0/6. Five later jobs are queued, not launched.
- UIEB train log has completed optimizer update 225000/400000 at 2026-10-02 17:33 UTC. Validation for 225000 is in progress; no complete `validation_0225000` directory existed at 17:49 UTC. Last completed 220000 validation: 91 samples, mean PSNR 24.700184545589067, SSIM 0.9298928353747178.
- A PID-start-time-checked monitor (`scripts/park_s2_at_checkpoint.py`) waits for the atomic replacement of the next `last.pt`, pauses only that PID, verifies the new complete checkpoint in a separate CPU process and checks the logged step matches. Only then does it send SIGTERM/SIGCONT to that PID to prevent dispatch of the remaining five jobs. Receipt: `runs/explore_ag_single_seed_v2_20261003/old_queue_park_receipt.json`. No source in the frozen `mpa_diff` package was modified while this queue runs.

## Recoverable checkpoints

- `runs/s2_grouped_v1/uieb/seed_20260927/last.pt`: step 220000, SHA-256 `bf749975e47f15daee293cb61ca86185b5730355b3969b2e1f453d01542fe075`.
- `runs/s2_grouped_v1/uieb/seed_20260927/best.pt`: step 85000, SHA-256 `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`.
- Both loaded in a separate CPU process; architecture, source and manifest hashes matched; all 465 model tensors were finite. `last.pt` includes 224 optimizer states, scheduler, model, global step, Python/NumPy/torch/device RNG, shuffled order of 702 training samples and cursor 394. Its source hash was `ea6bbef128d3b864b214cc26638f8a811e75575b41a96559a350c4657733dac9`. AMP scaler is null because the run uses float32.
- No new parent model has been selected. Parent selection will use only source validation with the frozen v2 sampler. The historical best is not automatically the common parent.

## Environment, data and budget

- Alibaba DSW PPU-ZW810E is exposed through `torch.cuda`; project `.venv` uses vendor PyTorch 2.0.0a0+nv2303, Python 3.8.10 and NumPy 1.23.5. No dependency installation or PyTorch replacement was performed.
- Frozen grouped UIEB manifest: 702 train / 91 validation / 97 test. Frozen grouped LSUI manifest: 3423 train / 429 validation / 427 test. Data identities and external-source overlap need further audit before final Benchmark.
- The original training protocol is RGB x0, 336x336, effective batch 4, DDPM1000, 5000-step validation/checkpoint, 400000 planned steps. The new protocol prohibits launching another 400000-step run automatically.
- `/mnt/workspace` had about 18 GiB free at audit time. New budget begins 2026-10-02 17:43 UTC, includes legacy PPU occupation after that point, caps cumulative single-device occupation at 168 hours and reserves 16 hours for final Benchmark. See `runs/explore_ag_single_seed_v2_20261003/budget.json`.
- No source test, LSUI test, U45 or other domain test will be used for sampler, parent or branch selection.

## Open checks

Follow-up, 2026-10-02 19:03 UTC: the legacy UIEB queue parked at step 225000 and exited after independent verification of the complete `last.pt` checkpoint (SHA-256 `113f3c964a1151b8d4bbe642e1f24b46527b3cf403998202f4c65f99a86f7c0f`). Its five later 400000-step jobs have not started. The preservation receipt is `runs/explore_ag_single_seed_v2_20261003/old_queue_park_receipt.json`; the prior complete 220000-step checkpoint is retained separately. The persistent single-device dispatcher is PID 2159773 in tmux session `uie_ag_v2`, with cumulative budget in `runs/explore_ag_single_seed_v2_20261003/budget.json`. It is running E0 full source validation, with the last observed progress at 56/91 images. The September 28 snapshot and the original queue status file are not live progress indicators. No parent has yet been selected and no exploration training branch has started as of this follow-up.

## 2026-10-02 18:49 UTC update: safe handoff completed

- The monitor observed atomic replacement of `last.pt` and issued SIGTERM/SIGCONT only to the identity-checked legacy PID after an independent CPU load. Receipt: `runs/explore_ag_single_seed_v2_20261003/old_queue_park_receipt.json`. The old PID and monitor have exited; its `queue.lock` is free. `queue_status.json` still says `running` because the signal interrupted the old queue before its status writer; the process and lock are authoritative here. No later seed or LSUI job started.
- The verified recoverable checkpoint is step **225000**, SHA-256 `113f3c964a1151b8d4bbe642e1f24b46527b3cf403998202f4c65f99a86f7c0f`. A second independent process loaded it and checked the model finite, nonempty optimizer state, scheduler, Python/NumPy/torch/device RNG, 702-item sampler order, cursor 36, source and manifest hashes, and matching final train-log step. The complete 220000-step predecessor was preserved before replacement at `runs/explore_ag_single_seed_v2_20261003/legacy_step_220000.pt`, SHA-256 `bf749975e47f15daee293cb61ca86185b5730355b3969b2e1f453d01542fe075`.
- The new dispatcher started in `tmux` session `uie_ag_v2` and recorded 3693.345 seconds of legacy PPU time since the fixed budget start, 2026-10-02 17:43 UTC. Its first task is E0 on UIEB validation; no new 400000-step run was launched. `budget.json` and `dispatch_state.json` under the new run root are the live machine-readable record.
- The source test is not fully scene-independent: validation `UIEB/61_img_` and test `UIEB/356_img_` show the same diver and reef in differently sized/cropped images. Frozen manifests remain unchanged; the final report must give a sensitivity result excluding the flagged test image and avoid claiming a completely unseen-scene UIEB test. The other dHash candidates inspected so far were false positives. Details: `data_overlap_audit.json` in the new run root.
