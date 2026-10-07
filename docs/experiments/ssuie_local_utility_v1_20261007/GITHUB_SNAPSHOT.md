# GitHub source snapshot

This snapshot includes the recovered V3 compatibility fixes, the SS-UIE local
utility implementation, protocol configuration, tests, and experiment reports.
The current experiment ended at `STOP_PRIOR_UNUSED`; this source publication
does not establish an effective algorithm or launch further training.

## Protocol dependency

The frozen source reads its scientific contract from a file outside the
repository. `PROTOCOL_GUIDE_V1_1.md` preserves that file byte for byte.
Its SHA256 is
`e4cc66157d67b0a4068601c5afd6c859411c795d900e423932ef22130c177ad7`.
To restore the expected layout after cloning, run from the repository root:

```bash
mkdir -p ../TEMP-FILE-STATION
cp -n docs/experiments/ssuie_local_utility_v1_20261007/PROTOCOL_GUIDE_V1_1.md \
  ../TEMP-FILE-STATION/SSUIE_C局部效用学习_本轮完整执行指南_20261008.md
```

The copy command preserves any existing uploaded guide. Verify its SHA256
against the value above before using the frozen configuration.

## Separate dependencies and evidence

Datasets, model weights, metric weights, upstream source checkouts, compiled
PPU extensions, virtual environments, generated images, and `runs/` evidence
are excluded from Git. A clone alone is not a full experiment recovery.
Use the separately generated code/protocol, review, visual, weight/recovery,
PPU wheel, and metric-weight archives described in `FINAL_REPORT.md`.
Same-server archives do not constitute a verified independent backup.

The backbone is the **SS-UIE official public simplified implementation**,
at upstream commit `88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df`.
Its official weight SHA256 is
`977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99`.
Preserve the vendor PyTorch/torchvision environment and use the documented
PPU dependencies; do not replace it with NVIDIA CUDA wheels.

See `REPRODUCE_AND_RESUME.md` and `RUN_COMMANDS_OFFICIAL_RESUME.md` for the
actual interfaces and prior execution records. Restoring the terminal run
does not authorize a new recipe, extra seeds, or additional training.
