# Reproduce and resume

From /mnt/workspace/uie-prior-utility, use the existing vendor environment .venv.

```bash
.venv/bin/python -B -m uie_next.cli verify-backbone --config configs/uie_next/protocol.yaml
.venv/bin/python -B -m uie_next.cli smoke --config configs/uie_next/protocol.yaml
.venv/bin/python -B -m uie_next.cli plan-budget --config configs/uie_next/protocol.yaml
.venv/bin/python -B -m uie_next.cli run --config configs/uie_next/protocol.yaml --resume
.venv/bin/python -B -m uie_next.cli status --config configs/uie_next/protocol.yaml
.venv/bin/python -B -m uie_next.cli closeout --config configs/uie_next/protocol.yaml
```
run uses a per-run dispatcher lock and device budget lease, atomically saves every 250 updates and at exit. It restores optimizer/scheduler and all named RNG streams; terminal scientific stops do not dispatch again.
Dependencies are hash-linked: official backbone, policy, candidate, roles, scales and source snapshot. Checkpoint file receipts permit integrity verification. No old V3 run is restarted.
Independent backup is not verified; same-server ZIPs do not survive loss of the server disk.
