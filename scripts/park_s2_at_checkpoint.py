"""Park the known S2 queue at its next atomic, recoverable checkpoint."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


IN_MOVED_TO = 0x80


def process_identity(pid):
    stat = Path('/proc/{}/stat'.format(pid)).read_text()
    return stat.rsplit(')', 1)[1].split()[19]


def audit_checkpoint(python, checkpoint, training_log, source_root):
    code = '''import json, math, sys, torch
from mpa_diff.engine.checkpoint import architecture_hash, source_hash
from mpa_diff.utils.io import sha256
p,log=sys.argv[1:]
s=torch.load(p,map_location="cpu")
required=("model","optimizer","scheduler","rng","data_sampler","step","config","architecture_hash","source_sha256","manifest_sha256")
assert all(k in s for k in required)
assert s["architecture_hash"]==architecture_hash(s["config"])
assert s["source_sha256"]==source_hash()
assert s["manifest_sha256"]==sha256(s["config"]["data"]["manifest"])
assert all(torch.isfinite(v).all().item() for v in s["model"].values() if torch.is_tensor(v))
assert s["optimizer"]["state"] and s["scheduler"]
assert set(("python","numpy","torch","cuda")) <= set(s["rng"])
assert len(s["data_sampler"]["order"])==702
with open(log,"rb") as f:
    f.seek(0,2)
    n=f.tell()
    f.seek(max(0,n-2048))
    tail=f.read().splitlines()
logged=json.loads(tail[-1])["step"]
assert logged==s["step"], (logged,s["step"])
print(json.dumps({"step":s["step"],"sha256":sha256(p),"logged_step":logged,"source_sha256":s["source_sha256"],"sampler_cursor":s["data_sampler"]["cursor"]}))'''
    result = subprocess.run([python, '-c', code, str(checkpoint), str(training_log)],
                            cwd=str(source_root), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--log', type=Path, required=True)
    ap.add_argument('--python', required=True)
    ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('--expected-start', required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    if process_identity(args.pid) != args.expected_start:
        raise SystemExit('PID identity changed; refusing to signal')
    if 'mpa_diff.cli.stages --execute' not in Path('/proc/{}/cmdline'.format(args.pid)).read_bytes().replace(b'\0', b' ').decode():
        raise SystemExit('PID is not the known S2 queue')
    old_inode = checkpoint.stat().st_ino
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    wd = libc.inotify_add_watch(fd, os.fsencode(checkpoint.parent), IN_MOVED_TO)
    if wd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_add_watch')
    print('Armed for {} inode {} PID {}'.format(checkpoint, old_inode, args.pid), flush=True)
    while True:
        select.select([fd], [], [])
        event = os.read(fd, 4096)
        if b'last.pt\0' not in event or checkpoint.stat().st_ino == old_inode:
            continue
        if process_identity(args.pid) != args.expected_start:
            raise SystemExit('PID identity changed at checkpoint')
        os.kill(args.pid, signal.SIGSTOP)
        stopped_at = time.time()
        try:
            info = audit_checkpoint(args.python, checkpoint, args.log, root)
            info.update(pid=args.pid, stopped_unix=stopped_at, verified_unix=time.time(),
                        previous_inode=old_inode, checkpoint=str(checkpoint), action='SIGTERM_after_verified_checkpoint')
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            temp = args.receipt.with_suffix('.tmp')
            temp.write_text(json.dumps(info, indent=2) + '\n')
            temp.replace(args.receipt)
            os.kill(args.pid, signal.SIGTERM)
            os.kill(args.pid, signal.SIGCONT)
            print(json.dumps(info), flush=True)
            return
        except Exception as exc:
            print('Checkpoint not accepted; continuing queue: {}'.format(exc), file=sys.stderr, flush=True)
            old_inode = checkpoint.stat().st_ino
            os.kill(args.pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
