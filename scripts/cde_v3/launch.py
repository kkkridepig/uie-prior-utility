"""Start or inspect one persistent local pipeline without resetting its ledger."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *


def alive(record):
    path = Path('/proc') / str(record.get('pid', 0))
    if not path.exists():
        return False
    try:
        return ('scripts/cde_v3/pipeline.py' in (path/'cmdline').read_bytes().decode().replace('\0', ' ')
                and (path/'stat').read_text().split()[21] == record.get('process_start_ticks'))
    except (OSError, UnicodeError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume', action='store_true', required=True)
    p.parse_args()
    with (RUN/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = RUN/'pipeline_process.json'
        old = read(target) if target.exists() else {}
        if alive(old):
            print('Already running:', old['pid'])
            return
        # The pipeline's own lock also protects against direct invocations.
        with (RUN/'pipeline.lock').open('a') as plock:
            fcntl.flock(plock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        b = read(RUN/'budget.json')
        assert not b.get('active'), 'An existing device job must be audited before launch'
        assert read(RUN/'cost_prediction.json')['dispatch_allowed']
        env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
        command = [str(ROOT/'.venv/bin/python'), 'scripts/cde_v3/pipeline.py', '--resume']
        log = RUN/'pipeline.log'
        with log.open('a') as output:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        record = {'pid': child.pid, 'command': command, 'log': str(log),
                  'started_unix': time.time(), 'process_start_ticks': (Path('/proc')/str(child.pid)/'stat').read_text().split()[21],
                  'source_sha256': code_hash(), 'ledger_preserved': True,
                  'charged_hours_at_start': b['charged_seconds']/3600,
                  'cost_prediction_sha256': sha(RUN/'cost_prediction.json')}
        if old:
            append(RUN/'pipeline_process_history.jsonl', old)
        write(target, record)
        print(record)


if __name__ == '__main__':
    main()
