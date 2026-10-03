"""Start the authorized long-running queue with logs and a durable PID receipt."""
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'runs/s2_grouped_v1';out.mkdir(parents=True,exist_ok=True)
    receipt=out/'launcher.json'
    if receipt.exists():
        old=json.loads(receipt.read_text());pid=old['pid']
        proc=Path('/proc')/str(pid)/'cmdline'
        if proc.exists() and b'mpa_diff.cli.stages' in proc.read_bytes():raise SystemExit('Queue already active, PID '+str(pid))
    command=[str(ROOT/'.venv/bin/python'),'-u','-m','mpa_diff.cli.stages','--execute','--profile','grouped','--output','runs/s2_grouped_v1']
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with (out/'queue.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'pid':process.pid,'started_utc':datetime.now(timezone.utc).isoformat(),'command':command,'log':str(out/'queue.log'),'reports':str(ROOT/'docs/experiments'),'note':'Process persists after terminal disconnect; restart after server reboot using this same script to resume checkpoints.'}
    receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
