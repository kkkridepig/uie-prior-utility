"""Read-only status; does not initialize a PPU context or alter the ledger."""
import argparse,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/prior_utility_cde_v3_20261004'
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--json',action='store_true'); a=p.parse_args()
    state=json.loads((RUN/'dispatch_state.json').read_text()); budget=json.loads((RUN/'budget.json').read_text())
    progress={}
    for path in RUN.rglob('progress.json'):
        progress[str(path.relative_to(RUN))]=json.loads(path.read_text())
    payload={'status':state['status'],'stage':state.get('stage'),'device_hours':budget['charged_seconds']/3600,'limit_hours':budget['limit_hours'],'final_reserved_hours':budget['final_reserve_hours'],'active':budget.get('active'),'completed_tasks':list(state['completed']),'failures':state.get('failures',{}),'progress':progress}
    print(json.dumps(payload,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
