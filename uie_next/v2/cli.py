"""Actual V2 state machine; never calls the legacy V1 run/closeout entrypoints."""
import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path
from ..records import ROOT,read,write,sha,digest,append
from ..budget import BudgetStop
from .context import State,OLD,STAGES
from .diagnostics import Diagnostics
from .acceptance import acceptance
from .selection import choose_producer,standalone,eligibility,rescue_choice


class Stop(RuntimeError):
    def __init__(self,status,reason):super().__init__(reason);self.status=status


def run_d0(s,d):
    d.load();d.verify_reuse();probe=d.probe();inventory=read(s.run/'diagnostics/checkpoint_inventory.json')
    if 'D0_MODEL_DIAGNOSTICS' not in s.state['completed']:
        for item in inventory:
            for role,rows,operation in [('model_fit_probe',probe,'probe_diagnostic'),('model_val',d.data.role('model_val'),'model_diagnostic')]:
                d.scan(item,role,rows,operation,sensitivity=item['head']=='B1' and role=='model_val')
        summaries=d.export();s.complete('D0_MODEL_DIAGNOSTICS',{'summaries':sha(s.run/'diagnostics/checkpoint_summary.json')})
    summaries=read(s.run/'diagnostics/checkpoint_summary.json')
    if 'MODEL_PRODUCER_FREEZE' not in s.state['completed']:
        mv=[r for r in summaries if r['role']=='model_val'];selection=choose_producer([r for r in mv if r['head']=='B1'])
        write(s.run/'selection/allowed_checkpoint_pool.json',{'standalone':inventory,'producer':[r for r in inventory if r['head']=='B1' and r['training_step']>0]})
        write(s.run/'selection/standalone_selection.json',{h:standalone([r for r in mv if r['head']==h]) for h in ['B1','B3']})
        selection['B3_oracle_diagnostic_best']=choose_producer([r for r in mv if r['head']=='B3'],False)
        selection.update(protocol_sha256=s.ctx.protocol_sha256,source_snapshot=sha(s.run/'source_snapshot.json'),scores_sha256=sha(s.run/'diagnostics/checkpoint_summary.json'))
        write(s.run/'selection/producer_selection_before_utility_val.json',selection)
        if not selection['selected']:
            choice=rescue_choice(next(r for r in mv if r['head']=='B1' and r['training_step']==4000))
            write(s.run/'selection/rescue_choice.json',choice)
        s.complete('MODEL_PRODUCER_FREEZE',{'selection':sha(s.run/'selection/producer_selection_before_utility_val.json')})
    if 'D0_UTILITY_DIAGNOSTICS' not in s.state['completed']:
        for item in inventory:d.scan(item,'utility_val',d.data.role('utility_val'),'candidate_diagnostic',sensitivity=item['head']=='B1')
        d.export();s.complete('D0_UTILITY_DIAGNOSTICS',{'summaries':sha(s.run/'diagnostics/checkpoint_summary.json')})
    selection=read(s.run/'selection/producer_selection_before_utility_val.json')
    if selection['selected']:
        cid=selection['selected']['checkpoint_id'];rows=read(s.run/'diagnostics/checkpoint_summary.json')
        transfer=next(r for r in rows if r['checkpoint_id']==cid and r['role']=='utility_val');q=eligibility(transfer)
        write(s.run/'selection/producer_transfer_gate.json',{'selected_checkpoint_id':cid,'qualification':q,'scores':transfer,
              'passed':q!='NO_QUALIFIED_PRODUCER_ON_MODEL_VAL','selection_changed_after_utility_val':False})
        if q=='NO_QUALIFIED_PRODUCER_ON_MODEL_VAL':raise Stop('STOP_PRODUCER_TRANSFER_GATE','唯一model_val producer未通过utility_val资格，不改选第二名、不救援。')
        write(s.run/'selection/producer_freeze.json',{'producer':selection['selected'],'transfer':transfer,'protocol':s.ctx.protocol_sha256,
              'selection_hash':sha(s.run/'selection/producer_selection_before_utility_val.json'),'role_sha256':sha(s.run/'roles.jsonl')})
        s.complete('PRODUCER_FROZEN',{'freeze':sha(s.run/'selection/producer_freeze.json')})
    return bool(selection['selected'])


def execute(s):
    lock=(s.run/'.pipeline.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('V2 dispatcher already active')
    append(s.run/'commands.jsonl',{'argv':sys.argv,'pid':os.getpid(),'cwd':os.getcwd()})
    try:
        if s.state['scientific_status'].startswith(('STOP_','INCONCLUSIVE_','CONFIRMATION_','QUALITY_ONLY_')):
            from .delivery import deliver
            return deliver(s)
        s.audit();d=Diagnostics(s);acceptance(d)
        from .supplemental_acceptance import accept
        accept(s,d)
        # Snapshot is the D0 code identity, independent of subsequent NEW files.
        if not (s.run/'source_snapshot.json').exists():s.source_snapshot()
        else:
            snapshot=read(s.run/'source_snapshot.json')
            if any(sha(ROOT/p)!=h for p,h in snapshot.items()):raise Stop('BLOCKED_ENGINEERING','D0冻结源码改变，需要记录影响与重生成受影响配对结果。')
        s.live(scientific_status='RUNNING')
        from .backups import stage_backup
        stage_backup(s,'ENGINEERING_ACCEPTANCE')
        from .training import profile_plan
        profile_plan(s,d)
        exists=run_d0(s,d)
        if not exists:
            from .training import rescue
            rescue(s,d)
        if 'PRODUCER_FROZEN' in s.state['completed']:
            from .training import utility_matrix
            utility_matrix(s,d)
            from .evaluation import evaluate_all
            evaluate_all(s,d)
    except Stop as exc:s.live(scientific_status=exc.status,stop_reason=str(exc))
    except BudgetStop as exc:s.live(scientific_status='INCONCLUSIVE_BUDGET',stop_reason=str(exc))
    except BaseException as exc:
        s.live(scientific_status='BLOCKED_ENGINEERING',stop_reason=type(exc).__name__+': '+str(exc))
        s.ctx.text(s.run/'logs/last_exception.log',traceback.format_exc())
        append(s.run/'events.jsonl',{'event':'engineering_failure','error':str(exc)})
    finally:lock.close()
    from .delivery import deliver
    return deliver(s)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','acceptance','run','status','closeout']);a=p.parse_args();s=State()
    if a.command=='audit':s.audit();result=s.state
    elif a.command=='acceptance':result=acceptance(Diagnostics(s))
    elif a.command=='status':result={'state':read(s.run/'state.json'),'budget':read(s.run/'budget.json')}
    elif a.command=='run':result=execute(s)
    else:
        from .delivery import deliver
        result=deliver(s)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
