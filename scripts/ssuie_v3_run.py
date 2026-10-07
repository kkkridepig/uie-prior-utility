"""Run audited V3 stages; no legacy pipeline invocation."""
import argparse,traceback,time
import torch
from uie_next.records import write,read,sha,append
from uie_next.v3.context import State,Data
from uie_next.v3.diagnostics import Diagnostics,extract_stats
from uie_next.v3.ridge_probe import run_probes


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','stage1','ridge','closeout','all','verify']);a=p.parse_args();s=State();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    if a.stage=='audit':s.audit();return
    if 'D1_D6_DIAGNOSTICS' in s.state['completed']:
        from uie_next.v3.recovery import verify_completed
        receipt=verify_completed(s)
        if a.stage=='verify':write(s.run/'tests/completed_stage_recovery.json',receipt);return
    elif a.stage=='verify':raise RuntimeError('no completed stage to verify')
    if 'RECOVERY_AUDIT' not in s.state['completed']:raise RuntimeError('audit first')
    if not (s.run/'source_snapshot.json').exists():s.snapshot()
    if a.stage in ['stage1','all']:
        if not (s.run/'tests/cpu_pass.json').exists():raise RuntimeError('CPU contracts not accepted')
        if 'D1_D6_DIAGNOSTICS' not in s.state['completed']:
            with s.device_job('STAGE1_REAL_ACCEPTANCE_AND_D1_D6',estimate=1200,stage=1):
                d=Diagnostics(s);d.load();d.accept();rec=d.scan();d.gradients()
            s.complete('D1_D6_DIAGNOSTICS',{'images':len(rec),'gradient_log':sha(s.run/'diagnostics/D5_loss_gradient_components.jsonl')})
    if a.stage in ['ridge','stage1','all']:
        if 'D1_D6_DIAGNOSTICS' not in s.state['completed']:raise RuntimeError('D1-D6 first')
        if 'STAGE1_DECISION' not in s.state['completed']:extract_stats(s,Data(s));run_probes(s)
    if a.stage in ['all','closeout']:
        dec=read(s.run/'stage1_decision.json')
        if dec['route'].startswith('TRAIN'):
            from uie_next.v3.training import run_stage2
            run_stage2(s)
        from uie_next.v3.delivery import closeout
        closeout(s)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        s=State();append(s.run/'events.jsonl',{'event':'error','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        s.live(status='INCONCLUSIVE_BUDGET' if type(exc).__name__=='BudgetStop' else 'ENGINEERING_ERROR',error=str(exc));raise
