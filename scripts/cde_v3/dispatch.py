"""Serial device-budget runner. Persistent lock, attempt IDs and durable heartbeat.

Direct --run jobs are for audited task invocations. --dry-run reports dependencies.
"""
import argparse, fcntl, os, signal, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import RUN,DOC,ROOT,read,write,sha

PLAN=[('P0','audit, role freeze, math/CPU/PPU and resume contracts, 100-step profiles'),('E','fixed24 parent+v2_base DDPM/DDIM diagnostics then source_dev DDIM'),('C','matched5000 BASE_CONT_V3, C_BANK, C_RGB_CONTROL then oracle'),('C','only oracle pass: ALL_ONLY, route-fit labels and CE/shuffle/utility, contamination gate'),('C','only utility pass: DATA_MATCHED then freeze winning configuration'),('D','only correct C numerical failure with complete metrics: one D pilot'),('REPEAT','only passing method: two seeds, all required controls'),('FINAL','final_freeze before holdout/regression, metrics, panels, lineage export')]

def command_identity(command):
    return [str((ROOT/command[0]).resolve())]+command[1:]

def update_live(state,budget):
    failures={k:dict(v,resolved_by_completion=k in state['completed']) for k,v in state.get('failures',{}).items()}
    (DOC/'LIVE_STATUS.md').write_text('# V3 实时状态\n\n'+f"状态：{state['status']}；已计设备时间 {budget['charged_seconds']/3600:.5f} / 72 h，最终预留12 h。\n\n活动：{budget.get('active')}\n\n完成任务：{list(state['completed'])}\n\n当前阻塞原因：{state.get('reason','无')}\n\n历史失败/安全分段停止（resolved_by_completion表示后续已完成）：{failures}\n\n此页是状态，不是方法有效性证明。\n")

def run_job(job,package,command,max_hours):
    RUN.mkdir(parents=True,exist_ok=True); DOC.mkdir(parents=True,exist_ok=True)
    with (RUN/'device.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=read(RUN/'budget.json'); s=read(RUN/'dispatch_state.json')
        if job in s['completed']:
            if command_identity(s['completed'][job]['command'])!=command_identity(command): raise ValueError('Completed ID reused for different command')
            return 0
        if b.get('active'):
            old=b['active']; pid=old.get('pid')
            if pid and Path('/proc/%s'%pid).exists(): raise RuntimeError('Previously tracked device job still alive; do not duplicate')
            # Unknown tail conservatively charged up to its deadline, never reset the ledger.
            tail=max(0,min(time.time(),old['deadline'])-old['last_heartbeat'])
            b['charged_seconds']+=tail
            b['events'].append({'id':old['id'],'package':old['package'],'elapsed_seconds':old['charged']+tail,'exit_code':None,'status':'interrupted_unknown_tail_conservatively_charged'})
            b['active']=None; write(RUN/'budget.json',b)
        failures=[e for e in b['events'] if e['id']==job and e.get('exit_code')!=0]
        if len(failures)>=2: raise RuntimeError('Single retry already consumed')
        package_used=sum(e['elapsed_seconds'] for e in b['events'] if e['package']==package)
        ceiling=(72 if package=='FINAL' else 60)*3600
        available=min(ceiling-b['charged_seconds'],b['package_caps'][package]*3600-package_used,max_hours*3600)
        if available<180: raise RuntimeError('Insufficient reserved device budget')
        start=time.time(); deadline=start+available
        log=RUN/'logs'/('%s_attempt%d.log'%(job,len(failures)+1)); log.parent.mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CDE_DEADLINE=str(deadline),PYTHONUNBUFFERED='1')
        with log.open('a') as f:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            b['active']={'id':job,'package':package,'pid':p.pid,'start':start,'deadline':deadline,'last_heartbeat':start,'charged':0}
            s.update(status='running',stage=package,active_job=job)
            s.pop('reason',None)
            write(RUN/'budget.json',b); write(RUN/'dispatch_state.json',s); update_live(s,b)
            last=start; sent=False
            while p.poll() is None:
                time.sleep(2); now=time.time(); delta=now-last; last=now
                b['charged_seconds']+=delta; b['active'].update(last_heartbeat=now,charged=now-start)
                write(RUN/'budget.json',b)
                if now>=deadline-90 and not sent:
                    os.kill(p.pid,signal.SIGTERM); sent=True
                if now>=deadline+30 and p.poll() is None:
                    # Do not silently abandon a child and let another dispatcher reuse the device.
                    s['status']='blocked_process_did_not_exit'; write(RUN/'dispatch_state.json',s); update_live(s,b)
                    raise RuntimeError('Child ignored safe stop; device lock must be inspected')
                if int(now-start)%30<2: update_live(s,b)
        elapsed=time.time()-start; tail=max(0,time.time()-last); b['charged_seconds']+=tail
        event={'id':job,'package':package,'elapsed_seconds':elapsed,'exit_code':p.returncode,'command':command,'log':str(log)}
        b['events'].append(event); b['active']=None
        if p.returncode==0: s['completed'][job]=event; s['status']='ready_for_next_dependency'
        else: s['failures'][job]=event; s['status']='blocked_task_failure'
        s.pop('active_job',None); write(RUN/'budget.json',b); write(RUN/'dispatch_state.json',s); update_live(s,b)
        return p.returncode

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dry-run',action='store_true'); p.add_argument('--run'); p.add_argument('--package',choices=['P0','E','C','D','REPEAT','FINAL','REPAIR']); p.add_argument('--max-hours',type=float,default=4); p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
    if a.dry_run:
        b=read(RUN/'budget.json') if (RUN/'budget.json').exists() else {}
        cost=read(RUN/'cost_prediction.json') if (RUN/'cost_prediction.json').exists() else None
        print({'plan':PLAN,'device_limit_hours':b.get('limit_hours',72),'reserve_hours':b.get('final_reserve_hours',12),'charged_hours':b.get('charged_seconds',0)/3600,'active':b.get('active'),'current_cost_prediction':cost,'read_only':True}); return
    if not a.run or not a.package or not a.command: p.error('--run, --package and command required')
    c=a.command[1:] if a.command[0]=='--' else a.command
    sys.exit(run_job(a.run,a.package,c,a.max_hours))
if __name__=='__main__': main()
