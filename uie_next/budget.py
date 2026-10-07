import fcntl
import time
from pathlib import Path
from .records import read,write,append


class BudgetStop(RuntimeError):
    pass


class DeviceBudget:
    """One device lease; conservatively accounts for an unreconciled orphan."""
    def __init__(self,run,max_hours=16,reserve_hours=3.5,clock=time.time):
        self.run=Path(run);self.run.mkdir(parents=True,exist_ok=True)
        self.path=self.run/'budget.json';self.clock=clock
        self.lease=None;self.max_hours=max_hours;self.reserve_hours=reserve_hours
        if not self.path.exists(): write(self.path,{'max_device_hours':max_hours,'reserve_device_hours':reserve_hours,'used_device_seconds':0.,'active':None})
        state=read(self.path)
        if state['max_device_hours']!=max_hours or state['reserve_device_hours']!=reserve_hours:
            raise ValueError('budget identity changed; no automatic new allowance')

    def __enter__(self):
        self.lease=(self.run/'.device.lock').open('a+')
        try: fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lease.close();self.lease=None;raise BudgetStop('device already leased')
        state=read(self.path)
        if state['active']:
            elapsed=max(0,self.clock()-state['active']['last_accounted_at'])*state['active']['devices']
            state['used_device_seconds']+=elapsed;state['active']=None;write(self.path,state)
            append(self.run/'budget_ledger.jsonl',{'event':'conservative_orphan_charge','device_seconds':elapsed})
        return self

    def remaining(self,final=False):
        self.tick();s=read(self.path)
        return (s['max_device_hours']-(0 if final else s['reserve_device_hours']))*3600-s['used_device_seconds']

    def start(self,job,estimate_seconds,final=False,devices=1):
        if self.lease is None: raise RuntimeError('budget lock required')
        if devices!=1: raise ValueError('protocol permits exactly one device')
        s=read(self.path)
        if s['active']: raise BudgetStop('previous job active')
        if estimate_seconds<=0 or self.remaining(final)<estimate_seconds: raise BudgetStop('insufficient reserved budget')
        s=read(self.path);s['active']={'job':job,'devices':devices,'last_accounted_at':self.clock(),'final':final}
        write(self.path,s);append(self.run/'budget_ledger.jsonl',{'event':'start','job':job,'time':self.clock()})

    def tick(self):
        s=read(self.path)
        if s['active']:
            now=self.clock();delta=max(0,now-s['active']['last_accounted_at'])*s['active']['devices']
            s['used_device_seconds']+=delta;s['active']['last_accounted_at']=now;write(self.path,s)

    def guard(self,save_period_seconds=0):
        s=read(self.path);final=bool(s['active'] and s['active']['final'])
        if self.remaining(final)<max(save_period_seconds,0): raise BudgetStop('save checkpoint and stop before limit')

    def stop(self):
        self.tick();s=read(self.path)
        if s['active']:
            append(self.run/'budget_ledger.jsonl',{'event':'stop','job':s['active']['job'],'used_device_seconds':s['used_device_seconds']})
            s['active']=None;write(self.path,s)

    def __exit__(self,*args):
        try: self.stop()
        finally:
            if self.lease: self.lease.close();self.lease=None
