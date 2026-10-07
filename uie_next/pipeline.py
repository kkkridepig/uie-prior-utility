import fcntl
from pathlib import Path
from .records import write,read,append,digest


def paths(config):
    run=Path(config['runtime']['run_dir']);run.mkdir(parents=True,exist_ok=True)
    doc=run.parents[1]/'docs/experiments'/config['runtime']['run_id']
    doc.mkdir(parents=True,exist_ok=True)
    return run,doc


def advance(config,handlers):
    """Only verified dependency handlers may advance a phase; duplicate runs lock out."""
    run,_=paths(config);lock=(run/'.pipeline.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise RuntimeError('this run already has an active dispatcher')
    try:
        path=run/'state.json'
        state=read(path) if path.exists() else {'schema_version':1,'run_id':config['runtime']['run_id'],
                'config_hash':digest(config),'completed':[],'scientific_status':'NOT_TESTED','sealed_eval_released':False}
        if state['config_hash']!=digest(config):raise ValueError('same run identity cannot change protocol silently')
        for name,handler in handlers:
            if name in state['completed']:continue
            result=handler()
            if not result.get('passed'):
                state.update(last_blocked_stage=name,scientific_status=result['status'],blockers=result.get('reasons',[]))
                write(path,state);return state
            state['completed'].append(name);state.pop('last_blocked_stage',None)
            append(run/'dispatch_events.jsonl',{'stage':name,'result':result});write(path,state)
        write(path,state);return state
    finally:lock.close()
