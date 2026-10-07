"""Stage-local evidence packages; never claim same-disk ZIP is independent storage."""
from pathlib import Path
from ..records import ROOT,read,write,sha
from ..reporting import archive
from .context import OLD


def stage_backup(s,name):
    path=s.run/'backups'/(name+'.json')
    if path.exists():return read(path)
    snapshot=s.run/'source_snapshot.json'
    if not snapshot.exists():return {'status':'pending_source_freeze'}
    source=[ROOT/p for p in read(snapshot)]+[snapshot,s.run/'protocol_source.md',s.run/'protocol_resolved.yaml',s.run/'roles.jsonl',s.run/'exposure_ledger.json']
    weights=[Path(s.config['backbone']['checkpoint'])]+list((OLD/'checkpoints').rglob('*.pt'))+list((s.run/'checkpoints').rglob('*.pt'))
    code=archive(ROOT.parent/(s.ctx.run_id+'_'+name+'_source_protocol.zip'),sorted(set(source)),ROOT)
    weight=archive(ROOT.parent/(s.ctx.run_id+'_'+name+'_weights_recovery.zip'),sorted(set(weights)),ROOT)
    record={'source_protocol':code,'weights_recovery':weight,'independent_backup_verified':False,'same_server_disk_only':True}
    write(path,record);return record
