"""Completed stages are reusable only with intact identities and expected rows."""
import csv,json
from pathlib import Path
from ..records import read,sha,write
from .context import ROLE_SHA,PROTOCOL_SHA


def require_hashes(root,manifest):
    root=Path(root)
    for name,expected in manifest.items():
        p=root/name
        if not p.is_file() or sha(p)!=expected:raise ValueError('recovery artifact missing or changed: '+name)


def verify_completed(s):
    if sha(s.run/'protocol_source.md')!=PROTOCOL_SHA or sha(s.run/'roles.jsonl')!=ROLE_SHA:raise ValueError('protocol/role recovery mismatch')
    probe=read(s.run/'diagnostics/probe_manifest.json');n=len(probe['sample_ids'])+136
    if 'D1_D6_DIAGNOSTICS' in s.state['completed']:
        parts=list((s.run/'diagnostics/parts').glob('*.json'))
        if len(parts)!=n:raise ValueError('incomplete D1-D6 part count')
        ids=set();source=sha(s.run/'source_snapshot.json')
        for p in parts:
            r=read(p)
            if not r.get('complete') or r['identity']['source']!=source:raise ValueError('D1-D6 resume identity mismatch')
            ids.add(r['meta']['sample_id'])
        roles=[json.loads(x) for x in (s.run/'roles.jsonl').read_text().splitlines()]
        expected=set(probe['sample_ids'])|{r['sample_id'] for r in roles if r['role']=='utility_val'}
        if ids!=expected:raise ValueError('D1-D6 paired IDs incomplete')
        grads=[json.loads(x) for x in (s.run/'diagnostics/D5_loss_gradient_components.jsonl').read_text().splitlines()]
        if len(grads)!=2 or any(r['optimizer_steps']!=0 or not r['old_weights_unchanged'] for r in grads):raise ValueError('D5 receipt incomplete')
    if 'STAGE1_DECISION' in s.state['completed']:
        d=read(s.run/'stage1_decision.json');require_hashes(s.run,d['input_hashes'])
        for filename,count in [('D7_oof_per_image.csv',443),('D7_dev_per_image.csv',136)]:
            rows=list(csv.DictReader((s.run/'diagnostics'/filename).open()))
            if len(rows)!=count or len({r['sample_id'] for r in rows})!=count:raise ValueError('D7 row/ID count incomplete')
        frozen=s.run/'selection/stage1_policy_freeze.json'
        if frozen.exists() and read(frozen)['decision_sha256']!=sha(s.run/'stage1_decision.json'):raise ValueError('stage1 frozen decision changed')
        if d['route']=='STOP_NO_PREDICTABILITY_SIGNAL' and s.state['sealed_eval_released']:raise ValueError('illegal sealed release')
    b=read(s.run/'budget.json')
    if b['active'] is not None:raise ValueError('unreconciled active device event')
    return {'passed':True,'state_and_hashes_and_expected_samples_verified':True,'does_not_require_old_tensor_cache':True,'sealed_released':s.state['sealed_eval_released']}
