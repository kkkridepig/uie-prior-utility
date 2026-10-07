"""Explicit, auditable admission of the user-supplied official public weight."""
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests

from .records import ROOT, RUN, DOC, append, digest, jsonl, read, sha, write
from .schema import load


SOURCE='https://drive.google.com/file/d/1YxyagMCbApON8dRdiQTaQG65g3tnZkPt/view?usp=sharing'
LABEL='SS-UIE official public simplified implementation'


def adopt(config):
    provenance=RUN/'backbone_provenance.json'
    target=Path(config['backbone']['checkpoint'])
    upload=ROOT.parent/'TEMP-FILE-STATION/SS_UIE.pth'
    expected=config['backbone']['checkpoint_sha256']
    if provenance.exists():
        record=read(provenance)
        if sha(target)!=expected or record['checkpoint_sha256']!=expected:
            raise ValueError('Adopted official weight identity changed.')
        return record
    receipt=read(RUN/'backbone_upload_verification_20261007/receipt_retry.json')
    if not receipt['passed'] or receipt['sha256']!=expected or sha(upload)!=expected:
        raise ValueError('Strict uploaded weight verification missing or mismatched.')
    state=read(RUN/'state.json')
    if state.get('formal_training_updates',0):
        raise ValueError('Backbone admission cannot alter an already trained run.')
    history=RUN/'audit_history/pre_official_resume'
    history.mkdir(parents=True,exist_ok=True)
    for name in ('intake.json','state.json','exposure_ledger.json','roles.jsonl','all_pairs.jsonl',
                 'split_freeze.json','cross_dataset_audit.json','backbone_verification.json','budget_plan.json'):
        source=RUN/name
        if source.exists() and not (history/name).exists():shutil.copy2(source,history/name)
    for source in DOC.glob('*.md'):
        if not (history/'docs'/source.name).exists():
            (history/'docs').mkdir(exist_ok=True);shutil.copy2(source,history/'docs'/source.name)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(target)!=expected:raise ValueError('Refuse to overwrite a different checkpoint.')
    else:
        temporary=target.with_suffix('.copying')
        shutil.copyfile(upload,temporary)
        if sha(temporary)!=expected:raise ValueError('Uploaded weight copy checksum failed.')
        temporary.replace(target)
    claims=[]
    for cid in (2894406180,2904097810):
        api='https://api.github.com/repos/LintaoPeng/SS-UIE/issues/comments/'+str(cid)
        response=requests.get(api,timeout=30);response.raise_for_status();data=response.json()
        if data['user']['login']!='LintaoPeng' or data['author_association']!='OWNER':
            raise ValueError('Unexpected authorship of public implementation limitation.')
        path=RUN/('source_evidence/issuecomment_%d.json'%cid);write(path,data)
        claims.append({'url':data['html_url'],'api_url':api,'receipt_sha256':sha(path),
                       'author':data['user']['login'],'author_association':data['author_association'],
                       'statement':data['body'].split('\r\n')[0]})
    guide=ROOT.parent/'TEMP-FILE-STATION/SSUIE_C局部效用学习_本轮完整执行指南_20261007.md'
    record={'accepted':True,'checkpoint':str(target),'checkpoint_sha256':expected,'size_bytes':target.stat().st_size,
            'source_url':SOURCE,'source_evidence_kind':'explicit_user_download_declaration_and_locked_official_README',
            'independently_redownloaded_and_compared':False,'upstream_published_checksum_available':False,
            'backbone_label':LABEL,'paper_complete_model':False,'unpublished_MCSS_reconstructed':False,
            'upstream_commit':receipt['identity']['commit'],'strict_device_receipt_sha256':sha(RUN/'backbone_upload_verification_20261007/receipt_retry.json'),
            'author_limitations':claims,'training_source_scope':'Official README declares LSUI; per-image author split not available locally.',
            'guide_sha256':sha(guide),'adopted_at_utc':datetime.now(timezone.utc).isoformat()}
    write(provenance,record)
    ledger=read(RUN/'exposure_ledger.json');old_roles=[(r['sample_id'],r['role'],r['group_id']) for r in ledger['samples']]
    for row in ledger['samples']:
        row['backbone_provenance_verified']=True
        row['checkpoint_sha256']=expected
        row['source_evidence'].append('backbone_provenance.json:'+digest(record))
        row['exposure_scope']='Official public simplified LSUI-source checkpoint; complete local LSUI/UIEB audit; author train membership not mapped.'
    assert old_roles==[(r['sample_id'],r['role'],r['group_id']) for r in ledger['samples']]
    ledger.update(upstream_claim=record['training_source_scope'],backbone_provenance=record,
                  sealed_eval_is_new_blind_test=False)
    write(RUN/'exposure_ledger.json',ledger);jsonl(RUN/'roles.jsonl',ledger['samples']);jsonl(RUN/'all_pairs.jsonl',ledger['samples'])
    freeze=read(RUN/'split_freeze.json');freeze.update(roles_sha256=sha(RUN/'roles.jsonl'),audit_status='DOCUMENTED_NONOVERLAP_WITH_DECLARED_SOURCE_SCOPE')
    write(RUN/'split_freeze.json',freeze)
    audit=read(RUN/'cross_dataset_audit.json');audit['status']='DOCUMENTED_NONOVERLAP_WITH_DECLARED_SOURCE_SCOPE';write(RUN/'cross_dataset_audit.json',audit)
    append(RUN/'protocol_amendments.jsonl',{'kind':'official_backbone_admission_and_user_clarification',
        'old_config_hash':state['config_hash'],'new_config_hash':digest(config),'reason':'User supplied official public simplified checkpoint and documented author limitations.',
        'guide_file_contains_new_section_3_5':False,'user_message_is_explicit_supplement':True,
        'roles_and_groups_changed':False,'scientific_network_or_loss_changed':False,'budget_reset':False,
        'history_directory':str(history),'source_provenance':str(provenance)})
    append(RUN/'protocol_amendments.jsonl',{'kind':'pretraining_engineering_fix','reason':'Correct cosine schedule off-by-one: LR is for the update about to execute, including exactly 1e-5 at final update.',
        'formal_training_updates_before_fix':0,'test':'test_learning_rate_is_set_for_the_update_being_executed'})
    state.update(config_hash=digest(config),scientific_status='RESUMING_PREFLIGHT',blockers=[],closeout_status='REOPENED_AFTER_OFFICIAL_BACKBONE_ADMISSION',
                 engineering_status='CPU_CORE_TESTED_REAL_INTEGRATION_PENDING',backbone_label=LABEL)
    write(RUN/'state.json',state)
    intake=read(RUN/'intake.json');intake.update(config_sha256=digest(config),guide_sha256=sha(guide),backbone_provenance_sha256=sha(provenance))
    write(RUN/'intake.json',intake)
    return record


if __name__=='__main__':
    import json
    print(json.dumps(adopt(load(ROOT/'configs/uie_next/protocol.yaml')),ensure_ascii=False,indent=2))
