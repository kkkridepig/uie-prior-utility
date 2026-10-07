"""Separate code, text review, visual and complete weight-recovery packages."""
import shutil
import subprocess

from .records import ROOT,GUIDE,sha,write,read
from .pipeline import paths
from .reporting import archive,text

TEXT_EXT={'.py','.toml','.yaml','.yml','.json','.jsonl','.csv','.xml','.md','.txt','.log','.sh','.html'}
CODE_EXT={'.py','.toml','.yaml','.yml','.sh','.txt','.cu','.cpp','.h','.hpp','.cuh'}


def package_evidence(config):
    run,doc=paths(config);out=ROOT.parent;tag=config['runtime']['run_id']+'_official_resume'
    code=[]
    for folder in ['uie_next','tests/uie_next','configs/uie_next']:
        code.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in CODE_EXT)
    for name in ['pyproject.toml','requirements-ssuie-ppu.txt','requirements-recovery-ppu.txt','LICENSE']:
        if (ROOT/name).is_file():code.append(ROOT/name)
    for folder in ['ss_uie','mamba_1_1_1','causal_conv1d_1_1_1']:
        source=ROOT/'third_party'/folder
        code.extend(p for p in source.rglob('*') if p.is_file() and p.suffix in CODE_EXT and 'build' not in p.parts and '.git' not in p.parts)
        for name in ['README.md','LICENSE','NOTICE']:
            if (source/name).is_file():code.append(source/name)
    updated=run/'protocol_source_v1_1.md'
    text(updated,GUIDE.read_text());code.append(updated)
    for name in ['roles.jsonl','exposure_ledger.json','split_freeze.json','protocol_resolved.yaml','source_snapshot.json','training_manifest_hashes.json']:
        if (run/name).is_file():code.append(run/name)
    source=archive(out/(tag+'_code_protocol.zip'),code,ROOT)
    visual=[p for p in (run/'figures').rglob('*') if p.is_file()]
    vis=archive(out/(tag+'_visual.zip'),visual,ROOT)
    weights=[p for p in (run/'checkpoints').rglob('*') if p.is_file() and p.suffix in {'.pt','.json'}]
    from pathlib import Path
    weights.append(Path(config['backbone']['checkpoint']))
    for name in ['baseline_policy.json','normalization_stats.json','checkpoint_selection.json','calibration_selection.json','selection_freeze_before_eval.json','backbone_provenance.json','protocol_resolved.yaml']:
        if (run/name).is_file():weights.append(run/name)
    recovery=archive(out/(tag+'_weights_recovery.zip'),weights,ROOT)
    wheels=list((ROOT/'downloads/ssuie').glob('*.whl'))
    runtime=archive(out/(tag+'_ppu_runtime_wheels.zip'),wheels,ROOT) if wheels else {'status':'not_available'}
    backup={'independent_backup_verified':False,'independent_mount_verified':False,'same_server_disk_only':True,
            'official_backbone_weight_included':True,'formal_weight_count':len(list((run/'checkpoints').rglob('*.pt'))),
            'source':source,'weights_recovery':recovery,'visual':vis,'runtime_wheels':runtime,
            'review_zip':str(out/(tag+'_review.zip')),
            'verification_scope':'Each ZIP listed member CRC and SHA256; not raw datasets or all server files. The authoritative source snapshot and input manifests have their separate identities.'}
    write(run/'closeout/backup_status.json',backup)
    # Dispatcher stdout is still open during closeout.  Hash an explicit stable
    # snapshot and exclude the live stream from the immutable verification set.
    active=run/'logs/official_resume_dispatch.log'
    if active.exists():shutil.copyfile(active,run/'closeout/dispatcher_log_snapshot.log')
    excluded={'archive_receipts.json','file_manifest_sha256.json'}
    eligible=[p for folder in [run,doc] for p in folder.rglob('*') if p.is_file() and p.suffix in TEXT_EXT
              and p.name not in excluded and p!=active]
    manifest={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(eligible)}
    manifest_path=run/'closeout/file_manifest_sha256.json'
    write(manifest_path,{'verification_scope':'Listed text evidence and docs only; excludes live stdout, manifest itself, archive receipts, raw datasets, image payloads, weights and caches.',
                         'files':manifest})
    review=archive(out/(tag+'_review.zip'),eligible+[manifest_path],ROOT)
    result={'source':source,'runtime_wheels':runtime,'visual':vis,'review':review,'weights_recovery':recovery,
            'review_excludes_weights_images_raw_data':True,'visual_contains_private_dataset_derivatives':True,
            'independent_backup_verified':False,'guide_file':str(GUIDE),'guide_sha256':sha(GUIDE)}
    write(run/'closeout/archive_receipts.json',result)
    return result
