"""Content-only candidate audit. Does not evaluate enhancement or inspect test scores."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from mpa_diff.data.manifest import read_manifest

setup()
parent=torch.load(PARENT,map_location='cpu'); cfg=parent['config']; del parent
lsui=read_manifest(ROOT/'manifests/lsui_recon_grouped_v1.jsonl')
uieb=read_manifest(ROOT/cfg['data']['manifest'])
# Only LSUI validation candidates; remove every S1 author example by normalized stem.
s1=read_manifest(ROOT/'manifests/s1_author_examples.jsonl')
excluded_stems={Path(r['image_path']).stem for r in s1}
root=ROOT/'data/LSUI'
# Resolve against the actual grouped data configuration, never infer source root.
import yaml
lc=yaml.safe_load((ROOT/'configs/data/lsui_grouped.yaml').read_text()); root=ROOT/lc['data_root']
known=[(r,ROOT/cfg['data']['data_root']) for r in uieb]+[(r,root) for r in lsui if r['split']=='test']
candidates=[r for r in lsui if r['split']=='val' and Path(r['image_path']).stem not in excluded_stems]
def fp(path):
    with Image.open(path) as im:
        a=np.asarray(im.convert('L').resize((9,8),Image.BILINEAR))
        return (a[:,1:]>a[:,:-1]).flatten()
knownprints=np.stack([fp(base/r['image_path']) for r,base in known]); knownhash={r['image_sha256'] for r,base in known}
kept=[]; suspects=[]
for r in candidates:
    actual=sha(root/r['image_path']); assert actual==r['image_sha256']
    assert sha(root/r['reference_path'])==r['reference_sha256']
    diff=(knownprints!=fp(root/r['image_path'])).sum(1); nearest=int(diff.argmin())
    if actual in knownhash or diff[nearest]<=8:
        suspects.append({'sample_id':r['sample_id'],'nearest':known[nearest][0]['sample_id'],'dhash_distance':int(diff[nearest]),'reason':'exact_overlap' if actual in knownhash else 'unresolved_scene_candidate_not_declared_leakage'})
    else: kept.append(r)
# Exclude entire candidate scene if any member is suspect.
bad={r['scene_id'] for r in candidates if r['sample_id'] in {s['sample_id'] for s in suspects}}
kept=[r for r in kept if r['scene_id'] not in bad]
report={'status':'candidate_only_pending_exposure_review','root':str(root),'candidates':len(candidates),'kept_count':len(kept),'kept':kept,'suspects':suspects,'comparison_known':len(known),'method':'exact hashes + 64bit dHash <=8; unresolved suspects excluded conservatively, not classified as confirmed leakage','limitations':['perceptual hash does not prove scene independence','LSUI validation never used in v2 known main queue; historical S1 examples excluded by stem','pretrained DA/VGG upstream exposure unknown','no image scores accessed'],'candidate_sha256':digest(kept)}
write(RUN/'holdout_candidate_audit.json',report)
(DOC/'HOLDOUT_AUDIT.md').write_text('# 新留出候选审计\n\nLSUI旧val候选 '+str(len(candidates))+' 张，保守排除近重复/未决场景后 '+str(len(kept))+' 张。未读取增强分数。\n\n当前仅为候选，不自动成为已确认新留出：需核查完整历史暴露，dHash不保证场景无关，DA/VGG上游暴露仍未知。疑似关系不等于已确认泄漏。详见holdout_candidate_audit.json。\n')
print({k:v for k,v in report.items() if k not in ('kept','suspects')})
