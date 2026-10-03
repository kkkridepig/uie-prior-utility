import copy,json
from pathlib import Path
import pytest
import torch
from mpa_diff.config import load_config
from mpa_diff.data.synthetic import generate
from mpa_diff.data.manifest import read_manifest,audit
from mpa_diff.data.cache import cache_key
from mpa_diff.engine.trainer import train


def test_manifest_leakage_and_stale(tmp_path):
    root=tmp_path/'data';path=tmp_path/'manifest.jsonl';generate(root,path,n=8)
    rows=read_manifest(path);rows[-1]['scene_id']=rows[0]['scene_id']
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='leakage'):audit(path,root)
    rows[-1]['scene_id']='different';rows[0]['image_sha256']='invalid';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='Stale'):audit(path,root)


def test_cache_identity():
    c=load_config('configs/base/synthetic_smoke.yaml');x=torch.rand(1,3,16,16)
    k=cache_key(x,c);x2=x.clone();x2[0,0,0,0]+=.01;assert k!=cache_key(x2,c)
    c['depth']['checkpoint_sha256']='changed';assert k!=cache_key(x,c)


def test_resume_exact(tmp_path):
    c=load_config('configs/base/synthetic_smoke.yaml');c['train'].update(total_steps=4,decay_start=1,checkpoint_every=2,validation_every=4,effective_batch=1,micro_batch=1,gradient_accumulation=1)
    c['model'].update(base_channels=4,dropout=.1);c['data']['resize_hw']=[16,16];c['sampler']['steps']=1
    c['runtime']['cache_dir']=str(tmp_path/'cache');c['runtime']['output']=str(tmp_path/'continuous')
    train(c);whole=torch.load(tmp_path/'continuous/last.pt',map_location='cpu')
    c['runtime']['output']=str(tmp_path/'resume');train(c,stop_after=2);train(c,resume=tmp_path/'resume/last.pt')
    split=torch.load(tmp_path/'resume/last.pt',map_location='cpu')
    assert whole['scheduler']==split['scheduler'];assert whole['data_sampler']==split['data_sampler']
    for key in whole['model']:torch.testing.assert_close(whole['model'][key],split['model'][key],rtol=0,atol=0)
    for key,state in whole['optimizer']['state'].items():
        for name,value in state.items():torch.testing.assert_close(value,split['optimizer']['state'][key][name],rtol=0,atol=0)


def test_depth_cache_batch_independent_and_lossless(tmp_path):
    from mpa_diff.priors.provider import PriorProvider
    from mpa_diff.data.cache import static_cached
    c=load_config('configs/base/synthetic_smoke.yaml');c['runtime']['cache_dir']=str(tmp_path)
    provider=PriorProvider(c).eval()
    image=torch.rand(2,3,16,16);image[0]=.4
    original=provider.static(image);cached=static_cached(provider,image,c)
    again=static_cached(provider,image.flip(0),c)
    assert len(list((tmp_path/'depth_v2').glob('*.npz')))==2
    for name in ('raw','distance_proxy','physical_coordinate','confidence','valid_mask'):
        torch.testing.assert_close(getattr(original['depth'],name),getattr(cached['depth'],name),rtol=0,atol=0)
        torch.testing.assert_close(getattr(original['depth'],name).flip(0),getattr(again['depth'],name),rtol=0,atol=0)
    for name in ('histogram','ambient','edges','wavelet'):
        torch.testing.assert_close(original[name],cached[name],rtol=0,atol=0)


def test_content_duplicates_require_same_explicit_group(tmp_path):
    root=tmp_path/'data';path=tmp_path/'manifest.jsonl';generate(root,path,n=8)
    rows=read_manifest(path);rows[1]['image_path']=rows[0]['image_path'];rows[1]['image_sha256']=rows[0]['image_sha256']
    def write():path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write()
    with pytest.raises(ValueError,match='Duplicate content'):audit(path,root)
    rows[1]['scene_id']=rows[0]['scene_id'];write()
    report=audit(path,root);assert report['same_split_grouped_duplicate_files']
    rows[1]['split']='test';write()
    with pytest.raises(ValueError,match='Duplicate content'):audit(path,root)
