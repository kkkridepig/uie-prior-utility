"""Strict configuration: unsupported experiment variants cannot silently run."""
import copy
from pathlib import Path
import yaml

DEFAULT = {
 'schema_version':1,
 'experiment':{'name':'mpa_recon_v1','evidence_profile':'reconstruction','seed':20260927,'parent_checkpoint':None},
 'data':{'manifest':'manifests/uieb_recon_702_91_97_v1.jsonl','data_root':'data','resize_hw':[336,336],'input_color':'srgb','augmentation':'none','split_policy':'fixed_manifest'},
 'depth':{'provider':'depth_anything_v2','encoder':'vits','checkpoint':None,'checkpoint_sha256':None,'source_root':'third_party/Depth-Anything-V2','frozen':True,'input_size':518,'raw_kind':'relative_inverse','physics_coordinate':'thesis_raw_minmax','epsilon':1e-6},
 'physics':{'mode':'shared_global','color_space':'srgb_approx','background':'pil_gaussian_legacy','joint_beta_training':True},
 'histogram':{'mode':'histogan_rgbuv','bins':64,'bandwidth':.02,'max_input_size':150},
 'frequency':{'shallow_prior':'sobel_haar','include_low':False},
 'model':{'base_channels':32,'channel_mult':[1,2,3,4],'residual_blocks':1,'dropout':.1},
 'diffusion':{'representation':'rgb01','prediction_type':'x0','train_steps':1000,'schedule':'linear','beta_start':1e-6,'beta_end':.02,'variance':'fixed_small'},
 'loss':{'pixel':'masked_mse','perceptual_backbone':'vgg19','perceptual_weight':.1,'vgg_checkpoint':None,'vgg_sha256':None},
 'train':{'optimizer':'adam','learning_rate':1e-4,'betas':[.9,.999],'weight_decay':0.,'total_steps':400000,'decay_start':200000,'final_learning_rate':1e-6,'effective_batch':4,'micro_batch':1,'gradient_accumulation':4,'precision':'float32','ema':False,'validation_every':5000,'checkpoint_every':5000},
 'sampler':{'name':'ddpm','steps':1000,'eta':0.,'clip_intermediate_x0':False},
 'extensions':{k:'disabled' for k in 'ABCDEFG'},
 'runtime':{'device':'cuda','threads':2,'output':'runs/mpa_recon_v1','cache_dir':'data/cache/mpa','test_fixture':False}
}


def merge_strict(base, update, prefix=''):
    for key,value in update.items():
        if key not in base: raise ValueError('Unknown config key: '+prefix+key)
        if isinstance(base[key],dict):
            if not isinstance(value,dict): raise ValueError('Expected mapping: '+prefix+key)
            merge_strict(base[key],value,prefix+key+'.')
        else: base[key]=value
    return base


def validate(c):
    t=c['train']; d=c['diffusion']; s=c['sampler']; fixture=c['runtime']['test_fixture']
    fixed={'schema_version':1}
    if c['schema_version']!=1: raise ValueError('Unsupported schema')
    for section, pairs in {'data':{'input_color':'srgb','augmentation':'none','split_policy':'fixed_manifest'},'physics':DEFAULT['physics'],'frequency':DEFAULT['frequency'],'diffusion':{'representation':'rgb01','prediction_type':'x0','schedule':'linear','variance':'fixed_small'},'train':{'optimizer':'adam','precision':'float32','ema':False},'depth':{'encoder':'vits','frozen':True,'raw_kind':'relative_inverse','epsilon':1e-6},'loss':{'pixel':'masked_mse','perceptual_backbone':'vgg19'}}.items():
        for k,v in pairs.items():
            if c[section][k]!=v: raise ValueError('Unsupported baseline option: '+section+'.'+k)
    if any(x!='disabled' for x in c['extensions'].values()): raise ValueError('S3–S7 extensions require separate implementation; unavailable in S0–S2 baseline')
    if t['micro_batch']*t['gradient_accumulation']!=t['effective_batch']: raise ValueError('Effective batch mismatch')
    if min(t['total_steps'],t['micro_batch'],t['gradient_accumulation'],t['validation_every'],t['checkpoint_every'])<1: raise ValueError('Positive training counts required')
    if not 0<=t['decay_start']<t['total_steps']: raise ValueError('Decay start must precede final step')
    if c['depth']['provider'] not in ('synthetic_fixture','depth_anything_v2'): raise ValueError('Unknown depth provider')
    if not fixture and (c['depth']['provider']=='synthetic_fixture' or c['loss']['perceptual_weight']==0): raise ValueError('Test double / disabled VGG allowed only in explicit test_fixture runs')
    if fixture and c['experiment']['evidence_profile']!='synthetic_test_only': raise ValueError('Fixture must be labelled synthetic_test_only')
    if c['depth']['physics_coordinate'] not in ('thesis_raw_minmax','distance_proxy'): raise ValueError('Unknown depth coordinate')
    if c['model']['base_channels']<4 or c['model']['base_channels']%2 or c['model']['channel_mult']!=[1,2,3,4] or c['model']['residual_blocks']<1: raise ValueError('Invalid HIN topology')
    if len(c['data']['resize_hw'])!=2 or min(c['data']['resize_hw'])<1: raise ValueError('Invalid image size')
    if s['name'] not in ('ddpm','ddim') or not 1<=s['steps']<=d['train_steps']: raise ValueError('Invalid sampler')
    if s['name']=='ddpm' and s['steps']!=d['train_steps']: raise ValueError('DDPM requires T steps')
    if c['histogram']['mode'] not in ('histogan_rgbuv','thesis_shared_uv_rgb_weight'): raise ValueError('Unknown histogram')
    return c


def load_config(path):
    data=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,dict): raise ValueError('Config must be a mapping')
    return validate(merge_strict(copy.deepcopy(DEFAULT),data))
