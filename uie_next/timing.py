"""Live deployment without reference access or quality-evaluation caches."""
import csv
import time

import cv2
import numpy as np
import torch

from .backbones.ssuie import postprocess
from .records import read, sha, write


def statistics(values):
    values=np.asarray(values,dtype=float)
    return {'calls':len(values),'mean_seconds':float(values.mean()),'median_seconds':float(np.median(values)),
            'p90_seconds':float(np.quantile(values,.90)),'p95_seconds':float(np.quantile(values,.95))}


class Deployment:
    def __init__(self,e,models,selection):
        self.backbone=e.backbone;self.models=models;self.selection=selection;self.policy=e.data.policy
        self.se2=read(e.run/'normalization_stats.json')['s_e2']

    @torch.no_grad()
    def __call__(self,image,method):
        policy=self.selection[method]['policy']
        if method.startswith('B0_'):
            return postprocess(self.backbone.model(image),method[3:])
        base=self.backbone(image)
        if policy.get('return_base') or method in ['B2','B4'] and policy.get('alpha')==0:return base
        from .priors.heuristic import make_prior,rgb_prior
        if method in ['B3','B4']:P,V=rgb_prior(image);head=self.models[method]
        else:
            fields=make_prior(image);P,V=fields['P'],fields['V'];head=self.models['B1']
        candidate=head(image,base,P,V)
        if method in ['B1','B2','B3','B4']:return (base+policy.get('alpha',1.)*(candidate-base)).clamp(0,1)
        pred=self.models[method](image,base,candidate,se2=self.se2)
        from .scientific_evaluation import decode_policy
        return decode_policy(method,pred,base,candidate,policy)[0]


def preprocess(decoded,device):
    if decoded is None or decoded.dtype!=np.uint8 or decoded.ndim!=3 or decoded.shape[2]!=3:raise ValueError('Deployment requires decoded uint8 BGR input.')
    rgb=cv2.cvtColor(cv2.resize(decoded,(256,256),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2,0,1).copy()).float().div(255)[None].to(device)


def measure(e,deploy,methods,repeats,warmup,output_prefix):
    rows=e.data.role('model_val')[:24]
    decoded=[cv2.imread(r['input_path'],cv2.IMREAD_UNCHANGED) for r in rows]
    inputs=[preprocess(x,'cuda:0') for x in decoded]
    raw=[];summary={};output=e.run/'timing/probe.png';output.parent.mkdir(parents=True,exist_ok=True)
    for method in methods:
        summary[method]={}
        for path in ['model_only','service_no_save','service_with_png']:
            def call(index):
                x=inputs[index] if path=='model_only' else preprocess(decoded[index],'cuda:0')
                result=deploy(x,method)
                if path!='model_only':
                    rgb=result[0].permute(1,2,0).contiguous().cpu().numpy()
                    if path=='service_with_png':
                        if not cv2.imwrite(str(output),cv2.cvtColor(np.round(rgb*255).clip(0,255).astype(np.uint8),cv2.COLOR_RGB2BGR)):raise RuntimeError('PNG timing write failed.')
            for n in range(warmup):e.guard();call(n%len(rows))
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();seconds=[]
            for n in range(repeats):
                e.guard();torch.cuda.synchronize();begin=time.perf_counter();call(n%len(rows));torch.cuda.synchronize()
                elapsed=time.perf_counter()-begin;seconds.append(elapsed)
                raw.append({'method':method,'path':path,'repeat':n,'sample_id':rows[n%len(rows)]['sample_id'],'seconds':elapsed})
            summary[method][path]={**statistics(seconds),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    path=e.run/('timing/'+output_prefix+'_raw.csv')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    result={'methods':summary,'device':torch.cuda.get_device_name(0),'batch':1,'shape':[3,256,256],'precision':'float32',
            'warmup_calls_per_path':warmup,'measured_calls_per_path':repeats,'distinct_input_images':len(rows),
            'decoded_CPU_input_start_for_service':True,'reads_reference':False,'reads_model_cache':False,'raw_sha256':sha(path),'outlier_exclusions':0}
    write(e.run/('timing/'+output_prefix+'.json'),result);return result


def tie_costs(e,models,selection):
    result=benchmark(e,models,selection)
    return {m:result['methods'][m]['service_no_save']['mean_seconds'] for m in selection}


def benchmark(e,models,selection):
    path=e.run/'timing/deployment.json'
    identity={'selection':selection,'checkpoints':{m:sha(e.run/'checkpoints'/m/'selection.json') for m in models},
              'backbone_sha256':e.config['backbone']['checkpoint_sha256'],'timing_source':sha(__file__)}
    if path.exists():
        saved=read(path)
        if saved['identity']!=identity:raise ValueError('Deployment timing identity changed.')
        return saved
    result=measure(e,Deployment(e,models,selection),list(selection),100,10,'deployment')
    parameters={}
    for name,model in models.items():
        passes=2 if name in ['O','O-NI','O-NP','O-ND'] else 1
        mac=passes*sum(256*256*m.out_channels*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
                       for m in model.modules() if isinstance(m,torch.nn.Conv2d))
        parameters[name]={'trained_parameter_count':sum(p.numel() for p in model.parameters()),'module_MACs_256x256':mac,
                          'shared_forward_passes':passes,'MAC_scope':'new full-resolution convolution module only; not SS-UIE FFT/selective scan total FLOPs'}
    result['module_parameters_and_MACs']=parameters
    result['frozen_backbone_parameters']=sum(p.numel() for p in e.backbone.parameters())
    result['identity']=identity;write(path,result);return result
