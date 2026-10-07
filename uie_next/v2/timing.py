"""Fresh isolated process with no reference or metric network for deployment timing."""
import argparse
import math
import time
from pathlib import Path
import numpy as np
import torch
import cv2
from ..records import ROOT,read,write,sha
from ..backbones.ssuie import load_official,postprocess
from ..data.manifest import image_tensor
from ..priors.heuristic import make_prior,rgb_prior
from ..models.controls import Controller
from .context import State
from .diagnostics import checkpoint_model
from .evaluation import apply_policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);a=p.parse_args();s=State();method=a.method
    torch.set_num_threads(2);backbone,identity=load_official(ROOT,s.config['backbone']['checkpoint'],s.config['backbone']['checkpoint_sha256']);backbone.to('cuda:0')
    registry=read(s.run/'method_registry.json') if (s.run/'method_registry.json').exists() else None
    controller=None;candidate=None;policy={};entry=None
    if registry:
        reg=registry[method];sel=read(s.run/'selection/calibration_selection.json')['methods'][method];policy=sel['policy'];entry=reg['candidate']
        if entry:candidate=checkpoint_model(entry)
        if reg['controller']:
            controller=Controller(method).to('cuda:0');controller.load_state_dict(torch.load(reg['controller']['path'],map_location='cpu')['model_state'],strict=True);controller.eval().requires_grad_(False)
    elif method!='B0_clip01':
        head='B1' if method.startswith('B1') else 'B3';stand=read(s.run/'selection/standalone_selection.json')[head]
        entry=next(i for i in read(s.run/'diagnostics/checkpoint_inventory.json') if i['checkpoint_id']==stand['checkpoint_id']);candidate=checkpoint_model(entry)
    rows=sorted([__import__('json').loads(x) for x in (s.run/'roles.jsonl').read_text().splitlines() if __import__('json').loads(x)['role']=='model_fit'],key=lambda r:r['sample_id'])[:20]
    def infer(image):
        base=backbone(image)
        if policy.get('return_base') or method=='B0_clip01':return base
        if method=='B0_official_minmax_float':return postprocess(backbone.model(image),'official_minmax_float')
        if entry['head']=='B3':P,V=rgb_prior(image)
        else:f=make_prior(image);P,V=f['P'],f['V']
        j=candidate(image,base,P,V)
        if controller:pred=controller(image,base,j,se2=read(s.run/'normalization_stats.json')['s_e2'])
        else:pred=None
        return apply_policy(method,policy,base,j,pred)[0]
    raw=[];service=[];image=image_tensor(rows[0]['input_path'])[None].to('cuda:0');torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(20):infer(image)
        for _ in range(100):
            torch.cuda.synchronize();start=time.perf_counter();infer(image);torch.cuda.synchronize();raw.append(time.perf_counter()-start)
        for n,row in enumerate(rows):
            torch.cuda.synchronize();start=time.perf_counter();x=image_tensor(row['input_path'])[None].to('cuda:0');out=infer(x);torch.cuda.synchronize()
            arr=(out[0].cpu().permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8);path=s.run/'timing/service_png'/method/(str(n)+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(path),cv2.cvtColor(arr,cv2.COLOR_RGB2BGR));service.append(time.perf_counter()-start)
    summarize=lambda vs:{'mean':float(np.mean(vs)),'p50':float(np.median(vs)),'p95':float(np.quantile(vs,.95))}
    write(s.run/'timing'/(method+'.json'),{'method':method,'model':summarize(raw),'service':summarize(service),'model_raw_seconds':raw,'service_raw_seconds':service,
       'peak_single_process_allocated_bytes':torch.cuda.max_memory_allocated(),'parameters_backbone':sum(p.numel() for p in backbone.parameters()),
       'parameters_candidate':sum(p.numel() for p in candidate.parameters()) if candidate else 0,'parameters_controller':sum(p.numel() for p in controller.parameters()) if controller else 0,
       'official_weight':identity,'policy':policy,'no_reference_reads':True,'no_metric_network_loaded':True,'no_cache_used':True,'device':torch.cuda.get_device_name(0)})

if __name__=='__main__':main()
