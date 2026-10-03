import json
import math
from pathlib import Path
import numpy as np
import torch
from mpa_diff.metrics.image import psnr,ssim,finite_json
from mpa_diff.utils.io import read_image,save_image,write_json


@torch.no_grad()
def evaluate_rows(model,rows,root,size,seed,output=None,save_images=True,diagnostic_limit=10):
    model.eval(); records=[]
    device=next(model.parameters()).device
    for row_index,row in enumerate(rows):
        x=read_image(Path(root)/row['image_path'],size).to(device)[None]
        pred,diag,prior=model.enhance(x,seed)
        record={'sample_id':row['sample_id'],'scene_id':row['scene_id'],'split':row['split'],'seed':seed,'nfe':diag['nfe'],'psnr':None,'ssim':None,'raw_psnr':None,'png_psnr':None,'png_ssim':None,'lpips':None,'uciqe':None,'uiqm':None,'uranker':None}
        if row['reference_path']:
            y=read_image(Path(root)/row['reference_path'],size).to(device)
            record.update(psnr=finite_json(psnr(pred[0],y)),ssim=ssim(pred[0],y),raw_psnr=finite_json(psnr(x[0],y)))
            quantized=(pred[0]*255).round()/255
            record.update(png_psnr=finite_json(psnr(quantized,y)),png_ssim=ssim(quantized,y))
        if output and save_images:
            sid=row['sample_id'].replace('/','__'); dest=Path(output)/'images'/sid
            save_image(dest/'input.png',x[0]); save_image(dest/'enhanced.png',pred[0])
            if row['reference_path']: save_image(dest/'reference.png',y)
            save_image(dest/'physical.png',prior.physical_image[0,...,:x.shape[-2],:x.shape[-1]])
            if row_index < diagnostic_limit:
                torch.save({'depth_raw':prior.depth.raw.cpu(),'distance_proxy':prior.depth.distance_proxy.cpu(),'physical_coordinate':prior.depth.physical_coordinate.cpu(),'confidence':prior.depth.confidence.cpu(),'fields':{k:v.detach().cpu() for k,v in prior.fields.items()},'sampler':diag},dest/'diagnostics.pt')
            record['prediction_path']=str((dest/'enhanced.png').resolve())
            record['reference_path']=str((Path(root)/row['reference_path']).resolve()) if row['reference_path'] else None
        records.append(record)
    scores=[r['psnr'] for r in records if isinstance(r['psnr'],(int,float))]
    perfect=sum(r['psnr']=='+inf' for r in records)
    mean_psnr=float('inf') if perfect else (float(np.mean(scores)) if scores else None)
    summary={'count':len(records),'mean_psnr':finite_json(mean_psnr) if mean_psnr is not None else None,'mean_ssim':float(np.mean([r['ssim'] for r in records if r['ssim'] is not None])) if scores or perfect else None,'perfect_reconstructions':perfect,'missing_metrics':{'lpips':'pretrained metric not provisioned','uciqe':'not yet independently audited','uiqm':'not yet independently audited','uranker':'weights unavailable'},'test_fixture':model.config['runtime']['test_fixture'],'diagnostic_policy':{'limit':diagnostic_limit,'selection':'first manifest entries, independent of scores','images':save_images},'protocol':'RGB float clipped once; SSIM valid 11x11 sigma1.5; PNG metrics separate'}
    if output:
        out=Path(output);out.mkdir(parents=True,exist_ok=True)
        (out/'per_image.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        write_json(out/'summary.json',summary)
        failures=sorted([r for r in records if isinstance(r['psnr'],(int,float))],key=lambda r:r['psnr'])[:10]
        write_json(out/'worst_cases.json',failures)
    return summary,records
