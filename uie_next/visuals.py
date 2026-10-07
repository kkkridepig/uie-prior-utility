"""Fixed case selection and common-scale offline diagnostic panels."""
import hashlib
import math

import numpy as np
import torch
from PIL import Image,ImageDraw

from .records import digest,jsonl,read,sha,write


def rgb(value):
    a=value.detach().cpu().permute(1,2,0).numpy()
    return Image.fromarray(np.round(np.clip(a,0,1)*255).astype(np.uint8))


def signed(value,scale):
    a=value.detach().cpu().squeeze().numpy()
    v=np.clip(a/scale,-1,1)
    # A shared diverging range: blue negative, white zero, red positive.
    result=np.stack([1-np.maximum(-v,0),1-np.abs(v),1-np.maximum(v,0)],-1)
    return Image.fromarray(np.round(result*255).astype(np.uint8))


def panel(images):
    columns=6;rows=math.ceil(len(images)/columns)
    canvas=Image.new('RGB',(columns*256,rows*280),'white');draw=ImageDraw.Draw(canvas)
    for n,(label,im) in enumerate(images):
        x=(n%columns)*256;y=(n//columns)*280
        draw.text((x+4,y+4),label,fill='black');canvas.paste(im.resize((256,256)),(x,y+24))
    return canvas


def case_ids(rows,first,second):
    a={r['sample_id']:r for r in rows if r['method']==first};b={r['sample_id']:r for r in rows if r['method']==second}
    if set(a)!=set(b):raise ValueError('Case-selection pairing mismatch.')
    score=sorted((a[s]['psnr']-b[s]['psnr'],s) for s in a)
    median=float(np.median([x[0] for x in score]));chosen={}
    for reason,subset in [('worst_5',score[:5]),('best_5',score[-5:]),
                          ('nearest_median_5',sorted(score,key=lambda x:(abs(x[0]-median),x[1]))[:5]),
                          ('fixed_hash_5',sorted(score,key=lambda x:hashlib.sha256(x[1].encode()).hexdigest())[:5])]:
        for delta,sid in subset:
            chosen.setdefault(sid,{'sample_id':sid,'reasons':[],'delta_db':delta})['reasons'].append(reason)
    return list(chosen.values())


def export_baseline(e):
    """Twenty preselected model_val images; no sealed access, no GPU work."""
    from .data.manifest import image_tensor
    path=e.run/'figures/baseline/selection.json'
    if path.exists():return read(path)
    roles=[r for r in __import__('uie_next.data.roles',fromlist=['read_roles']).read_roles(e.config['data']['manifest']) if r['role']=='model_val']
    rows=sorted(roles,key=lambda r:r['sample_id'])[:20];cases=[]
    path.parent.mkdir(parents=True,exist_ok=True)
    for n,row in enumerate(rows):
        state=torch.load(e.run/'cache/base'/('%s.pt'%digest(row['sample_id'])),map_location='cpu')
        bases=state['tensors']
        pairs=[('Input',rgb(image_tensor(row['input_path']))),('Reference',rgb(image_tensor(row['reference_path']))),
               ('B0 clip01',rgb(bases['clip01'])),('B0 official minmax float',rgb(bases['official_minmax_float']))]
        out=path.parent/('%02d.png'%n);panel(pairs).save(out)
        cases.append({'sample_id':row['sample_id'],'role':'model_val','path':str(out),'sha256':sha(out),
                      'selection_rule':'first 20 stable IDs, independent of quality','resolution':[256,256]})
    write(path,{'cases':cases,'scientific_scope':'baseline preprocessing diagnostics, not method evidence'})
    return read(path)


def export_candidates(e,rows,rgb_head):
    from .math.utility import labels,oracle
    path=e.run/'figures/candidates/case_selection.json';path.parent.mkdir(parents=True,exist_ok=True)
    cases=case_ids(rows,'B1','B0')
    with torch.no_grad():
        for n,case in enumerate(cases):
            row=e.data.by_id[case['sample_id']]
            batch=e.data.batch([row['sample_id']],'candidate_diagnostic',True)
            j1=batch['candidates'][:,0];base=batch['base'];target=batch['target']
            jrgb=e.data.real_candidate(batch['image'],base,rgb_head,True)
            block,alpha=oracle(base,j1,target,32);lab=labels(base,j1,target)
            images=[('Input',rgb(batch['image'][0])),('Reference',rgb(target[0])),('J0',rgb(base[0])),('Prior candidate B1',rgb(j1[0])),
                    ('RGB candidate B3',rgb(jrgb[0])),('Oracle block32 (not deployable)',rgb(block[0])),
                    ('Actual r [-.25,.25]',rgb(((j1-base)/.25*.5+.5).clamp(0,1)[0])),
                    ('True U [-.01,.01]',signed(lab['U'][0],.01)),('Oracle alpha [0,1]',rgb(alpha[0].expand(3,-1,-1)))]
            out=path.parent/('%02d.png'%n);panel(images).save(out)
            case.update(panel=str(out),sha256=sha(out),role='utility_val',uses_reference_for_offline_oracle=True)
    write(path,{'cases':cases,'selection_pair':['B1','B0'],'scientific_scope':'candidate diagnostics only; not O method evidence'})
    return read(path)


def export_panels(e,models,rows,role):
    from .math.utility import labels
    from .scientific_evaluation import image_features,get_output
    operation='final' if role=='sealed_eval' else 'develop'
    if operation=='final':e.data.guard.check(e.data.role(role)[0]['sample_id'],'final')
    selection=read(e.run/'calibration_selection.json');primary=selection['primary_control']
    path=e.run/('figures/'+role+'/case_selection.json');path.parent.mkdir(parents=True,exist_ok=True)
    cases=case_ids(rows,'O',primary)
    with torch.no_grad():
        for n,case in enumerate(cases):
            row=e.data.by_id[case['sample_id']]
            batch,j1,rgb_head,matched,preds,bases=image_features(e,models,row,operation)
            o,alpha=get_output(e,'O',batch,j1,rgb_head,matched,preds,bases,selection['methods']['O']['policy'])
            control,_=get_output(e,primary,batch,j1,rgb_head,matched,preds,bases,selection['methods'][primary]['policy'])
            lab=labels(batch['base'],j1,batch['target']);vhat=preds['O']['v_hat']
            error=(o-batch['target']).abs().mean(1,keepdim=True).div(.25).clamp(0,1).expand(-1,3,-1,-1)
            residual=(j1-batch['base']).div(.25).mul(.5).add(.5).clamp(0,1)
            zoom=rgb(o[0]).crop((96,96,160,160)).resize((256,256),Image.NEAREST)
            images=[('Input',rgb(batch['image'][0])),('Reference',rgb(batch['target'][0])),('SS-UIE J0',rgb(batch['base'][0])),
                    ('Candidate J1',rgb(j1[0])),('Primary '+primary,rgb(control[0])),('O',rgb(o[0])),
                    ('O fixed center zoom',zoom),('O absolute error [0,.25]',rgb(error[0])),('Actual r [-.25,.25]',rgb(residual[0])),
                    ('alpha [0,1]',rgb(alpha[0].expand(3,-1,-1))),('v_hat [-.25,.25]',signed(vhat[0],.25)),('true v [-.25,.25]',signed(lab['v'][0],.25))]
            out=path.parent/('%02d.png'%n);panel(images).save(out)
            case.update(panel=str(out),sha256=sha(out),primary_control=primary,resolution=[256,256],
                        reference_is_offline_only=True,r_range=[-.25,.25],v_range=[-.25,.25],error_range=[0,.25])
    write(path,{'role':role,'cases':cases,'rules':'five best/worst/median/hash IDs, then deduplicate; fixed center crop; shared display scales'})
    return read(path)
