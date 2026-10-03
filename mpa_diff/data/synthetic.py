"""Explicit synthetic TEST fixtures; never real UIEB/LSUI evidence."""
import json
from pathlib import Path
import numpy as np
import torch
from mpa_diff.physics.renderer import render,linear_to_srgb
from mpa_diff.data.manifest import FIELDS,audit
from mpa_diff.utils.io import save_image,sha256,write_json


def generate(root,manifest,n=16,seed=20260927,size=32):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    rng=np.random.RandomState(seed); rows=[]
    for i in range(n):
        # Smooth independently drawn colour fields and shapes.
        coarse=torch.tensor(rng.uniform(.08,.9,(1,3,4,4)),dtype=torch.float32)
        clean=torch.nn.functional.interpolate(coarse,size=(size,size),mode='bilinear',align_corners=False)
        distance=torch.tensor(rng.uniform(.1,1.5,(1,1,4,4)),dtype=torch.float32)
        distance=torch.nn.functional.interpolate(distance,size=(size,size),mode='bilinear',align_corners=False)
        kd=torch.tensor([.8,.4,.2])[None,:,None,None]; kb=kd*.7; ambient=torch.tensor([.12,.3,.4])[None,:,None,None]
        degraded,fields=render(clean,distance,kd,kb,ambient)
        sid='fixture_%03d'%i
        for name,tensor in (('input',linear_to_srgb(degraded)),('reference',linear_to_srgb(clean))): save_image(root/name/(sid+'.png'),tensor[0])
        (root/'truth').mkdir(exist_ok=True)
        np.savez(root/'truth'/(sid+'.npz'),distance=distance.numpy(),kappa_d=kd.numpy(),kappa_b=kb.numpy(),ambient=ambient.numpy())
        row=dict.fromkeys(FIELDS)
        row.update(sample_id=sid,dataset='synthetic_test_fixture',split='train' if i<n-4 else ('val' if i<n-2 else 'test'),image_path='input/'+sid+'.png',reference_path='reference/'+sid+'.png',scene_id=sid,source_url='generated:mpa_diff.data.synthetic',license='project-generated test fixture',image_sha256=sha256(root/'input'/(sid+'.png')),reference_sha256=sha256(root/'reference'/(sid+'.png')))
        rows.append(row)
    path=Path(manifest); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=audit(path,root); report.update(test_fixture=True,seed=seed,physics_space='linear_rgb',depth_truth='truth/*.npz; not used as deployment conditions')
    write_json(str(path)+'.audit.json',report)
    return report
