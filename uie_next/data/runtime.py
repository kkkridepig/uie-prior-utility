"""Audited role access and float32 caches for the frozen public backbone."""
from pathlib import Path

import torch

from .cache import get, put
from .manifest import image_tensor
from .roles import RoleGuard, read_roles
from ..backbones.ssuie import postprocess
from ..priors.heuristic import interventions, make_prior, rgb_prior
from ..records import digest, sha


class RuntimeData:
    def __init__(self, config, backbone, final_freeze=None):
        self.config=config
        self.run=Path(config['runtime']['run_dir'])
        self.rows=read_roles(config['data']['manifest'])
        self.guard=RoleGuard(self.rows,final_freeze)
        self.by_id=self.guard.rows
        self.backbone=backbone
        self.policy=None
        self.candidate=None
        self.candidate_hash=None

    def role(self, name):
        return sorted((r for r in self.rows if r['role']==name),key=lambda r:r['sample_id'])

    def identity(self, row, candidate_hash=None, views='base_both_policies'):
        c=self.config
        return {'input_sha256':row['input_sha256'],'preprocess':c['preprocess'],
                'backbone_commit':c['backbone']['commit'],'backbone_weight_sha256':c['backbone']['checkpoint_sha256'],
                'baseline_policy':self.policy if candidate_hash else ['clip01','official_minmax_float'],
                'candidate_weight_sha256':candidate_hash,'prior_spec':c['prior'],
                'intervention_spec':c['prior']['train_views'],'view_id':views,
                'shape':[3,256,256],'dtype':'float32'}

    def input(self, row):
        if sha(row['input_path'])!=row['input_sha256']:raise ValueError('Input file identity changed.')
        return image_tensor(row['input_path'])

    def target(self, row, operation):
        self.guard.check(row['sample_id'],operation)
        if sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('Reference file identity changed.')
        return image_tensor(row['reference_path'])

    def base_pair(self, row, image=None):
        # This method accepts only rows already admitted for the current stage.
        # Sealed inputs cannot be read ahead of the final-release guard.
        if row['role']=='sealed_eval':self.guard.check(row['sample_id'],'final')
        identity=self.identity(row)
        path=self.run/'cache/base'/('%s.pt'%digest(row['sample_id']))
        if path.exists():return get(path,identity)
        image=self.input(row) if image is None else image
        with torch.no_grad():
            raw=self.backbone.model(image[None].to('cuda:0'))
            tensors={p:postprocess(raw,p)[0].cpu() for p in ['clip01','official_minmax_float']}
        put(path,identity,tensors)
        return tensors

    def sample(self, sample_id, operation, views=False):
        row=self.guard.check(sample_id,operation)
        image=self.input(row)
        target=self.target(row,operation)
        base=self.base_pair(row,image)[self.policy]
        result={'image':image,'base':base,'target':target}
        if views:result['candidates']=self.views(row,image,base)
        return result

    def batch(self, ids, operation, views=False):
        samples=[self.sample(i,operation,views) for i in ids]
        return {k:torch.stack([s[k] for s in samples]).to('cuda:0') for k in samples[0]}

    def views(self, row, image=None, base=None):
        if self.candidate is None or self.candidate_hash is None:raise RuntimeError('Candidate must be frozen before utility cache.')
        identity=self.identity(row,self.candidate_hash,'seven_field_recomputed_views')
        path=self.run/'cache/candidates'/self.candidate_hash/('%s.pt'%digest(row['sample_id']))
        if path.exists():return get(path,identity)['candidates']
        image=self.input(row) if image is None else image
        base=self.base_pair(row,image)[self.policy] if base is None else base
        with torch.no_grad():
            image=image[None].to('cuda:0');base=base[None].to('cuda:0')
            fields=interventions(image)
            values=torch.stack([self.candidate(image,base,fields[n]['P'],fields[n]['V'])[0]
                                for n in self.config['prior']['train_views']])
        put(path,identity,{'candidates':values})
        return values.cpu()

    def real_candidate(self, image, base, model, rgb=False):
        if rgb:P,V=rgb_prior(image)
        else:
            field=make_prior(image);P,V=field['P'],field['V']
        return model(image,base,P,V)
