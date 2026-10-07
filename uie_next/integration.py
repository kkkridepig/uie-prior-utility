"""Real public-backbone integration, disposable training and fresh-process resume."""
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch

from .checkpoint import atomic_torch
from .data.cache import get,put
from .experiment import tensor_state_hash
from .math.utility import labels
from .priors.heuristic import interventions
from .records import sha, write,digest
from .training import optimizer


def accept_real(e):
    receipt_path=e.run/'tests/real_integration/receipt.json'
    if receipt_path.exists() and __import__('json').loads(receipt_path.read_text()).get('passed'):return
    xml=e.run/'tests/pytest_official_resume_cpu.xml'
    suites=ElementTree.parse(xml).getroot()
    failures=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites.iter('testsuite'))
    if failures:raise RuntimeError('CPU mathematical/protocol tests must pass before real integration.')
    directory=receipt_path.parent;directory.mkdir(parents=True,exist_ok=True)
    source_ids=[r['sample_id'] for r in e.data.role('model_fit')[:8]]
    utility_ids=[r['sample_id'] for r in e.data.role('utility_fit')[:8]]
    with e.device_job('S2_real_backbone_gradient_smoke_and_resume',1800):
        before=tensor_state_hash(e.backbone.model);e.backbone.train()
        assert not any(m.training for m in e.backbone.model.modules())
        assert all(not p.requires_grad for p in e.backbone.parameters())
        model_batch=e.data.batch(source_ids,'candidate_train')
        utility_batch=e.data.batch(utility_ids,'utility_train')
        with torch.no_grad():
            raw=e.backbone.model(model_batch['image'][:1]);batch=e.backbone.model(model_batch['image'][:2])
            raw_second=e.backbone.model(model_batch['image'][1:2])
            cached=e.data.base_pair(e.data.by_id[source_ids[0]])[e.data.policy].to('cuda:0')
            assert torch.allclose(raw[0].clamp(0,1) if e.data.policy=='clip01' else e.backbone(model_batch['image'][:1])[0],cached,atol=1e-5,rtol=1e-4)
            assert torch.allclose(torch.cat([raw,raw_second]),batch,atol=1e-5,rtol=1e-4)
        rng=e.fresh_streams();rng['init'].manual_seed(2026100701)
        candidate=e.factory('B1',rng);opt,schedule=optimizer(candidate,1000)
        param_ids={id(p) for group in opt.param_groups for p in group['params']}
        assert param_ids.isdisjoint(id(p) for p in e.backbone.parameters())
        candidate_initial=tensor_state_hash(candidate);candidate_losses=[]
        for step in range(100):
            e.guard();result=e.update('B1',candidate,opt,schedule,rng,temporary=model_batch)
            candidate_losses.append(result['loss'])
        candidate_changed=tensor_state_hash(candidate)!=candidate_initial
        candidate.requires_grad_(False).eval();candidate_before=tensor_state_hash(candidate)
        with torch.no_grad():
            fields=interventions(utility_batch['image'])
            candidates=torch.stack([candidate(utility_batch['image'],utility_batch['base'],f['P'],f['V']) for f in fields.values()],1)
            v2=[];u2=[]
            for view in range(7):
                lab=labels(utility_batch['base'],candidates[:,view],utility_batch['target'])
                for image in range(8):
                    active=lab['active'][image]
                    if active.any():v2.append(float(lab['v'][image][active].square().mean()))
                    u2.append(float(lab['U'][image].square().mean()))
            scales={'s_v':max(float(np.sqrt(np.mean(v2))),1e-3),'s_U':max(float(np.sqrt(np.mean(u2))),1e-4),
                    's_e2':max(float((utility_batch['base']-utility_batch['target']).square().mean()),1e-4)}
        fixed={k:v[:4] for k,v in utility_batch.items()};fixed['candidates']=candidates[:4,:2]
        rng=e.fresh_streams();rng['init'].manual_seed(2026100702)
        controller=e.factory('O',rng);opt,schedule=optimizer(controller,1000)
        controller_initial=tensor_state_hash(controller);utility_losses=[]
        for step in range(100):
            e.guard();result=e.update('O',controller,opt,schedule,rng,temporary=fixed,scales=scales)
            utility_losses.append(result['loss'])
        assert tensor_state_hash(candidate)==candidate_before
        assert tensor_state_hash(e.backbone.model)==before
        controller_changed=tensor_state_hash(controller)!=controller_initial
        assert all(id(p) not in param_ids for p in e.backbone.parameters())
        # Cache and live paths are compared on sixteen independently permitted inputs.
        cache_errors=[]
        with torch.no_grad():
            for sample_id in source_ids+utility_ids:
                row=e.data.by_id[sample_id];image=e.data.input(row)[None].to('cuda:0')
                live=e.backbone(image)[0].cpu();cached=e.data.base_pair(row)[e.data.policy]
                cache_errors.append(float((live-cached).abs().max()))
        fixture={'model_batch':{k:v.detach().cpu() for k,v in model_batch.items()},
                 'utility_batch':{k:v.detach().cpu() for k,v in fixed.items()},'scales':scales,
                 'source_ids':source_ids,'utility_ids':utility_ids,
                 'backbone_weight_sha256':e.config['backbone']['checkpoint_sha256'],
                 'baseline_policy_hash':sha(e.run/'baseline_policy.json'),
                 'temporary_candidate_state':{k:v.cpu() for k,v in candidate.state_dict().items()},
                 'not_formal_training_or_method_weight':True}
        fixture_path=directory/'recovery_fixture.pt';atomic_torch(fixture_path,fixture)
        del candidate,controller,opt,schedule,candidates,model_batch,utility_batch,fixed
        torch.cuda.empty_cache()
        commands=[]
        for mode in ['continuous','first','resume']:
            command=[sys.executable,'-B','-m','uie_next.real_resume_probe',mode,'--directory',str(directory)]
            p=subprocess.run(command,capture_output=True,text=True,timeout=300)
            (directory/(mode+'.log')).write_text(p.stdout+p.stderr)
            commands.append({'argv':command,'exit_code':p.returncode})
            if p.returncode:raise RuntimeError('Real cached-backbone recovery test failed; inspect '+mode+'.log')
        full=torch.load(directory/'continuous.pt',map_location='cpu')
        resumed=torch.load(directory/'resumed.pt',map_location='cpu')
        error=max(float((value-resumed['model_state'][k]).abs().max()) for k,value in full['model_state'].items())
        result={'passed':candidate_changed and controller_changed and np.mean(candidate_losses[-10:])<np.mean(candidate_losses[:10])
                         and np.mean(utility_losses[-10:])<np.mean(utility_losses[:10]) and error<=1e-5 and max(cache_errors)<=1e-5,
                'official_public_simplified_backbone_used':True,'paper_complete_model':False,'device':torch.cuda.get_device_name(0),
                'candidate_updates':100,'utility_updates':100,'formal_training_updates':0,'training_samples_per_role':8,
                'candidate_initial_loss_mean10':float(np.mean(candidate_losses[:10])),'candidate_final_loss_mean10':float(np.mean(candidate_losses[-10:])),
                'utility_initial_loss_mean10':float(np.mean(utility_losses[:10])),'utility_final_loss_mean10':float(np.mean(utility_losses[-10:])),
                'candidate_parameter_update':candidate_changed,'utility_parameter_update':controller_changed,
                'T27_backbone_parameters_and_BN_unchanged':True,'T28_candidate_unchanged_during_utility':True,
                'cache_live_sample_count':16,'cache_live_max_abs_errors':cache_errors,
                'fresh_process_recovery_max_parameter_error':error,'fresh_process_recovery_same_scheduler':full['scheduler_state']==resumed['scheduler_state'],
                'commands':commands,'fixture_sha256':sha(fixture_path),'sealed_eval_accessed':False}
        write(receipt_path,result);write(directory/'loss_curves.json',{'candidate':candidate_losses,'O':utility_losses})
        if not result['passed']:raise RuntimeError('Real integration or small-sample loss/recovery acceptance failed.')


def supplemental_real(e):
    """Round-trip J1 and its labels, then exercise reference-free live inference."""
    directory=e.run/'tests/real_integration';path=directory/'supplemental_receipt.json'
    if path.exists() and __import__('json').loads(path.read_text()).get('passed'):return
    with e.device_job('S2_real_seven_view_cache_and_reference_free_deployment',300):
        fixture=torch.load(directory/'recovery_fixture.pt',map_location='cpu')
        rng=e.fresh_streams();candidate=e.factory('B1',rng)
        candidate.load_state_dict(fixture['temporary_candidate_state']);candidate.requires_grad_(False).eval()
        errors=[];label_errors=[]
        with torch.no_grad():
            for sid in fixture['source_ids']+fixture['utility_ids']:
                row=e.data.by_id[sid];operation='candidate_train' if row['role']=='model_fit' else 'utility_train'
                batch=e.data.batch([sid],operation);fields=interventions(batch['image'])
                live=torch.stack([candidate(batch['image'],batch['base'],f['P'],f['V'])[0] for f in fields.values()])
                identity=e.data.identity(row,sha(directory/'recovery_fixture.pt'),'disposable_real_cache_test')
                cached_path=directory/'candidate_cache'/('%s.pt'%digest(sid));put(cached_path,identity,{'candidates':live})
                cached=get(cached_path,identity)['candidates'].to('cuda:0')
                errors.append(float((cached-live).abs().max()))
                for vi in range(7):
                    a=labels(batch['base'],live[vi:vi+1],batch['target']);b=labels(batch['base'],cached[vi:vi+1],batch['target'])
                    label_errors.append(max(float((a[k]-b[k]).abs().max()) for k in ['r','a','b','c','u','v','U']))
        from .timing import Deployment
        # No target path, role or reference object is supplied to this adapter.
        class Context:pass
        context=Context();context.backbone=e.backbone;context.data=e.data;context.run=directory
        write(directory/'normalization_stats.json',fixture['scales'])
        controller=e.factory('O',e.fresh_streams()).eval()
        deploy=Deployment(context,{'B1':candidate,'O':controller},{'O':{'policy':{'tau':0.,'lambda':1e-4}}})
        output=deploy(batch['image'],'O')
        result={'passed':max(errors)<=1e-5 and max(label_errors)<=1e-5 and bool(torch.isfinite(output).all()),
                'sample_count':16,'seven_real_views_per_image':True,'J1_cache_max_abs':max(errors),'label_max_abs':max(label_errors),
                'reference_free_live_deployment':True,'not_formal_training_or_method_weights':True,'sealed_eval_accessed':False}
        write(path,result)
        if not result['passed']:raise RuntimeError('Real cache/label/deployment check failed.')
