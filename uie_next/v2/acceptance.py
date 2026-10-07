"""Full matrix device acceptance on fit-only engineering fixtures."""
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree
import numpy as np
import torch
from ..records import ROOT,sha,write,read,digest
from ..training import streams,initialize,optimizer,pair_objective,drop_prior
from ..checkpoint import save,atomic_torch
from ..models.candidate import Candidate
from ..models.controls import Controller
from ..math.utility import labels
from ..priors.heuristic import make_prior,rgb_prior,interventions,recompute,shift_field
from ..experiment import tensor_state_hash
from ..scientific_evaluation import prediction_utility_statistics
from .context import METHOD_ORDER,OLD
from .diagnostics import checkpoint_model,metrics,csv_write,exact_strategies
from .selection import grid,choose_grid


def acceptance(d):
    s=d.s;path=s.run/'tests/acceptance_receipts.json'
    if path.exists() and read(path).get('passed'):
        if 'ENGINEERING_ACCEPTANCE' not in s.state['completed']:
            s.complete('ENGINEERING_ACCEPTANCE',{'receipt':sha(path)})
        return read(path)
    xml=ElementTree.parse(s.run/'tests/pytest_cpu.xml').getroot()
    if any(int(x.get('errors',0))+int(x.get('failures',0)) for x in xml.iter('testsuite')):raise RuntimeError('CPU tests failed')
    d.load();d.verify_reuse();items=read(s.run/'diagnostics/checkpoint_inventory.json')
    producer=checkpoint_model(next(x for x in items if x['checkpoint_id']=='B1_004000'))
    zero=checkpoint_model(next(x for x in items if x['checkpoint_id']=='B3_000000'))
    ids_model=[r['sample_id'] for r in d.data.role('model_fit')[:8]];ids_utility=[r['sample_id'] for r in d.data.role('utility_fit')[:8]]
    scores=[];updates={};profiles={};directory=s.run/'tests/real_matrix';directory.mkdir(parents=True,exist_ok=True)
    fixture={'engineering_fixture_only':True,'source_role':'model_fit_and_utility_fit','no_real_calibration_or_sealed':True}
    with s.device_job('ENGINEERING_ACCEPTANCE_11_METHODS',1200):
        before=tensor_state_hash(d.backbone);d.backbone.train()
        assert not any(m.training for m in d.backbone.model.modules())
        model_batch=d.data.batch(ids_model,'candidate_train');batch=d.data.batch(ids_utility,'utility_train')
        with torch.no_grad():
            p,v=rgb_prior(model_batch['image']);assert torch.equal(zero(model_batch['image'],model_batch['base'],p,v),model_batch['base'])
            raw=d.backbone.model(model_batch['image'][:2]);one=torch.cat([d.backbone.model(model_batch['image'][i:i+1]) for i in range(2)])
            assert torch.allclose(raw,one,atol=1e-5,rtol=1e-4)
            fields=interventions(batch['image']);c=torch.stack([producer(batch['image'],batch['base'],fields[k]['P'],fields[k]['V']) for k in s.config['prior']['train_views']],1)
            squares=[];U=[]
            for i in range(8):
                iv=[]
                for v in range(7):
                    l=labels(batch['base'][i:i+1],c[i:i+1,v],batch['target'][i:i+1]);act=l['active']
                    if act.any():iv.append(float(l['v'][act].square().mean()))
                    U.append(float(l['U'].square().mean()))
                if iv:squares.append(float(np.mean(iv)))
            scales={'s_v':max(float(np.sqrt(np.mean(squares))),1e-3),'s_U':max(float(np.sqrt(np.mean(U))),1e-4),
                    's_e2':max(float((batch['base']-batch['target']).square().mean()),1e-4),**fixture}
        write(directory/'scales.json',scales);candidate_hash=tensor_state_hash(producer)
        for method in METHOD_ORDER:
            rng=streams(2026100700);model=initialize(Candidate if method=='B4' else lambda:Controller(method),rng['init']).to('cuda:0')
            if method=='B4':model.load_state_dict(zero.state_dict(),strict=True)
            opt,sched=optimizer(model,3000);losses=[];gradnorm=[];torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
            fixed_c=c[:4,:2].detach();bb={k:v[:4] for k,v in batch.items()};projections=[]
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                if method=='B4':
                    image=torch.cat([model_batch['image'][:4],batch['image'][:4]]);base=torch.cat([model_batch['base'][:4],batch['base'][:4]]);target=torch.cat([model_batch['target'][:4],batch['target'][:4]])
                    P,V=rgb_prior(image);P,V,_=drop_prior(P,V,rng['missing_prior']);output=model(image,base,P,V);loss=(output-target).square().mean()
                else:
                    cc=fixed_c[:,:1].repeat(1,2,1,1,1) if method=='O-NI' else fixed_c
                    loss,parts=pair_objective(model,bb['image'],bb['base'],cc,bb['target'],scales)
                loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                assert torch.isfinite(gn) and float(gn)>0 and torch.isfinite(loss)
                opt.step();sched.step();losses.append(float(loss));gradnorm.append(float(gn))
            torch.cuda.synchronize();profiles[method]={'two_updates_seconds':time.monotonic()-begin,'parameters':sum(p.numel() for p in model.parameters()),
                'peak_shared_resident_bytes':torch.cuda.max_memory_allocated(),'single_method_peak_not_measured':True,**fixture}
            assert all(p.grad is None for p in producer.parameters()) and all(p.grad is None for p in d.backbone.parameters())
            with torch.no_grad():
                candidates=[]
                for condition in ['nominal','tau_050','tau_150','ambient_green','shift_right_4']:
                    f=make_prior(bb['image'])
                    if condition.startswith('tau_'):f=recompute(bb['image'],f['d'],f['A'],f['tau']*(.5 if condition=='tau_050' else 1.5))
                    elif condition=='ambient_green':f=recompute(bb['image'],f['d'],(f['A']+bb['image'].new_tensor([0,.06,0]).view(1,3,1,1)).clamp(.05,.95))
                    elif condition=='shift_right_4':
                        dd,vv=shift_field(f['d'],4,axis=-1);f=recompute(bb['image'],dd,f['A'],V=vv)
                    candidate=producer(bb['image'],bb['base'],f['P'],f['V'])
                    pred=None
                    if method=='B4':
                        pp,vv=rgb_prior(bb['image']);endpoint=model(bb['image'],bb['base'],pp,vv)
                    else:pred=model(bb['image'],bb['base'],candidate,se2=scales['s_e2']);endpoint=pred['output']
                    lab=labels(bb['base'],candidate,bb['target']);lp=d.lpips(endpoint,bb['target']).cpu().tolist()
                    for i,m in enumerate(metrics(endpoint,bb['target'])):
                        scores.append({'method_id':method,'sample_id':ids_utility[i],'condition':condition,**m,'lpips':lp[i],
                           **prediction_utility_statistics({} if pred is None else {k:v[i:i+1] for k,v in pred.items()}, {k:v[i:i+1] for k,v in lab.items()}),**fixture})
                grid_scores=[]
                for interface,actual_step in zip([0,750,1500,2250,3000],[2,2,2,2,2]):
                    # Five-point interface only: aliases of actual two-update fixture, never scientific training.
                    for index,policy in enumerate(grid(method)):
                        if policy.get('return_base'):out=bb['base']
                        elif method=='B4':out=bb['base']+policy['alpha']*(endpoint-bb['base'])
                        else:out=model(bb['image'],bb['base'],fixed_c[:,0],se2=scales['s_e2'],**policy)['output']
                        met=metrics(out,bb['target']);base_met=metrics(bb['base'],bb['target']);lpv=d.lpips(out,bb['target']);lbase=d.lpips(bb['base'],bb['target'])
                        grid_scores.append({'step':interface,'actual_fixture_updates':actual_step,'strategy_index':index,
                            'return_base':policy.get('return_base',False),'psnr':np.mean([m['psnr'] for m in met]),
                            'ssim_delta':np.mean([m['ssim']-b['ssim'] for m,b in zip(met,base_met)]),'lpips_delta':float((lpv-lbase).mean()),**fixture})
                chosen=choose_grid(grid_scores);write(directory/(method+'_mock_grid_and_calibration.json'),{'table':grid_scores,'chosen':chosen,'mock_calibration':chosen,**fixture})
            identity={'method':method,'context':fixture,'backbone':s.config['backbone']['checkpoint_sha256']}
            save(directory/(method+'.pt'),model,opt,sched,2,identity,rng,include_cuda=True,extra=fixture)
            cpu_state=copy.deepcopy(model.state_dict());restored=Candidate() if method=='B4' else Controller(method);restored.load_state_dict(cpu_state,strict=True)
            updates[method]={'actual_updates':2,'losses':losses,'gradient_norms':gradnorm,'strict_restore':True,'checkpoint_sha256':sha(directory/(method+'.pt')),**fixture}
            del model,opt,sched
        assert tensor_state_hash(producer)==candidate_hash and tensor_state_hash(d.backbone)==before
        torch.cuda.synchronize()
    csv_write(directory/'all_methods_metrics.csv',scores);write(directory/'matrix_receipt.json',updates);write(directory/'profile.json',profiles)
    # Independent process proves no live-object or workspace-state dependency in fixture restore.
    command=[sys.executable,'-m','uie_next.v2.acceptance','--restore-directory',str(directory)]
    result=subprocess.run(command,cwd='/tmp',env={**__import__('os').environ,'PYTHONPATH':str(ROOT)},capture_output=True,text=True)
    (directory/'fresh_process.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('fresh-process fixture restore failed')
    rec={'passed':True,'CPU_test_count':sum(int(x.get('tests',0)) for x in xml.iter('testsuite')),'all_11_methods_actual_updates':updates,
         'fixed_model_fit_ids':ids_model,'fixed_utility_fit_ids':ids_utility,'zero_step_exact':True,'backbone_BN_and_parameters_frozen':True,
         'candidate_frozen':True,'all_quality_and_stress_rows':len(scores),'new_process_strict_restore':True,'fit_only_engineering_fixture':True,
         'original_backbone_not_complete_paper_model':True,'base_reuse_receipt':sha(s.run/'tests/base_reuse.json')}
    write(path,rec);s.complete('ENGINEERING_ACCEPTANCE',{'receipt':sha(path)});return rec


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--restore-directory',required=True);a=p.parse_args();directory=Path(a.restore_directory)
    for method in METHOD_ORDER:
        state=torch.load(directory/(method+'.pt'),map_location='cpu');m=Candidate() if method=='B4' else Controller(method)
        m.load_state_dict(state['model_state'],strict=True);assert state['global_step']==2;assert state['extra']['engineering_fixture_only']
    print('strict restored all 11 engineering fixtures in fresh process')
