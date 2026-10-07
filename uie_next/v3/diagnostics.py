"""D1-D6 read-only model diagnostics plus identity-bound sufficient statistics."""
import json,time
import numpy as np
import torch
from ..records import ROOT,sha,read,write,digest,append
from ..backbones.ssuie import load_official
from ..models.controls import Controller
from ..v2.diagnostics import checkpoint_model,inventory,metrics,csv_write
from ..v2.evaluation import apply_policy
from ..evaluation import VerifiedLPIPS
from ..math.utility import labels,decision
from ..training import pair_objective
from ..v2.statistics import paired_stats
from .context import OLD,V1,PRODUCER_SHA,Data
from .moments import geometry,oracle_alpha,regret,mse_from,psnr,spatial_variants,matched_null
from .descriptors import descriptors


def load_controller(method,step=3000):
    p=OLD/'checkpoints'/method/('step_%06d.pt'%step);side=read(str(p)+'.json')
    if sha(p)!=side['sha256']:raise ValueError('old controller hash changed')
    st=torch.load(p,map_location='cpu')
    if st['global_step']!=step:raise ValueError('old step mismatch')
    m=Controller(method);m.load_state_dict(st['model_state'],strict=True)
    return m.to('cuda:0').eval().requires_grad_(False)


def tensor_np(x):return x.detach().cpu().double().numpy()


def stat_error(pred,true):
    x=np.asarray(pred).reshape(-1);y=np.asarray(true).reshape(-1);d=x-y
    corr=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
    return {'rmse':float(np.sqrt(np.mean(d*d))),'mae':float(np.abs(d).mean()),'pearson':corr,'pearson_status':'finite' if corr is not None else 'null_constant_array','sign_agreement':float(np.mean(np.sign(x)==np.sign(y)))}


def oracle_rows(base,j,target):
    out=[]
    for block in [256,32,1]:
        g=geometry(base,j,target,block);alpha=oracle_alpha(g['A'],g['B']);m=mse_from(g['mse0'],g['A'],g['B'],alpha)
        out.append({'space':{256:'global',32:'block32',1:'pixel'}[block],'mse':float(m[0]),'psnr':float(psnr(m)[0])})
    # Explicit legacy pixel active threshold version is distinct.
    g=geometry(base,j,target,1);al=oracle_alpha(g['A'],g['B']);al=np.where(np.sqrt(g['A'])>1e-6,al,0)
    out.append({'space':'pixel_active_legacy','mse':float(mse_from(g['mse0'],g['A'],g['B'],al)[0]),'psnr':float(psnr(mse_from(g['mse0'],g['A'],g['B'],al))[0])})
    return out


class Diagnostics:
    def __init__(self,s):self.s=s;self.data=Data(s);self.backbone=None
    def load(self):
        self.backbone,self.backbone_identity=load_official(ROOT,self.s.config['backbone']['checkpoint'],self.s.config['backbone']['checkpoint_sha256'])
        self.backbone.to('cuda:0');inv=inventory();self.producer=checkpoint_model(next(i for i in inv if i['checkpoint_id']=='B1_003000'))
        self.rgb=checkpoint_model(next(i for i in inv if i['checkpoint_id']=='B3_003000'))
        self.controllers={m:load_controller(m) for m in ['O','O-NP','O-NI','O-ND']}
        self.scales=read(OLD/'normalization_stats.json');self.cal=read(OLD/'selection/calibration_selection.json')['methods']
        self.lpips=VerifiedLPIPS(ROOT/'weights/cde_v3/vgg16-397923af.pth','cuda:0')
    def accept(self):
        from ..priors.heuristic import make_prior,interventions
        probe=self.data.probe();rows=probe[:8];diffs=[];torch.cuda.synchronize();start=time.monotonic()
        with torch.no_grad():
            for r in rows:
                b=self.data.sample(r,True);I=b['image'][None].cuda();base=self.backbone(I);f=interventions(I)
                live=torch.stack([self.producer(I,base,f[k]['P'],f[k]['V'])[0] for k in self.s.config['prior']['train_views']]).cpu()
                db=float((base[0].cpu()-b['base']).abs().max());dc=float((live-b['views']).abs().max());diffs.append({'sample_id':r['sample_id'],'base_max_abs':db,'candidate_max_abs':dc})
                if not torch.allclose(base[0].cpu(),b['base'],atol=1e-5,rtol=1e-4) or not torch.allclose(live,b['views'],atol=1e-5,rtol=1e-4):raise ValueError('cache real inference mismatch')
        torch.cuda.synchronize();sec=(time.monotonic()-start)/len(rows)
        receipt={'passed':True,'strict_backbone_load':True,'strict_producer_load':True,'real_synchronize':True,'real_cache_pairs':diffs,'seconds_per_real_7_view_image':sec,'metric_identity':self.lpips.identity}
        write(self.s.run/'tests/real_recovery.json',receipt)
        predicted=1.3*(sec*len(self.data.probe())+len(self.data.role('utility_val'))*sec*2+len(self.data.probe())*sec*3+120)
        write(self.s.run/'stage1_cost_plan.json',{'real_measurement':sec,'D1_D6_predicted_seconds_safety_1_3':predicted,'stage1_cap':3600,'ridge_CPU_not_device':True,'protected_closeout_seconds':12600})
        if predicted>3600:raise __import__('uie_next.budget',fromlist=['BudgetStop']).BudgetStop('stage1 profile exceeds cap')
        self.s.complete('STAGE1_ACCEPTANCE',receipt)
    def measure(self,output,target):
        t=torch.as_tensor(np.asarray(output),dtype=torch.float32,device='cuda:0');y=torch.as_tensor(np.asarray(target),dtype=torch.float32,device='cuda:0')
        m=metrics(torch.from_numpy(np.asarray(output)),torch.from_numpy(np.asarray(target)))[0]
        with torch.no_grad():m['lpips']=float(self.lpips(t,y)[0])
        return m
    def scan(self):
        probe=self.data.probe();rows=probe+self.data.role('utility_val');done=[]
        prefix=self.s.run/'diagnostics/parts';prefix.mkdir(parents=True,exist_ok=True)
        identity={'protocol':self.s.ctx.protocol_sha256,'source':sha(self.s.run/'source_snapshot.json'),'roles':sha(self.s.run/'roles.jsonl'),'producer':PRODUCER_SHA,
                  'O':sha(OLD/'checkpoints/O/step_003000.pt'),'metric':self.lpips.identity}
        with torch.no_grad():
            for row in rows:
                self.s.guard();path=prefix/(digest(row['sample_id'])+'.json')
                if path.exists():
                    rec=read(path)
                    if rec['identity']!=identity or not rec.get('complete'):raise ValueError('diagnostic resume identity mismatch')
                    continue
                b=self.data.sample(row,True);I=b['image'][None].cuda();base=b['base'][None].cuda();j=b['candidate'][None].cuda();Y=b['target'][None].cuda()
                N=tensor_np(base);J=tensor_np(j);T=tensor_np(Y);g=geometry(N,J,T);a=g['a'];bb=g['b'];active=np.sqrt(a)>1e-6
                truth={'b':bb,'U':2*bb-a,'v':np.where(active,bb/np.where(active,np.sqrt(a),1),0)}
                preds={m:model(I,base,j,se2=self.scales['s_e2']) for m,model in self.controllers.items()}
                p=preds['O'];err={k:stat_error(tensor_np(p[k+'_hat']),v) for k,v in truth.items()}
                meta={'sample_id':row['sample_id'],'group_id':row['group_id'],'role':row['role'],'input_sha256':row['input_sha256'],'reference_sha256':row['reference_sha256']}
                orig_m=self.measure(N,T);d1=[];d2=[];d3=[];d4=[];d6=[]
                ors=oracle_rows(N,J,T);pixel=next(x['mse'] for x in ors if x['space']=='pixel')
                for lamb in [1e-4,1e-3]:
                    out,alpha=decision(base,p,p['b_hat'],0,lamb);al=tensor_np(alpha);outn=tensor_np(out)
                    trueal=np.where(active,np.clip(bb/(a+lamb),0,1),0);trueout=N+trueal*g['r'];lm=float(((outn-T)**2).mean());lt=float(((trueout-T)**2).mean())
                    rg,sq,bd=regret(a,bb,al);replacement=lm-lt;restriction=lt-pixel
                    if abs(lm-pixel-replacement-restriction)>1e-10 or abs(float(rg.mean())-(lm-pixel))>1e-8:raise ValueError('regret decomposition mismatch')
                    d1.append({**meta,'lambda':lamb,'tau':0,'policy_status':'V2_formal' if lamb==1e-3 else 'existing_diagnostic_setting',**self.measure(outn,T),
                        'delta_J0_db':float(psnr([lm])[0])-orig_m['psnr'],'alpha_mean':float(al.mean()),'alpha_q05':float(np.quantile(al,.05)),
                        'alpha_q50':float(np.quantile(al,.5)),'alpha_q95':float(np.quantile(al,.95)),'alpha_zero_rate':float((al==0).mean()),'alpha_one_rate':float((al==1).mean()),
                        'coverage_over_0_05':float((al>.05).mean()),'reference_only_true_b_regularized_mse':lt,'reference_only_true_b_regularized_psnr':float(psnr([lt])[0]),
                        'reference_only_oracle_regret_mse':lm-pixel,'reference_only_prediction_replacement_mse':replacement,'reference_only_regularization_active_loss_mse':restriction,
                        'reference_only_regret_square_mse':float(sq.mean()),'reference_only_regret_boundary_mse':float(bd.mean()),'actual_mse_gain':orig_m['mse']-lm,
                        **{'prediction_'+k+'_'+z:v for k,vv in err.items() for z,v in vv.items()}})
                    if lamb==1e-3:
                        for name,av in spatial_variants(al,a,row['sample_id']).items():
                            oo=N+av*g['r'];met=self.measure(oo,T);d2.append({**meta,'variant':name,**met,'delta_J0_db':met['psnr']-orig_m['psnr'],'alpha_mean':float(av.mean())})
                for m,pred in preds.items():
                    for setting,policy in [('common',{'tau':0,'lamb':1e-3}),('V2_legal',self.cal[m]['policy'])]:
                        out,alpha=apply_policy(m,policy,base,j,pred);met=self.measure(tensor_np(out),T)
                        d3.append({**meta,'method':m,'setting':setting,'checkpoint_step':3000,'checkpoint_sha256':sha(OLD/'checkpoints'/m/'step_003000.pt'),
                             'policy':json.dumps(policy),**met,'delta_J0_db':met['psnr']-orig_m['psnr'],'prediction_b_mae_vs_O':float((pred['b_hat']-p['b_hat']).abs().mean()),
                             'alpha_mae_vs_O_common':float((alpha-preds['O']['b_hat'].div(a.shape[0] if False else preds['O']['a']+1e-3).clamp(0,1)).abs().mean()),'output_mae_vs_O_common':float((out-(base+preds['O']['b_hat'].div(preds['O']['a']+1e-3).clamp(0,1)*(j-base))).abs().mean())})
                if row['role']=='utility_fit':
                    views=b['views'].double().numpy();nom=g['r'][0];rowsdiff=[];nomg=geometry(N,J,T,32);nomalpha=oracle_alpha(nomg['A'],nomg['B'])
                    for k,name in enumerate(self.s.config['prior']['train_views']):
                        gj=geometry(N,views[k:k+1],T,32);rr=gj['r'][0];delta=rr-nom;rowsdiff.append(delta.reshape(-1))
                        cos=float((rr*nom).sum()/max(np.linalg.norm(rr)*np.linalg.norm(nom),1e-30));rblock=rr.reshape(3,8,32,8,32).transpose(1,3,0,2,4).reshape(64,-1);nblock=nom.reshape(3,8,32,8,32).transpose(1,3,0,2,4).reshape(64,-1)
                        bc=(rblock*nblock).sum(1)/np.maximum(np.linalg.norm(rblock,axis=1)*np.linalg.norm(nblock,axis=1),1e-30)
                        du=(2*gj['b']-gj['a'])-(2*bb-a);dB=gj['B']-nomg['B'];da=oracle_alpha(gj['A'],gj['B'])-nomalpha
                        d4.append({**meta,'view':name,'residual_rms':float(np.sqrt((rr*rr).mean())),'direction_cosine':cos,'block_cosine_mean':float(bc.mean()),'block_cosine_q05':float(np.quantile(bc,.05)),
                          'U_difference_rms':float(np.sqrt((du*du).mean())),'U_difference_rms_over_s_U':float(np.sqrt((du*du).mean())/self.scales['s_U']),
                          'U_difference_mean':float(du.mean()),'U_difference_q05':float(np.quantile(du,.05)),'U_difference_q95':float(np.quantile(du,.95)),
                          'B_difference_mean':float(dB.mean()),'B_difference_abs_mean':float(np.abs(dB).mean()),'B_difference_positive_rate':float((dB>0).mean()),
                          'optimal_region_alpha_difference_mean':float(da.mean()),'optimal_region_alpha_difference_abs_mean':float(np.abs(da).mean()),
                          'nonmicro_target_difference_fraction':float((np.abs(du)>.01*self.scales['s_U']).mean()),'nonmicro_threshold_descriptive_only':.01*self.scales['s_U']})
                    mat=np.stack(rowsdiff[1:]);eig=np.maximum(np.linalg.eigvalsh(mat@mat.T),0)[::-1];sv=np.sqrt(eig);probs=eig/eig.sum() if eig.sum()>0 else eig
                    rank=float(np.exp(-np.sum(probs[probs>0]*np.log(probs[probs>0])))) if eig.sum()>0 else 0.
                    for entry in d4:entry.update(intervention_singular_values=json.dumps(sv.tolist()),effective_rank=rank)
                jr=self.data.legacy.real_candidate(I,base,self.rgb,rgb=True);rgb=tensor_np(jr)
                for candidate,output in [('B1_original',J),('B3_003000',rgb)]:
                    for o in oracle_rows(N,output,T):d6.append({**meta,'direction':candidate,'null_index':None,'retained_energy_ratio':1.,'eta_zero_rate':0.,**o})
                for k in [1,2]:
                    rn,rm,eta=matched_null(N[0],g['r'][0],row['sample_id'],k);energy=float((rm*rm).sum()/max((g['r'][0]**2).sum(),1e-30))
                    if not np.allclose(np.mean(rn*rn,0),np.mean(rm*rm,0),atol=1e-14,rtol=1e-10):raise ValueError('null energy mismatch')
                    for name,rr in [('null',rn),('matched_original',rm)]:
                        for o in oracle_rows(N,N+rr[None],T):d6.append({**meta,'direction':name,'null_index':k,'retained_energy_ratio':energy,'eta_zero_rate':float((eta==0).mean()),**o})
                write(path,{'identity':identity,'complete':True,'base':orig_m,'oracles':ors,'meta':meta,'D1':d1,'D2':d2,'D3':d3,'D4':d4,'D6':d6})
                done.append(row['sample_id'])
                if len(done)%16==0:self.s.live(diagnostic_images_completed=len(done),status='D1_D6_DIAGNOSTICS')
        allrec=[read(prefix/(digest(r['sample_id'])+'.json')) for r in rows]
        for name,filename in [('D1','D1_prediction_regret_per_image.csv'),('D2','D2_spatial_controls.csv'),('D3','D3_common_policy_ablations.csv'),('D4','D4_intervention_information.csv'),('D6','D6_matched_null_and_rgb_oracles.csv')]:csv_write(self.s.run/'diagnostics'/filename,[x for rec in allrec for x in rec[name]])
        dev=[r for r in allrec if r['meta']['role']=='utility_val'];base=float(np.mean([r['base']['psnr'] for r in dev]));ov=float(np.mean([x['psnr'] for r in dev for x in r['D1'] if x['lambda']==1e-3]))
        if abs(base-24.3558853231)>1e-4 or abs(ov-24.3693422345)>1e-4:raise ValueError('V2 frozen PSNR regression failed')
        write(self.s.run/'tests/V2_real_regression.json',{'passed':True,'J0':base,'O':ov,'tolerance_db':1e-4,'n_images':len(dev),'references_used_only_in_diagnostics':True})
        return allrec
    def gradients(self):
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        manifest=read(self.s.run/'diagnostics/probe_manifest.json');rows=[r for r in self.data.probe() if r['group_id'] in manifest['gradient_groups']];rows=sorted(rows,key=lambda r:r['sample_id']);m=self.controllers['O'];m.requires_grad_(True);before={k:v.clone() for k,v in m.state_dict().items()};logs=[]
        for n in range(0,len(rows),4):
            self.s.guard();part=rows[n:n+4];batch=[self.data.sample(r,True) for r in part];I=torch.stack([b['image'] for b in batch]).cuda();base=torch.stack([b['base'] for b in batch]).cuda();Y=torch.stack([b['target'] for b in batch]).cuda();cand=torch.stack([b['views'][:2] for b in batch]).cuda()
            total,parts=pair_objective(m,I,base,cand,Y,self.scales)
            # Reconstruct projection directly, never by subtracting two large
            # branches from total or differentiating detached log values.
            pp=[m(I,base,cand[:,k],se2=self.scales['s_e2']) for k in range(2)]
            ll=[labels(base,cand[:,k],Y) for k in range(2)]
            active=torch.stack([l['active'] for l in ll],1).float()
            vh=torch.stack([p['v_hat'] for p in pp],1);vt=torch.stack([l['v'] for l in ll],1)
            h=torch.nn.functional.huber_loss((vh-vt)/max(float(self.scales['s_v']),1e-3),torch.zeros_like(vh),reduction='none')*active
            counts=active.flatten(2).sum(-1);valid=(counts>0).float()
            per_view=h.flatten(2).sum(-1)/counts.clamp_min(1.)
            per_image=(per_view*valid).sum(1)/valid.sum(1).clamp_min(1.)
            image_valid=(valid.sum(1)>0).float();proj=(per_image*image_valid).sum()/image_valid.sum().clamp_min(1.)
            live=[proj,parts['pair_live'],parts['dec_live']];weights=[1,.25,.1];params=list(m.parameters())
            grads=[torch.cat([g.reshape(-1) for g in torch.autograd.grad(l,params,retain_graph=True)]) for l in live];wg=[g*w for g,w in zip(grads,weights)];summed=sum(wg);gt=torch.cat([g.reshape(-1) for g in torch.autograd.grad(total,params)])
            diff=float((summed-gt).abs().max());norm=float(gt.norm());clip=min(1.,1/(norm+1e-6));cos={str(i)+'_'+str(j):float(torch.dot(grads[i],grads[j])/(grads[i].norm()*grads[j].norm()).clamp_min(1e-30)) for i in range(3) for j in range(i+1,3)}
            if not torch.allclose(summed,gt,atol=1e-5,rtol=1e-4):
                write(self.s.run/'tests/gradient_sum_failure.json',{'max_abs':diff,'total_norm':norm,'sum_norm':float(summed.norm()),'error_norm':float((summed-gt).norm()),'max_total_abs':float(gt.abs().max()),'violating_elements':int(((summed-gt).abs()>1e-5+1e-4*gt.abs()).sum())})
                raise ValueError('gradient sum mismatch')
            logs.append({'sample_ids':[r['sample_id'] for r in part],'pair':['nominal','tau_075'],'raw_loss':[float(x) for x in live],'weighted_loss':[float(x)*w for x,w in zip(live,weights)],
              'raw_gradient_norm':[float(g.norm()) for g in grads],'weighted_gradient_norm':[float(g.norm()) for g in wg],'gradient_cosines':cos,'total_gradient_norm':norm,
              'cancellation_ratio':norm/max(sum(float(g.norm()) for g in wg),1e-30),'common_clip_coefficient':clip,'joint_clipped_component_norms':[float(g.norm())*clip for g in wg],
              'sum_max_abs':diff,'optimizer_steps':0,'old_weights_unchanged':True})
        if any(not torch.equal(v,before[k]) for k,v in m.state_dict().items()):raise ValueError('diagnostic changed old model')
        m.requires_grad_(False)
        from ..records import jsonl
        jsonl(self.s.run/'diagnostics/D5_loss_gradient_components.jsonl',logs)
        return logs


def extract_stats(s,data):
    path=s.run/'diagnostics/sufficient_stats.npz';meta=s.run/'diagnostics/sufficient_stats.json';rows=data.role('utility_fit')+data.role('utility_val')
    identity={'roles':sha(s.run/'roles.jsonl'),'producer':PRODUCER_SHA,'source_snapshot':sha(s.run/'source_snapshot.json'),'preprocess':s.config['preprocess'],'descriptor_version':sha(ROOT/'uie_next/v3/descriptors.py'),'formula':sha(ROOT/'uie_next/v3/moments.py'),'dtype':'CPU_float64_summaries_float32_descriptors'}
    if path.exists():
        m=read(meta)
        if m['identity']!=identity or m['npz_sha256']!=sha(path) or len(m['rows'])!=579:raise ValueError('stats identity mismatch')
        return
    arrays={k:[] for k in ['A','B','mse0','f_region','f_global']};rmeta=[];start=time.monotonic();torch.set_num_threads(2)
    for row in rows:
        b=data.sample(row);I=b['image'][None].numpy();N=b['base'][None].numpy();J=b['candidate'][None].numpy();T=b['target'][None].numpy();g=geometry(N,J,T,32);fr,fg=descriptors(I,N,J)
        arrays['A'].append(g['A'].reshape(64));arrays['B'].append(g['B'].reshape(64));arrays['mse0'].append(g['mse0'][0]);arrays['f_region'].append(fr[0].reshape(26,64).T);arrays['f_global'].append(fg[0,:,0,0])
        rmeta.append({k:row[k] for k in ['sample_id','group_id','role','input_sha256','reference_sha256']})
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**{k:np.array(v) for k,v in arrays.items()});tmp.replace(path)
    write(meta,{'identity':identity,'rows':rmeta,'npz_sha256':sha(path),'cpu_wall_seconds':time.monotonic()-start,'device_lease_held':False})


def ridge_quality(s):
    """Supplementary full-reference quality, without changing D7's fixed gate."""
    from .ridge_probe import predict
    from .moments import expand
    path=s.run/'diagnostics/D7_quality_per_image.csv'
    if path.exists():return
    data=Data(s);meta=read(s.run/'diagnostics/sufficient_stats.json');stats=dict(np.load(s.run/'diagnostics/sufficient_stats.npz'));models=read(s.run/'diagnostics/D7_models.json');folds=read(s.run/'diagnostics/D7_fold_manifest.json')['folds'];out=[]
    with s.device_job('D7_PAIRED_OOF_AND_DEV_SSIM_LPIPS',estimate=600,stage=1):
        lp=VerifiedLPIPS(ROOT/'weights/cde_v3/vgg16-397923af.pth','cuda:0')
        with torch.no_grad():
            for i,rowmeta in enumerate(meta['rows']):
                s.guard();row=data.by_id[rowmeta['sample_id']];b=data.sample(row);N=b['base'][None].double().numpy();J=b['candidate'][None].double().numpy();T=b['target'][None].double().numpy()
                mm=models['fold_models'][str(folds[row['group_id']])] if row['role']=='utility_fit' else models['full_fit_models']
                ag,_=predict(mm['global'],stats,np.array([i]));ar,_=predict(mm['region'],stats,np.array([i]));fixed=mm['global']['constants']['alpha_fit_star']
                outputs={'J0':N,'FIXED_FIT':N+fixed*(J-N),'RIDGE_GLOBAL':N+expand(ag.reshape(1,1,8,8),32)*(J-N),'RIDGE_REGION':N+expand(ar.reshape(1,1,8,8),32)*(J-N)}
                tt=torch.tensor(T,dtype=torch.float32,device='cuda:0');stack=torch.tensor(np.concatenate(list(outputs.values())),dtype=torch.float32,device='cuda:0');lps=lp(stack,tt.expand_as(stack)).cpu().tolist()
                for k,(name,oo) in enumerate(outputs.items()):out.append({**rowmeta,'evaluation':'oof' if row['role']=='utility_fit' else 'utility_val','method':name,**metrics(torch.from_numpy(oo),torch.from_numpy(T))[0],'lpips':lps[k]})
    csv_write(path,out)
    summaries={};group_rows=[]
    for role in ['oof','utility_val']:
        for method in ['J0','FIXED_FIT','RIDGE_GLOBAL','RIDGE_REGION']:
            p=[r for r in out if r['evaluation']==role and r['method']==method];grouped={}
            for r in p:grouped.setdefault(r['group_id'],[]).append(r)
            for group,rr in grouped.items():group_rows.append({'evaluation':role,'method':method,'group_id':group,'n_images':len(rr),**{m:float(np.mean([x[m] for x in rr])) for m in ['psnr','ssim','lpips']},'group_type':'inherited_content_proxy_not_verified_scene'})
            summaries[role+'|'+method]={'image_weighted':{m:float(np.mean([x[m] for x in p])) for m in ['psnr','ssim','lpips']},'group_equal_sensitivity':{m:float(np.mean([np.mean([x[m] for x in rr]) for rr in grouped.values()])) for m in ['psnr','ssim','lpips']},'seed':20261007,'new_neural_training_seeds':0}
    csv_write(s.run/'diagnostics/D7_quality_per_group.csv',group_rows);write(s.run/'diagnostics/D7_quality_summary.json',summaries)
