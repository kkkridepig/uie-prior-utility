"""One fixed image-weighted ridge per scale and content-group OOF."""
import hashlib,json,time
import numpy as np
from ..records import read,write,sha,digest
from ..v2.diagnostics import csv_write
from ..v2.statistics import paired_stats
from .moments import fixed_alpha,relative_alpha,oracle_alpha,mse_from,psnr
from .descriptors import standardize,fit_standardization


def constants(data,indices):
    A=data['A'][indices];B=data['B'][indices];m=data['mse0'][indices]
    alpha,scores=fixed_alpha(m,A,B);ref=float(np.clip(alpha,.05,.95));sA=max(float(A.mean()),1e-8)
    return {'alpha_fit_star':alpha,'alpha_ref':ref,'s_A':sA,'lambda0':.05*sA,'source_indices_sha256':digest(indices.tolist())}


def features(data,indices,scale):
    r=data['f_region'][indices].astype(np.float64);g=data['f_global'][indices].astype(np.float64)
    if scale=='global':return g[:,None,:]
    return np.concatenate([r,np.broadcast_to(g[:,None,:],r.shape)],axis=-1)


def fit(data,indices,scale):
    c=constants(data,indices);X=features(data,indices,scale);A=data['A'][indices];B=data['B'][indices]
    if scale=='global':A=A.mean(1,keepdims=True);B=B.mean(1,keepdims=True)
    C=B-c['alpha_ref']*A;sC=max(float(np.sqrt(np.mean(C*C))),1e-6);c['s_C']=sC
    flat=X.reshape(-1,X.shape[-1]);st=fit_standardization(flat);Xn=standardize(flat,st)
    Xn=np.column_stack([Xn,np.ones(len(Xn))]);y=(C/sC).reshape(-1)
    penalty=np.eye(Xn.shape[-1])*.01;penalty[-1,-1]=0
    coef=np.linalg.solve(Xn.T@Xn/len(Xn)+penalty,Xn.T@y/len(Xn))
    return {'scale':scale,'constants':c,'standardization':st,'coefficients':coef.tolist(),'ridge_penalty':.01,'intercept_penalized':False,'source_count':len(indices)}


def predict(model,data,indices,kappa=1):
    X=features(data,indices,model['scale']);Xn=standardize(X,model['standardization']);coef=np.array(model['coefficients']);c=model['constants']
    Chat=(Xn@coef[:-1]+coef[-1])*c['s_C'];A=data['A'][indices]
    if model['scale']=='global':A=A.mean(1,keepdims=True)
    alpha=relative_alpha(A,Chat,c['alpha_ref'],c['lambda0'])*kappa
    return np.broadcast_to(alpha,data['A'][indices].shape),Chat


def score(data,indices,alpha):return psnr(mse_from(data['mse0'][indices],data['A'][indices],data['B'][indices],alpha))


def summarize(rows):
    result={}
    for role in ['oof','utility_val']:
        part=[r for r in rows if r['evaluation']==role];groups=[r['group_id'] for r in part]
        base=np.array([r['J0'] for r in part]);fixed=np.array([r['FIXED_FIT'] for r in part]);entry={}
        for method in ['FIXED_FIT','RIDGE_GLOBAL','RIDGE_REGION','oracle_global','oracle_region']:
            v=np.array([r[method] for r in part]);entry[method]={'psnr':float(v.mean()),'delta_base':float((v-base).mean()),
              'harm_rate':float(np.mean(v-base<-.1)),'vs_base':paired_stats(v-base,groups),'vs_fixed':paired_stats(v-fixed,groups)}
        entry['region_minus_global']=paired_stats(np.array([r['RIDGE_REGION']-r['RIDGE_GLOBAL'] for r in part]),groups)
        entry['oracle_region_minus_global']=float(np.mean([r['oracle_region']-r['oracle_global'] for r in part]))
        result[role]=entry
    return result


def decide(summary,integrity=True):
    checks={}
    for scale in ['global','region']:
        name='RIDGE_'+scale.upper();v={}
        for role in ['oof','utility_val']:
            s=summary[role];p=s[name]
            v[role+'_fixed_gain']=p['vs_fixed']['mean_image']>=.03
            v[role+'_base_gain']=p['delta_base']>=.03
            v[role+'_harm']=p['harm_rate']<=s['FIXED_FIT']['harm_rate']+.05
        v['integrity']=integrity;checks[scale]=v
    extra={'oof_region_gain':summary['oof']['region_minus_global']['mean_image']>=.02,
           'dev_region_gain':summary['utility_val']['region_minus_global']['mean_image']>=.02,
           'dev_oracle_gain':summary['utility_val']['oracle_region_minus_global']>=.05}
    region=all(checks['region'].values()) and all(extra.values());glob=all(checks['global'].values())
    route='TRAIN_REGION' if region else 'TRAIN_GLOBAL' if glob else 'STOP_NO_PREDICTABILITY_SIGNAL'
    return {'route':route,'checks':checks,'region_extra_checks':extra,'predictable_global':glob,'predictable_region':all(checks['region'].values()),
            'not_passed':[k+':'+n for k,v in checks.items() for n,x in v.items() if not x]+['region:'+k for k,v in extra.items() if not v],
            'thresholds_are_feasibility_only':True,'sealed_eval_released':False}


def run_probes(s):
    start=time.monotonic();path=s.run/'diagnostics/sufficient_stats.npz';data=dict(np.load(path,allow_pickle=False));meta=read(s.run/'diagnostics/sufficient_stats.json');rowsmeta=meta['rows']
    fitidx=np.array([i for i,r in enumerate(rowsmeta) if r['role']=='utility_fit']);validx=np.array([i for i,r in enumerate(rowsmeta) if r['role']=='utility_val'])
    groups=sorted({rowsmeta[i]['group_id'] for i in fitidx},key=lambda g:hashlib.sha256(('v3-oof-20261007|'+g).encode()).digest());folds={g:j%5 for j,g in enumerate(groups)}
    foldrec={'folds':folds,'source_ids':[rowsmeta[i]['sample_id'] for i in fitidx],'role_sha256':sha(s.run/'roles.jsonl'),'rule':'hash-sort then j mod 5; content groups never split'}
    freeze=s.run/'diagnostics/D7_fold_manifest.json'
    if freeze.exists() and read(freeze)!=foldrec:raise ValueError('OOF fold mismatch')
    write(freeze,foldrec);table=[];models={}
    for fold in range(5):
        tr=np.array([i for i in fitidx if folds[rowsmeta[i]['group_id']]!=fold]);va=np.array([i for i in fitidx if folds[rowsmeta[i]['group_id']]==fold])
        mg=fit(data,tr,'global');mr=fit(data,tr,'region');models[str(fold)]={'global':mg,'region':mr};ar=mg['constants']['alpha_fit_star']
        predg,_=predict(mg,data,va);predr,_=predict(mr,data,va)
        vals={'J0':psnr(data['mse0'][va]),'FIXED_FIT':score(data,va,np.full_like(data['A'][va],ar)),
              'RIDGE_GLOBAL':score(data,va,predg),'RIDGE_REGION':score(data,va,predr),
              'oracle_global':score(data,va,np.broadcast_to(oracle_alpha(data['A'][va].mean(1,keepdims=True),data['B'][va].mean(1,keepdims=True)),data['A'][va].shape)),
              'oracle_region':score(data,va,oracle_alpha(data['A'][va],data['B'][va]))}
        for j,i in enumerate(va):table.append({**rowsmeta[i],'evaluation':'oof','fold':fold,'training_source_hash':digest(tr.tolist()),'alpha_fit_star':ar,**{k:float(v[j]) for k,v in vals.items()}})
    allmodels={scale:fit(data,fitidx,scale) for scale in ['global','region']}
    ar=allmodels['global']['constants']['alpha_fit_star'];ag,_=predict(allmodels['global'],data,validx);arr,_=predict(allmodels['region'],data,validx)
    vals={'J0':psnr(data['mse0'][validx]),'FIXED_FIT':score(data,validx,np.full_like(data['A'][validx],ar)),
          'RIDGE_GLOBAL':score(data,validx,ag),'RIDGE_REGION':score(data,validx,arr),
          'oracle_global':score(data,validx,np.broadcast_to(oracle_alpha(data['A'][validx].mean(1,keepdims=True),data['B'][validx].mean(1,keepdims=True)),data['A'][validx].shape)),
          'oracle_region':score(data,validx,oracle_alpha(data['A'][validx],data['B'][validx]))}
    for j,i in enumerate(validx):table.append({**rowsmeta[i],'evaluation':'utility_val','fold':None,'training_source_hash':digest(fitidx.tolist()),'alpha_fit_star':ar,**{k:float(v[j]) for k,v in vals.items()}})
    table=sorted(table,key=lambda r:(r['evaluation'],r['sample_id']));csv_write(s.run/'diagnostics/D7_oof_per_image.csv',[r for r in table if r['evaluation']=='oof']);csv_write(s.run/'diagnostics/D7_dev_per_image.csv',[r for r in table if r['evaluation']=='utility_val'])
    write(s.run/'diagnostics/D7_models.json',{'fold_models':models,'full_fit_models':allmodels,'fit_ids_hash':digest([rowsmeta[i]['sample_id'] for i in fitidx])})
    summary=summarize(table);write(s.run/'diagnostics/D7_summary.json',summary)
    decision=decide(summary);decision.update(input_hashes={str(p.relative_to(s.run)):sha(p) for p in [path,freeze,s.run/'diagnostics/D7_models.json',s.run/'diagnostics/D7_oof_per_image.csv',s.run/'diagnostics/D7_dev_per_image.csv',s.run/'source_snapshot.json']},cpu_wall_seconds=time.monotonic()-start)
    write(s.run/'stage1_decision.json',decision);s.complete('D7_OOF_AND_DEV_PROBES',{'summary':sha(s.run/'diagnostics/D7_summary.json')});s.complete('STAGE1_DECISION',decision)
    s.live(status=decision['route']);return decision
