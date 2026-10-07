"""Measure every prescribed job and freeze common update counts before results."""
import copy
import shutil
import time

import numpy as np
import torch
import yaml

from .experiment import ScientificStop, average_psnr
from .models.candidate import Candidate
from .priors.heuristic import interventions
from .records import ROOT, jsonl, sha, write,read,digest
from .training import ORDER, optimizer


def profile_and_freeze(e,profile_only=False):
    if (e.run/'protocol_resolved.yaml').exists():return
    profile=e.run/'profile_measurements.json'
    measurement_identity={'backbone':e.config['backbone']['checkpoint_sha256'],'roles':sha(e.run/'roles.jsonl'),
                          'policy':sha(e.run/'baseline_policy.json'),
                          'compute_source':{n:sha(ROOT/'uie_next'/n) for n in ['training.py','losses.py','experiment.py','models/candidate.py','models/controls.py','models/utility.py']}}
    if profile.exists() and read(profile).get('measurement_identity')!=measurement_identity:
        old=read(profile);directory=e.run/'audit_history/profile_before_dispatch';directory.mkdir(parents=True,exist_ok=True)
        profile.replace(directory/('%s.json'%digest(old)))
    if not profile.exists():
        with e.device_job('S3_full_control_matrix_throughput_profile',3600):
            model_ids=[r['sample_id'] for r in e.data.role('model_fit')[:16]]
            utility_ids=[r['sample_id'] for r in e.data.role('utility_fit')[:16]]
            model=e.data.batch(model_ids[:8],'candidate_train')
            utility=e.data.batch(utility_ids[:8],'utility_train')
            rng=e.fresh_streams();rng['init'].manual_seed(2026100791)
            temporary=e.factory('B1',rng)
            with torch.no_grad():
                # A disposable nonzero candidate is only a workload fixture.
                torch.nn.init.normal_(temporary.net.output.weight,std=.005)
                views=interventions(utility['image'])
                candidates=torch.stack([temporary(utility['image'],utility['base'],v['P'],v['V']) for v in views.values()],1)
            paired={k:v[:4] for k,v in utility.items()};paired['candidates']=candidates[:4,:2]
            matched={k:torch.cat([model[k][:4],utility[k][:4]]) for k in model}
            scales={'s_v':.05,'s_U':.003,'s_e2':max(float((utility['base']-utility['target']).square().mean()),1e-4)}
            raw=[];jobs={}
            for method in ['B1','B3']+ORDER:
                e.guard();rng=e.fresh_streams();rng['init'].manual_seed(2026100792)
                head=e.factory(method,rng);opt,schedule=optimizer(head,1000)
                batch=matched if method=='B4' else model if method in ['B1','B3'] else paired
                for step in range(10):e.update(method,head,opt,schedule,rng,temporary=batch,scales=scales)
                torch.cuda.synchronize();seconds=[];torch.cuda.reset_peak_memory_stats()
                for step in range(30):
                    begin=time.perf_counter();e.update(method,head,opt,schedule,rng,temporary=batch,scales=scales)
                    torch.cuda.synchronize();seconds.append(time.perf_counter()-begin)
                    raw.append({'method':method,'measured_update':step+1,'seconds':seconds[-1],'effective_sources':8 if method in ['B1','B3','B4'] else 4})
                # Include real cache reads, decoding, H2D transfer and forward in
                # the validation estimate; throughput adds its measured I/O cost.
                ids=model_ids if method in ['B1','B3','B4'] else utility_ids
                operation='candidate_train' if method in ['B1','B3','B4'] else 'utility_train'
                io=0.;validation=0.
                with torch.no_grad():
                    for sample_id in ids:
                        begin=time.perf_counter();small=e.data.batch([sample_id],operation);torch.cuda.synchronize();io+=time.perf_counter()-begin
                        begin=time.perf_counter()
                        if method in ['B1','B3','B4']:out=e.data.real_candidate(small['image'],small['base'],head,method in ['B3','B4'])
                        else:
                            f=interventions(small['image'])['nominal'];j1=temporary(small['image'],small['base'],f['P'],f['V'])
                            out=head(small['image'],small['base'],j1,se2=scales['s_e2'])['output']
                        average_psnr(out,small['target']);torch.cuda.synchronize();validation+=time.perf_counter()-begin
                jobs[method]={'seconds_per_update_compute':float(np.mean(seconds)),
                              'seconds_per_update_io_estimate':io/16*(8 if method in ['B1','B3','B4'] else 4),
                              'seconds_per_update':float(np.mean(seconds))+io/16*(8 if method in ['B1','B3','B4'] else 4),
                              'validation_seconds_per_image':(io+validation)/16,
                              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                              'parameters':sum(p.numel() for p in head.parameters()),'warmup_updates':10,'measured_updates':30,
                              'validation_measured_images':16,'temporary_not_formal_training':True}
                del head,opt,schedule;torch.cuda.empty_cache();e.budget.tick();e.live(profile_job=method)
            image=model['image'][:1]
            with torch.no_grad():
                for _ in range(10):e.backbone(image)
                torch.cuda.synchronize();base_seconds=[]
                for _ in range(30):
                    begin=time.perf_counter();e.backbone(image);torch.cuda.synchronize();base_seconds.append(time.perf_counter()-begin)
                cache_seconds=[];metric_seconds=[]
                for sample_id in utility_ids:
                    row=e.data.by_id[sample_id];x=e.data.input(row)[None].to('cuda:0');y=e.data.target(row,'utility_train')[None].to('cuda:0')
                    begin=time.perf_counter();j0=e.backbone(x);fields=interventions(x)
                    for f in fields.values():temporary(x,j0,f['P'],f['V'])
                    torch.cuda.synchronize();cache_seconds.append(time.perf_counter()-begin)
                    begin=time.perf_counter();e.metrics(j0,y);torch.cuda.synchronize();metric_seconds.append(time.perf_counter()-begin)
            result={'jobs':jobs,'measurement_identity':measurement_identity,'backbone_seconds_per_image':float(np.mean(base_seconds)),
                    'seven_view_seconds_per_image':float(np.mean(cache_seconds)),
                    'full_metric_seconds_per_image':float(np.mean(metric_seconds)),
                    'base_warmup':10,'base_measurements':30,'cache_and_metrics_measured_images':16,
                    'candidate_fixture_is_disposable':True,'device':torch.cuda.get_device_name(0),
                    'batch_and_precision':'prescribed effective sources, float32','safety_factor':1.3}
            write(profile,result);jsonl(e.run/'timing/raw_training_profile.jsonl',raw)
            del temporary,candidates,paired,model,utility,matched
    if profile_only:return __import__('json').loads(profile.read_text())
    measured=__import__('json').loads(profile.read_text());factor=1.3
    # Account for the actual seven-view float cache read.  The workload fixture
    # is disposable and does not enter formal labels or normalization scales.
    fixture=e.run/'tests/real_integration/recovery_fixture.pt'
    begin=time.perf_counter()
    for _ in range(16):torch.load(fixture,map_location='cpu')
    payload=torch.load(fixture,map_location='cpu')
    measured_read=(time.perf_counter()-begin)/16
    fixture_images=payload['utility_batch']['candidates'].shape[0]
    candidate_read=measured_read/max(fixture_images,1)*7/2
    counts={name:len(e.data.role(name)) for name in ['model_fit','model_val','utility_fit','utility_val','calibration','sealed_eval']}
    jobs=copy.deepcopy(measured['jobs'])
    for m in ORDER:
        if m!='B4':
            jobs[m]['candidate_cache_read_seconds_per_source_estimate']=candidate_read
            jobs[m]['seconds_per_update']+=4*candidate_read
            jobs[m]['validation_seconds_per_image']+=candidate_read
    used=__import__('json').loads((e.run/'budget.json').read_text())['used_device_seconds']
    base_images=sum(counts.values())
    candidate_images=sum(counts[n] for n in ['utility_fit','utility_val','calibration','sealed_eval'])
    # Float32 both baseline policies plus seven candidate views, including the
    # unopened final role in capacity arithmetic, never in actual image access.
    capacity=int((base_images*2+candidate_images*7)*3*256*256*4*1.25)
    free=shutil.disk_usage(ROOT).free
    if free<capacity:raise ScientificStop('INCONCLUSIVE_BUDGET','Insufficient disk for prescribed float32 caches plus 25% margin: %d required, %d free.'%(capacity,free))
    cache_cost=factor*(base_images*measured['backbone_seconds_per_image']+candidate_images*measured['seven_view_seconds_per_image'])
    candidate_extra=factor*counts['utility_val']*(20*measured['full_metric_seconds_per_image']+measured['seven_view_seconds_per_image'])
    final_cost=factor*(counts['calibration']*120+counts['utility_val']*110+counts['sealed_eval']*110)*measured['full_metric_seconds_per_image']
    final_cost+=factor*18*3*110*(measured['backbone_seconds_per_image']+.04)
    candidate_count=None;utility_count=None
    def utility_prediction(n):
        return sum(factor*(jobs[m]['seconds_per_update']*n+5*counts['model_val' if m=='B4' else 'utility_val']*jobs[m]['validation_seconds_per_image']) for m in ORDER)
    for n in [4000,3000,2000,1000]:
        cost=sum(factor*(jobs[m]['seconds_per_update']*n+5*counts['model_val']*jobs[m]['validation_seconds_per_image']) for m in ['B1','B3'])+candidate_extra
        if cost<=3.5*3600 and used+cost+utility_prediction(1000)+cache_cost+max(3.5*3600,final_cost)<=16*3600:
            candidate_count=n;candidate_cost=cost;break
    for n in [3000,2000,1500,1000]:
        cost=utility_prediction(n)
        if cost<=6.5*3600 and candidate_count is not None and used+candidate_cost+cost+cache_cost+max(3.5*3600,final_cost)<=16*3600:
            utility_count=n;utility_cost=cost;break
    budget_ok=candidate_count is not None and utility_count is not None
    total=used+(candidate_cost if candidate_count else 0)+(utility_cost if utility_count else 0)+cache_cost+max(3.5*3600,final_cost)
    budget_ok=budget_ok and total<=16*3600
    for method,values in jobs.items():
        n=candidate_count if method in ['B1','B3'] else utility_count
        val_count=counts['model_val'] if method in ['B1','B3','B4'] else counts['utility_val']
        values.update(updates=n,validation_images_per_checkpoint=val_count,
                      predicted_total_seconds=factor*(values['seconds_per_update']*(n or 0)+5*val_count*values['validation_seconds_per_image']))
    plan={'status':'BUDGET_FROZEN' if budget_ok else 'INCONCLUSIVE_BUDGET','jobs':jobs,
          'candidate_updates':candidate_count,'utility_updates':utility_count,'safety_factor':factor,
          'profile_sha256':sha(profile),'used_before_freeze_device_seconds':used,
          'cache_predicted_device_seconds':cache_cost,'candidate_diagnostics_predicted_device_seconds':candidate_extra,
          'final_predicted_device_seconds':final_cost,'final_reserved_device_seconds':3.5*3600,
          'total_predicted_device_seconds_with_reserve':total,'maximum_device_seconds':16*3600,
          'cache_capacity_bytes_with_25_percent_margin':capacity,'disk_free_bytes':free,
          'all_13_training_jobs_included':True,'source_counts':counts}
    plan['budget_allocation_note']='At least 3.5 hours protected; unused front-stage allowance may cover an evaluation estimate above 3.5 hours, without increasing the 16-hour cap.'
    write(e.run/'budget_plan.json',plan)
    if not budget_ok:raise ScientificStop('INCONCLUSIVE_BUDGET','Measured full matrix plus required evaluation cannot fit authorized budget; no formal comparison dispatched.')
    resolved=copy.deepcopy(e.config);resolved['training'].update(candidate_updates=candidate_count,utility_updates=utility_count)
    from .reporting import text
    text(e.run/'protocol_resolved.yaml',yaml.safe_dump(resolved,sort_keys=False))
    write(e.run/'training_manifest_hashes.json',{'roles_sha256':sha(e.run/'roles.jsonl'),'exposure_ledger_sha256':sha(e.run/'exposure_ledger.json'),
            'baseline_policy_sha256':sha(e.run/'baseline_policy.json'),'source_snapshot_sha256':sha(e.run/'source_snapshot.json'),
            'backbone_sha256':e.config['backbone']['checkpoint_sha256'],'resolved_protocol_sha256':sha(e.run/'protocol_resolved.yaml'),
            'budget_plan_sha256':sha(e.run/'budget_plan.json')})
