"""E engineering eligibility, noise variance and old-protocol score reconciliation."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *

def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def main():
    full=read(RUN/'E/full/summary.json'); subset=read(RUN/'E/subset/summary.json'); r=lines(RUN/'E/full/per_image.jsonl')
    comparisons=[]
    for name in ('PARENT','V2_BASE_CONT'):
        baseline=next(x for x in full if x['model']==name and x['nfe']==20)
        for nfe in (1,2,4,8):
            candidate=next(x for x in full if x['model']==name and x['nfe']==nfe)
            delta={k:candidate[k]-baseline[k] for k in ('psnr','ssim','lpips')}; speed=baseline['model_seconds_p50']/candidate['model_seconds_p50']
            valid=delta['psnr']>=-.05 and delta['ssim']>=-.002 and delta['lpips']<=.01 and speed>=1.5
            comparisons.append({'model':name,'nfe':nfe,'delta_vs20':delta,'model_speedup':speed,'service_speedup':baseline['service_seconds_p50']/candidate['service_seconds_p50'],'engineering_eligible':valid})
    noise_variance=[]
    for name in ('PARENT','V2_BASE_CONT'):
        for sid in sorted({x['sample_id'] for x in r}):
            for nfe in (1,2,4,8,20):
                rr=[x for x in r if x['model']==name and x['sample_id']==sid and x['nfe']==nfe]; assert len(rr)==3
                # Output pixel variance cannot be inferred from output hashes; score variance explicitly named.
                noise_variance.append({'model':name,'sample_id':sid,'nfe':nfe,'psnr_variance_over_noise':float(np.var([x['psnr'] for x in rr])),'unique_output_hashes':len({x['output_float32_sha256'] for x in rr})})
    legacy=lines(RUN/'E/legacy/per_image.jsonl'); oldpath=ROOT/'runs/explore_ag_single_seed_v2_20261003/parent_candidates/best_ddim_20/per_image.jsonl'
    old={r['sample_id']:r for r in lines(oldpath)}
    differences=[abs(x['psnr']-old[x['sample_id']]['psnr']) for x in legacy if x['model']=='PARENT']
    result={'efficiency_candidates':comparisons,'main_sampler_unchanged':'DDIM20','noise_score_variance':noise_variance,'legacy_v2_key_reproduction':{'count':len(differences),'max_psnr_absolute_error':max(differences),'old_record_sha256':sha(oldpath)},'subset_timing_limitation':'per-step trace synchronizes device every call; subset timing is instrumented diagnostic, full untraced timing used for deployment table','full_timing_boundaries':'prior uncached; model pad+prior+sample+crop+clip; service IO+decode+H2D+model+save; metrics excluded','fresh_clock_vs_historical':'same host model timings are not assumed equal to Sep28 snapshots'}
    write(RUN/'E/engineering_decision.json',result)
    text='# E 采样机制与效率诊断\n\n全部双权重、24图三噪声DDPM/DDIM轨迹及完整91图三噪声DDIM已完成。没有训练蒸馏学生。\n\n|权重|NFE|ΔPSNR|ΔSSIM|ΔLPIPS|模型加速|部署容差通过|\n|---|---:|---:|---:|---:|---:|---|\n'
    for x in comparisons:
        d=x['delta_vs20']; text+='|%s|%d|%.6f|%.6f|%.6f|%.3f|%s|\n'%(x['model'],x['nfe'],d['psnr'],d['ssim'],d['lpips'],x['model_speedup'],x['engineering_eligible'])
    text+='\n主算法比较仍固定DDIM20。上述是DDIM工程候选，不是原创方法。trace子集有每步同步开销，部署时延取未启用轨迹记录的完整验证；服务/模型/先验/采样的p50与p95分别见full/summary.json。LPIPS指标时间不混入模型推理。显存从prior之前重置。\n\nV2兼容键复核最大PSNR误差 '+str(max(differences))+'；该复核与含eval_noise_seed的v3表分开。状态/时间替代属于分布外机制诊断，不用参考作为部署输入，也不证明扩散理论无效。\n'
    (DOC/'E_DIAGNOSTICS.md').write_text(text)
if __name__=='__main__': main()
