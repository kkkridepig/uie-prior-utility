"""Resumable dependency-ordered E -> C pilot -> gated C/D repeats -> frozen export.

Scientific failure and execution/resource failure take separate terminal paths.
"""
import argparse,fcntl,json,os,shutil,sys,time,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from scripts.cde_v3.dispatch import run_job,update_live,command_identity
from scripts.cde_v3.eval_development import oracle

PYTHON=str(ROOT/'.venv/bin/python')

def job(name,package,script,*args,hours=4):
    command=[PYTHON,'scripts/cde_v3/'+script]+[str(a) for a in args]
    frozen=RUN/'implementation_freeze_resume_before_pilot.json'
    if frozen.exists(): assert read(frozen)['source_sha256']==code_hash(),'Source changed after pilot freeze'
    # Check a previous direct invocation's exact command before treating it as done.
    s=read(RUN/'dispatch_state.json')
    if name in s['completed']:
        if command_identity(s['completed'][name]['command'])!=command_identity(command):
            raise ValueError('Completed ID reused for different command: '+name)
        return
    code=run_job(name,package,command,hours)
    if code==75:
        # Safe chunk deadline is resumable; ledger remains cumulative.
        code=run_job(name,package,command,hours)
        if code==75: raise RuntimeError('budget_stop:'+name)
    if code:
        # At most one identical retry, recorded by device runner; no unbounded loops.
        code=run_job(name,package,command,hours)
        if code: raise RuntimeError('execution_failure:'+name)

def train_branch(seed,branch,package):
    root=RUN/'pilot'/str(seed)/branch
    job('%s_%s_train'%(seed,branch),package,'train_bank.py','--branch',branch,'--finetune-seed',seed,'--steps',5000,'--output',root,hours=6)
    job('%s_%s_eval'%(seed,branch),package,'eval_development.py','--checkpoint',root/'step_05000.pt','--output',root/'eval',hours=3)
    return root/'step_05000.pt'

def c_pilot(seed,package):
    root=RUN/'pilot'/str(seed)
    for b in ['BASE_CONT_V3','C_BANK','C_RGB_CONTROL']: train_branch(seed,b,package)
    result=oracle(seed)
    if not result['pass']: return False,'bank_no_selectable_gain'
    train_branch(seed,'C_ALL_ONLY',package)
    for role in ['route_fit','route_cal']:
        job('%s_labels_%s'%(seed,role),package,'build_utilities.py','--checkpoint',root/'C_BANK/step_05000.pt','--role',role,'--output',root/'labels'/role,hours=3)
    for kind in ['UTILITY','WINNER_CE','SHUFFLED']:
        job('%s_gate_%s'%(seed,kind),package,'train_selector.py','--labels',root/'labels/route_fit/utilities.pt','--kind',kind,'--finetune-seed',seed,'--output',root/'selectors'/kind,hours=1)
    job('%s_selector_eval'%seed,package,'eval_selectors.py','--seed',seed,hours=6)
    result=read(root/'gate_result.json')
    if not result['pass']: return False,'utility_not_learned'
    train_branch(seed,'BASE_CONT_DATA_MATCHED',package)
    matched=read(root/'BASE_CONT_DATA_MATCHED/eval/summary.json')['null']; candidate=result['summary']['UTILITY']
    passed=candidate['psnr']-matched['psnr']>=.1 and candidate['ssim']>=matched['ssim']-.002 and candidate['lpips']<=matched['lpips']+.01
    write(root/'data_matched_gate.json',{'pass':passed,'matched':matched,'candidate':candidate})
    return passed,'pilot_C_passed' if passed else 'data_matched_gain_not_supported'

def d_pilot(seed,package):
    root=RUN/'pilot'/str(seed)
    # Real D profiles and frozen-rms contracts are prerequisites, not inferred from C.
    if seed==20261004: job('D_preflight_profile',package,'preflight_d.py','--run',hours=1)
    for b in ['BASE_CONT_V3','D_SOBEL_STD','D_PHASE_STD','D_SOFTPHASE_STD']: train_branch(seed,b,package)
    job('%s_D_noise'%seed,package,'diagnose_d.py','--seed',seed,hours=2)
    values={b:read(root/b/'eval/summary.json')['null'] for b in ['BASE_CONT_V3','D_SOBEL_STD','D_PHASE_STD','D_SOFTPHASE_STD']}
    candidate=values['D_SOFTPHASE_STD']; name,strong=max([(k,v) for k,v in values.items() if k!='D_SOFTPHASE_STD'],key=lambda p:p[1]['psnr'])
    ok=candidate['psnr']>=strong['psnr']+.1 and candidate['ssim']>=strong['ssim']-.002 and candidate['lpips']<=strong['lpips']+.01
    result={'pass':ok,'candidate':candidate,'strongest':name,'controls':values,'status':'pilot_D_passed' if ok else 'pilot_no_supported_gain','metrics_complete':True}
    write(root/'d_gate_result.json',result)
    (DOC/'D_FALLBACK.md').write_text('# D 备选结果\n\n只由C科学数值失败触发，非实现阻塞。\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
    return ok

def make_decision(status,method,seeds):
    models={}; paths=[str(PARENT)]
    for seed in seeds:
        root=RUN/'pilot'/str(seed); mapping={'BASE_CONT_V3':str(root/'BASE_CONT_V3/step_05000.pt')}
        if method=='C_UTILITY':
            for name in ['C_UTILITY','C_BEST_FIXED','C_WINNER_CE','C_SHUFFLED']: mapping[name]=str(root/'C_BANK/step_05000.pt')
            for name in ['C_RGB_CONTROL','C_ALL_ONLY','BASE_CONT_DATA_MATCHED']: mapping[name]=str(root/name/'step_05000.pt')
            paths += [str(root/'selectors'/k/'last.pt') for k in ['UTILITY','WINNER_CE','SHUFFLED']]
        elif method=='D_SOFTPHASE_STD':
            for name in ['D_SOBEL_STD','D_PHASE_STD','D_SOFTPHASE_STD']: mapping[name]=str(root/name/'step_05000.pt')
        models[str(seed)]=mapping; paths+=list(mapping.values())
    result={'status':status,'method':method,'seeds':seeds,'models':models,'checkpoints':sorted(set(paths)),'strong_control':'BASE_CONT_DATA_MATCHED' if method=='C_UTILITY' else 'BASE_CONT_V3','parent_initializations':1,'seed_claim':'one pilot plus two same-protocol fine-tuning repeats; not independent pretraining','threshold_is_not_innovation':True}
    write(RUN/'development_decision.json',result); return result

def finish_report(status,reason=''):
    budget=read(RUN/'budget.json'); state=read(RUN/'dispatch_state.json'); state.update(status=status,reason=reason); write(RUN/'dispatch_state.json',state); update_live(state,budget)
    files={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and p.suffix in ('.json','.yaml','.pt','.png') and p.name not in ('artifact_manifest.json','budget.json','dispatch_state.json')}
    write(RUN/'artifact_manifest.json',files)
    decision=read(RUN/'development_decision.json') if (RUN/'development_decision.json').exists() else None
    text='# CDE V3 最终状态报告\n\n状态：`'+status+'`。'+reason+'\n\n累计设备时间 %.5f / 72 h，最终保留12 h；旧V2结果原样保留。\n\n'%(budget['charged_seconds']/3600)
    text+='数学/shape/null/FFT/DDIM、CPU/PPU证据、独立RNG与恢复测试分别见对应JSON及日志。测试通过不代表算法有效；闸门只决定研究投资，不代表论文创新。\n\n'
    text+='```json\n'+json.dumps(decision,indent=2,ensure_ascii=False)+'\n```\n\n'
    text+='E详情见E_DIAGNOSTICS.md和E目录；C见C_BANK_ORACLE.md、pilot/<seed>/oracle.json、gate_result.json及全部逐图记录；D见D_FALLBACK.md。新留出数据身份见HOLDOUT_AUDIT.md，旧test均为legacy_exposed_regression。缺少输出意味着未执行，不填虚构指标。\n\n'
    text+='复现/恢复：`PYTHONPATH=. scripts/mpa_python.sh scripts/cde_v3/pipeline.py --resume`；只读计划：`PYTHONPATH=. scripts/mpa_python.sh scripts/cde_v3/dispatch.py --dry-run`。恢复不重置账本或重复完成ID。权重为delta，必须结合lineage中的原父checkpoint；artifact_manifest.json列出产物与哈希。\n'
    if (RUN/'inference_layout_validation.json').exists():
        layout=read(RUN/'inference_layout_validation.json')
        drift=max(r['metric_abs_deltas']['psnr'] for r in layout['evidence'])
        text+='\n## 推理与预算证据\n\nchannels_last仅用于推理，训练仍为原NCHW；完整DDIM20 hard-null逐位通过。训练图布局数值核验最大PSNR漂移 %.9f dB。E效率表来自原NCHW配置，不与新布局的单点时间混算加速比。旧超预算预测和停止快照均保留。\n\n'%drift
        cost=read(RUN/'cost_prediction.json')
        text+='预测C / 2个重复 / 最终全对照设备小时：%.3f / %.3f / %.3f（含1.25倍余量）；这些是预测，实际成本以budget.json为准。\n\n'%(cost['C_pilot_predicted_hours_with_safety'],cost['repeats_predicted_hours_with_safety'],cost['final_3seed_full_controls_hours_with_safety'])
    text+='\n## 实际训练与方向状态\n\n'
    trials=[]
    for p in sorted((RUN/'pilot').glob('*/*/progress.json')):
        progress=read(p); trials.append({'finetune_seed':p.parent.parent.name,'branch':p.parent.name,'step':progress.get('step'),'target':progress.get('target'),'status':progress.get('status')})
    text+='```json\n'+json.dumps(trials,indent=2,ensure_ascii=False)+'\n```\n\n'
    text+='profiles/目录的100步及恢复验证属于工程验收，不算正式5000步pilot或算法正证据。完整科学比较见各seed的oracle.json、gate_result.json、data_matched_gate.json、d_gate_result.json（仅实际存在的文件有效）。\n\n'
    text+='D触发记录：'+('D_trigger.json存在，原因见该文件；是否跑完以实际D控制记录为准。' if (RUN/'D_trigger.json').exists() else '`not_triggered`；当前没有授权触发证据。')+'\n\n'
    text+='逐图导出：E/*/per_image.jsonl、pilot/<seed>/<branch>/eval/per_image.jsonl、selector_evaluation/per_image.jsonl、final/<seed>/<role>/per_image.jsonl。缺失者为未执行，不能宣称三seed或新留出确认。最终权重/指标/图像哈希见artifact_manifest.json，统一面板与失败例名单仅在实际final导出完成后存在。\n'
    (DOC/'FINAL_REPORT.md').write_text(text)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--resume',action='store_true'); args=p.parse_args(); setup()
    with (RUN/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Wait for already-authorized E direct job without duplicating or interrupting it.
        while read(RUN/'budget.json').get('active'):
            a=read(RUN/'budget.json')['active']
            if not Path('/proc/%s'%a['pid']).exists(): break
            time.sleep(10)
        try:
            for stage in ['subset','full','legacy']: job('E_'+stage,'E','diagnose_e.py','--stage',stage,hours=3)
            job('E_interior_sensitivity','E','state_sensitivity.py',hours=.5)
            subprocess.run([PYTHON,'scripts/cde_v3/summarize_e.py'],cwd=ROOT,check=True)
            job('P0_metrics_and_capacity','P0','review_preflight.py',hours=.5)
            job('P0_statistics','P0','prepare_statistics.py',hours=.5)
            job('P0_profile_resume','P0','profile.py','--run',hours=2)
            if (RUN/'inference_runtime_freeze.json').exists():
                job('P0_selector_profile_resume_v1','P0','profile_selector.py','--run',hours=.5)
            cost=read(RUN/'cost_prediction.json')
            if cost.get('prediction_version')==2:
                runtime=read(RUN/'inference_runtime_freeze.json')
                assert runtime['prediction_sha256']==sha(RUN/'cost_prediction_v2_channels_last.json')
                assert cost==read(RUN/'cost_prediction_v2_channels_last.json')
                assert runtime['runtime_sha256']==sha(ROOT/'scripts/cde_v3/inference.py')
                assert runtime['training_implementation']==train_identity()
            if not cost['dispatch_allowed'] or shutil.disk_usage(ROOT).free<cost['required_disk_gib_estimate']*2**30:
                make_decision('resource_blocked',None,[]); finish_report('resource_blocked','完整控制矩阵预测超包预算或导出所需磁盘不足，未派发C，不触发D。'); return
            # Retain the earlier stop report; it must not remain the apparent
            # final state while an authorized, validated resume is running.
            previous=DOC/'FINAL_REPORT_PREFLIGHT_STOP.md'
            if (DOC/'FINAL_REPORT.md').exists() and not previous.exists():
                shutil.copy2(DOC/'FINAL_REPORT.md',previous)
            make_decision('pilot_execution_in_progress',None,[])
            (DOC/'FINAL_REPORT.md').write_text('# CDE V3 执行中\n\n状态：`pilot_execution_in_progress`。尚无最终算法结论。旧预算闸门停止快照保留在FINAL_REPORT_PREFLIGHT_STOP.md。\n\n验证后的确定性channels_last推理已冻结，训练配方和全部科学闸门不变。完整控制矩阵预测已在原包上限内通过；实际开销仍由budget.json累计。\n\n实时状态见LIVE_STATUS.md，完成依赖见dispatch_state.json；达到数值、预算或错误停止条件后由调度写入最终报告。C正式pilot按BASE_CONT_V3 → C_BANK → C_RGB_CONTROL → 训练后oracle顺序执行。只有正确且指标完整的C数值失败才能触发D。\n')
            passed,reason=c_pilot(20261004,'C'); method=None
            if passed: method='C_UTILITY'
            else:
                write(RUN/'D_trigger.json',{'reason':reason,'C_correct_complete':True,'metrics_complete':True,'one_fallback_only':True})
                if d_pilot(20261004,'D'): method='D_SOFTPHASE_STD'
            seeds=[20261004]
            status='pilot_no_supported_gain'
            if method:
                for seed in [20261005,20261006]:
                    # For repeats, complete controls even if a repeat is not positive;
                    # do not use interim repeat gates to cherry-pick required controls.
                    if method=='C_UTILITY':
                        run_c_repeat(seed)
                    else: d_pilot(seed,'REPEAT')
                    seeds.append(seed)
                status=check_repeats(method,seeds)
            make_decision(status,method,seeds)
            job('FINAL_frozen_test','FINAL','freeze_and_test.py',hours=12)
            from scripts.cde_v3.final_summary import final_summary
            finalstatus=final_summary()
            finish_report(finalstatus)
        except Exception as exc:
            reason=type(exc).__name__+': '+str(exc)
            status='budget_stopped_with_checkpoints' if 'budget' in reason.lower() else 'implementation_blocked'
            finish_report(status,reason)
            raise


def run_c_repeat(seed):
    root=RUN/'pilot'/str(seed)
    for b in ['BASE_CONT_V3','C_BANK','C_RGB_CONTROL','C_ALL_ONLY','BASE_CONT_DATA_MATCHED']: train_branch(seed,b,'REPEAT')
    oracle(seed)
    for role in ['route_fit','route_cal']: job('%s_labels_%s'%(seed,role),'REPEAT','build_utilities.py','--checkpoint',root/'C_BANK/step_05000.pt','--role',role,'--output',root/'labels'/role,hours=3)
    for kind in ['UTILITY','WINNER_CE','SHUFFLED']: job('%s_gate_%s'%(seed,kind),'REPEAT','train_selector.py','--labels',root/'labels/route_fit/utilities.pt','--kind',kind,'--finetune-seed',seed,'--output',root/'selectors'/kind,hours=1)
    job('%s_selector_eval'%seed,'REPEAT','eval_selectors.py','--seed',seed,hours=6)


def check_repeats(method,seeds):
    deltas=[]; protections=[]
    for seed in seeds:
        root=RUN/'pilot'/str(seed)
        if method=='C_UTILITY':
            r=read(root/'gate_result.json'); values=[v['psnr'] for k,v in r['summary'].items() if k in ['BEST_FIXED','WINNER_CE','SHUFFLED']]
            controls=[]
            for b,mode in [('BASE_CONT_V3','null'),('BASE_CONT_DATA_MATCHED','null'),('C_RGB_CONTROL','all'),('C_ALL_ONLY','all')]:
                control=read(root/b/'eval/summary.json')[mode]; values.append(control['psnr']); controls.append(control)
            delta=r['summary']['UTILITY']['psnr']-max(values)
            candidate=r['summary']['UTILITY']
            protections.append(all(candidate['ssim']>=v['ssim']-.002 and candidate['lpips']<=v['lpips']+.01 for v in controls) and r['checks']['corruption'])
        else:
            r=read(root/'d_gate_result.json'); delta=r['candidate']['psnr']-max(v['psnr'] for k,v in r['controls'].items() if k!='D_SOFTPHASE_STD')
            candidate=r['candidate']; protections.append(all(candidate['ssim']>=v['ssim']-.002 and candidate['lpips']<=v['lpips']+.01 for k,v in r['controls'].items() if k!='D_SOFTPHASE_STD'))
        deltas.append(delta)
    passed=all(d>0 for d in deltas) and float(np.mean(deltas))>=.1 and all(protections)
    write(RUN/'repeat_evidence.json',{'method':method,'seeds':seeds,'strongest_control_deltas':deltas,'metric_and_corruption_protections':protections,'mean':float(np.mean(deltas)),'std':float(np.std(deltas,ddof=1)),'min':min(deltas),'max':max(deltas),'passed':passed,'interpretation':'conditional on one parent; pilot+two repeats'})
    return 'same_parent_multiseed_supported' if passed else 'multiseed_not_supported'

if __name__=='__main__': main()
