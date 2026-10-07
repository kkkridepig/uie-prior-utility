"""Close out observed results without inventing unrun phases or metrics."""
import csv
import json
from collections import defaultdict
from xml.etree import ElementTree

import numpy as np
import torch

from .records import ROOT,GUIDE,digest,jsonl,read,sha,write
from .reporting import archive,text
from .training import ORDER


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)


def csv_write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for row in rows for k in row})
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def export_metrics(e):
    rows=[];role_ids={r['sample_id']:r for r in __import__('uie_next.data.roles',fromlist=['read_roles']).read_roles(e.config['data']['manifest'])}
    for path in sorted((e.run/'metrics').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if 'psnr' not in row:continue
            sample=role_ids[row['sample_id']]
            row.update(run_id=e.config['runtime']['run_id'],training_seed=e.config['seed'],
                       input_path=sample['input_path'],reference_path=sample['reference_path'],
                       input_sha256=sample['input_sha256'],reference_sha256=sample['reference_sha256'],source_file=path.name)
            if 'method' not in row:row['method']='B0_'+row['policy']
            row.setdefault('condition','nominal');rows.append(row)
    grouped=defaultdict(list)
    for row in rows:grouped[(row['source_file'],row['method'],row['group_id'],row['role'],row['condition'],row.get('strategy_index'))].append(row)
    groups=[]
    for key,part in sorted(grouped.items(),key=lambda item:str(item[0])):
        entry=dict(zip(['source_file','method','group_id','role','condition','strategy_index'],key));entry['images']=len(part);entry['training_seed']=e.config['seed']
        for metric in ['psnr','ssim','lpips','mse']:
            if all(isinstance(r.get(metric),(int,float)) for r in part):entry[metric]=float(np.mean([r[metric] for r in part]))
        groups.append(entry)
    csv_write(e.run/'metrics/per_image.csv',rows);csv_write(e.run/'metrics/per_group.csv',groups)
    by_seed=defaultdict(list)
    for row in rows:by_seed[(row['source_file'],row['method'],row['role'],row['condition'],row.get('strategy_index'))].append(row)
    seeds=[]
    for key,part in by_seed.items():
        entry=dict(zip(['source_file','method','role','condition','strategy_index'],key));entry.update(training_seed=e.config['seed'],images=len(part))
        for metric in ['psnr','ssim','lpips','mse']:
            if all(isinstance(r.get(metric),(int,float)) for r in part):entry[metric]=float(np.mean([r[metric] for r in part]))
        seeds.append(entry)
    csv_write(e.run/'metrics/per_seed.csv',seeds)
    for filename in ['development_gate.json','confirmation_results.json']:
        path=e.run/filename
        if path.exists():write(e.run/('metrics/'+filename.replace('.json','_paired_comparisons.json')),read(path)['paired_comparisons'])
    stress=[]
    for filename in ['development_stress.json','confirmation_stress.json']:
        path=e.run/filename
        if path.exists():
            for family in read(path):
                for method,metrics in family['summary'].items():stress.append({'source':filename,'method':method,'condition':family['condition'],**metrics})
    csv_write(e.run/'metrics/stress_by_family.csv',stress)
    return {'per_image_rows':len(rows),'per_group_rows':len(groups),'stress_rows':len(stress)}


def receipts(e):
    tests=[]
    for path in sorted((e.run/'tests').glob('pytest*.xml')):
        root=ElementTree.parse(path).getroot();suites=list(root.iter('testsuite'))
        tests.append({'file':str(path),'sha256':sha(path),'tests':sum(int(s.get('tests',0)) for s in suites),
                      'failures':sum(int(s.get('failures',0)) for s in suites),'errors':sum(int(s.get('errors',0)) for s in suites)})
    result={'pytest':tests}
    for name in ['real_integration/receipt.json','real_integration/supplemental_receipt.json','runtime_probe.json']:
        path=e.run/'tests'/name
        if path.exists():result[name]=read(path)
    write(e.run/'closeout/test_receipts.json',result);return result


def inventory(e):
    items=[];selections={}
    backbone=__import__('pathlib').Path(e.config['backbone']['checkpoint'])
    items.append({'path':str(backbone),'role':'official_public_simplified_SS_UIE','bytes':backbone.stat().st_size,'sha256':sha(backbone)})
    for path in sorted((e.run/'checkpoints').rglob('*.pt')):
        receipt_path=path.with_suffix(path.suffix+'.json')
        if not receipt_path.exists() or sha(path)!=read(receipt_path)['sha256']:raise ValueError('Checkpoint integrity failure: '+str(path))
        state=torch.load(path,map_location='cpu')
        items.append({'path':str(path),'role':'formal_head_checkpoint','bytes':path.stat().st_size,'sha256':sha(path),
                      'global_step':state['global_step'],'method':state['identity']['method_id'],'complete_restore_state':all(k in state for k in ['model_state','optimizer_state','scheduler_state','rng'])})
    for path in sorted((e.run/'checkpoints').glob('*/selection.json')):selections[path.parent.name]=read(path)
    if selections:write(e.run/'checkpoint_selection.json',selections)
    write(e.run/'closeout/server_weight_inventory.json',items)
    return items,selections


def documents(e,selections,tests):
    run=e.run;doc=e.doc;status=e.state['scientific_status'];budget=read(run/'budget.json');audit=read(run/'cross_dataset_audit.json')
    policy=read(run/'baseline_policy.json') if (run/'baseline_policy.json').exists() else None
    roles=table(['Role','Groups','Pairs'],[(n,v['groups'],v['pairs']) for n,v in audit['role_counts'].items()])
    mapping=(doc/'RESUMPTION_CONTRACT_AND_TASKS.md').read_text().split('## 公式、接口、角色与验收对应',1)[1].split('## 执行任务',1)[0]
    text(doc/'IMPLEMENTATION_CONFORMANCE.md','# Implementation conformance\n\n'+mapping+
         '\nFull dispatch: cli.py -> experiment.py -> integration.py/profiling.py/scientific_evaluation.py/timing.py/visuals.py/delivery.py.\n'
         'Backbone identity: official public simplified implementation; no unpublished MCSS reconstruction.\n'
         'Implemented is distinct from executed: state.json and per-job selection receipts identify the actual completed stages. B4 selects on model_val.\n')
    text(doc/'DATA_AND_EXPOSURE_AUDIT.md','# Data and exposure audit\n\n'+roles+
         '\nFull local LSUI 4279/UIEB 890 input/reference audit; 4 cross-source edges, one UIEB pair excluded; '+str(audit['near_edge_count'])+' near edges and '+str(audit['exact_edge_count'])+' exact edges.\n'
         'Grouping is content_group_proxy, not verified semantic scenes. All samples are historically_analyzed.\n'
         'The official source declares LSUI; per-image author training membership remains unknown. Local LSUI is an exposed candidate development pool, not an author-split reproduction.\n'
         'UIEB documented_nonoverlap is scoped to the declared LSUI source and the available full-file cross-audit. This does not prove universal nonoverlap or a new blind test.\n'
         'RoleGuard rejects independent labels when provenance is unknown and sealed references until DEV_PASS plus freeze.\n'
         'Lists and hashes: all_pairs.jsonl, roles.jsonl, groups.json, exposure_ledger.json, near_duplicate_edges.jsonl, split_freeze.json.\n')
    baseline='Not run.' if policy is None else table(['Policy','model_val mean PSNR dB'],list(policy['mean_psnr'].items()))
    text(doc/'BASELINE_VERIFICATION.md','# Baseline verification\n\nSS-UIE official public simplified implementation; commit '+e.config['backbone']['commit']+
         '\nWeight SHA256: '+e.config['backbone']['checkpoint_sha256']+'\n\n'+baseline+
         '\nPolicy selection only on 671 model_val images; frozen '+str(policy['selected'] if policy else None)+'. This is not a reproduction of the full AAAI paper model or author Test-400 scores.\n'
         'Source provenance: backbone_provenance.json; strict uploaded-weight PPU receipt; author issue-comment evidence.\n'
         'Metric identity: baseline_policy.json (PSNR/SSIM and actual pretrained VGG LPIPS). Fixed 256x256 RGB float32 preprocessing.\n')
    test_table=table(['Receipt','Tests','Failures','Errors'],[(x['file'],x['tests'],x['failures'],x['errors']) for x in tests['pytest']])
    real=tests.get('real_integration/receipt.json',{})
    text(doc/'MATHEMATICAL_TEST_REPORT.md','# Mathematical and integration tests\n\n'+test_table+
         '\nReal official-backbone integration receipt:\n```json\n'+json.dumps(real,ensure_ascii=False,indent=2)+'\n```\n'
         'Temporary smoke/profile heads are disposable engineering artifacts, not formal method checkpoints.\n')
    matrix=[]
    for method in ['B1','B3']+ORDER:
        record=selections.get(method)
        matrix.append((method,record['updates'] if record else 'not_run',record['selected_step'] if record else '-',
                       record['device_seconds']/3600 if record else '-',record['convergence_unresolved'] if record else '-'))
    matrix_table=table(['Method','Updates','Selected step','Device hours','Convergence unresolved'],matrix)
    text(doc/'TRAINING_AND_CONTROL_MATRIX.md','# Actual training matrix\n\n'+matrix_table+
         '\nSeed 20261007 only. B4 initializes from selected B3; its model_fit/utility_fit 4+4 stream matches the additional data sources, not necessarily total exposure or wall time.\n'
         'B0 uses both frozen output policies; B2 is a calibrated fixed alpha on B1. All utility methods share frozen B1, fit-only scales and paired source/view streams.\n'
         'Five validation fractions 0/.25/.5/.75/1 and earlier-checkpoint tie rule; candidate/B4 selection=model_val, utility selection=utility_val nominal.\n')
    diagnostic=read(run/'candidate_diagnostics.json') if (run/'candidate_diagnostics.json').exists() else None
    result=read(run/'confirmation_results.json') if (run/'confirmation_results.json').exists() else read(run/'development_gate.json') if (run/'development_gate.json').exists() else None
    nominal='Not run: no complete deployable-control evaluation.'
    if result:
        nominal=table(['Method','PSNR','SSIM','LPIPS','delta vs J0','harm > .1dB'],[(m,v['psnr'],v['ssim'],v['lpips'],v['delta_vs_base'],v['harm_rate_over_0_1']) for m,v in result['summary'].items()])
    elif diagnostic:
        nominal=table(['Candidate diagnostic','Mean PSNR dB'],diagnostic['means_psnr'].items())+'\nReference-based oracle entries are not deployable methods.\n'
    failure=str(e.state.get('stop_reason','No recorded scientific/engineering stop.'))
    text(doc/'FAILURE_CASES.md','# Failure cases and pressure scope\n\n'+failure+
         '\nFixed visual records are in figures/baseline and, if reached, figures/utility_val or figures/sealed_eval.\n'
         'No unrun pressure family or sealed score is synthesized. All registered families are reported only when their phase executes.\n')
    commands=['.venv/bin/python -B -m uie_next.cli '+command+' --config configs/uie_next/protocol.yaml'+(' --resume' if command=='run' else '') for command in ['verify-backbone','smoke','plan-budget','run','status','closeout']]
    text(doc/'REPRODUCE_AND_RESUME.md','# Reproduce and resume\n\nFrom /mnt/workspace/uie-prior-utility, use the existing vendor environment .venv.\n\n```bash\n'+'\n'.join(commands)+'\n```\n'
         'run uses a per-run dispatcher lock and device budget lease, atomically saves every 250 updates and at exit. It restores optimizer/scheduler and all named RNG streams; terminal scientific stops do not dispatch again.\n'
         'Dependencies are hash-linked: official backbone, policy, candidate, roles, scales and source snapshot. Checkpoint file receipts permit integrity verification. No old V3 run is restarted.\n'
         'Independent backup is not verified; same-server ZIPs do not survive loss of the server disk.\n')
    report=['# FINAL_REPORT','',f'状态：`{status}`。'+('本轮单种子、历史暴露数据条件下的确认闸门通过。' if status=='CONFIRMATION_PASS_SINGLE_SEED' else '未建立已通过本轮质量与机制闸门的算法证据。'),
            '', '## 实际执行', '', '底座为 SS-UIE 官方公开简化实现，不等同于论文完整模型；未补写未开源 MCSS。',
            '固定上游提交 `'+e.config['backbone']['commit']+'`；权重 SHA256 `'+e.config['backbone']['checkpoint_sha256']+'`。',
            'GPU: PPU-ZW810E，单卡，float32，256x256；厂商 torch/torchvision 保留。',
            f'累计设备小时 {budget["used_device_seconds"]/3600:.6f} / {budget["max_device_hours"]}；训练 seed=20261007。',
            '',roles,'',matrix_table,'', '## 实现忠实性', '',
            '公式、输入输出、角色与验收逐项见 IMPLEMENTATION_CONFORMANCE.md；真实验收回执见 tests/real_integration。',
            '残差使用实际裁剪后的 J1-J0；RGB 共享逐像素 alpha；正负方向网络共享参数；字段干预后重算候选和真实标签。',
            '底座及其 BatchNorm 冻结，效用训练的候选冻结，部署路径不读取参考图；B4、门控、协方差融合、误差投影和四项消融均已实现。',
            '完整实现与实际运行范围分别由测试、state.json 和 checkpoint selection 记录；未运行阶段不称已验收效果。',
            '', '## 名义结果', '',nominal,'', '## 效应归因', '',
            'H1/H2/H3 只能依据强控制和主要消融的实际差值判断。oracle 是不可部署空间诊断，不能作为方法贡献；训练跑完或一个阈值通过不能证明论文创新。',
            '本轮没有通过全部质量、机制及封存闸门时，收益归因保持未支持或证据不足。',
            '', '## 不确定性', '',
            '仅一个新增模块训练种子；内容代理组不是核验场景；本地数据曾历史分析。UIEB 非重叠证据限作者声明的 LSUI 来源与可获得数据审计。',
            '参考图指标不证明真实水体颜色或深度恢复；单种子区间不涵盖训练随机性。',
            '', '## 失败与压力', '',failure,
            '封存集已释放：'+str(e.state.get('sealed_eval_released',False))+'. 未解封时没有封存指标或完成版 final freeze。',
            '', '## 成本', '',
            'budget_plan.json 包含所有13作业、验证、缓存和评估的测时预测；budget.json 与 budget_ledger.jsonl 是实际累计账本。',
            '正式部署测速只有 timing/deployment.json 存在时才已执行；禁用底座/候选缓存，包含实际模块、传输和PNG。',
            '', '## 恢复与备份', '',
            '全部正式 checkpoint、选择文件、逐图记录和身份在同名 runs；权重清单见 closeout/server_weight_inventory.json。',
            '审阅/视觉/源码/权重恢复 ZIP 的成员哈希与CRC核验范围见 archive_receipts.json；原始数据不进入交付ZIP。',
            '独立备份未完成（independent_backup_verified=false）。同盘包不能抵抗再次断电丢盘。',
            '', '## 后续', '',
            '若本轮通过，只支持进一步做新增模块多种子、独立来源数据和第二底座确认；未通过则停止当前配方，不自动增加预算、步数、种子或重启 A-G。']
    text(doc/'FINAL_REPORT.md','\n'.join(report)+'\n')
    text(doc/'CONFIRMATION_REPORT.md','# Confirmation report\n\n'+('Executed: confirmation_results.json.\n' if (run/'confirmation_results.json').exists() else 'not_run: required development gates have not passed; sealed_eval remains protected.\n'))


def deliver(e):
    e.doc.mkdir(parents=True,exist_ok=True)
    old=read(e.run/'legacy_preservation.json')
    changed=[p for p,h in old.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    write(e.run/'closeout/legacy_verification.json',{'files':len(old),'changed':changed,'all_hashes_unchanged':not changed})
    if changed:raise ValueError('Historical files changed: '+str(changed))
    export_metrics(e);test_records=receipts(e);weights,selections=inventory(e)
    if (e.run/'baseline_policy.json').exists():
        from .visuals import export_baseline
        export_baseline(e)
    e.state.update(closeout_status='S11_CLOSEOUT_COMPLETE',independent_backup_verified=False,
                   completed_training_updates=sum(x['updates'] for x in selections.values()))
    write(e.run/'state.json',e.state);documents(e,selections,test_records)
    from .backup import package_evidence
    archives=package_evidence(e.config)
    return {'state':e.state,'report':str(e.doc/'FINAL_REPORT.md'),'archives':archives}
