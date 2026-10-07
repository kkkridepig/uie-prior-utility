"""Chinese closeout, fixed-case visuals, archive CRC and member-hash verification."""
import hashlib
import math
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from ..records import ROOT,read,write,sha,digest
from ..reporting import archive
from ..math.utility import labels
from .context import OLD,STAGES
from .diagnostics import checkpoint_model,exact_strategies,metrics,csv_write


def selected_cases(rows,kind='endpoint'):
    vals={r['sample_id']:r for r in rows if r.get('prediction_kind',kind)==kind}
    ids=sorted(vals);hashed=sorted(ids,key=lambda i:hashlib.sha256(i.encode()).hexdigest())[:8]
    ranked=sorted(ids,key=lambda i:(vals[i].get('delta_psnr_db',0),i));mid=len(ranked)//2
    return list(dict.fromkeys(hashed+ranked[:4]+ranked[-4:]+ranked[max(0,mid-2):mid+2]))


def rgb(t):
    return Image.fromarray((t.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8))


def heat(t,scale,negative=False):
    a=t.detach().cpu().numpy();a=a.mean(0) if a.ndim==3 else a
    if negative:
        a=np.clip(a/max(scale,1e-12),-1,1);v=np.stack([np.maximum(a,0),np.zeros_like(a),np.maximum(-a,0)],-1)
    else:v=np.repeat(np.clip(a/max(scale,1e-12),0,1)[...,None],3,-1)
    return Image.fromarray((v*255).round().astype(np.uint8))


def panel(path,images):
    cols=4;rows=math.ceil(len(images)/cols);out=Image.new('RGB',(cols*256,rows*280),'white');draw=ImageDraw.Draw(out)
    for i,(name,img) in enumerate(images):x=(i%cols)*256;y=(i//cols)*280;out.paste(img,(x,y+24));draw.text((x+5,y+4),name,fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);out.save(path)


def diagnostic_visuals(s,d):
    summaries=read(s.run/'diagnostics/checkpoint_summary.json') if (s.run/'diagnostics/checkpoint_summary.json').exists() else []
    if not summaries:return
    selected_path=s.run/'selection/producer_selection_before_utility_val.json';p=read(selected_path).get('selected') if selected_path.exists() else None
    if p is None:p=next((r for r in summaries if r['head']=='B1' and r['training_step']==4000 and r['role']=='utility_val'),None)
    if p is None:return
    item=next(i for i in read(s.run/'diagnostics/checkpoint_inventory.json') if i['checkpoint_id']==p['checkpoint_id']);model=checkpoint_model(item)
    records=read(s.run/'diagnostics/parts'/item['checkpoint_id']/'utility_val.json')['metrics']
    ids=selected_cases(records);manifest=[]
    with s.device_job('CLOSEOUT_FIXED_DIAGNOSTIC_VISUALS',300,final=True):
        with torch.no_grad():
            for sample_id in ids:
                batch=d.data.batch([sample_id],'candidate_diagnostic');j=d.data.real_candidate(batch['image'],batch['base'],model,False)
                variants,_=exact_strategies(batch['base'],j,batch['target']);lab=labels(batch['base'],j,batch['target'])
                fixed=next(r['alpha_fixed'] for r in sorted([r for r in records if r['sample_id']==sample_id and r['prediction_kind'].startswith('fixed_')],key=lambda r:(-r['psnr'],r['alpha_fixed'])))
                # Fixed-alpha winner is reference-aided per-case illustration, explicitly nondeployable.
                images=[('Input',rgb(batch['image'][0])),('Reference (diagnostic)',rgb(batch['target'][0])),('J0 clip01',rgb(batch['base'][0])),
                  (item['checkpoint_id']+' endpoint',rgb(j[0])),('Pixel oracle NOT deployable',rgb(variants['oracle_pixel'][0][0])),
                  ('Block32 oracle NOT deployable',rgb(variants['oracle_block32'][0][0])),('Residual |r| fixed scale .05',heat(lab['r'][0].abs(),.05)),
                  ('True U fixed scale .005',heat(lab['U'][0],.005,True)),('Oracle alpha',heat(variants['oracle_pixel'][1][0],1))]
                path=s.run/'figures/diagnostic'/('%s.png'%sample_id.replace('/','_'));panel(path,images)
                manifest.append({'sample_id':sample_id,'checkpoint_id':item['checkpoint_id'],'checkpoint_sha256':item['checkpoint_sha256'],
                    'path':str(path),'sha256':sha(path),'deployment_output_not_available':True,'predicted_U_not_available':True,'oracle_not_deployable':True})
    write(s.run/'figures/visual_selection.json',{'rule':'8 sample-id SHA256 + worst4 + best4 + middle4; ties ID; fixed field scales, no zero-residual autostretch',
          'cases':manifest,'seen_by_agent_visual_inspection':False})


def visual_method_panels(s,d,rt,rows,selection,role):
    if (s.run/'figures'/(role+'_manifest.json')).exists():return
    cases=selected_cases([dict(r,prediction_kind='endpoint') for r in rows if r['method']=='O']);out=[]
    with s.device_job('VISUAL_'+role,300,final=True):
        with torch.no_grad():
            for sid in cases:
                b=d.data.batch([sid],'develop' if role=='utility_val' else 'final');c,p=rt.features(b['image'],b['base']);j=c[rt.registry['O']['candidate']['checkpoint_sha256']]
                result,alpha=rt.output('O',selection['methods']['O']['policy'],b['image'],b['base'],c,p)
                lab=labels(b['base'],j,b['target']);baseerr=(b['base']-b['target']).square().mean(1,keepdim=True);methoderr=(result-b['target']).square().mean(1,keepdim=True)
                images=[('Input',rgb(b['image'][0])),('Reference',rgb(b['target'][0])),('J0',rgb(b['base'][0])),('Producer endpoint',rgb(j[0])),
                  ('O fixed deployment',rgb(result[0])),('Residual fixed .05',heat(lab['r'][0].abs(),.05)),('True U fixed .005',heat(lab['U'][0],.005,True)),
                  ('Predicted U fixed .005',heat(p['O']['U_hat'][0],.005,True)),('Alpha',heat(alpha[0],1)),('MSE error delta fixed .005',heat((baseerr-methoderr)[0],.005,True))]
                path=s.run/'figures'/role/(sid.replace('/','_')+'.png');panel(path,images);out.append({'sample_id':sid,'path':str(path),'sha256':sha(path)})
    write(s.run/'figures'/(role+'_manifest.json'),{'cases':out,'fixed_rule':True,'selection':sha(s.run/'selection/calibration_selection.json')})


def benchmark(s,d,rt=None,selection=None):
    path=s.run/'timing/deployment.json'
    if path.exists():return
    targets=['B0_clip01','B1_standalone_best','B3_standalone_best'] if rt is None else list(dict.fromkeys(['B0_clip01','O',selection['primary_control'],'G1','G2','R0','B4']))
    directory=s.run/'timing';directory.mkdir(parents=True,exist_ok=True);results=[]
    # Release this process's device tensors before an isolated benchmark worker.
    d.backbone.to('cpu');d.lpips.net.to('cpu')
    if d.data.candidate is not None:d.data.candidate.to('cpu')
    if rt:
        for model in list(rt.candidates.values())+list(rt.controllers.values()):model.to('cpu')
    torch.cuda.empty_cache()
    for method in targets:
        cmd=[sys.executable,'-m','uie_next.v2.timing','--method',method]
        with s.device_job('TIMING_ISOLATED_'+method,300,final=True):
            proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        (directory/(method+'.log')).write_text(proc.stdout+proc.stderr)
        if proc.returncode:raise RuntimeError('isolated deployment timing failed '+method)
        results.append(read(directory/(method+'.json')))
    write(path,{'methods':results,'batch':1,'size':[256,256],'precision':'float32','warmup':20,'timed':100,'service_inputs':20,
         'no_quality_or_reference_reads':True,'per_method_fresh_process':True,'compiled':False,'file_cache_state':'OS page cache not explicitly evicted; documented warm filesystem'})
    d.backbone.to('cuda:0');d.lpips.net.to('cuda:0')
    if d.data.candidate is not None:d.data.candidate.to('cuda:0')
    if rt:
        for model in list(rt.candidates.values())+list(rt.controllers.values()):model.to('cuda:0')


def preservation(s):
    old=read(s.run/'legacy_preservation.json');changed=[p for p,h in old.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    write(s.run/'delivery/legacy_integrity.json',{'all_unchanged':not changed,'n_files':len(old),'changed':changed})
    return not changed


def deliver(s):
    from .context import STAGES
    if (s.run/'delivery/archive_receipts.json').exists() and s.state.get('closeout_complete'):
        return {'state':s.state,'reports':str(s.doc),'archives':read(s.run/'delivery/archive_receipts.json')}
    if not preservation(s):s.live(scientific_status='BLOCKED_ENGINEERING',stop_reason='V1历史文件hash改变，需定位，不能宣布完成。')
    completed=set(s.state['completed']);reason=s.state.get('stop_reason','前置未解锁')
    for stage in STAGES:
        if stage not in completed and stage!='CLOSEOUT':s.state['stages'][stage]={'status':'not_run','reason':reason}
    if (s.run/'diagnostics/checkpoint_summary.json').exists():
        from .diagnostics import Diagnostics
        d=Diagnostics(s);d.load();d.verify_reuse()
        diagnostic_visuals(s,d)
        if not (s.run/'timing/deployment.json').exists():benchmark(s,d)
    status=s.state['scientific_status'];budget=read(s.run/'budget.json');summaries=read(s.run/'diagnostics/checkpoint_summary.json') if (s.run/'diagnostics/checkpoint_summary.json').exists() else []
    selected=read(s.run/'selection/producer_selection_before_utility_val.json') if (s.run/'selection/producer_selection_before_utility_val.json').exists() else {}
    standalone=read(s.run/'selection/standalone_selection.json') if (s.run/'selection/standalone_selection.json').exists() else {}
    gate=read(s.run/'development_gate.json') if (s.run/'development_gate.json').exists() else None
    table='| 角色 | 检查点 | 端点PSNR | H32 | G32 | Hp | Gp | S |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in summaries:
        if r['training_step']>0:table+='| %s | %s | %.6f | %.6f | %.6f | %.6f | %.6f | %.8g |\n'%(r['role'],r['checkpoint_id'],r['endpoint_psnr'],r['H32'],r['G32'],r['Hp'],r['Gp'],r['S'])
    stages='| 阶段 | 状态 | 原因／回执 |\n|---|---|---|\n'+''.join('| %s | %s | %s |\n'%(name,s.state['stages'][name]['status'],s.state['stages'][name].get('reason','已保存身份与数量回执')) for name in STAGES)
    mv4000=next((r for r in summaries if r['checkpoint_id']=='B1_004000' and r['role']=='model_val'),None)
    trade='未完成八点诊断，MSE/PSNR权衡尚不能判断。'
    if mv4000:trade='旧B1_4000在model_val：均值MSE相对变化 %.6f%%，平均逐图PSNR变化 %.6f dB。平均MSE与平均PSNR权重不同；logMSE是PSNR线性换算，不是独立证据。'%((mv4000['endpoint_mse']/mv4000['baseline_mse']-1)*100,mv4000['endpoint_psnr']-mv4000['baseline_psnr'])
    rescue=read(s.run/'selection/rescue_choice.json') if (s.run/'selection/rescue_choice.json').exists() else None
    formal={m:(read(s.run/'checkpoints'/m/'selection.json')['selected_step'] if (s.run/'checkpoints'/m/'selection.json').exists() else 'not_run：'+reason) for m in __import__('uie_next.v2.context',fromlist=['METHOD_ORDER']).METHOD_ORDER}
    protocol='# V2 与V1差异\n\nV1保持STOP_PRIOR_UNUSED。V2将standalone与非零producer分开；8旧非零点先model_val选型，再utility_val检查；最多唯一预定救援。B4也在utility_val只读选网络，所有方法按检查点×部署网格选网络，再在calibration独立重新选参数。组bootstrap增为5000。厂商环境与网络容量未更换。\n\n工程修补：恢复原六文件；显式RunContext/schema2；门控U_hat缺失NA，不改变B1/B3历史结果。D0精确oracle用实际float32输出转CPU float64；同时另报active版本。源码变更不改写旧运行。\n'
    s.ctx.text(s.doc/'protocol_changes_from_v1.md',protocol);s.ctx.text(s.run/'protocol_changes_from_v1.md',protocol)
    audit=read(s.run/'data_audit.json') if (s.run/'data_audit.json').exists() else {}
    s.ctx.text(s.doc/'DATA_AND_EXPOSURE_AUDIT.md','# 数据与暴露审计\n\n'+json.dumps(audit,ensure_ascii=False,indent=2)+'\n\n全部本地图有历史分析暴露。LSUI上游训练逐图成员未知，保守视为来源池暴露；UIEB非重叠仅限作者声明LSUI来源与本地重复审计。177对是本轮封存，不是全新盲测。身份审计读取文件hash不等于模型评分。\n')
    s.ctx.text(s.doc/'TRAINING_AND_CONTROL_MATRIX.md','# 实际训练矩阵\n\n```json\n'+json.dumps(formal,ensure_ascii=False,indent=2)+'\n```\n\n11方法工程夹具各2更新不算正式训练。未运行项不是O科学失败。所有正式新增训练单种子20261007；底座冻结。\n')
    s.ctx.text(s.doc/'NEXT_DECISION.md','# 下一步决策\n\n当前状态：**'+status+'**。\n\n'+reason+'\n\n若停止在producer筛选：优先分析候选训练目标、修正可学性及迁移证据，不建议直接以oracle空间为依据宣称O创新或开启多种子。若完整O开发／确认通过，也只支持固定公开简化底座、单新增模块种子、历史暴露数据下的有限证据；下一轮需要独立数据、完整方法重训多种子和同构O_RGB对照。任何扩大训练预算须另行授权。\n')
    report='# SS-UIE 非零候选诊断与局部效用验证 V2 最终报告\n\n结论状态：**'+status+'**。\n\n'+reason+'\n\n'
    report+='底座是SS-UIE官方公开简化实现，不是论文完整模型。固定上游88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df，权重977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99。旧run只读保留，恢复六数据文件与V1快照匹配；已本地提交，未推送远端。\n\n'
    report+='## 八点及新增候选诊断\n\n'+table+'\nH32/Hp为不可部署oracle相对J0空间，G32/Gp相对五固定强度最好值的空间。S为六字段干预最大跨图MAE。空间与先验响应分别验证，均不等于部署收益。B3未安排先验干预，S=0只表示not_scheduled，不作为RGB机制结论。\n\n'+trade+'\n\n'
    report+='## 候选选择和救援\n\n```json\n'+json.dumps({'producer':selected.get('selected'),'standalone':standalone,'rescue_choice':rescue},ensure_ascii=False,indent=2)+'\n```\n\nproducer是局部修正来源；standalone允许step0安全回退。选择先冻结model_val，再读取utility_val，不能按后者改选。零点曾掩盖非零空间的可能性由上表判断；非零空间存在也不是O成立。\n\n'
    report+='## 五个核心回答\n\n1. 候选是否有空间：按上表与资格回执判断；不可部署oracle和实际部署结果分开。\n2. 训练／损失诊断：'+trade+'救援仅在无合格model_val producer时按预定唯一分支进行，未运行原因见阶段表。\n3. 可部署收益：'+('完整开发结果已保存development_gate.json，见质量与强控制表。' if gate else 'O正式质量与强控制未完成，不能声称有收益或已被充分证伪。')+'\n4. O独特机制：'+('必须结合主要消融、所有强控制和数据匹配B4，不能靠oracle归因。' if gate else '尚未得到完整O／强控制／消融证据，无机制增量结论。')+'\n5. 下一轮：见NEXT_DECISION.md；当前结果不构成创新成立或自动扩大训练依据。\n\n'
    report+='## 验收、阶段与证据边界\n\n'+stages+'\n工程测试与临时真底座11方法验收可证明所测接口，不证明完整正式实验已全部运行。所有not_run按表记录，不用工程夹具数字填充正式结果。逐图长表、残差／先验响应、5000次组配对区间在diagnostics/；组是content_group_proxy，非人工确认真实scene。区间仅单种子探索性，不调整开发多重选型，也不覆盖底座预训练波动。\n\n'
    report+='## 预算、封存与恢复\n\n累计设备小时 %.8f / 16，继承V1 %.8f小时；旧账本60秒保守额度仍保留。新设备验收、profile、缓存、诊断、失败尝试、训练和测速按事件账本计时。CPU文件分析另外记录，不伪报设备训练。\n\n177对封存解锁：%s；133对calibration已评分：%s。未解锁则没有sealed_per_image.csv或最终冻结评分。历史暴露没有因新run_id消除。\n\n独立备份未核验（independent_backup_verified=false）。同盘ZIP可供下载，但不抵抗服务器磁盘丢失；下载后需在本地核验SHA256，不能把Git源码或同盘压缩包当成权重的异地备份。\n'%(budget['used_device_seconds']/3600,budget.get('inherited_device_seconds',0)/3600,s.state['sealed_eval_released'],(s.run/'selection/calibration_selection.json').exists())
    for target in [s.doc/'FINAL_REPORT.md',s.run/'FINAL_REPORT.md']:s.ctx.text(target,report)
    s.ctx.text(s.run/'NEXT_DECISION.md',(s.doc/'NEXT_DECISION.md').read_text())
    s.ctx.text(s.doc/'RUN_COMMANDS.md','# 实际入口与恢复命令\n\n```bash\ncd /mnt/workspace/uie-prior-utility\nOMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m uie_next.v2.cli status\nOMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m uie_next.v2.cli run\n.venv/bin/python -m pytest tests/uie_next -q\n.venv/bin/python -m uie_next.v2.cli closeout\n```\n\n真实执行历史见commands.jsonl；不把本文件列出的可复现命令都称已执行。终止状态run仅交付，不重复派发。完整权重有optimizer／scheduler／RNG／采样流状态；原V1身份按清单恢复。\n')
    weights=[]
    for p in [Path(s.config['backbone']['checkpoint'])]+sorted((OLD/'checkpoints').rglob('*.pt'))+sorted((s.run/'checkpoints').rglob('*.pt')):
        weights.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'official_backbone':p==Path(s.config['backbone']['checkpoint'])})
    write(s.run/'delivery/server_weights.json',weights)
    s.ctx.text(s.run/'delivery/recovery_instructions.md','# 恢复与独立备份\n\n先下载报告、审阅轻量ZIP、完整源码协议ZIP、权重恢复ZIP；在本地运行sha256sum，与archive_receipts.json核对。源码ZIP必须含uie_next/data六文件，ZIP成员hash清单逐项验证；权重ZIP包含官方底座、旧候选所有完整恢复点、V2实际新增恢复点。数据图片不公开上传；恢复时需原数据及指定上游commit/PPU依赖，度量权重另列身份。source_protocol包含固定源码与协议／数据名单，不包含数据图片。visuals含生成的面板，单独包。\n\n没有授权独立存储目录或可核实客户端下载工具；本轮只能提供服务器可下载包，独立备份仍false，需用户本地下载核验后登记。\n')
    s.complete('CLOSEOUT',{'reports':str(s.doc)});s.live(closeout_complete=True,current_job='none: scientific closeout')
    manifest={str(p.relative_to(ROOT)):sha(p) for p in list(s.doc.rglob('*'))+list(s.run.rglob('*')) if p.is_file() and p.suffix not in ['.pt','.pth','.png'] and '/cache/' not in str(p) and 'delivery/manifest' not in str(p)}
    write(s.run/'delivery/manifest_sha256.json',manifest)
    source=read(s.run/'source_snapshot.json') if (s.run/'source_snapshot.json').exists() else {}
    source_paths=[ROOT/p for p in source]+[s.run/'protocol_source.md',s.run/'protocol_resolved.yaml',s.run/'source_snapshot.json',s.run/'roles.jsonl',s.run/'exposure_ledger.json']
    upstream=ROOT/'third_party/ss_uie'
    source_paths += [p for p in upstream.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts
        and p.suffix.lower() not in ['.png','.jpg','.jpeg','.bmp','.gif','.pth','.pt','.zip','.pdf']]
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    source_paths += [ROOT/p for p in tracked if p and (ROOT/p).is_file() and Path(p).suffix.lower() not in ['.png','.jpg','.jpeg','.bmp','.gif','.pth','.pt','.zip','.pdf']]
    review=[p for p in s.run.rglob('*') if p.is_file() and p.suffix not in ['.pt','.pth','.png'] and '/cache/' not in str(p)]+[p for p in s.doc.rglob('*') if p.is_file()]
    packages={'review':review,'source_protocol':source_paths,'weights_recovery':[Path(r['path']) for r in weights]+[s.run/'delivery/server_weights.json'],
              'visuals':list((s.run/'figures').rglob('*.png'))+list((s.run/'figures').rglob('*.json'))}
    receipts={}
    for name,paths in packages.items():
        output=ROOT.parent/(s.ctx.run_id+'_'+name+'.zip');paths=[p for p in paths if p.exists()]
        receipts[name]=archive(output,sorted(set(paths)),ROOT)
    write(s.run/'delivery/archive_receipts.json',{'packages':receipts,'independent_backup_verified':False,'same_server_disk_only':True,
          'package_scope':'source/roles/protocol; weights separately; generated visuals separately; no dataset image in review/source',
          'hash_scope':'ZIP CRC plus every included member SHA256; V1 legacy preservation snapshot separately verified'})
    return {'state':s.state,'reports':str(s.doc),'archive_receipts':str(s.run/'delivery/archive_receipts.json')}
