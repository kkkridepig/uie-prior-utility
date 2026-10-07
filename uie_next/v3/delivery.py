"""Scientific stop is a complete outcome; evidence and portable recovery packages."""
import csv,json,math,shutil,subprocess,time,zipfile,hashlib
from pathlib import Path
import numpy as np
from ..records import ROOT,sha,read,write,digest
from ..v2.statistics import paired_stats
from ..v2.diagnostics import csv_write
from .context import OLD,V1,RUN_ID


def readcsv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def summarize_diagnostics(s):
    summary={};comparisons={}
    for filename,keys in [('D1_prediction_regret_per_image.csv',['lambda']),('D2_spatial_controls.csv',['variant']),('D3_common_policy_ablations.csv',['method','setting']),('D4_intervention_information.csv',['view']),('D6_matched_null_and_rgb_oracles.csv',['direction','null_index','space'])]:
        rows=readcsv(s.run/'diagnostics'/filename);by={}
        for row in rows:
            key='|'.join([row['role']]+[row[k] for k in keys]);by.setdefault(key,[]).append(row)
        for key,part in by.items():
            vals={}
            for col in part[0]:
                try:
                    x=np.array([float(r[col]) for r in part]);
                    if np.isfinite(x).all():vals[col]=float(x.mean())
                except (ValueError,TypeError):pass
            vals.update(n_images=len(part),n_groups=len({r['group_id'] for r in part}));summary[filename+'|'+key]=vals
    d2=readcsv(s.run/'diagnostics/D2_spatial_controls.csv')
    for role in ['utility_fit','utility_val']:
        p=[r for r in d2 if r['role']==role];orig={r['sample_id']:r for r in p if r['variant']=='original'}
        for name in ['mean','energy_mean','shuffle']:
            part=[r for r in p if r['variant']==name];comparisons['D2|'+role+'|original_minus_'+name]=paired_stats([float(orig[r['sample_id']]['psnr'])-float(r['psnr']) for r in part],[r['group_id'] for r in part])
    d3=readcsv(s.run/'diagnostics/D3_common_policy_ablations.csv')
    for role in ['utility_fit','utility_val']:
        for setting in ['common','V2_legal']:
            p=[r for r in d3 if r['role']==role and r['setting']==setting];o={r['sample_id']:r for r in p if r['method']=='O'}
            for method in ['O-NP','O-NI','O-ND']:
                part=[r for r in p if r['method']==method];comparisons['D3|'+role+'|'+setting+'|O_minus_'+method]=paired_stats([float(o[r['sample_id']]['psnr'])-float(r['psnr']) for r in part],[r['group_id'] for r in part])
    # Same group bootstrap seed yields identical paired resampling for comparisons.
    d1=readcsv(s.run/'diagnostics/D1_prediction_regret_per_image.csv')
    for role in ['utility_fit','utility_val']:
        for lam in ['0.0001','0.001']:
            p=[r for r in d1 if r['role']==role and r['lambda']==lam]
            for k in ['delta_J0_db','reference_only_prediction_replacement_mse','reference_only_regularization_active_loss_mse','reference_only_oracle_regret_mse']:
                comparisons['D1|'+role+'|'+lam+'|'+k]=paired_stats([float(r[k]) for r in p],[r['group_id'] for r in p])
    d6=readcsv(s.run/'diagnostics/D6_matched_null_and_rgb_oracles.csv')
    for role in ['utility_fit','utility_val']:
        for space in ['global','block32','pixel']:
            pp=[r for r in d6 if r['role']==role and r['space']==space]
            b1={r['sample_id']:r for r in pp if r['direction']=='B1_original'}
            b3={r['sample_id']:r for r in pp if r['direction']=='B3_003000'}
            ids=sorted(b1);groups=[b1[i]['group_id'] for i in ids]
            comparisons['D6|'+role+'|'+space+'|B1_minus_B3']=paired_stats([float(b1[i]['psnr'])-float(b3[i]['psnr']) for i in ids],groups)
            null_deltas=[]
            for index in ['1','2']:
                null={r['sample_id']:r for r in pp if r['direction']=='null' and r['null_index']==index}
                matched={r['sample_id']:r for r in pp if r['direction']=='matched_original' and r['null_index']==index}
                delta=np.array([float(null[i]['psnr'])-float(matched[i]['psnr']) for i in ids]);null_deltas.append(delta)
                comparisons['D6|'+role+'|'+space+'|null_'+index+'_minus_matched']=paired_stats(delta,groups)
            comparisons['D6|'+role+'|'+space+'|mean_two_null_minus_matched']=paired_stats(np.mean(null_deltas,axis=0),groups)
    write(s.run/'diagnostics/D1_D6_summary.json',summary);write(s.run/'metrics/paired_intervals.json',comparisons)
    return summary,comparisons


def stage1_visuals(s):
    import torch
    from PIL import Image,ImageDraw
    from .diagnostics import Diagnostics,tensor_np
    from .moments import geometry,expand
    from .ridge_probe import predict
    from ..v2.evaluation import apply_policy
    path=s.run/'figures/manifest.json'
    if path.exists():return read(path)
    d=Diagnostics(s);rows=readcsv(s.run/'diagnostics/D1_prediction_regret_per_image.csv');rows=[r for r in rows if r['lambda']=='0.001'];selected=[]
    for role in ['utility_fit','utility_val']:
        part=[r for r in rows if r['role']==role];fixed=sorted(part,key=lambda r:hashlib.sha256(('v3-visual|'+r['sample_id']).encode()).digest())[:8]
        ordered=sorted(part,key=lambda r:(float(r['delta_J0_db']),r['sample_id']));mid=max(0,len(ordered)//2-2)
        for reason,rr in [('hash_fixed8',fixed),('worst4',ordered[:4]),('best4',ordered[-4:]),('middle4',ordered[mid:mid+4])]:
            for r in rr:
                found=next((x for x in selected if x['sample_id']==r['sample_id']),None)
                if found:found['selection_reasons'].append(reason)
                else:selected.append({'sample_id':r['sample_id'],'role':role,'selection_reasons':[reason],'O_minus_J0_db':float(r['delta_J0_db'])})
    models=read(s.run/'diagnostics/D7_models.json')['full_fit_models'];ref=models['global']['constants']['alpha_ref'];fixed=models['global']['constants']['alpha_fit_star'];cm_scale=models['region']['constants']['s_C'];results=[]
    with s.device_job('STAGE1_FIXED_AND_FAILURE_VISUALS',estimate=120,stage=1):
        d.load()
        with torch.no_grad():
            for record in selected:
                s.guard();row=d.data.by_id[record['sample_id']];b=d.data.sample(row);I=b['image'][None].cuda();N=b['base'][None].cuda();J=b['candidate'][None].cuda();Y=b['target'][None].cuda();p=d.controllers['O'](I,N,J);onp=d.controllers['O-NP'](I,N,J);out,alpha=apply_policy('O',d.cal['O']['policy'],N,J,p);ab,_=apply_policy('O-NP',d.cal['O-NP']['policy'],N,J,onp)
                base=tensor_np(N);cand=tensor_np(J);target=tensor_np(Y);g=geometry(base,cand,target,32);fixedout=base+fixed*(cand-base);C=g['B']-ref*g['A']
                # Ridge's own alpha_ref is common across scales; C is reference-only.
                from .descriptors import descriptors
                fr,fg=descriptors(tensor_np(I),base,cand);dd={'A':g['A'].reshape(1,64),'B':g['B'].reshape(1,64),'mse0':g['mse0'],'f_region':fr.reshape(1,26,64).transpose(0,2,1),'f_global':fg.reshape(1,26)}
                _,chat=predict(models['region'],dd,np.array([0]));chat=chat.reshape(1,1,8,8);err=((tensor_np(out)-target)**2).mean(1,keepdims=True)-((base-target)**2).mean(1,keepdims=True)
                tiles=[('I',tensor_np(I)[0]),('Y reference only',target[0]),('J0 / primary',base[0]),('B1 endpoint',cand[0]),('FIXED_FIT',fixedout[0]),('O old deployed',tensor_np(out)[0]),('O-NP old ablation',tensor_np(ab)[0]),('O alpha',tensor_np(alpha)[0]),('error diff fixed +/-0.02',err[0]),('RIDGE predicted C',expand(chat,32)[0]),('reference-only C',expand(C,32)[0])]
                panel=Image.new('RGB',(4*256,3*280),(30,30,30));draw=ImageDraw.Draw(panel)
                for k,(label,arr) in enumerate(tiles):
                    if arr.shape[0]==3:pix=np.clip(arr.transpose(1,2,0),0,1)
                    elif 'alpha' in label:pix=np.repeat(np.clip(arr[0],0,1)[...,None],3,-1)
                    else:
                        limit=.02 if 'error' in label else cm_scale;z=np.clip(arr[0]/limit,-1,1);pix=np.stack([1-np.maximum(-z,0),1-np.abs(z),1-np.maximum(z,0)],-1)
                    x=(k%4)*256;y=(k//4)*280;panel.paste(Image.fromarray(np.uint8(np.clip(pix*255,0,255))), (x,y+24));draw.text((x+4,y+5),label,fill='white')
                filename=digest(record['sample_id'])[:16]+'.png';dest=s.run/'figures'/filename;dest.parent.mkdir(parents=True,exist_ok=True);panel.save(dest)
                results.append({**record,'path':str(dest.relative_to(s.run)),'sha256':sha(dest),'color_limits':{'error':.02,'C':cm_scale},'PSNR_source':'float tensors; PNG only display','stage':'stage1_O_vs_J0_no_new_S','sealed':False})
    manifest={'panels':results,'n_panels':len(results),'visual_selection_uses_Y_only_for_prespecified_case_reporting':True,'independent_backup_verified':False,'C_scale_from_utility_fit':cm_scale,'CAL_read':False,'sealed_read':False}
    write(path,manifest);return manifest


def render_reports(s,summary,paired):
    dec=read(s.run/'stage1_decision.json');probe=read(s.run/'diagnostics/D7_summary.json');b=read(s.run/'budget.json');status=s.state['status'];stop=dec['route']=='STOP_NO_PREDICTABILITY_SIGNAL'
    grad=[json.loads(x) for x in (s.run/'diagnostics/D5_loss_gradient_components.jsonl').read_text().splitlines()]
    d1=summary['D1_prediction_regret_per_image.csv|utility_val|0.001'];d4=[v for k,v in summary.items() if k.startswith('D4_intervention_information.csv|utility_fit|') and not k.endswith('|nominal')]
    table='|集合|策略|PSNR|相对J0|相对FIXED_FIT|损害比例|\n|---|---|---:|---:|---:|---:|\n'
    for role,v in probe.items():
        for m in ['FIXED_FIT','RIDGE_GLOBAL','RIDGE_REGION']:
            p=v[m];table+=f"|{role}|{m}|{p['psnr']:.6f}|{p['delta_base']:+.6f}|{p['vs_fixed']['mean_image']:+.6f}|{p['harm_rate']:.6f}|\n"
    spatial='\n'.join('- '+k+': '+f"{v['mean_image']:+.8f} dB，95%组配对区间 {v['ci95']}" for k,v in paired.items() if k.startswith('D2|utility_val'))
    text=f'''# SS-UIE V3 两阶段诊断与条件训练最终报告

状态：**{status}**。底座是SS-UIE官方公开简化实现，不是论文完整模型。

## 身份、工程和数据

协议SHA256：`{s.ctx.protocol_sha256}`。起点0521bfc；新独立分支，V1/V2原件只读保留。producer固定B1_003000，SHA256 c1af746056f0419eeb5787283a5c5e28d15d1ceec95d1dd6cedd4ed7592879af；未重新选候选。角色清单保持原哈希e6176357cbccc2424c268c245ef7e8c3b7dd36c4ef7f2fbbd2fa9b4070b520ce。

厂商torch/PPU保持；CPU验收和真实8图缓存/实时严格加载回归见tests。初始系统Python缺cv2，已使用原项目.venv；没有安装或替换torch。D5最初float32梯度求和误差7.11e-5；关闭该诊断的TF32后最大差降至1.03e-6，保持原atol=1e-5/rtol=1e-4，未放宽。D1–D4/D6身份与推理不受影响。新描述子初始float32方差消减错误在正式结果前修补；继承base缓存分别位于V1/V2，修补了只找V1的路径错误，失败设备成本已入账。所有影响结果的双方记录必须重算，未择优保留。详细工程状态见IMPLEMENTATION_CONFORMANCE。

utility_fit 443图/440内容代理组；utility_val 136图/132组；CAL133与sealed177本阶段不评分。作者逐图训练名单未知；OOF不消除历史分析暴露，内容组不是人工核实场景。

## D1：空间、预测替换与正则限制

旧O正式lambda=1e-3、tau=0：DEV PSNR {d1['psnr']:.9f} dB，真实采用收益 {d1['delta_J0_db']:+.9f} dB。

像素精确oracle regret均值：{d1['reference_only_oracle_regret_mse']:.10g} MSE；带符号预测替换差：{d1['reference_only_prediction_replacement_mse']:.10g}；正则/active损失：{d1['reference_only_regularization_active_loss_mse']:.10g}。两项按MSE严格相加，不当非负归因百分比。边界项均值 {d1['reference_only_regret_boundary_mse']:.10g}；不能丢弃边界项。v/b/U的逐图RMSE、MAE、相关及符号一致率已保存，常数相关为null。另有lambda=1e-4完整对应表。

候选oracle只使用参考图作诊断，并非可部署方法收益。B1/B3与两匹配随机方向的全图/32块/像素oracle、缩幅能量保留比例和legacy active版本见D6；随机方向不读取Y生成，原方向也同幅缩小。

## D2–D5：位置、干预与梯度

{spatial}

O与O-NP/NI/ND的真实3000点分别报告统一lambda及旧合法策略，逐图输出和预测差都保留；不重新选点。七视图的方向、U/B/最优alpha变化与有效秩见D4。非微小目标差阈值0.01*s_U仅为描述，不触发路线。

D5共{len(grad)}个固定训练批次，不做optimizer.step；三项真实可微loss的梯度范数/夹角、抵消比例和共同裁剪系数均保存。共同裁剪系数范围 {min(x['common_clip_coefficient'] for x in grad):.6g}–{max(x['common_clip_coefficient'] for x in grad):.6g}。梯度不是对detach日志求得，旧参数不变。干预信息或梯度大小本身不能确诊欠训练或证明机制无用。

## D7：部署信息可预测性与唯一分派

{table}

路由：**{dec['route']}**。全部逐项条件见stage1_decision.json；固定0.01岭系数、五折内容组、每折独立固定强度/尺度/标准化，截距不罚。两个全量拟合对象只用443图，各在136图评估一次；未使用CAL调整阈值或特征。RIDGE是诊断拟合，不称完全无训练。

区域−全局：OOF {probe['oof']['region_minus_global']['mean_image']:+.6f} dB、DEV {probe['utility_val']['region_minus_global']['mean_image']:+.6f} dB；DEV块oracle−图oracle {probe['utility_val']['oracle_region_minus_global']:+.6f} dB。可行性+0.03/+0.02不是正式成功门槛。

## 条件训练、开发与封存

{'D7未通过，按协议科学停止：不启动MOM/DIRECT正式矩阵，不训练旧O或候选，不追加步数、参数网格或其他路线。条件训练调度与正式评估入口未实现/未运行；第二阶段profile/正式训练、网络选点、CAL、正式DEV闸门、sealed确认、正式新方法压力/全流程测速均not_run，原因是前置可预测性闸门未解锁。5505参数模块的合成测试通过不能替代正式收益。' if stop else '第二阶段结果见selection与独立训练/评测回执；本段在阶段完成后补充。'}

sealed_eval_released=false，177对保持本轮未评分；不称完全独立盲测。本轮无新增训练种子重复，未验证独立底座波动、独立数据或物理先验不可替代性。

## 预算、交付与下一步

累计 {b['used_device_seconds']:.6f}/57600设备秒（{b['used_device_seconds']/3600:.6f}/16小时），继承15623.031541585922秒已包含V1，不重复相加。新增含加载、缓存验收、失败、诊断与视觉，详见budget_ledger/device_events。纯CPU分析另记；其中29.683683秒因旧GPU模型仍驻留而保守计设备成本，不冒充GPU计算。未启动项目不填虚构训练时间。至少3.5小时收尾额度保持。

提供审阅、完整源码协议、视觉和完整权重恢复四包；ZIP CRC与逐成员SHA在服务器核验，六个原data模块与V3文件必须齐全。服务器同盘包不是独立备份，independent_backup_verified=false；没有用户本地工具连接，不能代用户在本地下载并声称异地校验。

工程忠实性、候选oracle空间、可见信息预测性、可部署强对照收益与特殊机制增量是五个独立结论。本轮结果不构成原创算法成立。{'当前有限统计/ridge没有足够预登记可预测性信号，停止此配方。先重新定义可见信息和学习问题的独立协议，不因oracle较大就直接多种子或无限加训。' if stop else '后续结论以完整第二阶段闸门为准。'}
'''
    quality=read(s.run/'diagnostics/D7_quality_summary.json')
    text+='\n## 补充质量、失败诊断与结论分层\n\n|集合|策略|SSIM|LPIPS|\n|---|---|---:|---:|\n'
    for role in ['oof','utility_val']:
        for method in ['J0','FIXED_FIT','RIDGE_GLOBAL','RIDGE_REGION']:
            q=quality[role+'|'+method]['image_weighted'];text+=f"|{role}|{method}|{q['ssim']:.6f}|{q['lpips']:.6f}|\n"
    text+='\n补充PSNR/SSIM/LPIPS、逐内容组和组等权敏感性见D7_quality_*，没有改变D7选择条件。固定强度来自utility_fit的10001值网格，alpha='+str(read(s.run/'diagnostics/D7_models.json')['full_fit_models']['global']['constants']['alpha_fit_star'])+'；没有采用旧CAL的事后0.2325。\n'
    text+='\n旧O在DEV的空间均值、能量均值和固定置乱对照均没有支持正的位置增量，区间也跨零。七视图的平均方向余弦约0.978–0.999、有效秩均值约2.902；干预U差RMS约0.023–0.106倍旧s_U，提示所测配对变化较弱，但不据此宣称干预不可能有用。D5加权pair梯度范数仅约0.013–0.018，projection约2.41–3.21，decision约0.12–0.39；这是固定两个批次的解释性证据，不是训练充分性结论。\n'
    text+='\n候选空间存在；本轮有限可见统计/ridge可预测性未达线；没有进入新神经训练，因此MOM/DIRECT相对强控制、矩监督/全局上下文/区域机制增量均未检验，不能记成正式训练后的科学失败。单种子、历史暴露数据与内容代理分组不支持独立泛化或原创性结论。未来不能直接用更长训练替代新信息假设。\n'
    s.ctx.text(s.doc/'FINAL_REPORT.md',text);s.ctx.text(s.run/'FINAL_REPORT.md',text)
    nxt=f"# 下一轮决策\n\n状态：{status}；唯一预登记路由：{dec['route']}。\n\n"+('停止本轮受限表示和探针配方，不启动神经控制器、候选续训或更长训练。失败不证明所有网络不可学习；oracle仍只是参考图辅助空间。未来先提出可审计的新信息/目标假设，再申请独立协议；不直接用加训练保证正结果。\n' if stop else '以第二阶段正式对照与闸门结果决定；不得自动追加预算。\n')+'\n封存177未评分；独立备份尚未核验。请优先下载并核验报告、审阅、源码和权重包。\n'
    s.ctx.text(s.doc/'NEXT_DECISION.md',nxt);s.ctx.text(s.run/'NEXT_DECISION.md',nxt)
    s.ctx.text(s.doc/'D1_D7_EVIDENCE.md','# D1–D7证据索引\n\n所有逐图表、梯度、折清单、完整ridge拟合与布尔闸门位于runs/'+RUN_ID+'/diagnostics/。D1/D2/D3含PSNR/SSIM/LPIPS。D6/D7是二次式PSNR/误差诊断，未将oracle列作可部署排名。配对区间5000次、seed20261017，内容组重采样后图像等权。\n\n'+table+'\n完整D1–D6汇总见D1_D6_summary.json，配对区间见metrics/paired_intervals.json。\n')


def package(s):
    d=s.run/'delivery';d.mkdir(parents=True,exist_ok=True)
    s.snapshot_final={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'uie_next').rglob('*.py'))+list((ROOT/'tests').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'configs').rglob('*.yaml'))+list((ROOT/'configs').rglob('*.json'))}
    write(s.run/'final_source_snapshot.json',s.snapshot_final)
    src=[ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p]+[ROOT/p for p in s.snapshot_final]+[ROOT/'.gitignore',ROOT/'pyproject.toml',ROOT/'requirements-mpa-ppu.txt']+list(s.doc.rglob('*.md'))
    src += [p for r in [V1,OLD] for p in r.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.yaml','.md'] and 'cache' not in p.relative_to(r).parts and 'audit_history' not in p.relative_to(r).parts and 'delivery' not in p.relative_to(r).parts]
    src += [p for p in s.run.rglob('*') if p.is_file() and p.suffix in ['.py','.npz','.npy']]
    src += [p for p in (s.run/'delivery').glob('*.bundle')]
    src += [s.run/'delivery/recovery_instructions.md']
    src+=[p for p in (ROOT/'third_party/ss_uie').rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix not in ['.pth','.pt','.png','.jpg']]
    evidence=[p for p in s.run.rglob('*') if p.is_file() and 'delivery' not in p.relative_to(s.run).parts and 'figures' not in p.relative_to(s.run).parts and p.suffix not in ['.pt','.png','.npz','.npy']]
    visuals=list((s.run/'figures').rglob('*')) if (s.run/'figures').exists() else []
    weights=[]
    for run in [V1,OLD,s.run]:weights += [p for p in (run/'checkpoints').rglob('*') if p.is_file() and p.suffix in ['.pt','.json','.jsonl']]
    backbone=Path(s.config['backbone']['checkpoint']);weights+=[backbone,ROOT/'weights/cde_v3/vgg16-397923af.pth']
    # LPIPS learned coefficients are package assets and must be portable as well.
    import lpips
    assets=Path(lpips.__file__).parent/'weights';extra=[p for p in assets.rglob('*') if p.is_file()]
    payloads={'review':evidence+list(s.doc.rglob('*.md')),'source_protocol':src+evidence,'visuals':[p for p in visuals if p.is_file()],'weights_recovery':weights+[s.run/'recovery_audit.json',s.run/'method_registry.json',OLD/'method_registry.json',OLD/'selection/calibration_selection.json']}
    packages={}
    for name,paths in payloads.items():
        unique={str(p.relative_to(ROOT)):p for p in paths if p.exists()}
        if name=='weights_recovery':unique.update({'runtime_assets/lpips/weights/'+str(p.relative_to(assets)):p for p in extra})
        member_hashes={n:sha(p) for n,p in unique.items()};manifest={'package':name,'run_id':RUN_ID,'members':member_hashes}
        membername='runs/'+RUN_ID+'/delivery/package_member_hashes/'+name+'.json';write(d/'package_member_hashes'/(''+name+'.json'),manifest)
        dest=ROOT.parent/(RUN_ID+'_'+name+'.zip');tmp=dest.with_suffix('.zip.tmp')
        reusable=False
        if dest.exists():
            try:
                with zipfile.ZipFile(dest) as previous:reusable=json.loads(previous.read(membername))==manifest
            except (KeyError,ValueError,zipfile.BadZipFile):pass
        if not reusable:
            with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for n,p in sorted(unique.items()):z.write(p,n)
                z.write(d/'package_member_hashes'/(''+name+'.json'),membername)
            tmp.replace(dest)
        with zipfile.ZipFile(dest) as z:
            assert z.testzip() is None
            for n,h in member_hashes.items():
                hh=hashlib.sha256(z.read(n)).hexdigest()
                if hh!=h:raise ValueError('member mismatch '+n)
            if name=='source_protocol':
                original=read(V1/'source_snapshot.json')
                for n in ['__init__.py','audit.py','cache.py','manifest.py','roles.py','runtime.py']:
                    f='uie_next/data/'+n
                    if hashlib.sha256(z.read(f)).hexdigest()!=original[f]:raise ValueError('data source package mismatch')
                for n,h in s.snapshot_final.items():
                    if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('final source package mismatch')
        packages[name]={'path':str(dest),'sha256':sha(dest),'size_bytes':dest.stat().st_size,'CRC_pass':True,'all_member_SHA256_pass':True,'members':len(member_hashes),'independent_copy':False}
    write(d/'manifest_sha256.json',packages);write(d/'SERVER_PACKAGE_VERIFICATION.json',{'packages':packages,'independent_backup_verified':False,'scope':'zip bytes and every payload member; not algorithm correctness'})
    (ROOT.parent/(RUN_ID+'_SHA256SUMS.txt')).write_text(''.join(v['sha256']+'  '+Path(v['path']).name+'\n' for v in packages.values()))
    return packages


def closeout(s):
    dec=read(s.run/'stage1_decision.json')
    if dec['route']=='STOP_NO_PREDICTABILITY_SIGNAL':
        for k in ['STAGE2_ACCEPTANCE_AND_PROFILE','TRAINING_SCHEDULE_FREEZE','COMPLETE_MATCHED_MATRIX','NETWORK_FREEZE','CALIBRATION','DEV_GATE','SEALED_ONCE']:
            s.state['stages'][k]={'status':'not_run','reason':'D7 pre-registered predictability gate not passed'}
        s.live(status=dec['route'])
    summary,paired=summarize_diagnostics(s);stage1_visuals(s)
    registry={'RIDGE_GLOBAL':{'object_file':'diagnostics/D7_models.json','scale':'global','requires_reference_at_deploy':False,'kappa':1,'state':'diagnostic_only_no_CAL'},
      'RIDGE_REGION':{'object_file':'diagnostics/D7_models.json','scale':'region','requires_reference_at_deploy':False,'kappa':1,'state':'diagnostic_only_no_CAL'},
      'O':{'source_run':OLD.name,'checkpoint':'checkpoints/O/step_003000.pt','weight_sha256':sha(OLD/'checkpoints/O/step_003000.pt'),'policy':{'tau':0,'lamb':.001},'requires_reference_at_deploy':False},
      'oracle':{'requires_reference':True,'deployable':False,'state':'diagnostic_only'}}
    registry['J0']={'source_run':V1.name,'kind':'backbone','baseline_policy':'clip01','backbone_sha256':s.config['backbone']['checkpoint_sha256'],'requires_reference_at_deploy':False}
    registry['B1_003000']={'source_run':V1.name,'checkpoint':'checkpoints/B1/step_003000.pt','candidate_sha256':sha(V1/'checkpoints/B1/step_003000.pt'),'kind':'endpoint','requires_reference_at_deploy':False}
    registry['B3_003000']={'source_run':V1.name,'checkpoint':'checkpoints/B3/step_003000.pt','candidate_sha256':sha(V1/'checkpoints/B3/step_003000.pt'),'kind':'endpoint','requires_reference_at_deploy':False}
    registry['FIXED_FIT']={'source_run':s.ctx.run_id,'alpha':read(s.run/'diagnostics/D7_models.json')['full_fit_models']['global']['constants']['alpha_fit_star'],'policy_source_role':'utility_fit','requires_reference_at_deploy':False}
    policies=read(OLD/'selection/calibration_selection.json')['methods']
    for method in ['O-NP','O-NI','O-ND']:
        registry[method]={'source_run':OLD.name,'checkpoint':'checkpoints/'+method+'/step_003000.pt','weight_sha256':sha(OLD/'checkpoints'/method/'step_003000.pt'),'common_policy':{'tau':0,'lamb':.001},'V2_legal_policy':policies[method]['policy'],'requires_reference_at_deploy':False,'diagnostic_step_fixed':3000}
    registry['conditional_formal_matrix']={'status':'not_run','reason':'D7 scientific stop; no neural training/CAL/DEV/sealed unlocked'}
    write(s.run/'method_registry.json',registry)
    # Verify inherited artifacts again; no V3 cache/selection writes are allowed there.
    expected=read(s.run/'inherited_files_sha256.json');changed=[p for p,h in expected.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    if changed:raise ValueError('old artifacts changed '+str(changed))
    write(s.run/'tests/old_artifacts_unchanged.json',{'passed':True,'files':len(expected),'changed':[]})
    s.complete('CLOSEOUT',{'scientific_status':s.state['status'],'sealed_eval_released':False})
    s.live(current_job='CLOSEOUT_COMPLETE',error=None)
    render_reports(s,summary,paired)
    s.ctx.text(s.doc/'RUN_COMMANDS.md','# 实际命令与恢复\n\n从仓库根执行；厂商torch保留，使用项目.venv。\n\n```bash\nOMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.ssuie_v3_run audit\nOMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/ssuie_v3 tests/uie_next\nOMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.ssuie_v3_run stage1\nOMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.ssuie_v3_run closeout\n```\n\n实际失败与成功日志及events保留在新run；不调用旧pipeline。新环境先恢复源码、厂商环境、数据，再解压weights_recovery到仓库根，逐项核对recovery_audit中的hash。角色原路径如变更需显式身份迁移，不能只按数量重划。数据本体不在交付包，SS-UIE上游源码固定commit，原Git对象另包/源码身份恢复要求见恢复说明。\n')
    s.ctx.text(s.run/'delivery/recovery_instructions.md','# 恢复与备份边界\n\n四包同时保留并核验SHA256、ZIP CRC及每包内members清单。源码包包含六个原始data模块与新V3源码/测试/协议及固定上游源码；权重包包含底座、所有继承正式检查点和感知指标权重，路径相对仓库根；不依赖服务器私有绝对权重路径。实际数据需另存并按原roles逐项hash核验；厂商PPU环境需独立保存，不能用普通NVIDIA wheel替换。\n\n服务器同盘ZIP不是独立备份，尚无本地下载执行通道。下载后再核验并保留回执；本轮不会自动推送数据、图片、权重或GitHub。\n')
    package(s)
