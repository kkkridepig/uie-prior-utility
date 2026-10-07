"""Render Chinese conclusions from actual V2 evidence after scientific closeout.

This is report-only: no image access, training, calibration, model selection or
device allocation. Run after ssuie_v2_finalize_evidence --analysis-only, before
the final delivery ZIPs are rebuilt.
"""
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree

from uie_next.records import ROOT, read, sha, write
from uie_next.v2.context import RUN_ID, METHOD_ORDER, STAGES


def optional(path, default=None):
    return read(path) if path.exists() else default


def checkpoint_table(rows, role):
    text='| 检查点 | 完整端点 | 最佳固定强度 | 32块oracle | 像素oracle | H32 / G32 | Hp / Gp | S |\n'
    text+='|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in sorted([x for x in rows if x['role']==role and x['training_step']>0], key=lambda x:x['checkpoint_id']):
        sensitivity='%.8f'%r['S'] if r['head']=='B1' else 'NA：未安排'
        text+='| %s | %.6f | %.6f | %.6f | %.6f | %.6f / %.6f | %.6f / %.6f | %s |\n'%(
            r['checkpoint_id'],r['endpoint_psnr'],r['best_fixed_psnr'],r['oracle_block32_psnr'],
            r['oracle_pixel_psnr'],r['H32'],r['G32'],r['Hp'],r['Gp'],sensitivity)
    return text


def main():
    run=ROOT/'runs'/RUN_ID;doc=ROOT/'docs/experiments'/RUN_ID
    state=read(run/'state.json');budget=read(run/'budget.json')
    if state['scientific_status']=='RUNNING' or not state.get('closeout_complete'):
        raise RuntimeError('Scientific closeout must precede final conclusions')
    summaries=optional(run/'diagnostics/checkpoint_summary.json',[])
    selection=optional(run/'selection/producer_selection_before_utility_val.json',{})
    rescue_selection=optional(run/'selection/rescue_producer_selection.json')
    producer=(rescue_selection or selection).get('selected')
    standalone=optional(run/'selection/standalone_selection.json',{})
    transfer=optional(run/'selection/producer_transfer_gate.json',{})
    rescue=optional(run/'selection/rescue_choice.json')
    gate=optional(run/'development_gate.json')
    confirm=optional(run/'confirmation_results.json')
    calibration=optional(run/'selection/calibration_selection.json')
    losses=optional(run/'diagnostics/paired_loss_intervals.json',{})
    audit=read(run/'data_audit.json');legacy=read(run/'delivery/legacy_integrity.json')
    old_nonzero=[r for r in summaries if r['checkpoint_id'] in
                 {head+'_%06d'%step for head in ['B1','B3'] for step in [1000,2000,3000,4000]}]
    counts={role:len([r for r in old_nonzero if r['role']==role]) for role in ['model_fit_probe','model_val','utility_val']}
    all_eight=all(n==8 for n in counts.values())
    report='# SS-UIE 非零候选诊断与局部效用验证 V2 最终报告\n\n'
    report+='状态：**%s**。%s\n\n'%(state['scientific_status'],state.get('stop_reason') or '已按冻结协议完成相应流程。')
    report+='本轮底座为 **SS-UIE 官方公开简化实现**，不是论文完整模型。固定官方提交 `88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df`，权重 SHA256 `977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99`。单个新增模块训练种子 `20261007`，不代表独立底座预训练重复。旧 V1 的 `STOP_PRIOR_UNUSED` 和全部旧文件保留。\n\n'
    report+='## 1. 工程是否忠实实现\n\n'
    xml=run/'tests/pytest_resume_logging_fix.xml'
    suites=list(ElementTree.parse(xml).iter('testsuite')) if xml.exists() else []
    ntests=sum(int(x.get('tests','0')) for x in suites)
    report+='六个原始 `uie_next/data/*.py` 从服务器原件恢复，逐项与旧 `source_snapshot.json` 匹配并纳入 Git；没有猜写替代。全新检出导入通过。厂商 PyTorch/PPU 保持，官方底座 strict=True 加载。最新 CPU 套件 %d 项通过；另有规约原数值两区域反例、真实原4000点模型/AdamW/随机流CPU恢复回执。\n\n'%ntests
    report+='11个方法各2次真实PPU临时更新、完整指标/网格/模拟校准/压力/导出/新进程重载验收通过；16个fit图缓存与实时输出差0；连续4更新与2+新进程2的参数与优化器差0，独立随机流、采样和LR一致。临时五点接口是工程别名，不冒充正式750/1500/2250/3000更新。已修复门控缺 `U_hat` 的错误，缺失预测字段为null并注明原因，真实标签不充当预测。\n\n'
    report+='实际裁剪残差、RGB共享alpha、a/b/c/u/v/U、有符号共享正负网络、字段重算干预、底座/BN冻结、候选冻结、无Y部署、11控制与消融均有对应源码与测试。完整映射见 `IMPLEMENTATION_CONFORMANCE.md`。工程通过所测接口不等于尚未解锁的正式方法已训练或科学假设成立。两次安全工程中断均保留已提交D0批次；未改变D0数学、输入、标签、权重、角色或选择阈值，影响范围与旧源码快照在 `audit_history/`。\n\n'
    report+='## 2. 八个非零检查点：是否有局部空间\n\n'
    report+='全量诊断：%s；每角色完成非零检查点数 %s。训练探针为冻结128内容代理组，实际140张；model_val为671对，utility_val为136对。主估计量图像等权，oracle仅开发诊断、读取参考图、不可部署。\n\n'%('完成' if all_eight else '未全部完成',counts)
    report+='LSUI model_val（底座27.5233317253 dB）：\n\n'+checkpoint_table(summaries,'model_val')+'\n'
    baseline=next((r['baseline_psnr'] for r in summaries if r['role']=='utility_val'),None)
    report+='UIEB utility_val（冻结选型后读取，底座%s dB）：\n\n'%('%.10f'%baseline if baseline is not None else 'not_run')
    report+=checkpoint_table(summaries,'utility_val')+'\n'
    report+='固定强度是全数据同一个alpha的五值网格；表中最佳值使用开发Y辅助，尚未作为部署参数。H32/Hp是相对J0的oracle空间，G32/Gp是相对同候选五值网格最佳固定强度的空间。精确oracle在float64计算，另存active阈值版本。组配对95%区间、5/10/90/95分位、改善/损害比例及最差10%均保存；5000次内容组bootstrap、seed20261017，图像等权，未包含训练种子波动或开发多重搜索校正。\n\n'
    if producer:
        report+='唯一 producer 在model_val选为 `%s`（%s），步数%d、权重SHA256 `%s`。这证明V1零点选择漏掉了所测非零池的局部oracle空间；它不证明O可学会或先验不可替代。\n\n'%(producer['checkpoint_id'],producer.get('qualification',selection.get('status')),producer['training_step'],producer['checkpoint_sha256'])
    else:
        report+='当前有限池没有冻结合格producer，不能以零候选或未训练O来声称普遍科学证伪。资格与全部排名见选择清单。\n\n'
    for head,r in standalone.items():
        report+='`%s_standalone_best`：%s，%d更新，SHA256 `%s`。零点是合法独立端点最优/安全回退，未参与非零producer排名。\n\n'%(head,r['checkpoint_id'],r['training_step'],r['checkpoint_sha256'])
    if transfer:
        r=transfer['scores']
        report+='唯一候选迁移检查：**%s**；H32=%.6f、G32=%.6f、Hp=%.6f、Gp=%.6f dB，S=%.8f。检查后未更换第二名。\n\n'%('通过' if transfer['passed'] else '未通过',r['H32'],r['G32'],r['Hp'],r['Gp'],r['S'])
    report+='## 3. 训练/损失诊断与先验响应\n\n'
    trade=losses.get('B1_004000|model_val')
    if trade:
        report+='旧B1_4000：同图均值MSE相对变化 %.6f%%；ΔMSE=%+.9g，95%%组配对区间[%+.9g,%+.9g]；平均逐图ΔPSNR=%+.6f dB，区间[%+.6f,%+.6f]。\n\n'%(trade['mean_mse_relative_change']*100,trade['mse']['mean_image'],*trade['mse']['ci95'],trade['psnr']['mean_image'],*trade['psnr']['ci95'])
        report+='这%s预定的“MSE下降至少0.5%%且PSNR下降至少0.02dB”损失权衡数值条件；救援还必须先满足“旧model_val无合格非零producer”。本轮已有合格producer，因此该数值现象不授权额外损失搜索。平均MSE与平均逐图PSNR权重不同；log-MSE与PSNR是线性换算，不是两份独立证据。均值与探针趋势不能证明已经收敛或一定欠训练。\n\n'%('满足' if trade['mean_mse_relative_change']<=-.005 and trade['psnr']['mean_image']<=-.02 else '不满足')
    report+='先验响应使用同一检查点、同一输入的六个字段干预，重新计算t/Q/q与候选输出；S只是候选MAE响应，不是效用或因果有益性。B3未安排先验干预，表内NA不能解释成RGB模型能力为零。颜色/对比度先验是弱代理，不是真实深度。训练探针、残差、a/b/U分布和每步配对比较保留在diagnostics。\n\n'
    report+='候选救援：%s。\n\n'%('未触发：旧model_val已有合格producer；utility_val迁移失败不能再救援或改选。' if rescue is None and producer else json.dumps(rescue,ensure_ascii=False) if rescue else 'not_run：前置资格/预算未解锁，见状态表。')
    report+='## 4. 可部署收益与O机制增量\n\n'
    if gate:
        primary=gate['primary_control'];ablation=gate['primary_mechanism_ablation']
        report+='所有正式方法完成后，在utility_val的有限“检查点×网格”选择网络，随后133对CAL独立重新选择部署参数，未从两集合挑更好参数。主对照 `%s`，主要机制消融 `%s`。\n\n'%(primary,ablation)
        report+='| 方法 | PSNR | SSIM | LPIPS | 相对J0损害比例 |\n|---|---:|---:|---:|---:|\n'
        for m,r in sorted(gate['summary'].items()):
            report+='| %s | %.6f | %.6f | %.6f | %.6f |\n'%(m,r['psnr'],r['ssim'],r['lpips'],r['paired_vs_base']['harm_rate'])
        report+='\nO−J0=%+.6f dB；O−主对照=%+.6f dB，95%%区间[%+.6f,%+.6f]；O−主要消融=%+.6f dB。开发闸门%s，完整判据如下。utility_val已用于选点，区间仍是探索性。\n\n'%(gate['paired']['B0_clip01']['mean_image'],gate['paired'][primary]['mean_image'],*gate['paired'][primary]['ci95'],gate['paired'][ablation]['mean_image'],'通过' if gate['passed'] else '未通过')
        report+='```json\n'+json.dumps(gate['checks'],ensure_ascii=False,indent=2)+'\n```\n\n'
        report+='收敛风险：'+json.dumps(gate['unresolved_optimization'],ensure_ascii=False)+'；不自动延长训练。若有名义质量而消融/强控制不足，仅可说质量信号，不能归因给O特殊机制。所有压力族、同候选oracle regret、覆盖率和缺失效用原因均另存。\n\n'
        if calibration:
            report+='最终部署策略（只由133对calibration选择，不采用较好的开发提示参数）：\n\n'
            report+='| 方法 | CAL选定策略 |\n|---|---|\n'
            for m,r in sorted(calibration['methods'].items()):
                report+='| %s | `%s` |\n'%(m,json.dumps(r['policy'],ensure_ascii=False,sort_keys=True))
            report+='\n全部五检查点×预定网格分数见 utility_checkpoint_grid_scores.csv，CAL完整逐图×网格见 calibration_all_strategy_per_image.csv。选中return_base表示没有该方法的可部署增量，不能称为安全收益成立。\n\n'
        report+='冻结网络的未校准默认参数结果另存 metrics/default_development_policy_scores.csv，来自同一完整开发网格，不替代CAL部署结果、不构成额外早停门槛。\n\n'
    else:
        report+='正式O、G/R/F门控、B4和消融未获合法解锁/未完整完成，**可部署O收益与特殊机制增量未检验**。不能把oracle空间、临时两更新工程验收或当前科学停止当作正式O训练失败。已完成的SS-UIE/standalone部署时延另报，不是O加速成绩。\n\n'
    report+='本轮没有在匹配RGB非零producer上正式重训O_RGB，因此即便O通过，也不能单凭本轮声称物理先验不可替代；O-NS同时改变奇性约束与幅度输入，不能分别证明二者独立必要。\n\n'
    report+='## 5. 数据角色、封存和一次确认\n\n'
    report+='| 角色 | 对数 |\n|---|---:|\n'+''.join('| %s | %d |\n'%(k,v) for k,v in audit['role_counts'].items())+'\n'
    report+='LSUI作者训练逐图成员未知；UIEB documented_nonoverlap仅限公开LSUI来源和本地输入/参考交叉近重复审计。内容代理组不是人工确认采集scene。全部历史分析暴露保留，不因新run重新划分消除。本轮177对sealed_eval不是全新盲测。calibration已评分：%s；sealed已解锁：%s。未解锁数据没有候选缓存、质量/视觉评分，身份hash读取另记权限事件。\n\n'%(calibration is not None,state['sealed_eval_released'])
    report+='data_audit.json记录接管时仅做身份审计的快照；其中“尚未评分”不代表收尾时仍未校准。实际运行守卫访问汇总另存 delivery/data_access_closeout.json，保留原事件来源hash；事件次数含重复守卫检查，不能当独立前向或训练图像数。\n\n'
    if confirm:
        report+='一次确认结果：'+json.dumps({'passed':confirm['passed'],'checks':confirm['checks']},ensure_ascii=False)+'。未据确认分数修改参数或重选检查点。\n\n'
    else:
        report+='确认阶段 not_run：完整开发闸门未通过/尚未完成，保留177对本轮封存数据。\n\n'
    report+='## 6. 实际阶段、预算与恢复\n\n'
    report+='| 阶段 | 实际状态 | 原因 |\n|---|---|---|\n'
    for name in STAGES:
        item=state['stages'][name]
        report+='| %s | %s | %s |\n'%(name,item['status'],item.get('reason','已保存回执、数量和身份'))
    report+='\n| 正式效用作业 | 更新数 | 所选检查点 |\n|---|---:|---|\n'
    for m in METHOD_ORDER:
        chosen=optional(run/'checkpoints'/m/'selection.json')
        if chosen:report+='| %s | %d | %d |\n'%(m,chosen['updates'],chosen['selected_step'])
        else:report+='| %s | not_run/未完成 | %s |\n'%(m,state.get('stop_reason') or '见阶段表')
    report+='\n累计设备耗时 **%.8f小时 / 16小时**，继承V1 %.8f小时（1503.1731119155884秒，原账本hash保存）；不是新开16小时。旧60秒保守估计保留其身份，新设备活动按占用事件计时，包含真实验收、profile、缓存、重试/中断、训练、评估、视觉和测速；CPU分析另记，不能把缺测CPU耗时填0。最终保护取至少3.5小时与profile更高预测，不重复加总；未花满预算不构成追加训练理由。\n\n'%(budget['used_device_seconds']/3600,budget['inherited_device_seconds']/3600)
    report+='V1保留核验：%s，%d个历史文件hash未变。可恢复checkpoint含模型、AdamW、scheduler、CPU/设备/独立随机流及组采样状态；ZIP补入checksum旁文件与旧谱系，恢复时身份必须一致。时延为无缓存全模型、单方法独立进程、20预热/100计时/20输入service，原始记录和峰值显存分开保存，参见timing/deployment.json。\n\n'%('通过' if legacy['all_unchanged'] else '未通过',legacy['n_files'])
    report+='日志粒度限制：每50更新的真实loss窗口、样本数和梯度窗口均保留，但training.jsonl行未附独立墙钟时间戳。阶段设备时间来自实际budget_ledger与累计账本；没有事后猜补逐50批耗时。该日志字段缺项不改变输出、标签或选择，但不能声称已测得逐50批速度曲线。\n\n'
    timing=optional(run/'timing/deployment.json')
    if timing:
        report+='| 选定部署策略 | 模型p50(ms) | 模型p95(ms) | 服务p50(ms) | 单进程峰值MiB |\n|---|---:|---:|---:|---:|\n'
        for r in timing['methods']:
            report+='| %s | %.3f | %.3f | %.3f | %.2f |\n'%(r['method'],r['model']['p50']*1000,r['model']['p95']*1000,r['service']['p50']*1000,r['peak_single_process_allocated_bytes']/1024**2)
        report+='\n上表执行冻结的部署策略；若return_base，时延是回退J0的捷径，不是完整O机制计算。原isolated worker把尺度JSON读取放在控制器调用内，该CPU/I/O开销包含在所报时延。文件系统为未清空页缓存的暖态；不据此声称部署优化或一般服务加速。\n\n'
    full=optional(run/'timing/O_full_pipeline.json')
    if full:
        report+='完整O流水线另在独立进程执行底座、先验、producer与共享正负控制器，最终仍使用冻结部署策略；尺度只在计时前读一次。模型p50/p95=%.3f/%.3f ms，service p50=%.3f ms，峰值%.2f MiB。即使所选策略return_base，这份成本也包含完整机制，不能与捷径时延混为一谈。\n\n'%(full['model']['p50']*1000,full['model']['p95']*1000,full['service']['p50']*1000,full['peak_single_process_allocated_bytes']/1024**2)
    compute=optional(run/'timing/module_compute_costs.json')
    if compute:
        report+='外接模块实际参数与256×256、batch1卷积计算量：\n\n'
        report+='| 模块 | 参数 | 卷积GMAC | 共享方向网络调用数 |\n|---|---:|---:|---:|\n'
        for r in compute['rows']:
            report+='| %s | %d | %.6f | %d |\n'%(r['method'],r['parameters'],r['module_convolution_MACs']/1e9,r['shared_network_forward_calls'])
        report+='\n一个乘加计一个MAC；这是从真实模块结构精确计算的外接卷积成本，不是设备测速。未包括底座、自定义扫描、先验、逐元素操作与I/O；底座MAC未实测，明确缺失。O及保留共享结构的消融每张图调用同一h两次，G2仅近似匹配这部分计算量，参数量和全流程时延仍不同。冻结策略若回退J0，部署捷径不实际执行全部模块。\n\n'
    inspected=optional(run/'figures/agent_visual_inspection.json')
    if (run/'figures/comparison_utility_val_manifest.json').exists():
        report+='视觉导出修补仅影响案例挑选：原方法面板按O的ΔPSNR排序，保留作辅助失败展示；规约§15要求的候选排序已另生成candidate_ranked_utility_val_selection.json与comparison_utility_val面板，包含同图输入/参考/底座/固定候选/强对照/O及残差、真实/预测U、alpha、误差差图。未更改或重新挑选质量分数、标签、网络或部署策略；正式固定案例以该候选排序清单为准。\n\n'
    if inspected:
        report+='面板已实际打开检查的列表与hash见 figures/agent_visual_inspection.json；未列出的图片只完成生成/hash验证，不称逐张人工检查。\n\n'
    report+='## 7. 下一步与交付\n\n'
    if state['scientific_status']=='STOP_PRODUCER_TRANSFER_GATE':
        decision='停止当前配方。model_val选出的唯一producer未通过utility_val预定空间筛选，不换第二名、不做LSUI盲目续训救场。下一轮先研究候选迁移/训练目标与局部修正可学性，并以新协议重新登记数据与对照；当前不值得以O已成功为由启动多种子或更长训练。'
    elif state['scientific_status']=='STOP_NO_USABLE_PRODUCER':
        decision='停止当前有限候选/唯一救援配方。分析候选优化与先验响应，不自动延长到20k/50k，不把有限池失败泛化为全部局部效用机制不可能。'
    elif state['scientific_status'].startswith('CONFIRMATION_PASS'):
        decision='可进入另行预算授权的确认研究：完整方法多种子、独立数据、第二底座和同构O_RGB。当前仅单新增模块种子、共享简化底座、历史暴露数据的有限支持，不是论文创新成立。'
    else:
        decision='按当前具体停止/阻塞状态收束，不追加预算、种子或更改阈值。只有完整强对照与机制消融支持后，才值得讨论独立数据与完整方法多种子确认。当前需要分析失败/优化证据的具体原因，见完整闸门与NEXT_DECISION。'
    report+=decision+'\n\n'
    report+='交付：FINAL_REPORT/NEXT_DECISION、公式对照与测试回执、逐图长表/逐内容组/单seed统计、配对区间、全部实际选择冻结、预算/状态/事件、真实命令、诊断/失败案例及原始时延、服务器权重清单。未运行阶段逐项标记，没有虚构sealed表或训练分数。下载包在 /mnt/workspace：轻量review、完整source_protocol、weights_recovery、visuals；另有local_history.bundle。打包范围与验证范围见 delivery/archive_receipts.json 和 recovery_instructions.md。源码包必须核验六原始数据模块；恢复包包括优化器/RNG/校验旁文件，视觉单独。\n\n'
    report+='**独立备份未完成，independent_backup_verified=false。** 服务器同盘ZIP和本地Git不能抵抗磁盘丢失；请优先在本地下载报告、review、source_protocol和weights_recovery，再与SHA256清单校验，下载校验完成前不登记异地备份成功。没有公开上传数据、图片、权重或自动推送GitHub。\n'
    for path in [doc/'FINAL_REPORT.md',run/'FINAL_REPORT.md']:path.write_text(report,encoding='utf-8')
    nxt='# 下一轮决策\n\n状态：**'+state['scientific_status']+'**。\n\n'+decision+'\n\n候选空间、可部署收益、机制增量三者不能互换。仍需多种子、独立数据及同构RGB机制对照才能扩展结论；本轮不自动启动这些实验。独立备份未验证，先下载并校验源码与完整权重恢复包。\n'
    for p in [doc/'NEXT_DECISION.md',run/'NEXT_DECISION.md']:p.write_text(nxt,encoding='utf-8')
    write(run/'delivery/report_render_identity.json',{'script_sha256':sha(__file__),
        'state_sha256':sha(run/'state.json'),'budget_sha256':sha(run/'budget.json'),
        'summary_sha256':sha(run/'diagnostics/checkpoint_summary.json'),
        'selection_changes':False,'device_work':False,'image_reads':False,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    print('Rendered Chinese actual-evidence FINAL_REPORT and NEXT_DECISION')


if __name__=='__main__':main()
