"""Evidence-based blocked closeout. Never creates unrun metric/selection records."""
import csv
import io
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile
from xml.etree import ElementTree
from .records import ROOT,write,read,sha,digest
from .pipeline import paths


def text(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=str(path.parent),delete=False) as f:
        f.write(content);f.flush();os.fsync(f.fileno());temp=f.name
    os.replace(temp,str(path))


def archive(path,files,base):
    path=Path(path)
    temp=path.with_name('.'+path.name+'.partial')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for f in sorted(set(files)):
            z.write(f,str(f.relative_to(base)))
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise ValueError('archive CRC failure')
        for f in sorted(set(files)):
            import hashlib
            if hashlib.sha256(z.read(str(f.relative_to(base)))).hexdigest()!=sha(f):raise ValueError('archive payload hash mismatch')
    os.replace(temp,path)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'members':len(set(files)),
            'crc_verified':True,'all_payload_sha256_verified':True}


def closeout(config):
    run,doc=paths(config);state=read(run/'state.json');budget=read(run/'budget.json');audit=read(run/'cross_dataset_audit.json')
    if state['scientific_status']!='BLOCKED_BACKBONE':raise RuntimeError('this closeout currently supports the actual blocked intake only')
    old=read(run/'legacy_preservation.json')
    changed=[p for p,h in old.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    if changed:raise RuntimeError('legacy preservation mismatch: '+str(changed))
    write(run/'closeout/legacy_verification.json',{'files':len(old),'changed':changed,'all_hashes_unchanged':True})
    xml=run/'tests/pytest_delivery.xml'
    if not xml.exists():raise RuntimeError('final test receipt missing')
    parsed=ElementTree.parse(str(xml));suite=parsed.getroot().find('testsuite')
    tests={k:int(suite.get(k,'0')) for k in ['tests','failures','errors','skipped']}
    if tests['failures'] or tests['errors']:raise RuntimeError('final tests did not pass')
    write(run/'tests/test_receipt.json',{'xml_sha256':sha(xml),'suite':tests,'T26_official_backbone':'BLOCKED',
           'T27_T28_official_integration':'BLOCKED; synthetic wrapper/head checks passed','formal_train_profile':'not_run'})
    gpu=read(run/'tests/ppu_local_heads.json');cpu=read(run/'tests/recovery_cpu/receipt.json')
    ppu=read(run/'tests/recovery_ppu/receipt.json') if (run/'tests/recovery_ppu/receipt.json').exists() else {'passed':False}
    runtime=read(run/'tests/runtime_probe.json') if (run/'tests/runtime_probe.json').exists() else {'passed':False,'reason':'runtime not validated'}
    third_party={}
    for folder in ['ss_uie','mamba_1_1_1','causal_conv1d_1_1_1']:
        source=ROOT/'third_party'/folder
        third_party[folder]={'commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                             'remote':subprocess.check_output(['git','-C',str(source),'remote','get-url','origin'],text=True).strip(),
                             'working_changes':subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()}
    write(run/'upstream_identity.json',third_party)
    write(run/'cpu_and_download_ledger.json',{'audit_wall_seconds':audit['cpu_wall_seconds'],
           'formal_training_cpu_seconds':0,'download_and_compile':'GPU=0; stage wall times not fully instrumented; logs provide attempts and limits',
           'missing_cpu_timings_are_not_zero':True})
    if sha(__import__('torch').__file__)!=read(run/'intake.json')['vendor_torch_init_sha256']:
        raise RuntimeError('vendor torch identity changed')
    weight_inventory=[]
    for path in sorted((ROOT/'weights').rglob('*.pth')):
        weight_inventory.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),
                                 'role':'metric_or_legacy_dependency_not_SS_UIE'})
    for path in sorted((run/'tests').rglob('*.pt')):
        weight_inventory.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),
                                 'role':'synthetic_head_recovery_fixture_not_scientific_method'})
    write(run/'closeout/server_weight_inventory.json',{'official_SS_UIE':'missing','formal_candidate_controller_weights':[],
                                                       'existing_files':weight_inventory})
    hours=budget['used_device_seconds']/3600
    table='| 角色 | 组 | 图像对 |\n|---|---:|---:|\n'+''.join('| %s | %d | %d |\n'%(n,v['groups'],v['pairs']) for n,v in audit['role_counts'].items())
    matrix=['B0 clip01','B0 official_minmax_float','B1','B2','B3','B4','G0','G1','G2','F0','R0','O','O-NI','O-NP','O-NS','O-ND','Q*']
    matrix_table='| 方法 | 正式更新 | 正式质量评估 |\n|---|---:|---|\n'+''.join('| %s | 0 | not_run：缺少正式底座 |\n'%n for n in matrix)
    mapping='''| 规范 | 实现 | 验收与范围 |
|---|---|---|
| 实际裁剪残差 r / RGB 共享 alpha | models/candidate.py; math/utility.py | T01–T06、T09、T11–T13 CPU |
| a,b,c,u,v,U；安全分母；oracle | math/utility.py | 二次恒等式、像素/块 oracle、错误预测损害 |
| 正负方向共享、有符号、正齐次 | models/utility.py | T07/T08/T10/T30；共享 h，不是整图选择器 |
| 弱颜色/对比度代理、全图 A、七视图重算 | priors/heuristic.py | T14/T24/T25；六压力视图已实现，未做正式压力评估 |
| 候选 115907；O 76896 等参数 | models/blocks.py, candidate.py, controls.py | CPU 参数计数与 PPU 256x256 单更新；不是正式训练 |
| 投影/同图成对/MSE 决策等权损失 | losses.py; training.py:pair_objective | T15/T29/T31；F0/R0 及所有 O 消融目标已实现 |
| 名义数据和独立随机流、先验缺失 | training.py:streams/GroupSampler/drop_prior | 命名流可恢复；完整数据采样训练尚未派发 |
| 底座 eval、BN 冻结；候选冻结 | backbones/ssuie.py; models/system.py | 合成封装验收通过；官方底座 BN 验收 BLOCKED |
| 无 Y 的部署接口与角色守卫 | models/system.py; data/manifest.py, roles.py | T16/T20/T21；未验收权重来源时角色守卫拒绝效用训练 |
| 内容组、跨输入/参考重复 | data/audit.py | 10338 图审计与 T17/T18；仅 content_group_proxy |
| 身份缓存、原子保存、恢复、预算 | data/cache.py; checkpoint.py; budget.py | T19/T32–T35；CPU/PPU 合成恢复，非 SS-UIE 恢复 |
| PSNR/SSIM/真实 VGG LPIPS | evaluation.py | T36；零 MSE 为 +inf，不沿用旧 PSNR 截断 |
| 固定校准网格/并列规则 | calibration.py | 网格/约束单测；真实 CAL 尚未读取评分 |
| 固定上游和严格加载、逐图 minmax | backbones/ssuie.py | T37 通过；T26 官方前向与模型批量一致未测 |
| 外目录包安装/真实 CLI | cli.py; pyproject.toml | T38；inspect/audit/verify/run/status/plan/closeout 可用 |
| S3–S10 正式全流程与完整服务测速 | 尚未完成验收和调度实现 | missing/blocked：不把入口存在写成全流程完成 |
'''
    text(doc/'IMPLEMENTATION_CONFORMANCE.md','# 实现对照与验收范围\n\n'+mapping+'\n主规范 §3.2/§19.2：底座未取得仍完成独立工程，但不得正式训练。当前工程整体未完成；CLI 的 run 只推进入场和依赖检查，不能据此认为会自动跑完 S3–S10。\n')
    text(doc/'DATA_AND_EXPOSURE_AUDIT.md','# 数据与暴露审计\n\n全量覆盖 LSUI 4279 对、UIEB 890 对，包含输入↔输入、参考↔参考、输入↔参考。固定 pHash Hamming≤4 或缩略图 MSE≤1e-5 保守连边；完整解码重复和既有内容组也连边。\n\n'+table+'\n检测 4 条跨数据集图像连边，均涉及 LSUI/2061 与 UIEB/917_img_；UIEB 该对已排除。接触图已实际打开核查，布局一致，颜色/参考处理不同。近重复共 %d 个候选边；完整重复 %d 条。\n\n组为 content_group_proxy，不能声称已核实真实场景。全部本地样本登记 historically_analyzed；sealed_eval 仅为本轮封存，未评分、未解封。LSUI 来源已被上游使用，但逐图作者训练成员未知，登记 unknown_upstream_exposure 并保守视为来源池已暴露；只用于允许的候选开发，不声称对应作者 Train-3879/Test-400。初稿的逐图 known_upstream_train 表述已更正，旧台账保留在 audit_history，角色样本/组与阈值不变，protocol_amendments记录更正。UIEB 的 documented_nonoverlap 只限 README 声明的 LSUI 来源和本地全量审计；实际 checkpoint 来源尚未核验。角色名单目前为暂定，RoleGuard 的 backbone_provenance_verified 仍为 false，不允许效用训练。\n\n实际清单见 runs 同名目录 all_pairs.jsonl、roles.jsonl、near_duplicate_edges.jsonl、groups.json、exposure_ledger.json、split_freeze.json；不根据质量成绩改拆分。\n'%(audit['near_edge_count'],audit['exact_edge_count']))
    text(doc/'BASELINE_VERIFICATION.md','# 底座验收\n\nSS-UIE 官方代码锁定 88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df，网络 3/16/4/4、H=W=256；上游源码未修改。未获得官方权重，因此无权重 hash、无严格加载回执、无官方裸网络/适配层回归、无双策略质量表、未产生 baseline_policy.json。\n\nGoogle Drive ID 1YxyagMCbApON8dRdiQTaQG65g3tnZkPt：服务器网络不可达/超时。百度官方分享链接：响应标题“链接不存在”，errno=117，空 file_list。另试 Google 官方 usercontent 入口也超时。保留下载日志和百度响应页面 hash。未从不明第三方替代权重，也未用随机 SS-UIE、旧 MPA-Diff、VGG 或深度权重替代底座。\n\n厂商 torch 2.0.0a0+nv2303/torchvision 保留。causal-conv1d 与 mamba-ssm 1.1.1 使用固定上游源码在本机编译、--no-deps 安装；不安装 NVIDIA torch 轮子。扩展运行回执：%s。\n\n后处理 clip01 与逐图 official_minmax_float 实现与常数/批次测试通过；最终混合不再次 minmax。\n'%('PASS，见 tests/runtime_probe.json' if runtime['passed'] else '未通过或未完成，见运行日志'))
    text(doc/'MATHEMATICAL_TEST_REPORT.md','# 数学与测试记录\n\n最终仓库测试：%d 项，失败 %d，错误 %d，跳过 %d；见 pytest_delivery.xml 与 logs/tests_delivery.log。\n\n%s\nCPU 新进程恢复（连续20 vs 10+新进程10）：最大权重差 %.8g；PPU 合成头恢复：%s。PPU 上 13 个作业各单更新、float32、256×256、真实梯度检查通过；没有官方 SS-UIE 输出，不能称真实底座集成、小样本正式训练或完整吞吐测量。\n\n旧 CPU/打包失败日志保留：同名测试收集冲突、恢复探针线程配置、wheel 工程依赖故障。修复后以最终回执为准，不把旧失败删除。GPU 确定性只覆盖合成局部头，不推断官方 selective_scan 全模型的逐位确定性。\n'%(tests['tests'],tests['failures'],tests['errors'],tests['skipped'],mapping,cpu['max_weight_abs_error'],str(ppu)))
    text(doc/'TRAINING_AND_CONTROL_MATRIX.md','# 实际训练与控制矩阵\n\n'+matrix_table+'\n正式训练 seed 20261007 仅预登记；正式候选、标签、尺度、校准、DEV 和封存指标全部 not_run。合成梯度/恢复检查不是科学训练，不将其 checkpoint 用作方法权重。\n\n预算计划列出两个候选与 11 个效用作业，但秒/更新、验证、缓存吞吐和共同步数均未实测；不可用合成单次更新速度外推正式完整矩阵。最多16设备小时，至少保留3.5，当前 %.8f 设备小时。缓存规划仅为浮点容量算术，不是已生成缓存。\n'%hours)
    text(doc/'FAILURE_CASES.md','# 失败与案例\n\n当前是 BLOCKED_BACKBONE，不是充分实验后的科学失败。无正式增强图片/逐图质量、无最差方法案例、无压力结果，不伪造面板。\n\n已保存并实际查看的图片只有跨数据集重复审计接触图 figures/cross_dataset_contact.png；它说明同场景与颜色处理差异，不说明任何增强方法优劣。视觉包只含此审计证据。\n\n边界测试明确演示错误 b_hat 可以损害 MSE；无自动不退化保证。名义先验/干预名称不生成正负标签。\n')
    text(doc/'REPRODUCE_AND_RESUME.md','''# 复现与恢复

在 /mnt/workspace/uie-prior-utility 使用已有 .venv，保留厂商 torch/torchvision。必要新增依赖见 requirements-ssuie-ppu.txt，安装带 --no-deps；上游与编译日志见 runs 和 third_party。

```bash
.venv/bin/python -m uie_next.cli status --config configs/uie_next/protocol.yaml
.venv/bin/python -m uie_next.cli run --config configs/uie_next/protocol.yaml --resume
.venv/bin/python -m pytest tests/uie_next -q
.venv/bin/python -m uie_next.cli closeout --config configs/uie_next/protocol.yaml
```

当前 run 会按同一身份重查底座依赖，返回 exit 2/BLOCKED_BACKBONE，正式训练未派发。它没有完成 S3–S10 的自动调度；不能在上传权重后据此声称完整流程已解锁。

未完成的下一步：合法官方 checkpoint/hash/来源回执、严格加载、20图裸模型回归、model_val 双策略选择、正式小样本/缓存对照验收、完整预算 profile，及 S3–S10 调度与收束。因此不提供虚构的正式训练命令。

tests/recovery_cpu 与 recovery_ppu 中是可加载的合成头完整恢复证据，含模型、优化器、学习率、随机流；只作工程回归，不能作为正式候选/控制器。真正正式模型与相应全身份 checkpoint 尚未产生。

预算 ledger 在同 run 跨恢复累计，设备/dispatcher 均有本地互斥锁，原子记录。同盘 ZIP 不构成独立备份；independent_backup_verified=false。未执行远端推送。
''')
    text(doc/'FINAL_REPORT.md','# 本轮最终报告\n\n结论：**BLOCKED_BACKBONE，未完成有效算法检验**。没有正式 SS-UIE 权重，未启动正式训练；不能评价正收益或机制归因，不是科学证伪。\n\n## 实际完成\n\n保留恢复仓库和既有用户改动，独立实现局部机制及规定控制；全量数据审计、固定角色初稿、CPU 数学/协议/指标/恢复测试、真实 PPU 局部头合成梯度和恢复检查。上游固定 commit；厂商 torch 保留。最终仓库测试 %d 项通过；设备累计 %.8f 小时 / 16，预留至少3.5小时仍保留。\n\n%s\n## 工程忠实性\n\n核心公式和网络已按规范实现，映射见 IMPLEMENTATION_CONFORMANCE.md。**工程整体尚未完成**：官方底座集成/BatchNorm 实测、正式吞吐/缓存、S3–S10 完整训练评测调度与服务测速未验收或未实现。CLI 入口存在不等于科学流程可完整运行。\n\n## 名义结果与归因\n\n%s\nH1/H2/H3 均未检验；没有正常部署收益、oracle headroom、强控制增量、机制消融增量或确认区间可报告。没有产生校准选择、selection freeze、正式逐图表或失败案例，sealed_eval 未释放。\n\n## 不确定性与成本\n\n数据组是内容代理，所有本地数据历史已分析；UIEB 非重叠结论限官方 README 的 LSUI 来源与可获得文件审计，checkpoint 尚未核验。即使未来单种子通过，也仍需新增模块多种子、独立来源与第二底座；当前不得建议以“成功创新”扩大训练。合成 PPU 单次耗时不属于正式服务测速，CPU 编译、下载和审计不伪报为 GPU 训练。\n\n## 恢复与备份\n\n运行目录 runs/ssuie_local_utility_v1_20261007；docs 同名实验目录。代码、审阅、视觉与合成恢复证据包在 /mnt/workspace；SHA256/CRC/逐成员哈希清单在 closeout。没有官方或正式训练权重包。独立备份未完成，independent_backup_verified=false；同一服务器磁盘的压缩包无法抵抗服务器文件丢失。旧记录/既有修改哈希复核不变，未公开上传或推送 GitHub。\n\n## 后续条件\n\n需要可获取且来源可核验的官方 SS-UIE checkpoint。补齐后先完成缺失的官方验收和正式调度实现，按当前同一累计16小时预算继续，不额外种子、不改网络/阈值、不重启 A–G。\n'%(tests['tests'],hours,table,matrix_table))
    state.update(closeout_status='S11_CLOSEOUT_COMPLETE',scientific_status='BLOCKED_BACKBONE',formal_training_updates=0,
                 sealed_eval_released=False,independent_backup_verified=False)
    write(run/'state.json',state)
    text(doc/'LIVE_STATUS.md','科学状态：BLOCKED_BACKBONE。正式更新0；封存未解封；累计GPU设备小时 %.8f。工程仅局部机制/隔离/恢复验收，完整官方流程未完成。\n'%hours)
    return {'state':state,'docs':str(doc),'tests':tests}
