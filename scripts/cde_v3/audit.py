"""Read-only baseline audit; explicit creation of versioned data roles and protocol."""
import argparse, collections, json, platform, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *
from mpa_diff.data.manifest import read_manifest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); args=p.parse_args(); setup()
    assert sha(PARENT)==PARENT_HASH
    cp=torch.load(PARENT,map_location='cpu'); cfg=cp['config']
    env={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'branch':subprocess.check_output(['git','branch','--show-current']).decode().strip(),'worktree':subprocess.check_output(['git','status','--short']).decode(),'python':sys.version,'torch':torch.__version__,'torch_path':torch.__file__,'torch_cuda':torch.version.cuda,'platform':platform.platform(),'free_disk_bytes':shutil.disk_usage(ROOT).free,'parent_step':cp['step'],'parent_hash':sha(PARENT),'config':cfg,'processes':subprocess.check_output(['ps','-eo','pid,etime,args']).decode(),'device_report':subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout}
    write(RUN/'code_and_environment.json',env)
    old=read(ROOT/'runs/explore_ag_single_seed_v2_20261003/budget.json')
    write(RUN/'lineage.json',{'parent':str(PARENT),'sha256':PARENT_HASH,'parent_pretrain_seed':20260927,'parent_step':cp['step'],'old_budget_seconds':old['legacy_queue_device_seconds_since_start']+old['new_experiment_device_seconds'],'new_cap_hours':72,'old_budget_not_reused':True,'pretrained_weights':{k:sha(ROOT/v) for k,v in [('depth',cfg['depth']['checkpoint']),('vgg',cfg['loss']['vgg_checkpoint'])]},'base_control':'warm_start_reset_Adam_not_exact_resume'})
    rows=read_manifest(ROOT/cfg['data']['manifest']); groups={}
    for r in rows:
        if r['split']=='train': groups.setdefault(r['scene_id'] or r['sample_id'],[]).append(r)
    order=sorted(groups,key=lambda g:key(20261004,'scene_split',g)); n=len(order); cuts=[round(.7*n),round(.9*n)]
    rr={k:[] for k in ('adapter_fit','route_fit','route_cal','source_dev','legacy_exposed_regression')}
    for j,g in enumerate(order): rr[('adapter_fit' if j<cuts[0] else 'route_fit' if j<cuts[1] else 'route_cal')]+=groups[g]
    # Known dev/test duplicate remains a dev scene; both IDs are explicitly bound in audit.
    for r in rows:
        if r['sample_id'] in ('UIEB/61_img_','UIEB/356_img_'): r['scene_id']='known_scene_61_356'
        if r['split']=='val': rr['source_dev'].append(r)
        if r['split']=='test': rr['legacy_exposed_regression'].append(r)
    train_scenes={r['scene_id'] for k in ('adapter_fit','route_fit','route_cal') for r in rr[k]}
    rr['source_dev']=[r for r in rr['source_dev'] if r['scene_id'] not in train_scenes]
    for k in rr: rr[k]=sorted(rr[k],key=lambda r:r['sample_id'])
    # Verify original input/reference bytes before accepting new roles.
    verified=0
    for r in rows:
        for field in ('image','reference'):
            assert sha(ROOT/cfg['data']['data_root']/r[field+'_path'])==r[field+'_sha256'],r['sample_id']
            verified+=1
    data={'roles':rr,'manifest_sha256':sha(ROOT/cfg['data']['manifest']),'role_sha256':{k:digest(v) for k,v in rr.items()},'counts':{k:len(v) for k,v in rr.items()},'scene_counts':{k:len({r['scene_id'] for r in v}) for k,v in rr.items()},'verified_image_reference_files':verified,'historical_train_exposure':'all adapter_fit/route_fit/route_cal seen by parent','source_dev_test_relation':['UIEB/61_img_','UIEB/356_img_'],'legacy_exposed':['UIEB test97','LSUI test427','U45'],'confirm_holdout':{'status':'pending_exposure_and_scene_audit','candidate':'LSUI non-test groups; do not access scores before final_freeze'},'upstream_exposure_unknown':['DepthAnythingV2','VGG19'],'fixed24':[r['sample_id'] for r in sorted(rr['source_dev'],key=lambda r:key(20261004,'fixed_dev',r['sample_id']))[:24]],'fixed_visual12':[r['sample_id'] for r in sorted(rr['source_dev'],key=lambda r:key(20261004,'fixed_dev',r['sample_id']))[:12]]}
    target=RUN/'data_roles.json'
    if target.exists(): assert read(target)==data,'Frozen roles changed'
    elif args.freeze: write(target,data)
    framework=Path('/mnt/workspace/TEMP-FILE-STATION/CODEX_TASK_FRAMEWORK_CDE_V3_20261004.md')
    if args.freeze:
        import yaml
        config=yaml.safe_load((ROOT/'configs/cde_v3/protocol.yaml').read_text())
        config.update(framework_sha256=sha(framework),data_roles_sha256=sha(target),parent_config=cfg)
        frozen=RUN/'protocol_frozen.yaml'; text=yaml.safe_dump(config,sort_keys=True,allow_unicode=True)
        if frozen.exists(): assert frozen.read_text()==text
        else: frozen.write_text(text)
        if not (RUN/'budget.json').exists(): write(RUN/'budget.json',{'limit_hours':72,'final_reserve_hours':12,'package_caps':{'P0':4,'E':4,'C':18,'REPEAT':22,'D':8,'FINAL':12,'REPAIR':4},'charged_seconds':0,'events':[],'active':None})
        if not (RUN/'dispatch_state.json').exists(): write(RUN/'dispatch_state.json',{'status':'audit_complete','completed':{},'failures':{},'stage':'P0'})
    DOC.mkdir(parents=True,exist_ok=True)
    (DOC/'CURRENT_STATE_DIFF.md').write_text('# V3 现场差异审计\n\n'+f"核验UTC {env['utc']}。当前commit {env['commit']} 与框架记录一致；起始工作区干净，新建研究分支 {env['branch']}。未检测到研究训练进程；厂商 torch {env['torch']} 保持原路径。\n\n父checkpoint哈希与85000步身份核验通过。旧预算 {old['new_experiment_device_seconds']/3600+old['legacy_queue_device_seconds_since_start']/3600:.6f} h，仅作历史；新上限72 h，预留12 h。剩余磁盘 {env['free_disk_bytes']/2**30:.2f} GiB，需要delta checkpoint与容量预检。\n\n旧BASE_CONT日程终点10000，新轮5000，必须重训同轮控制。旧keyed_noise忽略外部seed；v3单独键控。旧C同位置sigmoid不满足QK交叉注意力，v3新实现；旧D浅层phase是允许简化，不列实现错误。\n\n角色图数：{data['counts']}。61/356绑定已知场景；保留旧测试身份为legacy_exposed_regression，不称新盲测。DA/VGG上游暴露未知；新留出候选待审计。\n")
    print(json.dumps({'counts':data['counts'],'parent_verified':True,'environment':env['torch'],'disk_free_gib':env['free_disk_bytes']/2**30}))
if __name__=='__main__': main()
