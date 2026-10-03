"""One persistent Markdown record per run, updated from actual observed evidence."""
import json
import math
from datetime import datetime,timezone
from pathlib import Path
from mpa_diff.utils.io import canonical_hash,write_json


def report_path(config):
    if Path(config['runtime']['output']).is_absolute():return Path(config['runtime']['output'])/'experiment.md'
    slug='__'.join(Path(config['runtime']['output']).parts[-3:])
    slug=''.join(c if c.isalnum() or c in '_-' else '_' for c in slug)
    return Path('docs/experiments')/(slug+'.md')


def update_report(config,status,record=None,evaluation=None,error=None):
    out=Path(config['runtime']['output']);out.mkdir(parents=True,exist_ok=True)
    state_path=out/'progress.json'
    progress=json.loads(state_path.read_text()) if state_path.exists() else {}
    progress.update(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),seed=config['experiment']['seed'],planned_steps=config['train']['total_steps'])
    if record is not None:progress['latest_train']=record
    if evaluation is not None:progress['evaluation']=evaluation
    if error is not None:progress['error']=str(error)
    write_json(state_path,progress)
    recent=progress.get('latest_train',{});test=progress.get('evaluation')
    text=['# 实验记录：'+config['experiment']['name'], '',
          '- 状态：'+status,'- 更新（UTC）：'+progress['updated_utc'],
          '- 随机种子：'+str(config['experiment']['seed']),
          '- 配置身份：'+config['experiment']['evidence_profile'],
          '- 数据清单：`'+config['data']['manifest']+'`',
          '- 配置 SHA256：`'+canonical_hash(config)+'`',
          '- 分辨率：'+str(config['data']['resize_hw'])+'；全局 batch：'+str(config['train']['effective_batch']),
          '- 训练预算：'+str(config['train']['total_steps'])+' 步；采样器：'+config['sampler']['name']+' / '+str(config['sampler']['steps'])+' 次',
          '- 当前训练步：'+str(recent.get('step',0)),
          '- 原始产物目录：`'+str(out)+'`','', '## 当前观测','']
    if recent:text+=['```json',json.dumps(recent,ensure_ascii=False,indent=2),'```']
    else:text+=['尚无训练数值。']
    if test:text+=['','## 测试结果','','```json',json.dumps(test,ensure_ascii=False,indent=2),'```']
    text+=['','## 分析与边界','']
    if test:
        text+=['以上为本种子在冻结测试名单上的结果；不能当作三种子均值。缺失指标按结果中的原因保留空值，最差样本和逐图数值见运行目录的 test/。']
    else:
        text+=['当前只有训练进度或验证记录；训练损失不代表测试质量，尚未完成的测试指标保持空值。']
    if 'grouped' in config['data']['manifest']:
        text+=['本实验按内容关系分组隔离，保留原始样本。组由重复内容/近重复筛查推断，不等于真实海域或已标定场景；与论文原始划分不一致，不能宣称严格复现原表。']
    if config['runtime']['test_fixture']:text+=['这是合成测试替身实验，不能填入真实benchmark结果。']
    if error:text+=['','## 中断原因','',str(error)]
    path=report_path(config);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text('\n'.join(text)+'\n',encoding='utf-8');temp.replace(path)
    return str(path)
