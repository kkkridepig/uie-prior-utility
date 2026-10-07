from pathlib import Path
import yaml
from .records import ROOT,GUIDE,digest


SECTIONS={'schema_version','experiment','seed','split_seed','bootstrap_seed','budget','backbone','data','preprocess',
          'prior','candidate','utility','training','calibration','gates','runtime'}


def load(path):
    value=yaml.safe_load(Path(path).read_text())
    if value.get('schema_version') == 2:
        from .v2.context import load_config
        return load_config(path)
    if set(value)!=SECTIONS or value['schema_version']!=1:raise ValueError('unsupported/unknown protocol sections')
    guide=GUIDE
    text=guide.read_text();section=text.split('## 18. 可执行配置合同',1)[1]
    reference=yaml.safe_load(section.split('```yaml',1)[1].split('```',1)[0])
    adjustable={
        'backbone':{'checkpoint','checkpoint_sha256','output_policy'},
        'data':{'lsui_root','uieb_root','manifest','exposure_ledger'},
        'training':{'candidate_updates','utility_updates'},
        'runtime':{'run_id','run_dir','backup_root'},
        'budget':{'max_device_hours','final_reserved_device_hours'},
    }
    for section in SECTIONS:
        if isinstance(reference[section],dict):
            if set(reference[section])!=set(value[section]):raise ValueError('unknown or missing keys in '+section)
            for key,default in reference[section].items():
                if key not in adjustable.get(section,set()) and value[section][key]!=default:
                    raise ValueError('scientific recipe changed: '+section+'.'+key)
        elif value[section]!=reference[section]:raise ValueError('scientific identity changed: '+section)
    if value['budget']['max_device_hours']>16 or value['budget']['final_reserved_device_hours']<3.5:
        raise ValueError('budget/reserve violates authorization')
    if value['seed']!=20261007 or value['budget']['max_gpus']!=1:raise ValueError('no additional seed or GPU authorized')
    for name in ['candidate','utility']:
        steps=value['training'][name+'_updates']
        if steps is not None and steps not in value['training'][name+'_update_options']:
            raise ValueError('updates outside protocol grid')
    return value


def identity(config):
    return digest(config)
