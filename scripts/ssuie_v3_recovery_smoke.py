"""CPU strict-loading smoke in a fresh extracted source/weights directory."""
import argparse,hashlib,json,io
from pathlib import Path
import torch
from uie_next.records import ROOT,sha,read,write
from uie_next.backbones.ssuie import load_official
from uie_next.models.candidate import Candidate
from uie_next.models.controls import Controller


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();run=ROOT/'runs/ssuie_utility_v3_predictability_20261007';audit=read(run/'recovery_audit.json');checked={}
    for name,h in audit['weights'].items():
        path=ROOT/name
        if sha(path)!=h:raise ValueError('weight not portable or hash mismatch '+name)
        checked[name]=h
    official=next(ROOT/name for name,h in checked.items() if h=='977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99')
    model,identity=load_official(ROOT,official,expected_sha='977029a53e095640ca86ea3ca37a1bf2ce976a0d7a143076a4a29fc173ab3a99')
    # This is strictly a CPU loading check, not a PPU speed measurement.
    paths={}
    for method,part in [('B1','ssuie_local_utility_v1_20261007'),('B3','ssuie_local_utility_v1_20261007'),('O','ssuie_local_utility_v2_diagnostic_20261007')]:
        path=ROOT/'runs'/part/'checkpoints'/method/'step_003000.pt';st=torch.load(path,map_location='cpu');m=Candidate() if method in ['B1','B3'] else Controller(method);m.load_state_dict(st['model_state'],strict=True);paths[method]=sha(path)
    registry=read(ROOT/'runs/ssuie_local_utility_v2_diagnostic_20261007/method_registry.json')
    for method,item in registry.items():
        for slot in ['candidate','controller']:
            v=item[slot]
            if not v:continue
            old=Path(v['path']);parts=old.parts;idx=parts.index('runs');path=ROOT/Path(*parts[idx:]);st=torch.load(path,map_location='cpu')
            m=Candidate() if slot=='candidate' else Controller(method);m.load_state_dict(st['model_state'],strict=True)
    write(a.receipt,{'passed':True,'source_root':str(ROOT),'CPU_strict_load_only':True,'official':identity,'producer_and_rgb_and_O':paths,'all_registry_models_strict_loaded':True,'weight_hashes_checked':len(checked),'private_absolute_weight_paths_required':False,'data_not_in_package':True,'PPU_runtime_restore_not_tested_in_new_environment':True})
if __name__=='__main__':main()
