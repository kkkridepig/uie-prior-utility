"""Export evidence index and lightweight records ZIP; large weights stay in place."""
import argparse,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/prior_utility_cde_v3_20261004'
DOC=ROOT/'docs/experiments/PRIOR_UTILITY_CDE_V3_20261004'

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True); a=p.parse_args()
    target=Path(a.output)
    if target.exists(): raise FileExistsError('Version export filename rather than overwrite evidence')
    suffixes={'.json','.jsonl','.log','.md','.yaml','.txt','.csv'}
    entries=[f for root in (RUN,DOC,ROOT/'configs/cde_v3',ROOT/'scripts/cde_v3',ROOT/'tests/cde_v3') for f in root.rglob('*') if f.is_file() and (f.suffix in suffixes or f.suffix=='.py') and '__pycache__' not in f.parts]
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(set(entries)): archive.write(f,str(f.relative_to(ROOT)))
    with zipfile.ZipFile(target) as archive: assert archive.testzip() is None
    print(json.dumps({'archive':str(target),'files':len(set(entries)),'bytes':target.stat().st_size,'weights_and_image_exports':'retained separately under run directory'}))
if __name__=='__main__': main()
