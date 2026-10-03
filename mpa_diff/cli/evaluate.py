import argparse,json
from pathlib import Path
import yaml
from mpa_diff.engine.checkpoint import load_model,seed_all
from mpa_diff.engine.evaluator import evaluate_rows
from mpa_diff.data.manifest import audit,read_manifest
from mpa_diff.utils.io import read_image,write_json
from mpa_diff.metrics.image import psnr,ssim,finite_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--manifest');p.add_argument('--data-root');p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--predictions');p.add_argument('--protocol');a=p.parse_args()
    if a.predictions:
        if not a.protocol:p.error('--protocol required with --predictions')
        protocol=yaml.safe_load(Path(a.protocol).read_text())
        if protocol.get('name')!='rgb_float_ssim_valid_v1':raise ValueError('Unknown protocol')
        records=[]
        for line in Path(a.predictions).read_text().splitlines():
            row=json.loads(line);x=read_image(row['prediction_path']);y=read_image(row['reference_path'],protocol.get('resize_hw'))
            records.append({'sample_id':row['sample_id'],'png_psnr':finite_json(psnr(x,y)),'png_ssim':ssim(x,y)})
        write_json(a.output,records);return
    if not a.checkpoint:p.error('--checkpoint or --predictions required')
    seed_all(0);model,state=load_model(a.checkpoint,a.device);c=state['config']
    manifest=a.manifest or c['data']['manifest'];root=a.data_root or c['data']['data_root']
    audit(manifest,root);rows=[r for r in read_manifest(manifest) if r['split']==a.split]
    if not rows:raise ValueError('Requested split is empty')
    summary,_=evaluate_rows(model,rows,root,c['data']['resize_hw'],c['experiment']['seed'],a.output);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
