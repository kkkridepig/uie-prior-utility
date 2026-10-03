import argparse
from pathlib import Path
from mpa_diff.config import load_config
from mpa_diff.priors.provider import PriorProvider
from mpa_diff.priors.kernels import pad_image
from mpa_diff.data.cache import static_cached
from mpa_diff.data.manifest import audit,read_manifest
from mpa_diff.utils.io import read_image

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=load_config(a.config)
    audit(c['data']['manifest'],c['data']['data_root']);provider=PriorProvider(c).to(c['runtime']['device']).eval()
    for row in read_manifest(c['data']['manifest']):
        image=read_image(Path(c['data']['data_root'])/row['image_path'],c['data']['resize_hw'])[None].to(c['runtime']['device'])
        static_cached(provider,pad_image(image)[0],c)
        print(row['sample_id'],flush=True)
if __name__=='__main__':main()
