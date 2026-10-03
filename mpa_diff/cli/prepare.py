import argparse,json
from pathlib import Path
import yaml
from mpa_diff.data.manifest import prepare
from mpa_diff.data.synthetic import generate

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--synthetic-fixture',action='store_true');a=p.parse_args()
    if a.synthetic_fixture:r=generate('data/synthetic_fixture','manifests/synthetic_fixture.jsonl')
    elif a.config:r=prepare(yaml.safe_load(Path(a.config).read_text(encoding='utf-8')))
    else:p.error('Provide --config or --synthetic-fixture')
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
