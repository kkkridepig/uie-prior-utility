import argparse,json
from mpa_diff.config import load_config
from mpa_diff.engine.trainer import train

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--stop-after',type=int);a=p.parse_args()
    print(json.dumps(train(load_config(a.config),a.resume,a.stop_after),indent=2))
if __name__=='__main__':main()
