import argparse
from pathlib import Path
from mpa_diff.engine.checkpoint import load_model,seed_all
from mpa_diff.utils.io import read_image,save_image,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    seed_all(a.seed);model,state=load_model(a.checkpoint,a.device)
    image=read_image(a.input)[None].to(a.device)
    result,diag,_=model.enhance(image,a.seed);save_image(a.output,result[0]);write_json(str(a.output)+'.json',diag)
if __name__=='__main__':main()
