import argparse,time
import torch
from mpa_diff.engine.checkpoint import load_model,seed_all
from mpa_diff.utils.io import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sizes',nargs='+',default=['256','512','1080p']);p.add_argument('--device',default='cpu');p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',required=True);a=p.parse_args()
    seed_all(0);model,_=load_model(a.checkpoint,a.device);records=[]
    sync=lambda:torch.cuda.synchronize() if a.device.startswith('cuda') else None
    for size in a.sizes:
        hw=(1080,1920) if size=='1080p' else (int(size),int(size));x=torch.rand(1,3,*hw,device=a.device)
        model.enhance(x,0);times=[]
        if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        for i in range(a.repeats):
            sync();start=time.perf_counter();_,diag,_=model.enhance(x,0);sync();times.append(time.perf_counter()-start)
        records.append({'hw':hw,'seconds':times,'mean_seconds':sum(times)/len(times),'nfe':diag['nfe'],'peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,'includes':'depth/background/histogram/beta/highfreq/sampling; excludes file IO'})
    write_json(a.output,records)
if __name__=='__main__':main()
