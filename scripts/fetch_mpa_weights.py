"""Explicit acquisition with resumable byte ranges and integrity checks; no pip."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'vgg19-dcbb9e9d.pth':('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',574673361,'dcbb9e9d'),
 'depth_anything_v2_vits.pth':('https://hf-mirror.com/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',99218434,'715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378')}

def fetch(name,source):
    url,size,digest=source
    output=ROOT/'weights'/name
    parts=ROOT/'weights'/('.'+name+'.parts'); parts.mkdir(parents=True,exist_ok=True)
    chunk=4*1024*1024
    def part(start):
        end=min(size,start+chunk)-1; path=parts/str(start)
        if path.exists() and path.stat().st_size==end-start+1: return path
        for attempt in range(3):
            try:
                r=requests.get(url,headers={'Range':'bytes=%d-%d'%(start,end)},stream=True,timeout=(20,60))
                r.raise_for_status()
                if r.status_code!=206 or not r.headers.get('Content-Range','').startswith('bytes %d-'%start): raise ValueError('Server did not honor range')
                with path.with_suffix('.tmp').open('wb') as f:
                    for data in r.iter_content(262144): f.write(data)
                if path.with_suffix('.tmp').stat().st_size!=end-start+1: raise ValueError('Truncated part')
                path.with_suffix('.tmp').replace(path); return path
            except Exception:
                if attempt==2: raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        paths=list(pool.map(part,range(0,size,chunk)))
    h=hashlib.sha256()
    with output.with_suffix('.download').open('wb') as out:
        for p in paths:
            b=p.read_bytes();h.update(b);out.write(b)
    if not h.hexdigest().startswith(digest): raise ValueError('Integrity failure '+name)
    output.with_suffix('.download').replace(output)
    for p in paths:p.unlink()
    parts.rmdir()
    receipt={'url':url,'path':str(output),'sha256':h.hexdigest(),'size':size,'expected_hash':digest,'license':'Apache-2.0 (DA Small)' if name.startswith('depth') else 'torchvision code BSD-3-Clause; ImageNet pretrained weights terms separate'}
    (ROOT/'weights'/(name+'.receipt.json')).write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda pair:fetch(*pair),SOURCES.items()): pass
