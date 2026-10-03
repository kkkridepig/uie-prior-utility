"""Infer inseparable content groups before any training/model selection."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from mpa_diff.utils.io import sha256,write_json


def run(dataset):
    root=Path('data')/dataset.lower();dirs=('raw-890','reference-890') if dataset=='UIEB' else ('input','GT')
    images={p.stem:p for p in (root/dirs[0]).iterdir() if p.is_file()};refs={p.stem:p for p in (root/dirs[1]).iterdir() if p.is_file()}
    assert images.keys()==refs.keys()
    ids=sorted(images);parent=list(range(len(ids)));seen={};evidence=[];thumbs=[];pair_hashes={};duplicates=[];dimensions=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j,kind):
        a,b=find(i),find(j)
        if a!=b:parent[max(a,b)]=min(a,b)
        evidence.append({'a':ids[i],'b':ids[j],'reason':kind})
    for i,sid in enumerate(ids):
        hashes=[];sizes=[]
        for kind,files in [('input',images),('reference',refs)]:
            path=files[sid]
            with Image.open(path) as im:
                rgb=im.convert('RGB');a=np.array(rgb);sizes.append(im.size)
                digest=hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest();hashes.append(digest)
                if kind=='input':thumbs.append(np.asarray(rgb.convert('L').resize((16,16),Image.BILINEAR),dtype=np.float64).ravel()/255)
            # Same image in either role also implies inseparable content.
            if digest in seen:union(i,seen[digest],'identical_decoded_'+kind)
            else:seen[digest]=i
        if sizes[0]!=sizes[1]:dimensions.append({'id':sid,'input':sizes[0],'reference':sizes[1]})
        pair=tuple(hashes)
        if pair in pair_hashes:duplicates.append([sid,ids[pair_hashes[pair]]])
        else:pair_hashes[pair]=i
    tree=cKDTree(np.stack(thumbs))
    for i,j in sorted(tree.query_pairs(np.sqrt(256*1e-5))):union(i,j,'16x16_gray_MSE_below_1e-5')
    clusters={}
    for i,sid in enumerate(ids):clusters.setdefault(find(i),[]).append(sid)
    groups={sid:'content_group_'+ids[key] for key,group in clusters.items() for sid in group}
    out=Path('runs/data_intake');write_json(out/(dataset.lower()+'_groups.json'),groups)
    report={'dataset':dataset,'pairs':len(ids),'groups':len(clusters),'nontrivial_groups':[g for g in clusters.values() if len(g)>1],'evidence':evidence,'identical_pairs':duplicates,'dimension_mismatches':dimensions,'policy':'Conservative content-derived grouping, not measured scene identity; no test performance used. Retain original samples, isolate related inputs/references into same split.'}
    write_json(out/(dataset.lower()+'_content_audit.json'),report)
    selected=evidence[:12]
    if selected:
        canvas=Image.new('RGB',(640,200*len(selected)),'white');draw=ImageDraw.Draw(canvas)
        for row,e in enumerate(selected):
            for col,sid in enumerate([e['a'],e['b']]):
                with Image.open(images[sid]) as im:im=im.convert('RGB');im.thumbnail((310,160));canvas.paste(im,(col*320,row*200+30))
                draw.text((col*320+5,row*200+5),dataset+'/'+sid,fill='black')
        canvas.save(out/(dataset.lower()+'_group_review.jpg'))
    print(dataset,'pairs',len(ids),'groups',len(clusters),'duplicate pairs',len(duplicates),'dimension mismatches',len(dimensions),'group examples',report['nontrivial_groups'][:10],flush=True)

if __name__=='__main__':
    for dataset in ('UIEB','LSUI'):run(dataset)
