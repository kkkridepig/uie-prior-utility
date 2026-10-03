import json
import random
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from mpa_diff.utils.io import sha256, write_json

FIELDS='sample_id dataset split image_path reference_path scene_id sequence_id camera_id domain_id depth_path depth_units depth_kind task_annotation_path source_url license image_sha256 reference_sha256'.split()


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def audit(path, root, check_near=True):
    rows=read_manifest(path); root=Path(root); ids=set(); hashes={}; groups={}; fingerprints=[]; within_split_duplicates=[]
    if not rows: raise ValueError('Empty manifest')
    for r in rows:
        sid=r.get('sample_id'); missing=set(FIELDS)-set(r)
        if missing: raise ValueError('%s missing fields %s'%(sid,sorted(missing)))
        if sid in ids: raise ValueError('Duplicate sample_id: '+str(sid))
        ids.add(sid)
        if r['split'] not in ('train','val','test'): raise ValueError('Invalid split: '+sid)
        dims=[]
        for field in ('image','reference'):
            name=r[field+'_path']
            if name is None:
                if field=='image': raise ValueError('Missing image: '+sid)
                continue
            file=(root/name).resolve()
            if root.resolve() not in file.parents: raise ValueError('Path outside data_root: '+sid)
            if not file.is_file(): raise FileNotFoundError('%s: %s'%(sid,file))
            digest=sha256(file)
            if digest!=r[field+'_sha256']: raise ValueError('Stale hash: '+sid+' '+field)
            old=hashes.get(digest)
            if old and old[0]!=sid:
                if old[1]!=r['split'] or not r['scene_id'] or old[2]!=r['scene_id']:
                    raise ValueError('Duplicate content without same-split group: %s and %s'%(sid,old[0]))
                within_split_duplicates.append({'sample_id':sid,'other':old[0],'role':field,'split':r['split']})
            hashes[digest]=(sid,r['split'],r['scene_id'])
            with Image.open(file) as im:
                dims.append(im.size)
                if field=='image' and check_near:
                    a=np.asarray(im.convert('L').resize((16,16),Image.BILINEAR),dtype=np.float32)/255
                    fingerprints.append((sid,r['split'],a))
        if len(dims)==2 and dims[0]!=dims[1]: raise ValueError('Misaligned pair dimensions: '+sid)
        for group in ('scene_id','sequence_id'):
            if r[group] is not None:
                key=(r['dataset'],group,r[group]); old=groups.get(key)
                if old and old!=r['split']: raise ValueError('Scene/sequence leakage: '+sid)
                groups[key]=r['split']
    suspects=[]
    if fingerprints:
        features=np.stack([a.astype(np.float64).ravel() for _,_,a in fingerprints])
        for i,j in sorted(cKDTree(features).query_pairs(np.sqrt(256*1e-5))):
            sid,split,_=fingerprints[i];other,sp,_=fingerprints[j]
            if split!=sp:suspects.append([sid,other])
    if suspects: raise ValueError('Near-duplicate cross-split images require scene review: '+str(suspects[:20]))
    return {'samples':len(rows),'splits':dict(Counter(r['split'] for r in rows)),'manifest_sha256':sha256(path),'content_hashes_verified':True,'known_groups_isolated':True,'unknown_scene_ids':sum(r['scene_id'] is None for r in rows),'near_duplicate_check':'16x16 gray MSE < 1e-5; heuristic, not proof of scene independence','near_duplicates':suspects,'same_split_grouped_duplicate_files':within_split_duplicates}


def prepare(config):
    allowed={'dataset','data_root','image_dir','reference_dir','manifest','counts','seed','source_url','license','groups'}
    if set(config)-allowed: raise ValueError('Unknown data config keys')
    root=Path(config['data_root']).resolve()
    def files(directory):
        result={}
        for p in sorted((root/directory).rglob('*')):
            if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp'):
                if p.stem in result: raise ValueError('Duplicate filename stem: '+p.stem)
                result[p.stem]=p
        return result
    inputs=files(config['image_dir']); refs=files(config['reference_dir'])
    if not inputs or set(inputs)!=set(refs): raise ValueError('Unpaired / absent files: '+str(sorted(set(inputs)^set(refs))[:20]))
    names=sorted(inputs); random.Random(config.get('seed',20260927)).shuffle(names)
    counts=config['counts']
    if sum(counts)!=len(names): raise ValueError('Expected %s pairs; found %s'%(sum(counts),len(names)))
    groups=json.loads(Path(config['groups']).read_text()) if config.get('groups') else {}
    rows=[]
    for i,sid in enumerate(names):
        split='train' if i<counts[0] else ('val' if i<sum(counts[:2]) else 'test')
        row=dict.fromkeys(FIELDS)
        row.update(sample_id=config['dataset']+'/'+sid,dataset=config['dataset'],split=split,image_path=str(inputs[sid].relative_to(root)),reference_path=str(refs[sid].relative_to(root)),scene_id=groups.get(sid),source_url=config['source_url'],license=config['license'],image_sha256=sha256(inputs[sid]),reference_sha256=sha256(refs[sid]))
        rows.append(row)
    if groups:
        # Group preservation takes precedence over exact table counts.
        by_group={}
        for row in rows: by_group.setdefault(row['scene_id'] or row['sample_id'],[]).append(row)
        used=0
        for group in by_group.values():
            split='train' if used<counts[0] else ('val' if used<sum(counts[:2]) else 'test')
            for row in group: row['split']=split
            used+=len(group)
        if 'grouped' not in Path(config['manifest']).stem: raise ValueError('Group split requires separately named grouped manifest')
    path=Path(config['manifest']); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError('Refusing to overwrite frozen manifest: '+str(path))
    temporary=path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    report=audit(temporary,root)
    temporary.replace(path)
    write_json(str(path)+'.audit.json',report)
    return report
