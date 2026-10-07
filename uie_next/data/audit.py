"""Full input/reference content audit, conservative components, fixed role split."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from uie_next.records import ROOT, RUN, write, read, jsonl, sha,append


class Components:
    def __init__(self, n): self.parents = list(range(n))
    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]; i = self.parents[i]
        return i
    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parents[max(a,b)] = min(a,b)


def features(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Non-8-bit RGB or corrupt image: '+str(path))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h = hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    small = cv2.resize(gray, (32,32), interpolation=cv2.INTER_AREA)
    frequency = cv2.dct(small)[:8,:8].reshape(-1)[1:]
    bits = frequency > np.median(frequency)
    phash = sum(int(bit) << i for i,bit in enumerate(bits))
    thumb = cv2.resize(gray, (16,16), interpolation=cv2.INTER_AREA).reshape(-1)
    with Image.open(path) as im:
        orientation = im.getexif().get(274)
    abnormal = bool(gray.std() < 1e-4 or orientation not in (None,1))
    return {'decoded_sha256':h, 'size_wh':[image.shape[1], image.shape[0]],
            'phash':phash, 'near_constant':bool(gray.std()<1e-4),
            'orientation':orientation, 'abnormal':abnormal}, thumb


def exact_edges(entries, comps):
    seen={};edges=[]
    for i,e in enumerate(entries):
        h=e['decoded_sha256']
        if h in seen:
            other=seen[h];comps.join(e['owner'],entries[other]['owner'])
            if e['owner']!=entries[other]['owner']:
                edges.append({'left':other,'right':i,'exact_rgb':True,'hamming':None,'thumb_mse':0.})
        else:seen[h]=i
    return edges


def clarify_membership(rows):
    for row in rows:
        source_related=row['dataset']=='LSUI' or row['role']=='excluded_overlap'
        row['upstream_train_membership']='author_split_not_mapped' if source_related else 'checkpoint_provenance_pending'
        row['conservatively_treat_as_exposed']=source_related
        if source_related:row['upstream_exposure']='unknown_upstream_exposure'
        row['backbone_provenance_verified']=False
    return rows


def run_audit():
    run = RUN; run.mkdir(parents=True, exist_ok=True)
    final = run/'cross_dataset_audit.json'
    if final.exists():
        saved = json.loads(final.read_text())
        freeze=read(run/'split_freeze.json')
        if sha(run/'roles.jsonl')!=freeze['roles_sha256']:raise ValueError('Role freeze identity changed')
        for name,h in saved['manifest_hashes'].items():
            if sha(ROOT/name) != h: raise ValueError('Audit identity changed')
        for row in read(run/'exposure_ledger.json')['samples']:
            for key in ['input','reference']:
                if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('Audited data changed')
        if saved.get('membership_exposure_schema')!=2:
            ledger=read(run/'exposure_ledger.json')
            write(run/'audit_history/exposure_ledger_before_membership_clarification.json',ledger)
            write(run/'audit_history/split_freeze_before_membership_clarification.json',freeze)
            samples=clarify_membership(ledger['samples'])
            ledger.update(samples=samples,author_train_membership='unknown; LSUI source-pool relation is not individual train membership')
            write(run/'exposure_ledger.json',ledger);jsonl(run/'roles.jsonl',samples);jsonl(run/'all_pairs.jsonl',samples)
            freeze.update(roles_sha256=sha(run/'roles.jsonl'),membership_exposure_schema=2)
            write(run/'split_freeze.json',freeze)
            saved['membership_exposure_schema']=2;write(final,saved)
            append(run/'protocol_amendments.jsonl',{'kind':'audit_metadata_correction','reason':'README LSUI source does not establish per-image author train membership',
                'role_sample_ids_and_groups_changed':False,'thresholds_changed':False,'trained_models_or_metrics_exist':False,
                'history':'audit_history/exposure_ledger_before_membership_clarification.json'})
        return saved
    start=time.monotonic(); cv2.setNumThreads(2)
    rows=[]; entries=[]; thumbs=[]; manifest_hashes={}
    for dataset in ('lsui','uieb'):
        name='manifests/'+dataset+'_recon_grouped_v1.jsonl'; manifest_hashes[name]=sha(ROOT/name)
        source=[json.loads(line) for line in (ROOT/name).read_text().splitlines() if line.strip()]
        for row in source:
            record={'sample_id':row['sample_id'], 'dataset':row['dataset'],
                    'legacy_split':row['split'], 'legacy_scene_id':row['scene_id'],
                    'historically_analyzed':True, 'source_evidence':['frozen_local_manifest',
                      'SS-UIE README at 88b23a1: 3879 LSUI train / 400 test'],
                    'source_url':row['source_url']}
            owner=len(rows)
            for old,new in [('image','input'),('reference','reference')]:
                path=ROOT/'data'/dataset/row[old+'_path']; h=sha(path)
                if h != row[old+'_sha256']: raise ValueError('Original file changed: '+str(path))
                feat,thumb=features(path)
                record[new+'_path']=str(path);record[new+'_sha256']=h
                record['decoded_'+new+'_sha256']=feat['decoded_sha256']
                record[new+'_size_wh']=feat['size_wh']
                entries.append(dict(feat,owner=owner,kind=new,path=str(path),sample_id=row['sample_id'],dataset=row['dataset']))
                thumbs.append(thumb)
            record['abnormal']=any(e['abnormal'] for e in entries[-2:])
            if record['input_size_wh'] != record['reference_size_wh']: record['abnormal']=True
            rows.append(record)
            if len(rows)%500==0: print('Decoded audit pairs:',len(rows),flush=True)
    comps=Components(len(rows));edges=[];seen={};scenes={}
    for i,row in enumerate(rows):
        key=(row['dataset'],row['legacy_scene_id'])
        if row['legacy_scene_id']:
            if key in scenes: comps.join(i,scenes[key])
            else: scenes[key]=i
    edges.extend(exact_edges(entries,comps))
    near={}; hashes=np.asarray([e['phash'] for e in entries],dtype=np.uint64)
    lut=np.asarray([bin(i).count('1') for i in range(256)],dtype=np.uint8)
    for start_index in range(0,len(entries),128):
        xor=np.bitwise_xor(hashes[start_index:start_index+128,None],hashes[None,:])
        distances=lut[xor.view(np.uint8).reshape(len(xor),len(entries),8)].sum(-1)
        ii,jj=np.where(distances<=4)
        for a,b in zip(ii.tolist(),jj.tolist()):
            a+=start_index
            if a>=b or entries[a]['owner']==entries[b]['owner']: continue
            if entries[a]['abnormal'] or entries[b]['abnormal']: continue
            near[(a,b)]={'hamming':int(distances[a-start_index,b])}
    matrix=np.asarray(thumbs,dtype=np.float64)
    for a,b in cKDTree(matrix).query_pairs(np.sqrt(256*1e-5)):
        if entries[a]['owner']==entries[b]['owner'] or entries[a]['abnormal'] or entries[b]['abnormal']:continue
        near.setdefault((a,b),{})['thumb_mse']=float(np.mean((matrix[a]-matrix[b])**2))
    for (a,b),scores in sorted(near.items()):
        comps.join(entries[a]['owner'],entries[b]['owner'])
        scores.setdefault('hamming',bin(entries[a]['phash']^entries[b]['phash']).count('1'))
        scores.setdefault('thumb_mse',float(np.mean((matrix[a]-matrix[b])**2)))
        edges.append(dict(left=a,right=b,exact_rgb=False,**scores))
    groups=defaultdict(list)
    for i,row in enumerate(rows): groups[comps.find(i)].append(i)
    group_records=[];eligible=[];lsui_groups=[]
    for indexes in groups.values():
        ids=sorted(rows[i]['sample_id'] for i in indexes)
        gid='content_'+hashlib.sha256('|'.join(ids).encode()).hexdigest()[:20]
        mixed=len({rows[i]['dataset'] for i in indexes})>1
        abnormal=any(rows[i]['abnormal'] for i in indexes)
        record={'group_id':gid,'sample_ids':ids,'group_quality':'content_group_proxy','cross_dataset':mixed,'abnormal':abnormal}
        group_records.append(record)
        for i in indexes:
            rows[i].update(group_id=gid,group_quality='content_group_proxy',role='excluded')
            rows[i]['upstream_exposure']='unknown_upstream_exposure' if rows[i]['dataset']=='LSUI' or mixed else 'documented_nonoverlap'
            rows[i]['cross_dataset_group']=mixed
            rows[i]['exposure_scope']='declared LSUI source; actual checkpoint provenance still must be verified'
        if mixed or abnormal:continue
        if rows[indexes[0]]['dataset']=='UIEB':eligible.append(record)
        else:lsui_groups.append(record)
    rng=np.random.RandomState(20261007)
    eligible=sorted(eligible,key=lambda g:g['group_id']);eligible=[eligible[i] for i in rng.permutation(len(eligible))]
    g=len(eligible);cuts=[int(.5*g),int(.15*g),int(.15*g)]
    roles={};position=0
    for name,n in zip(['utility_fit','utility_val','calibration','sealed_eval'],cuts+[g-sum(cuts)]):
        roles[name]=[r['group_id'] for r in eligible[position:position+n]];position+=n
    lsui_groups=sorted(lsui_groups,key=lambda g:g['group_id'])
    lsui_groups=[lsui_groups[i] for i in np.random.RandomState(20261007).permutation(len(lsui_groups))]
    cut=int(.85*len(lsui_groups));roles['model_fit']=[r['group_id'] for r in lsui_groups[:cut]];roles['model_val']=[r['group_id'] for r in lsui_groups[cut:]]
    # A mixed source component is usable for candidate fitting only, never as an
    # independent utility group. Keep every component in a single role.
    mixed_lsui=[g for g in group_records if g['cross_dataset'] and not g['abnormal']]
    roles['model_fit'] += [g['group_id'] for g in mixed_lsui]
    role_by_group={gid:name for name,gids in roles.items() for gid in gids}
    for row in rows:
        row['role']=role_by_group.get(row['group_id'],'excluded')
        if row['dataset']=='UIEB' and row['cross_dataset_group']:row['role']='excluded_overlap'
    clarify_membership(rows)
    counts={name:{'groups':len(gids),'pairs':sum(r['role']==name for r in rows)} for name,gids in roles.items()}
    minimum={'utility_fit':(80,100),'utility_val':(25,40),'calibration':(25,40),'sealed_eval':(40,60)}
    deficits={role:{'required_groups':ng,'required_pairs':npairs,'actual':counts[role]} for role,(ng,npairs) in minimum.items() if counts[role]['groups']<ng or counts[role]['pairs']<npairs}
    for e in edges:
        e['left_sample']=entries[e['left']]['sample_id'];e['right_sample']=entries[e['right']]['sample_id']
        e['left_kind']=entries[e['left']]['kind'];e['right_kind']=entries[e['right']]['kind']
        e['cross_dataset']=entries[e['left']]['dataset']!=entries[e['right']]['dataset']
    jsonl(run/'all_pairs.jsonl',rows);jsonl(run/'roles.jsonl',rows)
    jsonl(run/'near_duplicate_edges.jsonl',edges);write(run/'groups.json',group_records)
    write(run/'exposure_ledger.json',{'upstream_claim':'README declares LSUI training, actual downloaded checkpoint not yet verified','historically_analyzed_all_local_samples':True,'samples':rows})
    contact_edges=sorted([e for e in edges if e['cross_dataset']],key=lambda e:(not e['exact_rgb'],e['left_sample'],e['right_sample']))[:24]
    if contact_edges:
        canvas=Image.new('RGB',(512,152*len(contact_edges)),(255,255,255));draw=ImageDraw.Draw(canvas)
        for n,e in enumerate(contact_edges):
            for col,key in enumerate(['left','right']):
                with Image.open(entries[e[key]]['path']) as im:canvas.paste(im.convert('RGB').resize((256,128)),(col*256,n*152))
                draw.text((col*256,n*152+129),entries[e[key]]['sample_id']+' '+entries[e[key]]['kind'],fill=(0,0,0))
        (run/'figures').mkdir(exist_ok=True);canvas.save(run/'figures/cross_dataset_contact.png')
        write(run/'figures/contact_selection.json',contact_edges)
    result={'status':'BLOCKED_DATA' if deficits else 'DATA_COUNTS_PASS_PENDING_BACKBONE_PROVENANCE','membership_exposure_schema':2,
            'pairs':len(rows),'image_records':len(entries),'manifest_hashes':manifest_hashes,
            'all_lsui_pairs_audited':4279,'all_uieb_pairs_audited':890,
            'cross_dataset_edge_count':sum(e['cross_dataset'] for e in edges),
            'uieb_excluded_overlap':sum(r['role']=='excluded_overlap' for r in rows),
            'near_edge_count':len(near),'exact_edge_count':sum(e['exact_rgb'] for e in edges),
            'role_counts':counts,'minimum_deficits':deficits,'group_quality':'content_group_proxy',
            'phash_threshold':4,'thumbnail_mse_threshold':1e-5,
            'cpu_wall_seconds':time.monotonic()-start,'gpu_seconds':0,
            'sealed_reference_scoring':False,'scope':'Full local LSUI/UIEB, input/reference cross comparisons; conservative unreviewed links; not semantic scene proof.'}
    write(run/'split_freeze.json',{'split_seed':20261007,'roles':roles,'counts':counts,'minimums_passed':not deficits,'roles_sha256':sha(run/'roles.jsonl'),'audit_status':result['status'],'sealed_eval_released':False})
    write(final,result);print(json.dumps(result,indent=2),flush=True)
    return result
