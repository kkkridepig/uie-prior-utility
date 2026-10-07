"""Standard-library verifier. Run on downloaded copies; no claim of remote backup."""
import argparse,hashlib,json,zipfile
from pathlib import Path,PurePosixPath
RUN='ssuie_utility_v3_predictability_20261007'
NAMES=['review','source_protocol','weights_recovery','visuals']


def hash_stream(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=Path('.'));ap.add_argument('--independent-copy',action='store_true');ap.add_argument('--receipt',type=Path,default=Path('V3_DOWNLOAD_VERIFICATION.json'));a=ap.parse_args();d=a.directory
    checks={line.split()[1]:line.split()[0] for line in (d/(RUN+'_SHA256SUMS.txt')).read_text().splitlines()};results={}
    for package in NAMES:
        filename=RUN+'_'+package+'.zip';p=d/filename
        with p.open('rb') as f:h=hash_stream(f)
        if h!=checks[filename]:raise ValueError('ZIP SHA mismatch '+filename)
        with zipfile.ZipFile(p) as z:
            members=z.namelist()
            if len(members)!=len(set(members)):raise ValueError('duplicate member')
            for n in members:
                if PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n:raise ValueError('unsafe path')
            if z.testzip() is not None:raise ValueError('CRC failure')
            mn='runs/'+RUN+'/delivery/package_member_hashes/'+package+'.json';m=json.loads(z.read(mn))
            if set(members)!=(set(m['members'])|{mn}):raise ValueError('uncovered payload')
            for n,hh in m['members'].items():
                with z.open(n) as f:actual=hash_stream(f)
                if actual!=hh:raise ValueError('member SHA mismatch '+n)
            if package=='source_protocol':
                original=json.loads(z.read('runs/ssuie_local_utility_v1_20261007/source_snapshot.json'))
                for n in ['__init__.py','audit.py','cache.py','manifest.py','roles.py','runtime.py']:
                    member='uie_next/data/'+n
                    if hashlib.sha256(z.read(member)).hexdigest()!=original[member]:raise ValueError('original data module mismatch')
                frozen=json.loads(z.read('runs/'+RUN+'/final_source_snapshot.json'))
                for n,hh in frozen.items():
                    if hashlib.sha256(z.read(n)).hexdigest()!=hh:raise ValueError('final source mismatch '+n)
            results[package]={'sha256':h,'CRC_pass':True,'member_SHA_pass':True,'members':len(m['members'])}
    receipt={'packages':results,'independent_backup_verified':a.independent_copy,'independent_copy_flag_is_user_attestation':True}
    a.receipt.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
