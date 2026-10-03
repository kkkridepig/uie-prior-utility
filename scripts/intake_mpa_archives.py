"""Inspect/extract only the three named uploaded archives into isolated staging."""
import json,os,stat,zipfile,sys
from pathlib import Path
import libarchive
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mpa_diff.utils.io import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]
UPLOAD=ROOT.parent/'TEMP-FILE-STATION'

def run():
    result=[]
    for name in ['raw-890-s.rar','reference-890.rar','LSUI.zip']:
        archive=UPLOAD/name;dest=ROOT/'data/intake'/name.replace('.rar','').replace('.zip','')
        receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha256(archive),'destination':str(dest),'files':0,'status':'pending'}
        try:
            if name.endswith('.zip'):
                with zipfile.ZipFile(archive) as z:
                    bad=z.testzip()
                    if bad:raise ValueError('ZIP CRC failure: '+bad)
            with libarchive.file_reader(str(archive)) as entries:
                for entry in entries:
                    path=dest/entry.pathname.replace('\\','/')
                    if dest.resolve() not in path.resolve().parents and path.resolve()!=dest.resolve():raise ValueError('Unsafe archive path '+entry.pathname)
                    if entry.isdir:continue
                    if not entry.isfile:raise ValueError('Nonregular archive member '+entry.pathname)
                    path.parent.mkdir(parents=True,exist_ok=True)
                    if path.exists():raise FileExistsError(str(path))
                    with path.open('xb') as f:
                        for block in entry.get_blocks():f.write(block)
                    receipt['files']+=1
            receipt['status']='extracted'
        except Exception as e:
            receipt.update(status='failed',error=type(e).__name__+': '+str(e))
        result.append(receipt);write_json(ROOT/'runs/data_intake/archives.json',result);print(json.dumps(receipt),flush=True)
    return result
if __name__=='__main__':run()
