"""Extract delivered source/weights into a new directory and strict-load on CPU."""
import argparse,json,shutil,subprocess,tempfile,zipfile,os
from pathlib import Path
from uie_next.records import ROOT,write
RUN='ssuie_utility_v3_predictability_20261007'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=ROOT.parent);a=ap.parse_args();directory=a.directory
    with tempfile.TemporaryDirectory(prefix='v3_portable_',dir=str(directory)) as t:
        target=Path(t)/'restored';target.mkdir()
        for package in ['source_protocol','weights_recovery']:
            with zipfile.ZipFile(directory/(RUN+'_'+package+'.zip')) as z:z.extractall(target)
        upstream=target/'third_party/ss_uie'
        if upstream.exists():shutil.rmtree(upstream)
        bundle=target/'runs'/RUN/'delivery/ssuie_upstream.bundle'
        subprocess.run(['git','clone',str(bundle),str(upstream)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','-C',str(upstream),'checkout','88b23a1247d2d92ee7cf8dcad8f3b5079b6a20df'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        receipt=target/'recovery_strict_load.json'
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        cmd=[str(ROOT/'.venv/bin/python'),'-m','scripts.ssuie_v3_recovery_smoke','--receipt',str(receipt)]
        proc=subprocess.run(cmd,cwd=target,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        dest=ROOT/'runs'/RUN/'delivery/portable_recovery.log';dest.write_text(proc.stdout+'\n'+proc.stderr)
        if proc.returncode:raise RuntimeError('portable strict load failed; see '+str(dest))
        value=json.loads(receipt.read_text());value.update(source_and_weights_from_verified_packages=True,temporary_new_directory=True,new_process=True,independent_backup_verified=False,scope='fresh source/weight paths on current vendor Python; not independently reinstalled PPU environment')
        write(ROOT/'runs'/RUN/'tests/portable_recovery.json',value)
        print(json.dumps(value,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
