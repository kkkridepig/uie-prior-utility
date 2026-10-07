"""Atomic evidence records and stable identities."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/ssuie_local_utility_v1_20261007'
DOC = ROOT / 'docs/experiments/ssuie_local_utility_v1_20261007'
GUIDE = ROOT.parent / 'TEMP-FILE-STATION/SSUIE_C局部效用学习_本轮完整执行指南_20261008.md'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=str(path.parent), delete=False, encoding='utf-8') as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n'); out.flush(); os.fsync(out.fileno()); temporary = out.name
    os.replace(temporary, str(path))
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as out:
        out.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n'); out.flush(); os.fsync(out.fileno())


def jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=str(path.parent), delete=False, encoding='utf-8') as out:
        for row in rows: out.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        out.flush(); os.fsync(out.fileno()); temporary = out.name
    os.replace(temporary, str(path))
