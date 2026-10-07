"""Verify LSUI against the frozen manifest, then publish an audited extraction."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from mpa_diff.data.manifest import audit, read_manifest
from scripts.cde_v3.common import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True)
    p.add_argument('--receipt', required=True)
    p.add_argument('--password')
    args = p.parse_args()
    archive = Path(args.archive).resolve()
    receipt = Path(args.receipt).resolve()
    destination = ROOT / 'data/lsui'
    manifest = ROOT / 'manifests/lsui_recon_grouped_v1.jsonl'
    if receipt.exists():
        raise FileExistsError('Receipt already exists: ' + str(receipt))
    if destination.exists():
        raise FileExistsError('Refusing to overwrite existing dataset: ' + str(destination))
    started = time.monotonic()
    snapshot = archive.stat()
    identity = (snapshot.st_size, snapshot.st_mtime_ns, snapshot.st_ino)
    manifest_hash = sha(manifest)
    result = {'archive': str(archive), 'archive_bytes': snapshot.st_size,
              'destination': str(destination), 'manifest': str(manifest),
              'manifest_sha256': manifest_hash, 'training_started': False,
              'utc_started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        result['archive_sha256'] = sha(archive)
        expected = {}
        rows = read_manifest(manifest)
        for row in rows:
            for role in ('image', 'reference'):
                name = row[role + '_path']
                if name in expected:
                    raise ValueError('Repeated manifest path: ' + name)
                expected[name] = row[role + '_sha256']
        with zipfile.ZipFile(archive) as bundle:
            password = args.password.encode() if args.password else None
            members = {}
            for item in bundle.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in item.filename or ':' in item.filename:
                    raise ValueError('Unsafe ZIP path: ' + item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError('ZIP symlink refused: ' + item.filename)
                if not path.parts or path.parts[0] != 'LSUI':
                    raise ValueError('Unexpected ZIP root: ' + item.filename)
                if item.is_dir():
                    continue
                name = PurePosixPath(*path.parts[1:]).as_posix()
                if name in members:
                    raise ValueError('Repeated ZIP member: ' + name)
                members[name] = item
            missing = sorted(set(expected) - set(members))
            extra = sorted(set(members) - set(expected))
            if missing or extra:
                raise ValueError('ZIP/manifest mismatch: ' + str({'missing': missing[:20], 'extra': extra[:20]}))
            result['uncompressed_bytes'] = sum(i.file_size for i in members.values())
            sizes = {}
            for number, (name, item) in enumerate(members.items(), 1):
                h = hashlib.sha256()
                with bundle.open(item, pwd=password) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(block)
                if h.hexdigest() != expected[name]:
                    raise ValueError('Frozen image hash mismatch: ' + name)
                with bundle.open(item, pwd=password) as stream, Image.open(stream) as image:
                    sizes[name] = image.size
                    image.verify()
                if number % 2000 == 0:
                    print('CRC, SHA256 and image format verified:', number, flush=True)
            for row in rows:
                if sizes[row['image_path']] != sizes[row['reference_path']]:
                    raise ValueError('Pair size mismatch: ' + row['sample_id'])
            result['archive_crc_verified_files'] = len(members)
            result['original_image_hashes_verified_files'] = len(members)
            result['image_format_checks_passed'] = True
            result['pair_dimensions_match'] = True
            free = shutil.disk_usage(destination.parent).free
            if free < result['uncompressed_bytes'] + 128 * 1024 * 1024:
                raise OSError('Insufficient disk space for safe extraction')
            staging = Path(tempfile.mkdtemp(prefix='.lsui-recovery-', dir=str(destination.parent)))
            result['staging_path'] = str(staging)
            for name, item in members.items():
                target = staging.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item, pwd=password) as source, target.open('xb') as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
            result['data_audit'] = audit(manifest, staging)
            current = archive.stat()
            if identity != (current.st_size, current.st_mtime_ns, current.st_ino):
                raise ValueError('Archive changed during verification/extraction')
            if sha(manifest) != manifest_hash:
                raise ValueError('Frozen manifest changed during verification')
            if destination.exists():
                raise FileExistsError('Dataset appeared during extraction; refusing overwrite')
            staging.rename(destination)
        result.update(status='verified_and_extracted', pairs=len(rows),
                      splits=dict(Counter(row['split'] for row in rows)),
                      manifest_unchanged=True)
    except Exception as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        write(receipt, result)
        raise
    result['elapsed_seconds'] = time.monotonic() - started
    write(receipt, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
