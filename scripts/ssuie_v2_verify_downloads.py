"""Verify downloaded V2 deliveries using only the Python standard library.

Run on the client's independently stored copies, not on the server, to obtain
an independent-copy receipt. A server execution verifies packages only.
"""
import argparse
import hashlib
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


RUN_ID = 'ssuie_local_utility_v2_diagnostic_20261007'
PACKAGES = ['review', 'source_protocol', 'weights_recovery', 'visuals']
SIX = ['__init__.py', 'audit.py', 'cache.py', 'manifest.py', 'roles.py', 'runtime.py']


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_member(z, name):
    h = hashlib.sha256()
    with z.open(name) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_zip(path, expected, package, require_six=True):
    actual = sha_file(path)
    if actual != expected:
        raise ValueError('ZIP SHA256 mismatch: ' + str(path))
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP member')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:
                raise ValueError('Unsafe ZIP member: ' + name)
        bad = z.testzip()
        if bad:
            raise ValueError('ZIP CRC mismatch: ' + bad)
        manifest_name = 'runs/' + RUN_ID + '/delivery/package_member_hashes/' + package + '.json'
        manifest = json.loads(z.read(manifest_name))
        if manifest['package'] != package:
            raise ValueError('Wrong package identity')
        members = manifest['members']
        if set(names) != set(members) | {manifest_name}:
            raise ValueError('ZIP payload not fully covered by member manifest')
        for name, h in members.items():
            if sha_member(z, name) != h:
                raise ValueError('Member SHA256 mismatch: ' + name)
        six = {}
        frozen_files = {}
        if package == 'source_protocol' and require_six:
            snapshot_name = 'runs/ssuie_local_utility_v1_20261007/source_snapshot.json'
            original = json.loads(z.read(snapshot_name))
            for name in SIX:
                member = 'uie_next/data/' + name
                h = sha_member(z, member)
                if h != original[member]:
                    raise ValueError('Recovered data source differs from V1: ' + member)
                six[member] = h
            frozen = json.loads(z.read('runs/' + RUN_ID + '/source_snapshot.json'))
            for member, h in frozen.items():
                if sha_member(z, member) != h:
                    raise ValueError('Frozen scientific source mismatch: ' + member)
                frozen_files[member] = h
        return {'path': str(Path(path).resolve()), 'sha256': actual,
                'bytes': Path(path).stat().st_size, 'crc_verified': True,
                'all_payload_members_sha256_verified': True,
                'payload_members': len(members), 'original_six_data_modules': six,
                'frozen_scientific_source_files': len(frozen_files)}


def self_check():
    """Synthetic corruption checks; never reads experiment images or weights."""
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'fixture.zip'
        member = 'uie_next/data/__init__.py'
        name = 'runs/' + RUN_ID + '/delivery/package_member_hashes/source_protocol.json'
        payload = b'original synthetic source\n'
        record = {'package': 'source_protocol', 'members': {member: hashlib.sha256(payload).hexdigest()}}

        def make(data, extra=None):
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr(member, data)
                z.writestr(name, json.dumps(record))
                if extra:
                    z.writestr(extra, b'bad')

        make(payload)
        verify_zip(path, sha_file(path), 'source_protocol', require_six=False)
        rejected = []
        for case, action in [
            ('wrong_external_hash', lambda: verify_zip(path, '0' * 64, 'source_protocol', False)),
            ('missing_six_original_sources', lambda: verify_zip(path, sha_file(path), 'source_protocol')),
        ]:
            try:
                action()
            except (ValueError, KeyError):
                rejected.append(case)
            else:
                raise AssertionError('Invalid delivery accepted: ' + case)
        make(b'tampered source\n')
        try:
            verify_zip(path, sha_file(path), 'source_protocol', False)
        except ValueError:
            rejected.append('tampered_member_with_updated_zip_hash')
        else:
            raise AssertionError('Tampered member accepted')
        make(payload, '../escape.py')
        try:
            verify_zip(path, sha_file(path), 'source_protocol', False)
        except ValueError:
            rejected.append('unsafe_member_path')
        else:
            raise AssertionError('Unsafe member accepted')
    return {'passed': True, 'fixture_only': True, 'rejected_cases': rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('.'))
    parser.add_argument('--checksums', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--independent-copy', action='store_true',
                        help='Use only after copying to a client or an independent storage location.')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        result = self_check()
    else:
        sums = args.checksums or args.directory / (RUN_ID + '_SHA256SUMS.txt')
        expected = {}
        for line in sums.read_text(encoding='utf-8').splitlines():
            if line.strip():
                h, filename = line.split(None, 1)
                if len(h) != 64 or filename in expected or Path(filename).name != filename:
                    raise ValueError('Invalid or duplicate checksum entry')
                expected[filename] = h
        results = {}
        for package in PACKAGES:
            filename = RUN_ID + '_' + package + '.zip'
            results[package] = verify_zip(args.directory / filename, expected[filename], package)
        result = {'passed': True, 'verified_utc': datetime.now(timezone.utc).isoformat(),
                  'run_id': RUN_ID, 'packages': results,
                  'checksums_sha256': sha_file(sums),
                  'independent_copy_user_attested': args.independent_copy,
                  'independent_storage_automatically_verified': False,
                  'backup_status': 'client_copy_hash_verified_storage_user_attested' if args.independent_copy
                                  else 'package_integrity_only_not_independent_backup',
                  'scope': 'Four ZIPs, CRC, every payload member, and original six data sources; '
                           'Git/dependency bundles are separately listed in SHA256SUMS.'}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
