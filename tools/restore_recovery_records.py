"""Verify archived V2/V3 evidence and optionally restore only run records."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/recovery/20261006/RECOVERY_MANIFEST.json"
RECORD_ARCHIVES = {
    "review_bundle_no_weights_no_images.zip": "",
    "uie-prior-utility-runs-records-20261003.zip": "uie-prior-utility/",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Restore missing run records; refuse conflicting files.")
    args = parser.parse_args()
    registry = json.loads(MANIFEST.read_text(encoding="utf-8"))
    writes = []
    identical = 0
    for record in registry["archives"]:
        archive = ROOT / record["path"]
        if digest(archive) != record["sha256"]:
            raise ValueError("Archive SHA256 mismatch: " + record["path"])
        if archive.name not in RECORD_ARCHIVES:
            continue
        prefix = RECORD_ARCHIVES[archive.name]
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("Archive CRC mismatch: " + record["path"])
            for member in bundle.infolist():
                if member.is_dir():
                    continue
                name = member.filename
                if prefix:
                    if not name.startswith(prefix):
                        raise ValueError("Unexpected archive prefix: " + name)
                    name = name[len(prefix):]
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
                    raise ValueError("Unsafe archive path: " + name)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Archive symlink refused: " + name)
                if not name.startswith("runs/"):
                    continue
                destination = ROOT.joinpath(*relative.parts)
                resolved_root = ROOT.resolve()
                resolved = destination.resolve()
                if resolved_root not in resolved.parents:
                    raise ValueError("Destination outside checkout: " + name)
                payload = bundle.read(member)
                if destination.exists():
                    if destination.read_bytes() != payload:
                        raise FileExistsError("Different existing record; nothing overwritten: " + name)
                    identical += 1
                else:
                    writes.append((destination, payload))
    if args.apply:
        for destination, payload in writes:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also refuses a file created after validation.
            with destination.open("xb") as stream:
                stream.write(payload)
    print(json.dumps({
        "mode": "apply" if args.apply else "verify_only",
        "archives_verified": len(registry["archives"]),
        "identical_existing_records": identical,
        "records_restored" if args.apply else "records_to_restore": len(writes),
        "weights_or_images_restored": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
