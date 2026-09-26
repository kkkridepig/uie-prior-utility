"""Official/author-linked sources. Downloads are never uploaded to GitHub."""
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

SOURCES = {
    "wwe-bundle": {
        "id": "1ceeWdqRdiIortVTSH0UNTZamHdMsZ3aA",
        "filename": "UnderWaterDataset.zip",
        "page": "https://github.com/chingheng0808/WWE-UIE",
        "description": "Author-provided splits: UIEB/LSUI/UFO-120/EUVP/C60/U45/UCCS",
    },
    "uieb-raw": {
        "id": "12W_kkblc2Vryb9zHQ6BfGQ_NKUfXYk13", "filename": "raw-890.zip",
        "page": "https://li-chongyi.github.io/proj_benchmark.html",
    },
    "uieb-reference": {
        "id": "1cA-8CzajnVEL4feBRKdBxjEe6hwql6Z7", "filename": "reference-890.zip",
        "page": "https://li-chongyi.github.io/proj_benchmark.html",
    },
    "uieb-challenging": {
        "id": "1Ew_r83nXzVk0hlkfuomWqsAIxuq6kaN4", "filename": "challenging-60.zip",
        "page": "https://li-chongyi.github.io/proj_benchmark.html",
    },
}


def safe_extract(archive, output):
    archive, output = Path(archive), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    def destination(name):
        dest = (output / name.replace("\\", "/")).resolve()
        if not dest.is_relative_to(output):
            raise ValueError(f"Unsafe archive member: {name}")
        return dest

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as f:
            for item in f.infolist():
                destination(item.filename)
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlink refused")
            # Validate all members before writing any.
            for item in f.infolist():
                if not item.is_dir():
                    dest = destination(item.filename)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        raise ValueError(f"Extraction would overwrite {dest}")
                    with f.open(item) as src, dest.open("wb") as dst:
                        import shutil
                        shutil.copyfileobj(src, dst)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as f:
            members = f.getmembers()
            for item in members:
                destination(item.name)
                if not (item.isfile() or item.isdir()):
                    raise ValueError("Archive links/devices refused")
            for item in members:
                if item.isfile():
                    dest = destination(item.name)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        raise ValueError(f"Extraction would overwrite {dest}")
                    with f.extractfile(item) as src, dest.open("wb") as dst:
                        import shutil
                        shutil.copyfileobj(src, dst)
    else:
        raise ValueError("Not a valid ZIP/TAR archive (a login/quota page may have been downloaded)")


def download(name, output, extract_to=None, source_url=None):
    if name not in SOURCES:
        raise ValueError(f"Unknown dataset source: {name}")
    source = SOURCES[name]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / source["filename"]
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        command = [sys.executable, "-m", "gdown", source_url or source["id"], "-O", str(partial), "--continue"]
        if source_url:
            command.append("--fuzzy")
        subprocess.run(command, check=True)
        if not (zipfile.is_zipfile(partial) or tarfile.is_tarfile(partial)):
            raise ValueError(f"Download is not an archive: {partial}")
        partial.replace(path)
    if extract_to:
        safe_extract(path, extract_to)
    from .data import sha256_file
    receipt = dict(source, requested_source=source_url or source["id"],
                   file=str(path), sha256=sha256_file(path))
    path.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
