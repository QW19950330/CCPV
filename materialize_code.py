"""Restore one approved code/runtime archive without editing its contents."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import stat
import zipfile

HERE = Path(__file__).resolve().parent

def longpath(value):
    path = Path(value).resolve()
    if os.name == "nt" and not str(path).startswith(chr(92) * 2 + "?" + chr(92)):
        return Path(chr(92) * 2 + "?" + chr(92) + str(path))
    return path

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def materialize(bundle, archives_root, output):
    entries = json.loads((HERE / "SOURCE_ARCHIVES.json").read_text(encoding="utf-8"))["archives"]
    selected = next(row for row in entries if row["bundle"] == bundle)
    archive = longpath(archives_root) / selected["filename"]
    destination = longpath(output)
    repo = longpath(HERE)
    if destination.exists():
        raise FileExistsError("Output must not exist.")
    if destination.is_relative_to(repo) or repo.is_relative_to(destination):
        raise ValueError("Choose an output outside the source clone.")
    if archive.stat().st_size != selected["bytes"] or sha256(archive) != selected["sha256"]:
        raise ValueError("Archive size/SHA differs from the approved identity.")
    with zipfile.ZipFile(archive) as source:
        members = []
        names = set()
        for info in source.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or ":" in str(name) or chr(92) in str(name):
                raise ValueError("Unsafe archive member.")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Archive symlinks are not accepted.")
            identity = str(name).casefold()
            if identity in names:
                raise ValueError("Duplicate/case-colliding archive member.")
            names.add(identity)
            members.append((info, name))
        if len(members) != selected["complete_members"]:
            raise ValueError("Archive member count differs.")
        destination.mkdir(parents=True)
        output_rows = []
        for info, name in members:
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with source.open(info) as src, target.open("xb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    digest.update(chunk)
                    count += len(chunk)
                    dst.write(chunk)
            if count != info.file_size:
                raise ValueError("Incomplete extracted member.")
            output_rows.append({"path":str(name),"bytes":count,"sha256":digest.hexdigest()})
    if sha256(archive) != selected["sha256"]:
        raise ValueError("Source archive changed during materialization.")
    return {"status":"PASS_EXACT_ARCHIVE_MATERIALIZATION","bundle":bundle,
            "archive":selected["filename"],"archive_sha256":selected["sha256"],
            "files":len(output_rows),"bytes":sum(r["bytes"] for r in output_rows),
            "training_updates":0,"model_forwards":0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True,
                        choices=["portable","runtime","diagnostic_exact","diagnostic_portable"])
    parser.add_argument("--archives-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.bundle, args.archives_root, args.out), indent=2))
