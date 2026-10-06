#!/usr/bin/env python3
"""Download and fingerprint the original inputs required by code/make.jl."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def fetch(item, check_only=False):
    target = ROOT / item["path"]
    if target.is_file():
        if digest(target) != item["sha256"]:
            raise RuntimeError(f"Fingerprint mismatch: {item['path']}; existing file left unchanged")
        print(f"Verified {item['path']}")
        return
    if check_only:
        raise RuntimeError(f"Missing input: {item['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(item["url"], headers={"User-Agent": "PPB-replication"})
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        downloaded = Path(temporary) / "download"
        with urllib.request.urlopen(request, timeout=120) as response, downloaded.open("wb") as output:
            shutil.copyfileobj(response, output)
        if digest(downloaded) != item["sha256"]:
            raise RuntimeError(f"Fingerprint mismatch in download: {item['path']}; nothing installed")
        downloaded.replace(target)
    print(f"Downloaded and verified {item['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="check existing files without downloading")
    args = parser.parse_args()
    inputs = json.loads((ROOT / "data" / "inputs.json").read_text())
    for item in inputs["files"]:
        fetch(item, args.check_only)
    print(f"All {len(inputs['files'])} inputs verified.")


if __name__ == "__main__":
    main()
