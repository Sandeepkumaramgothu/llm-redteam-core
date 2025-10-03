#!/usr/bin/env python
from __future__ import annotations
import argparse, pathlib, json, hashlib, zipfile, os

def sha256_of_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True, help="runs/<run_id>")
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    assert run_dir.is_dir(), f"Not a run dir: {run_dir}"

    # Decide export name
    out_zip = run_dir / "export.zip"

    # Collect files we care about (you can add/remove as needed)
    wanted = []
    for p in run_dir.rglob("*"):
        if p.is_file() and not p.name.endswith(".zip"):
            wanted.append(p)

    # Build manifest with SHA256 of key files
    manifest = {"files": []}
    for p in wanted:
        rel = p.relative_to(run_dir).as_posix()
        manifest["files"].append({
            "path": rel,
            "sha256": sha256_of_file(p)
        })
    (run_dir/"MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Write zip
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # include MANIFEST first
        z.write(run_dir/"MANIFEST.json", arcname="MANIFEST.json")
        # include the rest, but skip MANIFEST to avoid duplicates
        for p in wanted:
            if p.name == "MANIFEST.json":
                continue
            z.write(p, arcname=p.relative_to(run_dir).as_posix())

    print("Wrote:", out_zip)

if __name__ == "__main__":
    main()
