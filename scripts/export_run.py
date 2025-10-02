\
#!/usr/bin/env python
from __future__ import annotations
import argparse, pathlib, hashlib, json, os, zipfile, time

def sha256_of_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def export_run(run_dir: pathlib.Path):
    assert run_dir.is_dir(), f"Not a directory: {run_dir}"
    # 1) Build manifest
    files = []
    for root, _, fnames in os.walk(run_dir):
        for fn in fnames:
            p = pathlib.Path(root) / fn
            rel = p.relative_to(run_dir).as_posix()
            files.append({"path": rel, "sha256": sha256_of_file(p), "size": p.stat().st_size})
    manifest = {
        "run_dir": run_dir.as_posix(),
        "export_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "files": files,
    }
    (run_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 2) Zip it
    zip_out = run_dir.parent / f"{run_dir.name}.zip"
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, fnames in os.walk(run_dir):
            for fn in fnames:
                p = pathlib.Path(root) / fn
                z.write(p, p.relative_to(run_dir))
    print("Wrote:", zip_out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    args = ap.parse_args()
    export_run(pathlib.Path(args.run_dir))

if __name__ == "__main__":
    main()
