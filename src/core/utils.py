from __future__ import annotations
import os, json, time, pathlib

def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())

def ensure_dir(p: str | pathlib.Path) -> str:
    p = pathlib.Path(p); p.mkdir(parents=True, exist_ok=True); return str(p)

def write_json(path: str | pathlib.Path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path: str | pathlib.Path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
