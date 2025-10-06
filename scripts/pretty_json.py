
#!/usr/bin/env python3
from __future__ import annotations
import pathlib, json, io, sys

def pretty_json_file(p: pathlib.Path):
   try:
       obj = json.loads(p.read_text(encoding="utf-8"))
   except Exception as e:
       print(f"[skip] {p.name}: {e}")
       return
   out = p.with_name(p.stem + "_pretty.json")
   out.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
   print("wrote", out.name)

def pretty_jsonl_file(p: pathlib.Path):
   out = p.with_name(p.stem + "_pretty.jsonl")
   with io.open(p, "r", encoding="utf-8") as fin, io.open(out, "w", encoding="utf-8") as fou:
       for li, line in enumerate(fin, start=1):
           line=line.strip()
           if not line:
               fou.write("\n")
               continue
           try:
               obj = json.loads(line)
               fou.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
           except Exception as e:
               print(f"[warn] {p.name}:{li}: {e}")
               fou.write(line + "\n")
   print("wrote", out.name)

def run(run_dir: str):
   rd = pathlib.Path(run_dir)
   assert rd.exists(), f"Not found: {run_dir}"
   for p in rd.rglob("*.json"):
       if p.name.endswith("_pretty.json"): continue
       pretty_json_file(p)
   for p in rd.rglob("*.jsonl"):
       if p.name.endswith("_pretty.jsonl"): continue
       pretty_jsonl_file(p)

def main():
   import argparse
   ap = argparse.ArgumentParser()
   ap.add_argument("--run-dir", required=True)
   args = ap.parse_args()
   run(args.run_dir)

if __name__ == "__main__":
   main()
