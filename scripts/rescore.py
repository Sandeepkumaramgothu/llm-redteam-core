#!/usr/bin/env python
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import argparse, json
from src.reward.openassistant_reward import OpenAssistantReward, label_from_score
from src.reward.aux_detoxify import DetoxifyAux
from tqdm import tqdm

def latest_run():
    runs = sorted([p for p in pathlib.Path("runs").glob("*") if p.is_dir()],
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if list(r.glob("iter_*")): return r
    raise SystemExit("No runs with iter_* found.")

def rescore_run(run_dir: pathlib.Path, batch_size: int = 16):
    print("Using run:", run_dir.as_posix())
    rm = OpenAssistantReward()
    aux = DetoxifyAux()

    total = 0
    for it_dir in sorted(run_dir.glob("iter_*")):
        src = it_dir/"interactions.jsonl"
        if not src.exists() or src.stat().st_size == 0: continue

        rows = []
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                try: rows.append(json.loads(line))
                except: pass
        if not rows: continue

        prompts  = [r.get("adversarial_prompt","") for r in rows]
        responses= [r.get("response","") for r in rows]

        scores = []
        for i in tqdm(range(0, len(rows), batch_size), desc=f"[{it_dir.name}] OA-score"):
            scores.extend(rm.score_batch(prompts[i:i+batch_size], responses[i:i+batch_size]))

        aux_scores = []
        for i in tqdm(range(0, len(rows), batch_size), desc=f"[{it_dir.name}] Detoxify"):
            aux_scores.extend(aux.score_batch(responses[i:i+batch_size]))

        # attach labels
        out = []
        for r, s, ax in zip(rows, scores, aux_scores):
            r.update(label_from_score(s))
            for k,v in (ax or {}).items():
                r[f"aux_{k}"] = float(v)
            out.append(r)

        # Replace original JSONL (compact) and also write pretty JSON array for manual review
        tmp = it_dir/"interactions.scored.jsonl"
        with tmp.open("w", encoding="utf-8") as f:
            for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (it_dir/"interactions_pretty.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        bak = it_dir/"interactions.bak.jsonl"
        if src.exists(): src.replace(bak)
        tmp.replace(src)

        total += len(out)
        print(f"Scored {len(out)} rows in {it_dir.name}")
    print("Done. Total rescored rows:", total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="latest")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir) if args.run_dir != "latest" else latest_run()
    rescore_run(run_dir, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
