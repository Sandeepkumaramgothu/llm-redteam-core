#!/usr/bin/env python
from __future__ import annotations

import argparse, pathlib, json
import pandas as pd

def iter_jsonl(path: pathlib.Path):
    # ELI-10: read each line safely and yield a dict
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                # skip bad lines so we don't crash
                continue

def main():
    # ELI-10: let the user say which run folder to read
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir)

    rows = []  # ELI-10: we'll store each record as a dict (like a spreadsheet row)

    # ELI-10: go through every iteration folder like runs/<id>/iter_0001/
    iters = sorted([p for p in run_dir.glob("iter_*") if p.is_dir()])
    for iterd in iters:
        iter_id = iterd.name  # like "iter_0001"
        jl = iterd / "interactions.jsonl"
        if not jl.exists():
            continue
        # ELI-10: read each example and keep useful fields
        for rec in iter_jsonl(jl):
            rows.append({
                "iteration": iter_id,
                "seed": rec.get("seed"),
                "risk": rec.get("risk") or rec.get("risk_type"),
                "adversarial_prompt": rec.get("adversarial_prompt"),
                "response": rec.get("response"),
                "score": rec.get("score"),
                "toxicity_level": rec.get("toxicity_level"),
                "success": rec.get("success"),
                "meta": rec.get("meta")  # keep any extra info as is
            })

    # ELI-10: build the big table
    df = pd.DataFrame(rows)

    # ELI-10: write CSV (easy to download)
    (run_dir / "full_data.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    # ELI-10: write HTML so the report can link to it (no masking)
    # escape=False lets the text show exactly as-is (no hiding), and the <style> helps wrap long lines
    style = """
    <style>
    table { width: 100%; border-collapse: collapse; }
    th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; }
    td { word-break: break-word; white-space: pre-wrap; }
    </style>
    """
    html = f"<h1>Full Data (Unmasked)</h1><p>All prompts and responses, exactly as saved.</p>{style}" + df.to_html(index=False, escape=False)
    (run_dir / "full_data.html").write_text(html, encoding="utf-8")

    print("Wrote:", run_dir / "full_data.csv", "and", run_dir / "full_data.html")

if __name__ == "__main__":
    main()
