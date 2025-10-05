#!/usr/bin/env python
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import argparse, json, html, pandas as pd

def latest_run():
    runs = sorted([p for p in pathlib.Path("runs").glob("*") if p.is_dir()],
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if list(r.glob("iter_*")): return r
    raise SystemExit("No runs with iter_* found.")

def load_rows(run_dir: pathlib.Path):
    rows=[]
    for it in sorted(run_dir.glob("iter_*")):
        f=it/"interactions.jsonl"
        if not f.exists(): continue
        with f.open() as fh:
            for line in fh:
                try: rows.append(json.loads(line))
                except: pass
    if not rows: raise SystemExit("No interactions found.")
    return rows

def as_details(title: str, body: str) -> str:
    return f"""<details class='cell'><summary>{html.escape(title)}</summary><pre>{html.escape(body)}</pre></details>"""

def build(run_dir: pathlib.Path):
    out = run_dir
    rows = load_rows(run_dir)
    df = pd.DataFrame(rows)
    # choose columns to show
    cols = ["iteration","seed_id","risk","toxicity_level","success",
            "toxicity_prob","toxicity_score","aux_toxicity","aux_obscene","aux_insult",
            "seed","adversarial_prompt","response"]
    for c in cols:
        if c not in df.columns: df[c] = None

    # Pretty long text as expandable HTML
    show = df.copy()
    show["seed"] = show["seed"].fillna("")
    show["adversarial_prompt"] = show["adversarial_prompt"].fillna("")
    show["response"] = show["response"].fillna("")
    show["Seed"] = [as_details("see seed", s) for s in show["seed"]]
    show["Adversarial"] = [as_details("see adversarial", s) for s in show["adversarial_prompt"]]
    show["Response"] = [as_details("see response", s) for s in show["response"]]

    # CSV (full)
    df.to_csv(out/"interactions_full.csv", index=False)

    # HTML table (compact)
    small = show[["iteration","seed_id","risk","toxicity_level","success","toxicity_prob","toxicity_score",
                  "aux_toxicity","aux_obscene","aux_insult","Seed","Adversarial","Response"]]
    html_table = small.to_html(index=False, escape=False)

    # Write HTML page
    html_doc = f"""<html><head><meta charset='utf-8'>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: auto; }}
details.cell {{ margin: 4px 0; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
th {{ background: #f3f3f3; }}
</style>
<title>Interactions — {html.escape(run_dir.name)}</title></head>
<body>
<h1>Interactions — {html.escape(run_dir.name)}</h1>
<p><b>Rows:</b> {len(df)}</p>
{html_table}
</body></html>"""
    (out/"interactions_pretty.html").write_text(html_doc, encoding="utf-8")
    print("Wrote:", (out/"interactions_full.csv").as_posix(), "and", (out/"interactions_pretty.html").as_posix())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="latest")
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir) if args.run_dir != "latest" else latest_run()
    build(run_dir)

if __name__ == "__main__":
    main()
