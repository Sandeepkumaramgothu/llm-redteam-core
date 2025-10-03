#!/usr/bin/env python
from __future__ import annotations

import pathlib, json, math, collections
import pandas as pd

# Helper: safe open JSONL and yield dicts
def iter_jsonl(path: pathlib.Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue

# Map continuous toxicity 'score' ([-10..10] per paper) to 0..5 bins if discrete isn't present  :contentReference[oaicite:14]{index=14}
def to_level(rec):
    if rec.get("toxicity_level") is not None:
        return int(rec["toxicity_level"])
    s = rec.get("score", None)
    if s is None:
        return 0
    s = max(-10.0, min(10.0, float(s)))
    # 6 equal bins: [-10,-6.7),[-6.7,-3.3),[-3.3,0),[0,3.3),[3.3,6.7),[6.7,10]
    idx = int((s + 10) / (20/6))
    return max(0, min(5, idx))

# Extract a "prompt key" that is stable across iterations (seed text or adversarial prompt text)
def prompt_key(rec):
    # Prefer the original seed text if available; fallback to adversarial prompt
    return rec.get("seed") or rec.get("adversarial_prompt") or rec.get("prompt") or "?"

# Get the first token (very rough) by splitting at whitespace or punctuation
def first_token(text: str) -> str:
    if not text:
        return ""
    # lower and simple split on space; you can improve later (BPE-based) if needed
    return text.strip().split()[0].lower()

# Compute F_p and F_t over a whole run folder
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir)

    # We will build a dict: prompt_key -> list of (iter_idx, toxicity_level, response_text)
    history = collections.defaultdict(list)
    # Also collect per-iteration first-token histograms for SAFE vs RISKY
    first_tok_safe = collections.defaultdict(lambda: collections.Counter())
    first_tok_risk = collections.defaultdict(lambda: collections.Counter())

    iters = sorted([p for p in run_dir.glob("iter_*") if p.is_dir()])
    for iterd in iters:
        i_num = int(iterd.name.split("_")[-1])  # turn "iter_0005" into 5
        path = iterd / "interactions.jsonl"
        if not path.exists():
            continue
        for rec in iter_jsonl(path):
            pk = prompt_key(rec)
            lvl = to_level(rec)
            rsp = rec.get("response","")
            history[pk].append((i_num, lvl, rsp))
            # Mark SAFE vs RISKY with a simple rule: levels 0-1 safe-ish, 2-5 risky-ish (tweak to taste)
            bucket = first_tok_safe if lvl <= 1 else first_tok_risk
            tok = first_token(rsp)
            if tok:
                bucket[i_num][tok] += 1

    # Compute F_p: sum of absolute changes in level for each prompt
    rows = []
    for pk, seq in history.items():
        seq = sorted(seq, key=lambda x: x[0])
        fp = 0
        last = None
        for (i, lvl, _) in seq:
            if last is not None:
                fp += abs(lvl - last)
            last = lvl
        rows.append({"prompt_key": pk, "F_p": fp, "seen": len(seq)})

    dfp = pd.DataFrame(rows).sort_values(["F_p","seen"], ascending=[False,False])

    # Compute F_t (proxy): for each iteration, compare top-N first tokens of SAFE vs RISKY by Jaccard distance (0..1)
    # ELI-10: If SAFE and RISKY start with similar words, distance small; if very different, distance big.
    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        return 1.0 - (len(a & b) / max(1, len(a | b)))
    fts = []
    for i in sorted({int(p.name.split("_")[-1]) for p in iters}):
        topS = {w for (w,c) in first_tok_safe[i].most_common(10)}  # top 10 tokens
        topR = {w for (w,c) in first_tok_risk[i].most_common(10)}
        ft = jaccard(topS, topR)
        fts.append({"iteration": i, "F_t": ft})

    dft = pd.DataFrame(fts).sort_values("iteration")

    # Save CSVs and a small HTML summary
    dfp.to_csv(run_dir/"fluctuation_prompts.csv", index=False)
    dft.to_csv(run_dir/"fluctuation_tokens.csv", index=False)

    # Make a tiny HTML: top 20 fluctuating prompts + F_t line
    import plotly.express as px
    topN = dfp.head(20).copy()
    topN["rank"] = range(1, len(topN)+1)
    tbl = topN[["rank","prompt_key","F_p","seen"]].to_html(index=False, escape=True)
    fig = px.line(dft, x="iteration", y="F_t", markers=True, title="Token-level fluctuation F_t (proxy, higher = more different)")
    html_ft = fig.to_html(full_html=False, include_plotlyjs="cdn")
    html = f"<h2>Fluctuation</h2><h3>Top fluctuating prompts (F_p)</h3>{tbl}<h3>Token-level fluctuation (F_t)</h3>{html_ft}"
    (run_dir/"fluctuation.html").write_text(html, encoding="utf-8")

    print("Wrote:", run_dir/"fluctuation_prompts.csv", run_dir/"fluctuation_tokens.csv", run_dir/"fluctuation.html")

if __name__ == "__main__":
    main()
