#!/usr/bin/env python
# Line-by-line comments explain each step like you're 10:
from __future__ import annotations  # allows future Python typing goodies

import sys, pathlib  # system & path helpers
from typing import Dict, List, Tuple  # for clear type hints
import json, pandas as pd  # JSON reading and DataFrames for counting
import plotly.graph_objects as go  # Sankey diagram
import plotly.express as px        # Line charts for overview

# Turn a string path into a Path object
def _p(x: str) -> pathlib.Path:
    return pathlib.Path(x)

# Read a single interactions.jsonl (one line = one record) and return a DataFrame with columns we need
def load_interactions(jl_path: pathlib.Path) -> pd.DataFrame:
    rows = []  # will fill with dictionaries
    with open(jl_path, "r", encoding="utf-8") as f:  # open the file safely
        for line in f:  # go through each line
            try:
                r = json.loads(line)  # parse JSON
                # we keep only the simple fields needed for Sankey and metrics
                rows.append({
                    "risk": r.get("risk") or r.get("risk_type") or "Unknown",  # risk type name
                    "toxicity": r.get("toxicity_level") if r.get("toxicity_level") is not None else r.get("toxicity", None),  # level 0..5 if present
                    "score": r.get("score", None)  # optional numeric toxicity score
                })
            except Exception:
                # If a line is bad, skip it to be robust
                continue
    # Make a DataFrame even if rows is empty (won't crash downstream)
    df = pd.DataFrame(rows)
    # Normalize toxicity to 0..5 bins when missing: simple rule-of-thumb
    if "toxicity" in df and df["toxicity"].isna().any():
        # If numeric score exists, bin it; else fill with 0
        if "score" in df and df["score"].notna().any():
            # Map score [-10..+10] to bins 0..5 (paper uses 6 levels)  :contentReference[oaicite:9]{index=9}
            # We'll clamp then cut into 6 equal bins.
            s = df["score"].clip(-10, 10)
            bins = pd.cut(s, bins=6, labels=[0,1,2,3,4,5])
            df["toxicity"] = bins.astype(int)
        else:
            df["toxicity"] = df["toxicity"].fillna(0)
    return df

# Build a Sankey from counts of risk→toxicity for one iteration
def sankey_html(df: pd.DataFrame, title: str) -> str:
    # Count pairs risk→toxicity
    # Guard: drop rows with missing risk or toxicity
    df2 = df.dropna(subset=["risk", "toxicity"]).copy()
    # Force toxicity to int in [0..5] if possible
    try:
        df2["toxicity"] = df2["toxicity"].astype(int).clip(lower=0, upper=5)
    except Exception:
        # fallback: coerce
        df2["toxicity"] = pd.to_numeric(df2["toxicity"], errors="coerce").fillna(0).astype(int).clip(0,5)
    ct = df2.groupby(["risk","toxicity"]).size().reset_index(name="count")
    # Build node list: risks + toxicity levels (as strings)
    risks = sorted(ct["risk"].dropna().astype(str).unique().tolist())
    # Ensure all toxicity levels 0-5 are included in labels
    tox_labels_full = [f"Toxicity {i}" for i in range(6)]
    labels = risks + tox_labels_full
    # Map names to indices
    idx = {name: i for i, name in enumerate(labels)}
    # Build link arrays: source index, target index, value
    src, tgt, val = [], [], []
    for _, row in ct.iterrows():
        r = row["risk"]
        t_label = f"Toxicity {int(row['toxicity'])}" # Use the consistent label format
        if t_label not in labels: # Should not happen with full labels, but good check
            continue
        src.append(idx[r])  # risk source
        tgt.append(idx[t_label])  # tox target
        val.append(int(row["count"]))  # how many
    # Create sankey figure (kept simple, readable)
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=16),
        link=dict(source=src, target=tgt, value=val)
    )])
    fig.update_layout(title_text=title, font_size=12)
    # Return HTML string (no outer <html> so we can embed)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

# Build an Overview page: dual-axis ASR vs capability + diversity trend
def overview_html(metrics_csv: pathlib.Path, title: str="Overview: ASR vs Capability + Diversity") -> str:
    # Read metrics.csv which should have columns: iteration, ASR, cap_acc, div_self_bleu, etc.
    if not metrics_csv.exists():
        return "<p>No metrics.csv yet.</p>"
    df = pd.read_csv(metrics_csv)
    if not len(df):
        return "<p>No metrics rows yet.</p>"
    # Create two y-axes figure: ASR (left, lower is better) and Capability (right, higher is better)  :contentReference[oaicite:10]{index=10}
    fig = go.Figure()
    # Left axis: ASR line
    fig.add_trace(go.Scatter(x=df["iteration"], y=df["ASR"], name="ASR (↓)", mode="lines+markers"))
    # Right axis: Capability line if present
    if "cap_acc" in df.columns:
        fig.add_trace(go.Scatter(x=df["iteration"], y=df["cap_acc"], name="Capability (↑)", mode="lines+markers", yaxis="y2"))
    # Layout with secondary y-axis
    fig.update_layout(
        title=title,
        xaxis=dict(title="Iteration"),
        yaxis=dict(title="ASR", rangemode="tozero"),
        yaxis2=dict(title="Capability", overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h")
    )
    trend = fig.to_html(full_html=False, include_plotlyjs=False)
    # Diversity (Self-BLEU) separate simple line
    if "div_self_bleu" in df.columns:
        fig2 = px.line(df, x="iteration", y="div_self_bleu", markers=True, title="Diversity (Self-BLEU) — lower is better")
        div_html = fig2.to_html(full_html=False, include_plotlyjs=False)
    else:
        div_html = "<p>No diversity column found.</p>"
    # Stitch together
    return f"<h2>Training Overview</h2>{trend}<h3>Diversity Trend</h3>{div_html}"

# Main entry: create per-iteration Sankey HTML files and one overview.html inside the run folder
def main():
    import argparse  # parse CLI flags
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True, help="path to runs/<run_id>")
    args = ap.parse_args()
    run_dir = _p(args.run_dir)  # Path to the run
    # For each iter_* folder, read interactions and make a sankey_XX.html
    iters = sorted([p for p in run_dir.glob("iter_*") if p.is_dir()])
    for iterd in iters:
        jl = iterd / "interactions.jsonl"  # source file for that iteration
        if not jl.exists():  # skip if missing
            continue
        try: # Add try-except block here
            df = load_interactions(jl)  # load rows
            h = sankey_html(df, f"Sankey — {iterd.name}")  # create HTML string
            (iterd / "sankey.html").write_text(h, encoding="utf-8")  # save near the data
        except Exception as e: # Catch exceptions and print
            print(f"Error processing {jl}: {e}")
            continue # Skip to the next iteration if there's an error
    # Build the overview for training metrics
    mh = overview_html(run_dir / "metrics.csv")
    (run_dir / "overview.html").write_text(mh, encoding="utf-8")  # drop at run root
    print("Wrote per-iteration sankey.html and overview.html.")

if __name__ == "__main__":
    main()
