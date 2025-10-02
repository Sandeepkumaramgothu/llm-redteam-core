#!/usr/bin/env python
from __future__ import annotations
import argparse, pathlib, json, pandas as pd, plotly.express as px

def load_one_run(run_dir: pathlib.Path):
    st = json.load(open(run_dir / "status.json", "r", encoding="utf-8"))
    m = (run_dir / "metrics.csv")
    dfm = pd.read_csv(m) if m.exists() else pd.DataFrame(columns=["iteration","ASR","avg_toxicity","n","seconds"])
    last = dfm.sort_values("iteration").tail(1)
    tgt = st.get("models",{}).get("target","?")
    red = st.get("models",{}).get("red","?")
    # Shorten the run name a bit for axis readability
    short = run_dir.name.replace("baseline_", "")
    name = f"{short} — target={tgt.split('/')[-1]}"
    out = dict(name=name, run_dir=run_dir.as_posix(), target=tgt, red=red)
    if len(last):
        out.update(dict(
            final_iter=int(last["iteration"].values[0]),
            final_ASR=float(last["ASR"].values[0]),
            final_avg_toxicity=float(last["avg_toxicity"].values[0]),
            n=int(last["n"].values[0]),
            seconds=float(last["seconds"].values[0]),
        ))
    return out, dfm

def build_compare(root: pathlib.Path, out_html: pathlib.Path):
    runs = sorted([p for p in root.glob("*") if p.is_dir()])
    rows, timelines = [], []
    for r in runs:
        try:
            row, dfm = load_one_run(r)
            rows.append(row)
            if len(dfm):
                dfm = dfm.copy()
                dfm["run"] = row["name"]
                timelines.append(dfm)
        except Exception as e:
            print("Skip", r, "->", e)

    if not rows:
        raise RuntimeError("No valid runs found")

    dfr = pd.DataFrame(rows)
    # Handle missing columns gracefully
    for col in ["final_ASR", "final_avg_toxicity", "final_iter", "n", "seconds"]:
        if col not in dfr.columns: dfr[col] = None

    dfr = dfr.sort_values("final_ASR", ascending=False)
    html_table = dfr[["name","final_iter","final_ASR","final_avg_toxicity","n","seconds"]].to_html(index=False)

    # Bar: final ASR (0–100%) with labels
    fig_asr = px.bar(dfr, x="name", y="final_ASR", title="Final ASR by Run", text="final_ASR")
    fig_asr.update_traces(texttemplate="%{text:.1%}", hovertemplate="%{y:.2%}")
    fig_asr.update_layout(
        yaxis=dict(range=[0,1], tickformat=".0%", gridcolor="rgba(0,0,0,0.1)"),
        xaxis=dict(tickangle=-25, categoryorder="array", categoryarray=list(dfr["name"])),
        font=dict(size=16),
        height=520,
        margin=dict(l=10,r=10,t=60,b=120),
    )
    html_asr = fig_asr.to_html(full_html=False, include_plotlyjs="cdn")

    # Lines: ASR per iteration for each run (0–100%)
    if timelines:
        dft = pd.concat(timelines, ignore_index=True)
        fig_line = px.line(
            dft, x="iteration", y="ASR", color="run", markers=True,
            title="ASR over Iterations (all runs)"
        )
        fig_line.update_layout(
            yaxis=dict(range=[0,1], tickformat=".0%", gridcolor="rgba(0,0,0,0.1)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
            font=dict(size=16),
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        fig_line.update_traces(hovertemplate="Run=%{fullData.name}<br>Iter %{x} • ASR %{y:.2%}")
        html_line = fig_line.to_html(full_html=False, include_plotlyjs=False)
    else:
        html_line = "<p>No timelines available.</p>"

    html = f'''
<html><head><meta charset="utf-8"><title>Runs Comparison</title></head>
<body style="font-family: Arial, sans-serif; max-width: 1200px; margin:auto;">
  <h1>Runs Comparison</h1>
  <h2>Final Metrics Table</h2>
  {html_table}
  <h2>Final ASR (higher is worse in safety tests)</h2>
  {html_asr}
  <h2>ASR Over Iterations</h2>
  {html_line}
  <p style="color:#666;font-size:14px;margin-top:20px;">Tip: ASR = fraction of prompts flagged as risky by the toxicity threshold.</p>
</body></html>
'''
    out_html.write_text(html, encoding="utf-8")
    print("Wrote:", out_html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, required=True, help="folder that contains multiple run_* directories")
    ap.add_argument("--out", type=str, default="compare.html")
    args = ap.parse_args()
    build_compare(pathlib.Path(args.runs_root), pathlib.Path(args.out))

if __name__ == "__main__":
    main()
