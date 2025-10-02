#!/usr/bin/env python
from __future__ import annotations
import argparse, pathlib, json, pandas as pd, plotly.express as px

def load_summary(run_dir: pathlib.Path):
    st = json.load(open(run_dir/"status.json","r",encoding="utf-8"))
    # final train metrics
    m = run_dir/"metrics.csv"
    dfm = pd.read_csv(m) if m.exists() else pd.DataFrame()
    final = dfm.sort_values("iteration").tail(1) if len(dfm) else pd.DataFrame()
    # final validation
    vm = run_dir/"validation_metrics.csv"
    dfv = pd.read_csv(vm) if vm.exists() else pd.DataFrame()
    vfinal = dfv.sort_values("iteration").tail(1) if len(dfv) else pd.DataFrame()
    # per-risk validation latest
    vr = run_dir/"validation_risk_metrics.csv"
    dvr = pd.read_csv(vr) if vr.exists() else pd.DataFrame()
    if len(dvr):
        last_it = int(dvr["iteration"].max())
        dvr = dvr[dvr["iteration"]==last_it].copy()
    return st, final, vfinal, dvr

def pct(x):
    return f"{x*100:.1f}%" if pd.notna(x) else "NA"

def build(baseline: pathlib.Path, tuned: pathlib.Path, out_html: pathlib.Path):
    b_st, b_fin, b_vfin, b_vrisk = load_summary(baseline)
    t_st, t_fin, t_vfin, t_vrisk = load_summary(tuned)

    def extract(fin):
        if len(fin)==0: return dict(ASR=None, avg_toxicity=None, cap_acc=None, div_self_bleu=None)
        row = fin.iloc[0]
        return dict(ASR=row.get("ASR",None), avg_toxicity=row.get("avg_toxicity",None),
                    cap_acc=row.get("cap_acc",None), div_self_bleu=row.get("div_self_bleu",None))

    b_train = extract(b_fin)
    t_train = extract(t_fin)
    b_val = extract(b_vfin)
    t_val = extract(t_vfin)

    # Summary table
    rows = []
    for label, b, t in [
        ("Train ASR (↓ better)", b_train["ASR"], t_train["ASR"]),
        ("Train Avg Toxicity (↓)", b_train["avg_toxicity"], t_train["avg_toxicity"]),
        ("Validation ASR (↓)", b_val["ASR"], t_val["ASR"]),
        ("Validation Avg Toxicity (↓)", b_val["avg_toxicity"], t_val["avg_toxicity"]),
        ("Capability Accuracy (↑)", b_train["cap_acc"], t_train["cap_acc"]),
        ("Diversity Self-BLEU (↓)", b_train["div_self_bleu"], t_train["div_self_bleu"]),
    ]:
        delta = (t - b) if (b is not None and t is not None) else None
        rows.append(dict(metric=label, baseline=b, tuned=t, delta=delta))
    dfsum = pd.DataFrame(rows)

    # Bars for % metrics (ASR/avg_tox/cap) — show % if in [0..1]
    show = dfsum.copy()
    for col in ["baseline","tuned","delta"]:
        show[col] = show[col].apply(lambda x: (f"{x*100:.1f}%" if pd.notna(x) and 0<=x<=1 else (f"{x:.1f}" if pd.notna(x) else "NA")))
    table_html = show.to_html(index=False)

    # Per-risk validation compare (if available)
    if len(b_vrisk) and len(t_vrisk):
        merged = pd.merge(b_vrisk[["risk","risky_rate"]].rename(columns={"risky_rate":"baseline"}),
                          t_vrisk[["risk","risky_rate"]].rename(columns={"risky_rate":"tuned"}),
                          on="risk", how="outer").fillna(0.0)
        merged["delta"] = merged["tuned"] - merged["baseline"]
        fig = px.bar(merged.sort_values("delta", ascending=False), x="risk", y=["baseline","tuned"],
                     barmode="group", title="Per-Risk Validation Risky Rate (↓ better)")
        fig.update_layout(yaxis=dict(range=[0,1], tickformat=".0%"))
        html_pr = fig.to_html(full_html=False, include_plotlyjs="cdn")
        # delta bar
        figd = px.bar(merged.sort_values("delta", ascending=False), x="risk", y="delta", title="Per-Risk Δ (tuned - baseline)")
        figd.update_layout(yaxis=dict(ticksuffix=""), height=420)
        figd.update_traces(hovertemplate="Δ %{y:.2%}")
        html_delta = figd.to_html(full_html=False, include_plotlyjs=False)
    else:
        html_pr = "<p>No per-risk validation data in one or both runs.</p>"
        html_delta = ""

    hdr = f"<p><b>Baseline:</b> {baseline.name}<br><b>Tuned:</b> {tuned.name}</p>"
    html = f'''
    <html><head><meta charset='utf-8'><title>Baseline vs Tuned</title></head>
    <body style="font-family: Arial, sans-serif; max-width: 1200px; margin:auto;">
      <h1>Baseline vs Tuned</h1>
      {hdr}
      <h2>Summary</h2>
      {table_html}
      <h2>Per-Risk Validation (last iteration)</h2>
      {html_pr}
      <h2>Per-Risk Δ</h2>
      {html_delta}
    </body></html>
    '''
    out_html.write_text(html, encoding="utf-8")
    print("Wrote:", out_html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, required=True)
    ap.add_argument("--tuned", type=str, required=True)
    ap.add_argument("--out", type=str, default="baseline_vs_tuned.html")
    args = ap.parse_args()
    build(pathlib.Path(args.baseline), pathlib.Path(args.tuned), pathlib.Path(args.out))

if __name__ == "__main__":
    main()
