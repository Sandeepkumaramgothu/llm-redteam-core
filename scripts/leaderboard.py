#!/usr/bin/env python
from __future__ import annotations
import argparse, pathlib, json, pandas as pd, plotly.express as px

def summarize_run(run_dir: pathlib.Path) -> dict:
    st = json.load(open(run_dir/"status.json","r",encoding="utf-8"))
    m = run_dir/"metrics.csv"
    vm = run_dir/"validation_metrics.csv"
    dfm = pd.read_csv(m) if m.exists() else pd.DataFrame()
    dfv = pd.read_csv(vm) if vm.exists() else pd.DataFrame()
    lastm = dfm.sort_values("iteration").tail(1) if len(dfm) else pd.DataFrame()
    lastv = dfv.sort_values("iteration").tail(1) if len(dfv) else pd.DataFrame()
    def get(row, col): 
        return (float(row.iloc[0][col]) if (len(row) and col in row.columns) else None)
    # attach cost if available
    import pandas as _pd
    cost_total = None
    cost_per_risky = None
    cpath = run_dir/"costs.csv"
    if cpath.exists():
        _dfc = _pd.read_csv(cpath)
        try:
            cost_total = float(_dfc.tail(1)["usd_cost"].values[0])
        except Exception:
            cost_total = float(_dfc["usd_cost"].sum())
        mpath = run_dir/"metrics.csv"
        if mpath.exists():
            _dfm = _pd.read_csv(mpath)
            risky = int((_dfm["ASR"] * _dfm["n"]).sum())
            if risky > 0:
                cost_per_risky = cost_total / risky

    return {
        "run": run_dir.name,
        "red_model": st.get("models",{}).get("red","?"),
        "target_model": st.get("models",{}).get("target","?"),
        "train_ASR": get(lastm, "ASR"),
        "train_avg_toxicity": get(lastm, "avg_toxicity"),
        "cap_acc": get(lastm, "cap_acc"),
        "div_self_bleu": get(lastm, "div_self_bleu"),
        "val_ASR": get(lastv, "ASR"),
        "val_avg_toxicity": get(lastv, "avg_toxicity"),
        "seconds_last_iter": get(lastm, "seconds"),
        "cost_total_usd": cost_total,
        "cost_per_risky_usd": cost_per_risky,
    }

def build_leaderboard(root: pathlib.Path, out_csv: pathlib.Path, out_html: pathlib.Path):
    runs = sorted([p for p in root.glob("*") if p.is_dir() and (p/"status.json").exists()])
    if not runs:
        raise SystemExit("No run folders found in " + str(root))
    rows = [summarize_run(r) for r in runs]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    sortcol = "val_ASR" if df["val_ASR"].notna().any() else "train_ASR"
    df_sorted = df.sort_values(sortcol, ascending=True, na_position="last")
    metric_name = "Validation ASR" if sortcol=="val_ASR" else "Train ASR"

    fig = px.bar(df_sorted, x="run", y=sortcol, text=sortcol, title=f"Leaderboard — {metric_name} (lower is better)")
    fig.update_traces(texttemplate="%{text:.1%}", hovertemplate="%{y:.2%}")
    fig.update_layout(yaxis=dict(range=[0,1], tickformat=".0%"), xaxis=dict(tickangle=-25))

    html = f'''
    <html><head><meta charset='utf-8'><title>Leaderboard</title></head>
    <body style="font-family: Arial, sans-serif; max-width: 1200px; margin:auto;">
      <h1>Leaderboard</h1>
      <p>Sorted by {'Validation' if sortcol=='val_ASR' else 'Train'} ASR (lower is safer).</p>
      {fig.to_html(full_html=False, include_plotlyjs="cdn")}
      <h2>Details</h2>
      {df_sorted.to_html(index=False)}
    </body></html>
    '''
    out_html.write_text(html, encoding="utf-8")
    print("Wrote:", out_csv, "and", out_html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default="leaderboard.csv")
    ap.add_argument("--out-html", type=str, default="leaderboard.html")
    args = ap.parse_args()
    build_leaderboard(pathlib.Path(args.runs_root),
                      pathlib.Path(args.out_csv),
                      pathlib.Path(args.out_html))

if __name__ == "__main__":
    main()
