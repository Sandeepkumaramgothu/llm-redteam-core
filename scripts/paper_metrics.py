
#!/usr/bin/env python3
# Builds paper-style metrics and charts from a run folder.

import sys, os, json, pathlib
# ensure `src` can be imported when run as a script
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.metrics.compute import save_metrics, load_interactions_df

import pandas as pd
import plotly.express as px

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="run folder, e.g. runs/2025-..._openai-red_...")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Compute and write metrics.csv, by_risk.csv, metrics_summary.json
    summary = save_metrics(str(run_dir))

    # Also build simple HTML charts
    df = load_interactions_df(str(run_dir))

    out_dir = run_dir / "paper_metrics"
    out_dir.mkdir(exist_ok=True)

    # 1) Iteration metrics
    dfi = pd.read_csv(run_dir / "metrics.csv")
    fig_asr = px.line(dfi, x="iteration", y="ASR", markers=True, title="Attack Success Rate by Iteration")
    fig_tox = px.line(dfi, x="iteration", y="avg_toxicity_prob", markers=True, title="Avg Toxicity Probability by Iteration")

    # 2) Top risky categories
    if (run_dir / "by_risk.csv").exists():
        dfr = pd.read_csv(run_dir / "by_risk.csv")
        fig_risk = px.bar(dfr.head(15), x="risk", y="ASR", title="Top Risk Types by ASR")
        html_risk = fig_risk.to_html(full_html=False, include_plotlyjs=False)
    else:
        html_risk = "<p>No risk breakdown available.</p>"

    # 3) Save HTML report
    html = f'''
    <html><head><meta charset='utf-8'><title>Paper Metrics</title></head>
    <body style="font-family:Arial,sans-serif;max-width:1200px;margin:auto;">
      <h1>Paper Metrics</h1>
      <p><b>Total Rows:</b> {summary["rows"]} |
         <b>Iterations:</b> {summary["iterations"]} |
         <b>ASR overall:</b> {summary["ASR_overall"]:.3f} |
         <b>Avg toxicity prob overall:</b> {summary["avg_toxicity_prob_overall"]:.3f}</p>
      <h2>Attack Success Rate (per iteration)</h2>
      {fig_asr.to_html(full_html=False, include_plotlyjs="cdn")}
      <h2>Avg Toxicity Probability (per iteration)</h2>
      {fig_tox.to_html(full_html=False, include_plotlyjs=False)}
      <h2>Risk Breakdown</h2>
      {html_risk}
      <h2>Sample Rows</h2>
      <p>Tip: use interactions_pretty.html for expandable text.</p>
      <pre style="white-space:pre-wrap">{df.head(10).to_string()}</pre>
    </body></html>
    '''
    (out_dir / "paper_report.html").write_text(html, encoding="utf-8")
    print("Wrote:", out_dir / "paper_report.html")

if __name__ == "__main__":
    main()
