from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
#!/usr/bin/env python
import argparse, pathlib, json, os
import pandas as pd, plotly.express as px, plotly.graph_objects as go
from src.metrics.summary import build_summary_html

def _mask(s, n=80):
    s = (s or "").replace("\n"," ")
    return s[:n] + ("..." if len(s)>n else "")

def _pcty(fig):
    fig.update_layout(
        yaxis=dict(range=[0,1], tickformat=".0%", gridcolor="rgba(0,0,0,0.1)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
        font=dict(size=16),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

def _build_sankey(df, toxicity_thr: float):
    df = df.copy()
    df["outcome"] = df["toxicity"].apply(lambda x: "Risky (≥ thr)" if float(x) >= toxicity_thr else "Safe (< thr)")
    risks = sorted(df["risk"].unique().tolist())
    outcomes = ["Safe (< thr)", "Risky (≥ thr)"]
    nodes = risks + outcomes
    idx = {n:i for i,n in enumerate(nodes)}
    grp = df.groupby(["risk","outcome"]).size().reset_index(name="value")
    links = dict(
        source=[idx[r] for r in grp["risk"]],
        target=[idx[o] for o in grp["outcome"]],
        value=grp["value"].tolist()
    )
    sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, pad=18, thickness=18),
        link=links
    )])
    sankey.update_layout(title_text="From Risk Type ➜ Outcome (latest iteration)", font_size=14, height=480, margin=dict(l=10,r=10,t=50,b=10))
    return sankey.to_html(full_html=False, include_plotlyjs=False)

def build_report(run_dir: pathlib.Path):
    if run_dir.is_file():
        run_dir = run_dir.parent
    assert run_dir.is_dir(), f"Not a run dir: {run_dir}"

    # Executive Summary card
    summary_html = build_summary_html(run_dir)

    # Train metrics
    mpath = run_dir / "metrics.csv"
    dfm = pd.read_csv(mpath) if mpath.exists() else pd.DataFrame(columns=["iteration","ASR","avg_toxicity","div_self_bleu","cap_acc","n","seconds"])

    # Validation metrics
    vpath = run_dir / "validation_metrics.csv"
    dfv = pd.read_csv(vpath) if vpath.exists() else pd.DataFrame(columns=["iteration","ASR","avg_toxicity","n"])

    # Chart 1: Train ASR & Toxicity
    if len(dfm):
        fig1 = px.line(dfm, x="iteration", y=["ASR","avg_toxicity"], markers=True, title="TRAIN — ASR & Avg Toxicity")
        fig1.update_traces(hovertemplate="%{y:.2%} at iter %{x}")
        _pcty(fig1)
        html1 = fig1.to_html(full_html=False, include_plotlyjs="cdn")
        latest = int(dfm["iteration"].max())
    else:
        html1 = "<p>No metrics.csv yet.</p>"
        latest = 1

    # Chart 2: Validation ASR & Toxicity
    if len(dfv):
        figv = px.line(dfv, x="iteration", y=["ASR","avg_toxicity"], markers=True, title="VALIDATION — ASR & Avg Toxicity")
        figv.update_traces(hovertemplate="%{y:.2%} at iter %{x}")
        _pcty(figv)
        htmlv = figv.to_html(full_html=False, include_plotlyjs=False)
    else:
        htmlv = "<p>Validation disabled or no data yet.</p>"

    # Capability + Diversity (train)
    if "cap_acc" in dfm.columns and dfm["cap_acc"].notna().any():
        fig_cap = px.line(dfm, x="iteration", y=["cap_acc"], markers=True, title="Capability Accuracy (mini-quiz)")
        fig_cap.update_traces(hovertemplate="%{y:.2%} at iter %{x}")
        _pcty(fig_cap)
        html_cap = fig_cap.to_html(full_html=False, include_plotlyjs=False)
    else:
        html_cap = "<p>No capability data.</p>"

    if "div_self_bleu" in dfm.columns and dfm["div_self_bleu"].notna().any():
        fig_div = px.line(dfm, x="iteration", y=["div_self_bleu"], markers=True, title="Diversity (Self-BLEU, lower=better)")
        fig_div.update_layout(yaxis=dict(range=[0,100], ticksuffix=""), font=dict(size=16), height=500)
        html_div = fig_div.to_html(full_html=False, include_plotlyjs=False)
    else:
        html_div = "<p>No diversity data.</p>"

    # Per-risk heatmaps — train and validation (if present)
    rtrain = run_dir / "risk_metrics.csv"
    if rtrain.exists():
        dfr = pd.read_csv(rtrain)
        hm = dfr.pivot(index="risk", columns="iteration", values="risky_rate").fillna(0.0)
        fig_hm = px.imshow(hm*100.0, aspect="auto", text_auto=".0f", color_continuous_scale="RdYlGn_r", title="TRAIN — Per-Risk Risky Rate (%)")
        fig_hm.update_layout(height=520, font=dict(size=16), coloraxis_colorbar=dict(title="% risky"))
        html_hm_train = fig_hm.to_html(full_html=False, include_plotlyjs=False)
    else:
        html_hm_train = "<p>No per-risk train data.</p>"

    rval = run_dir / "validation_risk_metrics.csv"
    if rval.exists():
        dfvr = pd.read_csv(rval)
        hm2 = dfvr.pivot(index="risk", columns="iteration", values="risky_rate").fillna(0.0)
        fig_hm2 = px.imshow(hm2*100.0, aspect="auto", text_auto=".0f", color_continuous_scale="RdYlGn_r", title="VALIDATION — Per-Risk Risky Rate (%)")
        fig_hm2.update_layout(height=520, font=dict(size=16), coloraxis_colorbar=dict(title="% risky"))
        html_hm_val = fig_hm2.to_html(full_html=False, include_plotlyjs=False)
    else:
        html_hm_val = "<p>No per-risk validation data.</p>"


    # ----- Cost section -----
    # ----- Cost section -----
    import pandas as _pd
    import plotly.express as _px

    cost_path = run_dir / "costs.csv"
    if cost_path.exists():
        _dfc = _pd.read_csv(cost_path)

        # headline: total + optional cost per risky
        try:
            # Try to read TOTAL row if present
            total_row = _dfc.tail(1).iloc[0]
            total_cost = float(total_row["usd_cost"])
        except Exception:
            total_cost = float(_dfc["usd_cost"].sum())

        _mpath = run_dir / "metrics.csv"
        risky = 0
        if _mpath.exists():
            _dfm = _pd.read_csv(_mpath)
            risky = int((_dfm["ASR"] * _dfm["n"]).sum())

        if risky > 0:
            html_cost_headline = f"<p><b>Total cost:</b> ${total_cost:.4f} &nbsp; | &nbsp; <b>Cost per risky:</b> ${total_cost/risky:.4f}</p>"
        else:
            html_cost_headline = f"<p><b>Total cost:</b> ${total_cost:.4f}</p>"

        # bar: cost per numeric iteration only
        _dfc_iter = _dfc[_dfc["iteration"].apply(lambda x: str(x).isdigit())]
        if len(_dfc_iter):
            _figc = _px.bar(_dfc_iter, x="iteration", y="usd_cost", text="usd_cost", title="Cost per iteration (USD)")
            _figc.update_traces(texttemplate="$%{y:.4f}")
            _figc.update_layout(yaxis=dict(ticksuffix=""), xaxis=dict(dtick=1), height=420)
            html_cost_bar = _figc.to_html(full_html=False, include_plotlyjs=False)
        else:
            html_cost_bar = "<p>No iteration costs.</p>"

        html_cost_table = _dfc.to_html(index=False)
    else:
        html_cost_headline = "<p>No cost data yet. Run scripts/costs.py for this run.</p>"
        html_cost_bar = ""
        html_cost_table = ""
    # Latest iter details
    # Latest iter details

    iterd = run_dir / f"iter_{latest:04d}"
    if not iterd.exists():
        iters = sorted([p for p in run_dir.glob("iter_*") if p.is_dir()])
        if iters: iterd = iters[-1]
    rows = [json.loads(l) for l in open(iterd/'interactions.jsonl','r',encoding='utf-8')]
    dfi = pd.DataFrame(rows)

    st = json.load(open(run_dir/"status.json","r",encoding="utf-8"))
    thr = float(st.get("toxicity_threshold", 0.5))
    mdl_red = st.get("models",{}).get("red","?")
    mdl_tgt = st.get("models",{}).get("target","?")

    fig2 = px.box(dfi, x="risk", y="toxicity", points="all", title=f"TRAIN — Toxicity by Risk ({iterd.name})")
    fig2.update_traces(hovertemplate="%{y:.2%}")
    fig2.update_layout(yaxis=dict(range=[0,1], tickformat=".0%", gridcolor="rgba(0,0,0,0.1)"), xaxis=dict(tickangle=-25, gridcolor="rgba(0,0,0,0.05)"), font=dict(size=16), height=520, margin=dict(l=10,r=10,t=60,b=80))
    html2 = fig2.to_html(full_html=False, include_plotlyjs=False)

    sankey_html = _build_sankey(dfi, thr)

    dfi["seed"] = dfi["seed"].apply(_mask)
    dfi["adversarial_prompt"] = dfi["adversarial_prompt"].apply(_mask)
    dfi["response"] = dfi["response"].apply(_mask)
    table_html = dfi.sort_values("toxicity", ascending=True)[["risk","seed","adversarial_prompt","response","toxicity","success"]].to_html(index=False, escape=False)

    header_info = f"<p><b>Red:</b> {mdl_red}<br><b>Target:</b> {mdl_tgt}<br><b>Toxicity Threshold:</b> {thr}<br><b>Run:</b> {run_dir.name}</p>"

    html = f'''
    <html><head><meta charset='utf-8'><title>Run Report</title></head>
    <body style="font-family: Arial, sans-serif; max-width: 1100px; margin: auto;">
  <div style="margin:10px 0;padding:10px;border:1px solid #eee;border-radius:10px;">
    <h2>Per-Iteration Sankey Index</h2>
    <p><a href="sankey_index.html">Open list of all Sankey charts</a></p>
  </div>

  <div style="margin:10px 0;padding:10px;border:1px dashed #bbb;border-radius:10px;">
    <h2>Full Data (Unmasked)</h2>
    <p><a href="full_data.html">Open all prompts & responses (no masking)</a> &nbsp; | &nbsp;        <a href="full_data.csv">Download CSV</a></p>
  </div>

  <div style="margin:10px 0;padding:10px;border:1px solid #eee;border-radius:10px;">
    <h2>Adversarial Flow & Extras</h2>
    <p><a href="overview.html">Training Overview (ASR vs Capability + Diversity)</a></p>
    <p><a href="fluctuation.html">Fluctuation Summary (Fₚ & Fₜ)</a></p>
    <p><a href="validation_leaderboard.html">Validation Leaderboard</a> &nbsp; | &nbsp; <a href="validation_flow.html">Validation Flow</a></p>
    <p>Per-iteration Sankey files live under each <code>iter_XXXX/sankey.html</code>.</p>
  </div>
      {summary_html}
      <h1>Run Report</h1>
      {header_info}
      <h2>TRAIN — ASR & Toxicity</h2>
      {html1}
      <h2>VALIDATION — ASR & Toxicity</h2>
      {htmlv}
      <h2>Capability (mini-quiz)</h2>
      {html_cap}
      <h2>TRAIN — Diversity (Self-BLEU, lower=better)</h2>
      {html_div}
      <h2>TRAIN — Per-Risk Risky Rate</h2>
      {html_hm_train}
      <h2>VALIDATION — Per-Risk Risky Rate</h2>
      {html_hm_val}
      <h2>TRAIN — Toxicity by Risk ({iterd.name})</h2>
      {html2}
      <h2>TRAIN — Risk ➜ Outcome (Sankey)</h2>
      {sankey_html}
      <h2>Top Samples (masked)</h2>
      {table_html}
    </body></html>
    '''
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print("Wrote:", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True, help="Path to a run folder")
    args = ap.parse_args()
    build_report(pathlib.Path(args.run_dir))

if __name__ == "__main__":
    main()
