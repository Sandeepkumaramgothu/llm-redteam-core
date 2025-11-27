"""
Minimalist, Presentation-Grade Visualization Suite (Core 5)
- Iteration timeline
- Risk per iteration
- Jailbreak leaderboard
- ASR by risk type
- Refusal match rate
+ CSV Export Utilities (Required by PostRun)
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import html
import json

def safe_print(*args, **kwargs):
    """Print with robust error handling."""
    try:
        print(*args, **kwargs)
    except Exception:
        pass

def create_all_visualizations(df: pd.DataFrame, output_dir: Path) -> dict:
    """The essential analytic visuals for a clear red-teaming or alignment paper."""
    charts = {}

    # (1) Agreement calculation for all figures
    if (
        'response' in df.columns and
        'ft_response' in df.columns and
        'agrees_with_refusal' not in df.columns
    ):
        df['agrees_with_refusal'] = (
            df['response'].map(str).str.lower().str.strip()
            == df['ft_response'].map(str).str.lower().str.strip()
        )

    # (2) Timeline per iteration (ASR, severe, refusal, toxicity, reward)
    if 'iter' in df.columns and df['iter'].nunique() > 1:
        group = df.groupby('iter').agg(
            attack_success=('success', 'mean'),
            severe_rate=('risk_level', lambda x: ((x == 'L2') | (x == 'L3')).mean()),
            avg_toxicity=('toxicity', 'mean'),
            avg_reward=('reward', 'mean'),
            avg_refusal=('refusal', 'mean')
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=group['iter'], y=100*group['attack_success'], name='ASR (%)', line=dict(color='#e74c3c', width=3)))
        fig.add_trace(go.Scatter(x=group['iter'], y=100*group['severe_rate'], name='Severe Risk (%)', line=dict(color='#c0392b', dash='dash')))
        fig.add_trace(go.Scatter(x=group['iter'], y=group['avg_toxicity'], name='Avg Toxicity', line=dict(color='#f39c12', dash='dot')))
        fig.add_trace(go.Scatter(x=group['iter'], y=100*group['avg_refusal'], name='Refusal Rate (%)', line=dict(color='#16a085', dash='dot')))
        fig.add_trace(go.Scatter(x=group['iter'], y=group['avg_reward'], name='Avg Reward', line=dict(color='#3498db', dash='dashdot')))
        fig.update_layout(
            title="Timeline: Attack, Harm, Refusal (per Iteration)",
            xaxis_title="Iteration", yaxis_title="Score / %",
            legend_title="Metric", height=450
        )
        charts['timeline'] = fig

    # (3) Risk level per iteration (stacked bar)
    if 'risk_level' in df.columns and 'iter' in df.columns:
        order = ['L0','L1','L2','L3']
        levels = [col for col in order if col in df['risk_level'].unique()]
        risk_counts = df.pivot_table(index='iter', columns='risk_level', values='seed', aggfunc='count', fill_value=0)
        fig = go.Figure()
        for level in levels:
            fig.add_trace(go.Bar(x=risk_counts.index, y=risk_counts[level], name=level))
        fig.update_layout(barmode='stack', title="Risk Output Distribution per Iteration",
                          xaxis_title="Iteration", yaxis_title="Count", height=400)
        charts['risk_iter'] = fig

    # (4) Leaderboard Table (top jailbreaks)
    if 'success' in df.columns and 'risk_level' in df.columns:
        # Sort by severity (L3 > L2 > L1) then reward
        df_lb = df[df['success']].sort_values(['risk_level', 'reward', 'toxicity'], ascending=[False,False,False]).head(8)
        cols = ['iter','seed','risk_type','response','risk_level','reward','toxicity']
        # Ensure columns exist
        cols = [c for c in cols if c in df.columns]

        if not df_lb.empty:
            fig = go.Figure(go.Table(
                header=dict(values=[c.replace("_"," ").title() for c in cols], fill_color='#34495e', font=dict(color='white')),
                cells=dict(values=[df_lb[c] for c in cols], fill_color='#ecf0f1')
            ))
            fig.update_layout(title="Leaderboard: Top Jailbreaks (Most Harmful Responses)", height=420)
            charts['leaderboard'] = fig

    # (5) ASR by risk type (bar)
    if 'success' in df.columns and 'risk_type' in df.columns:
        asr_by_risk = df.groupby('risk_type')['success'].mean().sort_values(ascending=False) * 100
        fig = px.bar(
            x=asr_by_risk.values,
            y=asr_by_risk.index,
            orientation="h",
            text=asr_by_risk.round(1),
            title="ASR (Unsafe Output %) by Risk Type",
            labels={'x':'% Unsafe', 'y':'Risk Type'},
            color=asr_by_risk.values,
            color_continuous_scale='Reds'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_title="ASR (%)", yaxis_title="Risk Type",
                          height=400, yaxis_autorange="reversed")
        charts['asr_by_risk'] = fig

    # (6) Refusal match pie
    if 'agrees_with_refusal' in df.columns:
        match = df['agrees_with_refusal'].sum()
        mismatch = (~df['agrees_with_refusal']).sum() if df['agrees_with_refusal'].dtype==bool else (df['agrees_with_refusal']==False).sum()
        fig = px.pie(
            values=[match, mismatch],
            names=['Refusal Matched','Override/Jailbreak'],
            title="Fraction of Outputs Matched to Refusal (after Jailbreak)",
            color_discrete_sequence=["#16a085","#e74c3c"]
        )
        charts['refusal_match_pie'] = fig

    safe_print(f"✓ Generated {len(charts)} charts: {list(charts.keys())}")
    return charts

def generate_html_report(df, metrics, charts, output_path):
    """Generate simple HTML dashboard, compact mode."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html><html><head><meta charset='UTF-8'>")
        f.write('<title>AdversaFlow MiniReport</title>')
        f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body>')
        f.write("<h1 style='font-family:Arial,sans-serif;color:#764ba2'>AdversaFlow Paper Dashboard</h1>")
        f.write("<h2>Summary</h2>")
        if metrics:
            f.write("<pre style='background:#f3f4f6;padding:16px;border-radius:8px;'>")
            f.write(json.dumps(metrics, indent=2))
            f.write("</pre>")
        f.write("<h2>Visualizations</h2>")
        for name, fig in charts.items():
            f.write(f"<h3>{name.replace('_',' ').title()}</h3>")
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("</body></html>")
    safe_print(f"✓ HTML dashboard saved: {output_path}")

def generate_interactive_data_table(df: pd.DataFrame, output_path: Path):
    """Generate a basic HTML table with all columns (for data exploration)."""
    safe_print(f"\n📋 Generating interactive data table...")
    display_columns = list(df.columns)
    table_rows = ""
    for idx, row in df.iterrows():
        table_rows += "<tr>" + "".join(
            f"<td style='max-width:300px;overflow:auto'>{html.escape(str(row[col]))[:400]}</td>" for col in display_columns
        ) + "</tr>\n"
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AdversaFlow Data Table</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
</head>
<body>
    <h1 style='font-family:Arial,sans-serif;color:#764ba2'>AdversaFlow Data Explorer</h1>
    <table id="resultsTable" class="display" style="width:100%">
        <thead><tr>
            {''.join(f'<th>{html.escape(col)}</th>' for col in display_columns)}
        </tr></thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    <script>
    $(document).ready(function() {{
        $('#resultsTable').DataTable({{
            pageLength: 25
        }});
    }});
    </script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding='utf-8')
    safe_print(f"✓ Data table saved: {output_path}")

def save_iteration_summary_csv(df: pd.DataFrame, output_path: Path):
    """Write per-iteration summary Table to CSV."""
    if 'iter' in df.columns:
        iter_summary = df.groupby('iter').agg(
            n_seeds=('seed', 'nunique') if 'seed' in df.columns else ('reward','count'),
            asr=('success', 'mean'),
            avg_toxicity=('toxicity','mean'),
            avg_reward=('reward','mean'),
            nL3=('risk_level', lambda x: (x=='L3').sum()),
            nL2=('risk_level', lambda x: (x=='L2').sum()),
            nL1=('risk_level', lambda x: (x=='L1').sum()),
            nL0=('risk_level', lambda x: (x=='L0').sum())
        ).reset_index()

        # Calculate percentage of severe risks (L2+L3)
        total_risk = iter_summary['nL3'] + iter_summary['nL2'] + iter_summary['nL1'] + iter_summary['nL0']
        # Avoid division by zero
        total_risk = total_risk.replace(0, 1)
        iter_summary['severe_risk_pct'] = (iter_summary['nL3'] + iter_summary['nL2']) / total_risk * 100
        iter_summary['asr'] = iter_summary['asr'] * 100

        iter_summary.to_csv(output_path, index=False)
        safe_print(f"✓ Wrote iteration summary CSV: {output_path}")
    else:
        safe_print("WARNING: No 'iter' column in result DataFrame, cannot write iteration summary.")

def save_global_summary_csv(metrics: dict, output_path: Path):
    """Write single-row global summary."""
    # Flatten key metrics
    row = {
        'attack_success_rate': metrics.get('attack_success_rate', 0),
        'severe_risk_rate': metrics.get('severe_risk_rate', 0),
        'avg_toxicity': metrics.get('toxicity', {}).get('avg', 0),
        'avg_reward': metrics.get('reward', {}).get('avg', 0),
        'refusal_rate': metrics.get('refusal_rate', 0),
        'total_samples': metrics.get('total_samples', 0),
        'successful_attacks': metrics.get('successful_attacks', 0)
    }
    df = pd.DataFrame([row])
    df.to_csv(output_path, index=False)
    safe_print(f"✓ Wrote global summary CSV: {output_path}")
