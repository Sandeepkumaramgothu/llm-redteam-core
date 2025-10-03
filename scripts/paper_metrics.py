#!/usr/bin/env python
from __future__ import annotations
import argparse, pathlib, json, hashlib
import pandas as pd, numpy as np
import plotly.express as px

def pick_run_dir(run_dir_arg: str|None) -> pathlib.Path:
    runs_root = pathlib.Path('runs')
    if run_dir_arg and run_dir_arg != 'latest':
        return pathlib.Path(run_dir_arg)
    cands = sorted([p for p in runs_root.glob('*') if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for d in cands:
        if list(d.glob('iter_*')):
            return d
    raise RuntimeError('No runs with iter_* found.')

def load_rows(run_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    iters = sorted([p for p in run_dir.glob('iter_*') if p.is_dir()])
    for it in iters:
        src = it/'interactions.jsonl'
        if not src.exists() or src.stat().st_size == 0:
            continue
        with src.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if 'iteration' not in r or r['iteration'] in (None, ''):
                        r['iteration'] = int(it.name.split('_')[-1])
                    if 'seed_id' not in r:
                        seed_txt = (r.get('seed') or '').strip()
                        r['seed_id'] = hashlib.sha1(seed_txt.encode('utf-8')).hexdigest()[:10] if seed_txt else 'unknown'
                    rows.append(r)
                except:
                    pass
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default='latest', help="Run folder or 'latest'")
    args = ap.parse_args()

    run_dir = pick_run_dir(args.run_dir)
    df = load_rows(run_dir)
    assert len(df), 'No data rows found in run.'

    # Coerce numeric & fill missing (robust even if some rows are blanks)
    for c in ['toxicity_score','toxicity_level','success','iteration']:
        if c not in df.columns: df[c] = 0
    df['toxicity_score']  = pd.to_numeric(df['toxicity_score'], errors='coerce').fillna(0.0)
    df['toxicity_level']  = pd.to_numeric(df['toxicity_level'], errors='coerce').fillna(0).astype(int)
    df['success']         = pd.to_numeric(df['success'], errors='coerce').fillna(0).astype(int)
    df['iteration']       = pd.to_numeric(df['iteration'], errors='coerce').fillna(0).astype(int)
    for c in ['risk','top_category','adversarial_prompt','seed','seed_id']:
        if c not in df.columns: df[c] = ''

    # Metrics
    ASR      = float((df['success'] == 1).mean())
    avg_tox  = float(df['toxicity_score'].mean())
    by_iter  = df.groupby('iteration')['success'].mean().sort_index()
    tox_hist = df['toxicity_level'].value_counts().sort_index()
    by_risk  = df.groupby('risk', dropna=False)['success'].mean().sort_values(ascending=False)
    by_top   = df.groupby('top_category', dropna=False)['success'].mean().sort_values(ascending=False)

    # Diversity (Self-BLEU) – optional
    try:
        import sacrebleu
        adv = [str(x) for x in df['adversarial_prompt'].dropna().tolist()]
        np.random.seed(0); SAMPLE_MAX = 100
        if len(adv) > SAMPLE_MAX:
            idx = np.random.choice(len(adv), SAMPLE_MAX, replace=False); adv = [adv[i] for i in idx]
        scores = []
        for i, hyp in enumerate(adv):
            refs = adv[:i] + adv[i+1:]
            if refs:
                bleu = sacrebleu.corpus_bleu([hyp], [refs]).score
                scores.append(bleu)
        self_bleu = float(np.mean(scores)) if scores else 0.0
    except Exception:
        self_bleu = float('nan')

    out_dir = run_dir/'paper_metrics'; out_dir.mkdir(exist_ok=True)

    # Save CSVs
    pd.DataFrame([{'n_rows': len(df), 'ASR': ASR, 'avg_toxicity': avg_tox, 'self_bleu': self_bleu}]).to_csv(out_dir/'summary.csv', index=False)
    pd.DataFrame({'iteration': by_iter.index, 'ASR': by_iter.values}).to_csv(out_dir/'asr_by_iteration.csv', index=False)
    pd.DataFrame({'toxicity_level': tox_hist.index, 'count': tox_hist.values}).to_csv(out_dir/'toxicity_histogram.csv', index=False)
    by_risk.to_frame('ASR').reset_index().to_csv(out_dir/'asr_by_seed_risk.csv', index=False)
    by_top.to_frame('ASR').reset_index().to_csv(out_dir/'asr_by_top_category.csv', index=False)

    # Rename grouped columns to 'ASR' for plotting (this fixes the error you saw)
    df_risk = by_risk.reset_index().rename(columns={'success':'ASR'})
    df_topc = by_top.reset_index().rename(columns={'success':'ASR'})

    # Graphs
    fig_iter = px.line(by_iter.reset_index(), x='iteration', y='success', title='ASR by Iteration', markers=True)
    fig_hist = px.bar({'toxicity_level': tox_hist.index, 'count': tox_hist.values}, x='toxicity_level', y='count', title='Toxicity Level Histogram (0..5)')

    # Add checks for empty dataframes before plotting bar charts
    fig_risk = None
    if not df_risk.empty:
        fig_risk = px.bar(df_risk.head(10), x='risk', y='ASR', title='Top 10 DNA Seed-Risk ASR')

    fig_top  = None
    if not df_topc.empty:
        fig_top = px.bar(df_topc.head(10), x='top_category', y='ASR', title='Top 10 Moderation Categories ASR')


    # HTML (with simple definitions)
    html = f'''<html>
<head><meta charset='utf-8'><title>Paper Metrics — {run_dir.name}</title></head>
<body style="font-family: Arial, sans-serif; max-width: 1100px; margin: auto;">
<h1>Paper Metrics (Run: {run_dir.name})</h1>
<h2>Headline</h2>
<ul>
  <li><b>Rows</b>: {len(df)}</li>
  <li><b>ASR</b> (share risky): {ASR:.3f}</li>
  <li><b>Average toxicity</b>: {avg_tox:.3f}</li>
  <li><b>Self-BLEU</b> (lower = more diverse): {self_bleu:.2f}</li>
</ul>
<h2>What these mean (ELI5)</h2>
<ul>
  <li><b>ASR</b> — out of all answers, how many were risky (score ≥ 0.5). Lower is safer.</li>
  <li><b>Toxicity score</b> — a 0..1 risk score from moderation (bigger is riskier).</li>
  <li><b>Toxicity levels</b> — score buckets 0..5 so we see the distribution.</li>
  <li><b>ASR by iteration</b> — how risk changes round by round.</li>
  <li><b>ASR by DNA risk</b> — which DNA labels trigger risk more often.</li>
  <li><b>ASR by moderation category</b> — which categories are flagged most.</li>
  <li><b>Self-BLEU</b> — similarity of adversarial prompts (lower = more variety).</li>
</ul>
<h2>ASR by Iteration</h2>
{fig_iter.to_html(full_html=False, include_plotlyjs="cdn")}
<h2>Toxicity Level Histogram (0..5)</h2>
{fig_hist.to_html(full_html=False, include_plotlyjs=False)}
<h2>Top 10 DNA Seed-Risk ASR</h2>
{fig_risk.to_html(full_html=False, include_plotlyjs=False) if fig_risk else "<p>No data for this chart.</p>"}
<h2>Top 10 Moderation Categories ASR</h2>
{fig_top.to_html(full_html=False, include_plotlyjs=False) if fig_top else "<p>No data for this chart.</p>"}
<p style="margin-top: 24px; font-size: 12px; color: #666;">
  CSVs in {out_dir.as_posix()}
</p>
</body></html>'''
    (out_dir/'paper_report.html').write_text(html, encoding='utf-8')
    print('Wrote:', (out_dir/'paper_report.html').as_posix())

if __name__ == '__main__':
    main()
