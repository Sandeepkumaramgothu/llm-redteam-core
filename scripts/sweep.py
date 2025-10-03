#!/usr/bin/env python
from __future__ import annotations

import sys, pathlib # Import sys and pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1] # Get project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT)) # Add project root to sys.path

import argparse, pathlib, json, subprocess, time, os # Import os

import pandas as pd
import plotly.express as px

def run_once(config_path: str, model_name: str, extra_args=None) -> pathlib.Path:
    # Run your existing CLI for a SHORT run (e.g., 1–2 iterations) to keep sweeps quick
    ts = time.strftime("%Y%m%d-%H%M%S")
    # We pass target model via --target if your run.py supports it; else write a temp config
    cmd = ["python","scripts/run.py","--config",config_path,"--iterations","2","--seeds-per-iter","100","--target",model_name]

    # Add specific args based on model name if needed
    if model_name == "mistralai/Mistral-7B-Instruct-v0.3":
         cmd.append("--no-load-4bit") # Disable 4-bit for this model

    if extra_args: cmd += extra_args
    print("Running:", " ".join(cmd))

    # Change directory to project root before running subprocess
    original_cwd = os.getcwd()
    os.chdir(ROOT)

    # Capture stdout and stderr to diagnose CalledProcessError
    result = subprocess.run(cmd, capture_output=True, text=True, check=False) # Remove env=env

    # Change back to original directory
    os.chdir(original_cwd)

    print("Subprocess stdout:\n", result.stdout)
    print("Subprocess stderr:\n", result.stderr)
    result.check_returncode() # Re-raise CalledProcessError if returncode is non-zero

    # Find the newest run folder
    runs = sorted([p for p in pathlib.Path("runs").glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    return runs[-1]

def summarize_run(run_dir: pathlib.Path) -> dict:
    st = json.loads((run_dir/"status.json").read_text())
    model = st.get("models",{}).get("target","?")
    m = run_dir/"metrics.csv"
    df = pd.read_csv(m) if m.exists() else pd.DataFrame()
    last = df.sort_values("iteration").tail(1)
    out = {"run": run_dir.name, "model": model}
    for col in ["ASR","avg_toxicity","cap_acc","div_self_bleu","seconds"]:
        out[col] = (float(last[col].values[0]) if (len(last) and col in last.columns) else None)
    return out

def main():
    import os # Import os here
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--models", type=str, nargs="+", required=True, help="list of HF model names or API ids")
    args = ap.parse_args()

    results = []
    run_ids = []
    for m in args.models:
        try: # Add try-except block to catch errors per model run
            rd = run_once(args.config, m)
            run_ids.append(rd)
            results.append(summarize_run(rd))
        except subprocess.CalledProcessError as e:
            print(f"Error running model {m}: {e}")
            # Optionally, log this failure or add a placeholder to results
            results.append({"run": "failed", "model": m, "ASR": None, "avg_toxicity": None, "cap_acc": None, "div_self_bleu": None, "seconds": None})


    df = pd.DataFrame(results)
    # Save CSV
    out_csv = pathlib.Path("runs/sweep_summary.csv")
    out_csv.write_text(df.to_csv(index=False), encoding="utf-8")

    # Simple bar charts for quick compare
    figs = []
    for col in ["ASR","avg_toxicity","cap_acc","div_self_bleu"]:
        if col in df.columns and df[col].notna().any():
            fig = px.bar(df, x="model", y=col, text=col, title=f"{col} by model")
            figs.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    # Stitch into one summary HTML
    html = "<h1>Multi-model Sweep</h1>" + "".join(figs) + df.to_html(index=False)
    out_html = pathlib.Path("runs/sweep_summary.html")
    out_html.write_text(html, encoding="utf-8")
    print("Wrote:", out_csv, "and", out_html)
    print("Runs compared:", [p.as_posix() for p in run_ids])

if __name__ == "__main__":
    main()
