#!/usr/bin/env python
from __future__ import annotations
import argparse, json, pathlib, re
from typing import Dict, Any, List
import pandas as pd

# optional; only needed for OpenAI family tokenization
try:
    import tiktoken
except Exception:
    tiktoken = None

SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."

def _load_cost_map(cfg_path: pathlib.Path) -> Dict[str, Any]:
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    return cfg.get("costs", {})

def _guess_backend_and_price(model_name: str, costs: Dict[str, Any]) -> Dict[str, float]:
    '''
    Guess price table for the target model:
    - If exact model key exists under costs['openai'], use it.
    - Else if looks OpenAI (no slash and starts with 'gpt'), use gpt-4o-mini if present.
    - Else fall back to hf_default.
    '''
    openai_costs = costs.get("openai", {})
    if model_name in openai_costs:
        return openai_costs[model_name]
    # heuristic: openai models typically don't contain '/'
    if "/" not in model_name and model_name.startswith("gpt"):
        # try a default (user can add more to config)
        if "gpt-4o-mini" in openai_costs:
            return openai_costs["gpt-4o-mini"]
    return costs.get("hf_default", {"input_per_1k": 0.0, "output_per_1k": 0.0})

def _encoder_for_model(model_name: str):
    '''
    Pick a tiktoken encoder for OpenAI-ish models; default to cl100k_base.
    For HF models we still use cl100k_base as a rough proxy if needed.
    '''
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str, enc) -> int:
    if enc is None:
        # rough fallback: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)
    return len(enc.encode(text or ""))

def _iter_interactions(run_dir: pathlib.Path) -> List[pathlib.Path]:
    return sorted([p for p in run_dir.glob("iter_*") if p.is_dir()])

def estimate_run_cost(run_dir: pathlib.Path, cfg_path: pathlib.Path) -> Dict[str, Any]:
    # load model name from status
    st = json.loads((run_dir / "status.json").read_text())
    model = st.get("models", {}).get("target", "unknown-model")

    costs = _load_cost_map(cfg_path)
    price = _guess_backend_and_price(model, costs)

    enc = _encoder_for_model(model)

    rows = []
    total_in, total_out = 0, 0
    iters = _iter_interactions(run_dir)
    if not iters:
        raise SystemExit(f"No iteration folders in {run_dir}")

    for iterd in iters:
        path = iterd / "interactions.jsonl"
        if not path.exists():
            continue
        in_tokens, out_tokens, n = 0, 0, 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                # We approximate input as system + adversarial prompt
                inp = SYSTEM_PROMPT + "\n" + (r.get("adversarial_prompt") or "")
                out = r.get("response") or ""
                in_tokens += _count_tokens(inp, enc)
                out_tokens += _count_tokens(out, enc)
                n += 1
        total_in += in_tokens
        total_out += out_tokens
        cost_iter = (in_tokens/1000.0)*price["input_per_1k"] + (out_tokens/1000.0)*price["output_per_1k"]
        rows.append({
            "iteration": int(re.findall(r"\d+", iterd.name)[0]),
            "n": n,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "usd_cost": round(cost_iter, 6),
        })

    # totals
    cost_total = (total_in/1000.0)*price["input_per_1k"] + (total_out/1000.0)*price["output_per_1k"]
    df = pd.DataFrame(rows).sort_values("iteration")
    df.loc["TOTAL"] = ["", int(df["n"].sum()), int(total_in), int(total_out), round(cost_total, 6)]
    (run_dir / "costs.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    # compute $ per risky finding
    # need ASR and n from metrics.csv
    mpath = run_dir / "metrics.csv"
    risky = 0
    if mpath.exists():
        dfm = pd.read_csv(mpath)
        risky = int((dfm["ASR"] * dfm["n"]).sum())
    cpp = (cost_total / risky) if risky > 0 else None

    return {
        "model": model,
        "price_used": price,
        "total_input_tokens": int(total_in),
        "total_output_tokens": int(total_out),
        "total_cost_usd": round(cost_total, 6),
        "cost_per_risky_usd": (round(cpp, 6) if cpp is not None else None)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True, help="path to a single run folder")
    ap.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    cfg_path = pathlib.Path(args.config)
    out = estimate_run_cost(run_dir, cfg_path)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
