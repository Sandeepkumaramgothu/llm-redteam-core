#!/usr/bin/env python
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, os, time, json, psutil, platform
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
from src.core.config import load_config
from src.models.llm_hf import HFGenerator
from src.models.openai_llm import OpenAIChat

try:
    import tiktoken
except Exception:
    tiktoken = None

DUMMY_PROMPTS = [
    "Explain how rain happens in one friendly sentence.",
    "List three animals that live in the ocean.",
    "What is 12 + 7? Answer with just the number.",
    "Give a short bedtime wish (one sentence).",
    "Name the capital of Italy.",
]

@dataclass
class SysInfo:
    python: str
    torch: str
    cuda_available: bool
    cuda_device: str
    vram_total_gb: float
    vram_free_gb: float
    cpu_count: int
    ram_total_gb: float

def get_sysinfo() -> SysInfo:
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_total = torch.cuda.get_device_properties(0).total_memory/1e9 if cuda_avail else 0.0
    vram_free = torch.cuda.mem_get_info()[0]/1e9 if cuda_avail else 0.0
    vm = psutil.virtual_memory()
    return SysInfo(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_available=cuda_avail,
        cuda_device=dev_name,
        vram_total_gb=round(vram_total,2),
        vram_free_gb=round(vram_free,2),
        cpu_count=psutil.cpu_count(logical=True),
        ram_total_gb=round(vm.total/1e9,2),
    )

def count_tokens_openai_like(text: str, model_hint: str="gpt-4o-mini") -> int:
    if tiktoken is None:
        return int(len(text.split()) * 1.3)
    try:
        enc = tiktoken.encoding_for_model(model_hint)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def count_tokens_hf(text: str, tok) -> int:
    return len(tok(text, add_special_tokens=True).input_ids)

def run_profile(cfg_path: str, out_path: str, n_prompts: int=5) -> Dict[str, Any]:
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)
    backend = cfg["models"].get("backend", "hf")
    sysinfo = asdict(get_sysinfo())

    if backend == "openai":
        model_name = cfg["models"].get("target_model","gpt-4o-mini")
        chat = OpenAIChat(model=model_name,
                          max_tokens=cfg["models"].get("max_new_tokens",128),
                          temperature=cfg["models"].get("temperature",0.7))
        tk_count = lambda s: count_tokens_openai_like(s, model_hint=model_name)
        tok_name, quant = model_name, "n/a"
    else:
        model_name = cfg["models"]["target_model"]
        chat = HFGenerator(
            model_name=model_name,
            load_4bit=cfg["models"].get("load_4bit", True),
            device_map=cfg["models"].get("device_map","auto"),
            max_new_tokens=cfg["models"].get("max_new_tokens",128),
            temperature=cfg["models"].get("temperature",0.7),
            top_p=cfg["models"].get("top_p",0.95),
        )
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HUGGING_FACE_HUB_TOKEN"))
        tk_count = lambda s: count_tokens_hf(s, tok)
        tok_name = model_name.split("/")[-1]
        quant = "4-bit" if cfg["models"].get("load_4bit", True) else "fp16/bf16"

    prompts = DUMMY_PROMPTS[:max(1, min(n_prompts, len(DUMMY_PROMPTS)))]
    records = []
    total_in = total_out = 0
    total_secs = 0.0

    for i, p in enumerate(prompts, 1):
        sys_msg = "You are a helpful, harmless assistant. Be brief."
        inp_tokens = tk_count(p) + tk_count(sys_msg)
        t0 = time.time()
        out_text = chat.chat(sys_msg, p)
        dt = time.time() - t0
        out_tokens = tk_count(out_text)
        total_in += inp_tokens
        total_out += out_tokens
        total_secs += dt
        records.append({
            "idx": i,
            "prompt_preview": p[:60] + ("..." if len(p)>60 else ""),
            "response_preview": out_text[:120].replace("\n"," ") + ("..." if len(out_text)>120 else ""),
            "inp_tokens": inp_tokens,
            "out_tokens": out_tokens,
            "latency_sec": round(dt, 3),
            "tps_out": round(out_tokens/dt, 2) if dt>0 else None
        })
        print(f"[{i}/{len(prompts)}] {inp_tokens}→{out_tokens} tokens, {dt:.2f}s  ({(out_tokens/dt if dt>0 else 0):.1f} tok/s)")

    summary = {
        "backend": backend,
        "model": model_name,
        "tokenizer_hint": tok_name,
        "quantization": quant,
        "sysinfo": sysinfo,
        "n_samples": len(prompts),
        "totals": {"input_tokens": total_in, "output_tokens": total_out, "seconds": round(total_secs,3)},
        "throughput": {
            "output_tok_per_sec": round(total_out/total_secs, 2) if total_secs>0 else None,
            "avg_latency_sec": round(total_secs/len(prompts), 3) if prompts else None
        },
        "samples": records
    }

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\\nSaved profile to:", out_path)
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to baseline.yaml")
    ap.add_argument("--out", type=str, default="runs/model_profile.json")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    run_profile(args.config, args.out, n_prompts=args.n)

if __name__ == "__main__":
    main()
