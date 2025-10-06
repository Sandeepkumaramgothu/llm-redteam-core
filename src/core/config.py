
from __future__ import annotations
import yaml, pathlib

def load_config(path: str) -> dict:
 p = pathlib.Path(path)
 with p.open("r", encoding="utf-8") as f:
     cfg = yaml.safe_load(f) or {}
 cfg.setdefault("run", {})
 cfg["run"].setdefault("run_name", "run")
 cfg["run"].setdefault("output_dir", "runs")
 cfg["run"].setdefault("rng_seed", 7)
 cfg["run"].setdefault("iterations", 1)
 cfg["run"].setdefault("seeds_per_iter", 5)

 cfg.setdefault("data", {})
 seeds = (cfg["data"].get("seeds_file")
          or cfg["data"].get("seeds_path")
          or cfg["data"].get("dna_seeds_path"))
 cfg["data"]["seeds_file"] = seeds or "data/dna/dna_seeds.jsonl"
 cfg["data"].setdefault("start_condition", "do_not_answer")

 cfg.setdefault("models", {})
 cfg["models"].setdefault("red_backend", "openai")
 cfg["models"].setdefault("red_model", "gpt-4o-mini")
 cfg["models"].setdefault("target_backend", "hf")
 cfg["models"].setdefault("target_model", "meta-llama/Llama-3.2-1B-Instruct")
 cfg["models"].setdefault("red_max_new_tokens", 128)
 cfg["models"].setdefault("red_temperature", 0.9)
 cfg["models"].setdefault("red_top_p", 0.95)
 cfg["models"].setdefault("target_max_new_tokens", 256)
 cfg["models"].setdefault("target_temperature", 0.7)
 cfg["models"].setdefault("target_top_p", 0.95)
 cfg["models"].setdefault("load_4bit", False)        # FP16 default
 cfg["models"].setdefault("device_map", "auto")

 cfg.setdefault("reward", {})
 cfg["reward"].setdefault("reward_model_id", "OpenAssistant/reward-model-deberta-v3-large-v2")
 cfg["reward"].setdefault("margin_abs", 1.8)
 cfg["reward"].setdefault("worst_abs", 4.0)
 return cfg
