
#!/usr/bin/env python3
from __future__ import annotations
import argparse, pathlib, sys, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.core.config import load_config
from src.core.engine import run_loop

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--config", required=True)
 ap.add_argument("--resume", action="store_true")
 ap.add_argument("--iters", type=int, default=None, help="override iterations")
 ap.add_argument("--seeds", type=int, default=None, help="override seeds per iter")
 args = ap.parse_args()

 cfg = load_config(args.config)
 if args.iters is not None:
     cfg["run"]["iterations"] = int(args.iters)
 if args.seeds is not None:
     cfg["run"]["seeds_per_iter"] = int(args.seeds)

 # env checks
 if cfg["models"].get("target_backend","hf") == "hf":
     assert os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),          "HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) is required to load gated HF model."
 if cfg["models"].get("red_backend","openai") == "openai":
     assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is required."

 run_loop(cfg, resume=args.resume)

if __name__ == "__main__":
 main()
