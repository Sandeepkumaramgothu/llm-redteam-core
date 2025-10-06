
from __future__ import annotations
import os, json, time, random, pathlib, datetime
from typing import Dict, Any, List
from .seeds import load_all, sample_seeds
from ..models.llm_hf import HFGenerator
from ..models.red import make_red_model

def run_loop(cfg: Dict[str, Any], resume: bool = False):
 rng = random.Random(cfg["run"].get("rng_seed", 7))
 iters = int(cfg["run"]["iterations"])
 k = int(cfg["run"]["seeds_per_iter"])

 out_root = pathlib.Path(cfg["run"]["output_dir"])
 out_root.mkdir(parents=True, exist_ok=True)
 ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 run_name = f'{cfg["run"]["run_name"]}_{ts}'
 run_dir = out_root / run_name
 run_dir.mkdir(parents=True, exist_ok=False)

 status = {
     "models": {
         "red": f'{cfg["models"]["red_backend"]}:{cfg["models"]["red_model"]}',
         "target": f'{cfg["models"]["target_backend"]}:{cfg["models"]["target_model"]}',
     },
     "run": {"iterations": iters, "seeds_per_iter": k, "rng_seed": cfg["run"].get("rng_seed", 7)},
     "data": {"seeds_file": cfg["data"]["seeds_file"], "start_condition": cfg["data"].get("start_condition","")}
 }
 (run_dir/"status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

 red = make_red_model(cfg)
 tgt = HFGenerator(
     model_name=cfg["models"]["target_model"],
     max_new_tokens=cfg["models"]["target_max_new_tokens"],
     temperature=cfg["models"]["target_temperature"],
     top_p=cfg["models"]["target_top_p"],
     device_map=cfg["models"].get("device_map","auto"),
 )

 all_seeds = load_all(cfg["data"]["seeds_file"])
 assert all_seeds, f"No seeds loaded from {cfg['data']['seeds_file']}"

 for i in range(1, iters+1):
     t0 = time.time()
     it_dir = run_dir / f"iter_{i:04d}"
     it_dir.mkdir(parents=True, exist_ok=False)
     out_jl = (it_dir/"interactions.jsonl").open("w", encoding="utf-8")

     batch = sample_seeds(all_seeds, k, rng)
     n = 0
     for item in batch:
         seed_id = item.get("seed_id")
         risk = item.get("risk","unknown")
         seed = item.get("seed","")
         try:
             adv = red.adversarialize(seed)
         except Exception as e:
             adv = f"[red-error] {e}"
         try:
             resp = tgt.generate(adv)
         except Exception as e:
             resp = f"[target-error] {e}"
         row = {"iteration": i, "seed_id": seed_id, "risk": risk,
                "seed": seed, "adversarial_prompt": adv, "response": resp}
         out_jl.write(json.dumps(row, ensure_ascii=False) + "\n")
         n += 1
     out_jl.close()
     dt = time.time() - t0
     print(f"[iter {i}] wrote {n} rows in {dt:.1f}s to {it_dir/'interactions.jsonl'}")

 print("All done. Run folder:", run_dir.as_posix())
 return run_dir.as_posix()
