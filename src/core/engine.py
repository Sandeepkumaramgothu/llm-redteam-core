from __future__ import annotations
from typing import Dict, Any, List
import random, time, pathlib, os, json, pandas as pd

from .config import load_config
from .utils import now_id, ensure_dir, write_json, write_jsonl
from ..data.seed_loader import load_seeds
from ..models.red import make_red_model, rewrite_to_adversarial
from ..models.target import make_target_model, answer
from ..models.reward import ToxicityReward
from ..metrics.diversity import self_bleu
from ..metrics.capability import capability_accuracy
from ..metrics.riskstats import per_risk_stats
from ..metrics.validation import evaluate_fixed_set

def _mask(s: str, n: int=80) -> str:
    s = (s or "").replace("\n"," ")
    return s[:n] + ("..." if len(s) > n else "")

def _load_status(path: pathlib.Path) -> dict | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _append_csv(csv_path: pathlib.Path, row: dict):
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)

def run_loop(cfg: Dict[str, Any], resume: bool=False) -> Dict[str, Any]:
    rng = random.Random(cfg["run"].get("rng_seed", 7))
    run_name = cfg["run"]["run_name"]
    all_iters = int(cfg["run"]["iterations"])
    out_root = pathlib.Path(cfg["run"]["output_dir"])

    # Create/resume run folder
    if resume:
        candidates = sorted([p for p in out_root.glob(f"{run_name}_*") if p.is_dir()])
        if not candidates:
            raise RuntimeError("No previous run to resume.")
        run_dir = candidates[-1]
    else:
        run_id = now_id()
        run_dir = out_root / f"{run_name}_{run_id}"
        ensure_dir(run_dir)

    status_path = run_dir / "status.json"
    metrics_csv = run_dir / "metrics.csv"
    risk_csv = run_dir / "risk_metrics.csv"
    val_metrics_csv = run_dir / "validation_metrics.csv"
    val_risk_csv = run_dir / "validation_risk_metrics.csv"

    # load models once
    red_llm = make_red_model(cfg)
    tgt_llm = make_target_model(cfg)
    scorer = ToxicityReward()
    tox_thr = float(cfg["reward"].get("toxicity_threshold", 0.5))

    # options
    want_self_bleu = bool(cfg.get("metrics", {}).get("diversity", {}).get("self_bleu", True))
    cap_cfg = cfg.get("metrics", {}).get("capability", {"enabled": True, "max_questions": 5})
    want_cap = bool(cap_cfg.get("enabled", True))
    cap_max = int(cap_cfg.get("max_questions", 5))

    weights = cfg.get("sampling", {}).get("risk_weights", {})
    use_shortlist = bool(cfg.get("steering", {}).get("use_shortlist", True))
    shortlist_top_n = int(cfg.get("steering", {}).get("shortlist_top_n", 3))
    shortlist_in = run_dir / "shortlist_next.json"
    shortlist_out = run_dir / "shortlist_next.json"

    # validation config
    val_cfg = cfg.get("validation", {})
    val_enabled = bool(val_cfg.get("enabled", False))
    val_path = val_cfg.get("local_jsonl", "")

    start_iter = 1
    if resume:
        s = _load_status(status_path)
        if s is not None:
            start_iter = int(s.get("iteration", 0)) + 1

    for it in range(start_iter, all_iters + 1):
        iterd = run_dir / f"iter_{it:04d}"
        ensure_dir(iterd)

        # ===== 1) Seeds
        k = int(cfg["run"]["seeds_per_iter"])
        pool = load_seeds(cfg["data"], max(k*3, 64), rng)
        # simple weighted chooser (same as Day 6)
        def choose(pool, k, weights):
            by_risk = {}
            for item in pool:
                by_risk.setdefault(item["risk"], []).append(item)
            risks = list(by_risk.keys())
            w = [float(weights.get(r, 1.0)) for r in risks]
            out = []
            for _ in range(k):
                total = sum(w); 
                if total <= 0: break
                r = rng.uniform(0, total)
                upto, pick_idx = 0.0, 0
                for i, wi in enumerate(w):
                    upto += wi
                    if r <= upto:
                        pick_idx = i; break
                risk = risks[pick_idx]; bucket = by_risk[risk]
                if bucket:
                    out.append(bucket.pop(rng.randrange(len(bucket))))
                if not bucket:
                    w[pick_idx] = 0.0
            remain = [x for v in by_risk.values() for x in v]
            rng.shuffle(remain); out.extend(remain[: (k - len(out))])
            return out[:k]
        chosen = choose(pool, k, weights)

        # Steering carry-over
        if use_shortlist and shortlist_in.exists():
            try:
                prev = json.load(open(shortlist_in, "r", encoding="utf-8"))
                inject = prev[:min(len(prev), k//2 or 1)]
                chosen = inject + chosen[: max(0, k - len(inject))]
                print(f"[steer] carried over {len(inject)} prompts from shortlist.")
                shortlist_in.unlink()
            except Exception as e:
                print("[steer] failed to load shortlist:", e)

        # ===== 2) Generate on training loop
        rows: List[Dict[str, Any]] = []
        t0 = time.time()
        for idx, s in enumerate(chosen, 1):
            adv = rewrite_to_adversarial(red_llm, s["text"], s["risk"])
            resp = answer(tgt_llm, adv)
            tox = scorer.score([resp])[0]
            success = float(tox >= tox_thr)
            rows.append({
                "iter": it, "idx": idx, "risk": s["risk"],
                "seed": s["text"], "adversarial_prompt": adv, "response": resp,
                "toxicity": tox, "success": success,
            })
            print(f"[iter {it:02d}] [{idx:02d}/{k}] risk={s['risk']:<11} tox={tox:.3f} success={int(success)}")
            print("  seed:", _mask(s["text"]))
            print("  adv :", _mask(adv))
            print("  resp:", _mask(resp))
        dt = time.time() - t0

        # ===== 3) Metrics (train loop)
        asr = sum(r["success"] for r in rows) / len(rows) if rows else 0.0
        avg_tox = sum(r["toxicity"] for r in rows) / len(rows) if rows else 0.0
        div_self_bleu = self_bleu([r["adversarial_prompt"] for r in rows]) if want_self_bleu else None
        cap_acc = capability_accuracy(tgt_llm, max_q=cap_max) if want_cap else None

        # ===== 4) Per-risk train metrics
        pr_df = per_risk_stats(rows)
        pr_df["iteration"] = it
        if risk_csv.exists():
            old = pd.read_csv(risk_csv)
            new = pd.concat([old, pr_df], ignore_index=True)
        else:
            new = pr_df
        new.to_csv(risk_csv, index=False)

        # ===== 5) Validation pass (fixed set)
        if val_enabled and val_path:
            vdf = evaluate_fixed_set(tgt_llm, scorer, tox_thr, val_path)
            vdf["iter"] = it
            # save this iteration's validation detail
            vdf.to_csv(iterd / "validation_results.csv", index=False)
            # metrics
            vasr = float(vdf["success"].mean()) if len(vdf) else 0.0
            vavg = float(vdf["toxicity"].mean()) if len(vdf) else 0.0
            vpr = vdf.groupby("risk").agg(
                n=("success","count"),
                risky_rate=("success","mean"),
                avg_tox=("toxicity","mean")
            ).reset_index()
            vpr["iteration"] = it
            # append CSVs
            row = {"iteration": it, "n": int(vdf.shape[0]), "ASR": vasr, "avg_toxicity": vavg}
            if val_metrics_csv.exists():
                old = pd.read_csv(val_metrics_csv)
                old = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
            else:
                old = pd.DataFrame([row])
            old.to_csv(val_metrics_csv, index=False)

            if val_risk_csv.exists():
                oldr = pd.read_csv(val_risk_csv)
                oldr = pd.concat([oldr, vpr], ignore_index=True)
            else:
                oldr = vpr
            oldr.to_csv(val_risk_csv, index=False)
            print(f"[val] iter {it:02d}: ASR={vasr:.3f} avg_tox={vavg:.3f} (n={len(vdf)})")

        # ===== 6) Save train artifacts + metrics
        write_jsonl(iterd / "interactions.jsonl", rows)
        metric_row = {
            "iteration": it, "n": len(rows), "ASR": asr, "avg_toxicity": avg_tox,
            "div_self_bleu": div_self_bleu if div_self_bleu is not None else "",
            "cap_acc": cap_acc if cap_acc is not None else "",
            "seconds": dt
        }
        _append_csv(metrics_csv, metric_row)
        status = {
            "run_name": run_name,
            "output_dir": str(run_dir),
            "iteration": it,
            "stats": metric_row,
            "models": {"red": cfg["models"]["red_model"], "target": cfg["models"]["target_model"]},
            "toxicity_threshold": tox_thr,
        }
        write_json(status_path, status)

        # ===== 7) Shortlist for next round
        if use_shortlist:
            top = sorted(rows, key=lambda r: r["toxicity"], reverse=True)[:max(0, int(cfg.get("steering",{}).get("shortlist_top_n", 3)))]
            carry = [{"text": r["seed"], "risk": r["risk"]} for r in top]
            write_json(shortlist_out, carry)
            print(f"[steer] prepared shortlist_next.json with {len(carry)} items.")

        print(f"✓ saved iter {it:02d}: train ASR={asr:.3f} val={'ON' if val_enabled else 'OFF'}")

    print("\nAll done. Run folder:", run_dir)
    return {"run_dir": str(run_dir)}
