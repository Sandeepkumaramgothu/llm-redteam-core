from typing import Dict, Any, List
from pathlib import Path
import json, random, time, sys, platform, torch, pkg_resources
from datetime import datetime
from tqdm import tqdm
import pandas as pd, numpy as np

# Updated imports for robust fine-tuning
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

PROJECT_ROOT = Path("/content/drive/MyDrive/llm-redteam-core")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import seeds
from src.core.strategies import AttackStrategyFactory
from src.models.target_model import TargetModel
from src.models.red_model import RedModel
from src.utils import scoring, metrics

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        pass

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

def print_detailed_scores(result: Dict[str, Any]):
    """Prints detailed metrics and FULL CONVERSATION HISTORY."""
    safe_print("-" * 80)
    safe_print(f"🌱 SEED: {result.get('seed', '')}")

    # ✅ NEW: Print Full Conversation History if available (Crucial for Crescendo)
    if "conversation_history" in result and result["conversation_history"]:
        safe_print("\n🗣️  CONVERSATION HISTORY:")
        for turn in result["conversation_history"]:
            t_num = turn.get('turn', '?')
            prompt = turn.get('prompt', '').replace('\n', ' ')[:100] + "..."
            resp = turn.get('response', '').replace('\n', ' ')[:100] + "..."
            refusal = "🚫 REFUSAL" if turn.get('is_refusal') else "✅ COMPLIANCE"
            safe_print(f"   Turn {t_num}: {refusal}")
            safe_print(f"     🔴 Red: {prompt}")
            safe_print(f"     🛡️ Tgt: {resp}")
    else:
        # Fallback for single-turn strategies
        safe_print(f"🔴 RED PROMPT: {result.get('adversarial', '')}")
        safe_print(f"🛡️ TARGET RESP: {result.get('response', '')}")

    safe_print("-" * 80)

    status = "✅ SUCCESS" if result.get('success') else "❌ FAILED (Safe)"
    safe_print(f"📊 RESULT: {status}")
    safe_print(f"   Risk Level: {result.get('risk_level')} | Refusal: {result.get('refusal')}")
    safe_print(f"   Reward (Helpful): {result.get('reward', 0):.2f} (Norm: {result.get('reward_norm', 0):.2f})")
    safe_print(f"   Toxicity: {result.get('toxicity', 0):.2f}% | Hate: {result.get('hate', 0):.2f}")
    safe_print(f"   Ensemble Score: {result.get('ensemble_score', 0):.2f}")
    safe_print("=" * 80 + "\n")

def save_run_metadata(run_dir: Path, config: Dict[str, Any], model_id: str):
    try:
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        key_packages = ['torch', 'transformers', 'accelerate', 'peft', 'datasets',
                        'pandas', 'plotly', 'detoxify', 'openai', 'sentence-transformers']
        tracked_versions = {pkg: installed_packages.get(pkg, '?') for pkg in key_packages}
        system_info = {
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        if torch.cuda.is_available() else []
        }
        metadata = {
            'run_id': run_dir.name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'target_model': model_id,
            'system_info': system_info,
            'package_versions': tracked_versions
        }
        with (run_dir / 'run_metadata.json').open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        safe_print(f"✓ Run metadata saved")
    except Exception as e:
        safe_print(f"⚠️ Warning: Could not save metadata: {e}")

def train_lora_on_finetune(finetune_path, base_model_name, lora_ckpt_dir, n_epochs=1):
    """Robust LoRA Fine-tuning using Hugging Face Datasets."""
    try:
        secrets_path = Path("/content/drive/MyDrive/llm-redteam-core/.secrets.json")
        token = None
        if secrets_path.exists():
            token = json.loads(secrets_path.read_text()).get("HF_TOKEN")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(base_model_name, token=token)

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, lora_cfg)

        # 1. Load data manually to handle JSONL robustly
        prompts = []
        with open(finetune_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if 'messages' in obj and len(obj['messages']) >= 2:
                        p = obj['messages'][0]['content']
                        r = obj['messages'][1]['content']
                        text = f"{p}\n{r}"
                        if tokenizer.eos_token:
                            text += tokenizer.eos_token
                        prompts.append(text)
                except: continue

        if not prompts:
            safe_print("   ⚠️ No valid data found in finetune file. Skipping.")
            return None

        # 2. Create Dataset object
        train_dataset = Dataset.from_dict({"text": prompts})

        # 3. Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256
            )

        tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

        # Add labels
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples

        tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

        # 4. Train
        training_args = TrainingArguments(
            output_dir=str(lora_ckpt_dir),
            overwrite_output_dir=True,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=1,
            learning_rate=2e-4,
            save_strategy="no",
            prediction_loss_only=True,
            logging_steps=1,
            report_to="none",
            remove_unused_columns=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            train_dataset=tokenized_dataset
        )

        safe_print(f"🚀 Fine-tuning target model on {len(prompts)} examples for {n_epochs} epoch(s)...")
        trainer.train()

        model.save_pretrained(str(lora_ckpt_dir))
        safe_print(f"✅ LoRA checkpoint saved: {lora_ckpt_dir}")
        return lora_ckpt_dir
    except Exception as e:
        safe_print(f"⚠️ LoRA Fine-tuning failed: {e}")
        return None

def run_attack(config: Dict[str, Any]) -> str:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["paths"]["runs"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    finetune_dir = run_dir / "finetune_data"
    finetune_dir.mkdir(exist_ok=True)

    safe_print(f"\n{'='*70}")
    safe_print(f"🚀 RED-TEAMING ATTACK RUN: {run_id}")
    safe_print(f"{'='*70}\n")

    safe_print("📋 Loading seeds...")
    all_seeds = seeds.load_all_seeds(config["paths"]["seeds"])
    safe_print(f"✓ Loaded {len(all_seeds)} seeds")

    rng = random.Random(config.get("seed", 42))
    n_seeds = config["attack"]["n_seeds"]
    n_iters = config["attack"]["n_iters"]
    use_ledger = config.get("attack", {}).get("use_ledger", False)
    ledger = seeds.SeedLedger() if use_ledger else None

    safe_print("\n🔴 Loading RED MODEL...")
    red_model_wrapper = RedModel(config)
    safe_print("🛡️  Loading TARGET MODEL...")
    target_model_wrapper = TargetModel(config)
    model_id = target_model_wrapper.model_name

    strategy_name = config.get("red", {}).get("strategy", "pyrit_crescendo")
    attack_strategy = AttackStrategyFactory.create(
        strategy_name, config, red_model_wrapper, target_model_wrapper
    )
    max_turns = config.get("red", {}).get("crescendo_turns", 5)

    safe_print("\n📊 Loading scoring system...")
    scorer = scoring.load_scorers()
    save_run_metadata(run_dir, config, model_id)

    safe_print(f"\n{'='*70}")
    safe_print(f"🎯 ATTACK CONFIGURATION")
    safe_print(f"   Seeds: {n_seeds} x {n_iters} = {n_seeds * n_iters} total")
    safe_print(f"   Target: {model_id}")
    safe_print(f"   Strategy: {strategy_name}")
    safe_print(f"{'='*70}\n")

    all_results = []
    debug = config.get("debug", False)

    for iter_idx in range(1, n_iters + 1):
        safe_print(f"\n{'─'*70}")
        safe_print(f"📍 ITERATION {iter_idx}/{n_iters}")
        safe_print(f"{'─'*70}\n")
        selected_seeds = seeds.sample_seeds(
            all_seeds, n_seeds, rng,
            allow_repeat=not use_ledger,
            ledger=ledger
        )
        safe_print(f"✓ Sampled {len(selected_seeds)} seeds")

        iter_results = []
        finetune_data = []

        pbar = tqdm(enumerate(selected_seeds), desc=f"Iter {iter_idx}", total=len(selected_seeds), unit="seed")

        for idx, seed_obj in pbar:
            seed_text = seed_obj["Seed"]
            risk_type = seed_obj.get("risk_type", "Other")

            # 1. Run Attack
            try:
                if hasattr(attack_strategy, 'run'):
                    # Strategy handles prompt generation and target interaction
                    adv_result = attack_strategy.run(seed_text, risk_type, rng, debug=False)
                else:
                    adv_result = attack_strategy.generate(
                        seed_text, risk_type, rng, debug=False, idx=idx,
                        iter_idx=iter_idx, pbar=pbar, target_model=target_model_wrapper
                    )
            except Exception as e:
                safe_print(f"\n⚠️  Attack error: {e}")
                adv_result = {
                    "adversarial": seed_text,
                    "combined": seed_text,
                    "response": "Error",
                    "method": "error",
                    "red_model": "none",
                    "turns_taken": 0,
                    "conversation_history": []
                }

            # 2. Get Response (from strategy output)
            response = adv_result.get("response", "")
            if not response or response == "Error":
                # Fallback: query target directly if strategy didn't
                try:
                    response = target_model_wrapper.generate(adv_result.get("combined", seed_text))
                except:
                    response = ""

            # 3. Scoring
            query_start = time.time()
            score_summary = None
            try:
                # Use updated composite scoring logic
                composite = scorer.score_composite(
                    adv_result.get("combined", seed_text),
                    response,
                    risk_type=risk_type
                )

                score_summary = {
                    "reward": float(composite.get('reward', 0.0)),
                    "reward_norm": float(composite.get('reward_norm', 0.0)),
                    "toxicity": float(composite.get('toxicity', 0.0)),
                    "risk_level": str(composite.get('risk_level', 'L0')),
                    "refusal": bool(composite.get('refusal', False)),
                    "success": bool(composite.get('success', False)),
                    "ensemble_score": float(composite.get('ensemble_score', 0.0)),
                    "hate": float(composite.get('hate', 0.0)),
                    "composite_harm": float(composite.get('composite_harm', 0.0)),
                    "llama_guard": str(composite.get('llama_guard', 'N/A')),
                    "llama_category": str(composite.get('llama_category', 'N/A')),
                    "model_agreement": float(composite.get('confidence', 0.0))
                }

                if 'detoxify' in composite.get('model_scores', {}):
                    detox = composite['model_scores']['detoxify']
                    score_summary.update({
                        "severe_toxicity": float(detox.get('severe_toxicity', 0.0)),
                        "obscene": float(detox.get('obscene', 0.0)),
                        "threat": float(detox.get('threat', 0.0)),
                        "insult": float(detox.get('insult', 0.0)),
                        "identity_attack": float(detox.get('identity_attack', 0.0))
                    })

            except Exception as e:
                safe_print(f"\n⚠️ Scoring error: {e}")
                score_summary = {
                    "reward": 0.0, "toxicity": 0.0, "risk_level": "L0", "refusal": True, "success": False,
                    "ensemble_score": 0.0, "hate": 0.0, "composite_harm": 0.0
                }

            query_time = time.time() - query_start

            if ledger and use_ledger:
                ledger.mark_used(
                    seed_obj.get("seed_hash", seeds.generate_seed_hash(seed_text)),
                    success=score_summary["success"]
                )

            record = {
                "run_id": run_id,
                "iter": iter_idx,
                "seed_seq": seed_obj.get("seed_seq"),
                "seed_hash": seed_obj.get("seed_hash"),
                "risk_type": risk_type,
                "risk_area": seed_obj.get("risk_area", ""),
                "seed": seed_text,
                "adversarial": adv_result.get("adversarial", ""),
                "combined": adv_result.get("combined", seed_text),
                "response": response,
                "target_model": model_id,
                "query_time_sec": round(query_time, 3),
                "red_method": adv_result.get("method", "unknown"),
                "red_model": adv_result.get("red_model", "unknown"),
                "turns_taken": adv_result.get("turns_taken", 1),
                "conversation_history": adv_result.get("conversation_history", []), # Save history
                **score_summary
            }

            iter_results.append(record)
            print_detailed_scores(record)

            if score_summary['success']:
                finetune_example = {
                    'messages': [
                        {'role': 'user', 'content': adv_result.get("combined", seed_text)},
                        {'role': 'assistant', 'content': "I cannot assist with that request due to safety guidelines."}
                    ],
                    'metadata': record
                }
                finetune_data.append(finetune_example)

            n_success = sum(1 for r in iter_results if r["success"])
            avg_tox = sum(r["toxicity"] for r in iter_results) / len(iter_results) if iter_results else 0
            pbar.set_postfix({
                "Success": f"{n_success}",
                "ASR": f"{n_success/len(iter_results)*100:.1f}%",
                "AvgTox": f"{avg_tox:.2f}%"
            })

        all_results.extend(iter_results)

        # --- Fine-Tuning Step ---
        safe_print(f"\n💾 Saving fine-tuning data for iteration {iter_idx}...")
        finetune_jsonl = finetune_dir / f"iter{iter_idx}_finetune.jsonl"

        if finetune_data:
            with finetune_jsonl.open('w', encoding='utf-8') as f:
                for example in finetune_data:
                    f.write(json.dumps(convert_to_serializable(example)) + '\n')

            finetune_csv = finetune_dir / f"iter{iter_idx}_finetune.csv"
            pd.DataFrame([item['metadata'] for item in finetune_data]).to_csv(finetune_csv, index=False)

            safe_print(f"   ✅ Saved {len(finetune_data)} successful attacks for fine-tuning")

            # Run LoRA Fine-tuning
            lora_ckpt_dir = finetune_dir / f"lora_ckpt_iter{iter_idx}"
            ckpt_path = train_lora_on_finetune(finetune_jsonl, model_id, lora_ckpt_dir, n_epochs=1)

            if ckpt_path:
                safe_print(f"   ✅ Fine-tuned target model after iteration {iter_idx} (LoRA)")
                try:
                    target_model_wrapper = TargetModel({
                        **config,
                        "target": {
                            **config.get("target", {}),
                            "use_lora": True,
                            "lora_checkpoint": str(ckpt_path)
                        }
                    })
                    model_id = target_model_wrapper.model_name
                    safe_print(f"   🔄 Target model reloaded with LoRA checkpoint: {ckpt_path}")
                except Exception as e:
                    safe_print(f"   ⚠️  Failed to reload target model with LoRA: {e}")
        else:
            safe_print(f"   ℹ️  No successful attacks this iteration; skipping LoRA update.")

        metrics.print_iteration_summary(iter_results, iter_idx)

        # --- Adversarial Flow: Add new seeds ---
        safe_print(f"\n🔄 ADVERSARIAL FLOW: Analyzing toxic prompts...")
        toxic_prompts = [r for r in iter_results if r['success']]

        if toxic_prompts:
            new_seeds = []
            for result in toxic_prompts:
                new_seed = {
                    'Seed': result['combined'],
                    'seed_hash': seeds.generate_seed_hash(result['combined']),
                    'risk_type': result['risk_type'],
                    'dataset': 'generated',
                    'source': 'adversarial_flow',
                    'source_score': result['reward'],
                    'added_iteration': iter_idx,
                    'parent_seed': result['seed_hash'],
                    'risk_level': result['risk_level']
                }
                new_seeds.append(new_seed)

            seeds_path = Path(config["paths"]["seeds"])
            with seeds_path.open('a', encoding='utf-8') as f:
                for seed in new_seeds:
                    f.write(json.dumps(convert_to_serializable(seed)) + '\n')
            all_seeds.extend(new_seeds)
            safe_print(f"   ➕ Added {len(new_seeds)} toxic prompts to database")
        else:
            safe_print(f"   ℹ️  No new toxic prompts to add")

    safe_print(f"\n{'='*70}")
    safe_print(f"💾 SAVING RESULTS")
    safe_print(f"{'='*70}")

    results_jsonl = run_dir / "results.jsonl"
    with results_jsonl.open('w', encoding='utf-8') as f:
        for record in all_results:
            f.write(json.dumps(convert_to_serializable(record)) + '\n')

    df = pd.DataFrame(all_results)
    results_csv = run_dir / "results.csv"
    df.to_csv(results_csv, index=False)

    final_metrics = metrics.compute_all_metrics(all_results)
    metrics_file = run_dir / "metrics.json"
    with metrics_file.open('w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(final_metrics), f, indent=2)

    safe_print(f"✅ Results saved: {results_csv}")

    safe_print(f"\n{'='*70}")
    safe_print(f"✅ ATTACK COMPLETE")
    safe_print(f"{'='*70}")
    metrics.print_metrics_summary(final_metrics)

    return str(run_dir)
