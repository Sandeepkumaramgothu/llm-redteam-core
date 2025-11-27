"""
Main Experiment Runner
Run this to execute attacks with different strategies.
- Loads YAML config
- Converts all paths to absolute
- Runs core engine
- Triggers post-run metrics+dashboard analysis/report
"""

import sys
from pathlib import Path
import yaml

# Add project to path
PROJECT_ROOT = Path("/content/drive/MyDrive/llm-redteam-core")
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.engine import run_attack
from src.utils.postrun import analyze_and_visualize

def run_experiment(config_path: str):
    """Run complete experiment from YAML config file."""
    print(f"\n{'='*70}")
    print(f"  STARTING EXPERIMENT")
    print(f"  Config: {config_path}")
    print(f"{'='*70}\n")

    # Load config
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert relative paths to absolute (seeds, runs)
    for path_key in ['seeds', 'runs']:
        if path_key in config.get('paths', {}):
            path_val = Path(config['paths'][path_key])
            if not path_val.is_absolute():
                config['paths'][path_key] = str(PROJECT_ROOT / path_val)

    print(f"Experiment:         {config['name']}")
    print(f"Attack Strategy:    {config['red']['strategy']}")
    print(f"Target Model:       {config['target']['hf_model']}")
    print(f"Seeds:              {config['attack']['n_seeds']}  x  {config['attack']['n_iters']} iterations")
    print(f"Few-shot:           {config.get('red',{}).get('use_few_shot', False)}\n")

    # Run attack (main loop)
    run_dir = run_attack(config)

    # Analyze and summarize results
    report_path = analyze_and_visualize(run_dir)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Results: {run_dir}")
    print(f"  Report:  {report_path}")
    print(f"{'='*70}\n")

    return run_dir, report_path

if __name__ == "__main__":
    # This block only runs if file is executed directly (not when imported)
    # To run: python run_experiment.py configs/YOUR_CONFIG.yaml
    import sys
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        run_experiment(config_file)
    else:
        print("💡 Usage: python run_experiment.py configs/YOUR_CONFIG.yaml")
