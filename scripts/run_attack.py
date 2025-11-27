#!/usr/bin/env python3
"""
Main Attack Runner
==================
Command-line interface for running red-teaming attacks.

Usage:
    python run_attack.py --config configs/test.yaml
    python run_attack.py --config configs/base.yaml
"""

import sys
import argparse
from pathlib import Path
import yaml

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import engine
from src.utils import metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LLM red-teaming attack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 seeds, 1 iteration)
  python run_attack.py --config configs/test.yaml
  
  # Full attack (50 seeds, 3 iterations)
  python run_attack.py --config configs/base.yaml
  
  # Template-based (no OpenAI)
  python run_attack.py --config configs/template_only.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--override-seeds',
        type=int,
        help='Override n_seeds from config'
    )
    
    parser.add_argument(
        '--override-iters',
        type=int,
        help='Override n_iters from config'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.override_seeds:
        config['attack']['n_seeds'] = args.override_seeds
        print(f"Override: n_seeds = {args.override_seeds}")
    
    if args.override_iters:
        config['attack']['n_iters'] = args.override_iters
        print(f"Override: n_iters = {args.override_iters}")
    
    # Run attack
    print(f"\n🚀 Running attack with config: {config_path}")
    print(f"   Configuration: {config.get('name', 'Unnamed')}")
    
    try:
        run_dir = engine.run_attack(config)
        
        print(f"\n✅ Attack complete!")
        print(f"   Results: {run_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Attack interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n\n❌ Attack failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
