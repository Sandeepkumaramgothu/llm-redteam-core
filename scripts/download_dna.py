"""
Download DNA Dataset
====================
Downloads and prepares the Do-Not-Answer dataset from Hugging Face.
"""

from datasets import load_dataset
from pathlib import Path
import json
import hashlib
import argparse
from collections import Counter


def create_seed_hash(text: str) -> str:
    """Generate MD5 hash for seed."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:10]


def download_dna_dataset(
    output_path: str,
    max_samples: int = None,
    risk_types: list = None
):
    """
    Download and prepare DNA dataset.
    
    Args:
        output_path: Where to save seeds.jsonl
        max_samples: Limit number of samples (None = all)
        risk_types: Filter by risk types (None = all)
    """
    print("📥 Downloading Do-Not-Answer dataset from Hugging Face...\n")
    
    # Load dataset
    ds = load_dataset("LibrAI/do-not-answer")
    
    # Get appropriate split
    if 'train' in ds:
        data = ds['train']
    else:
        split_name = list(ds.keys())[0]
        data = ds[split_name]
        print(f"Using split: {split_name}")
    
    print(f"✓ Downloaded {len(data)} samples")
    
    # Apply max_samples limit
    if max_samples and max_samples < len(data):
        data = data.select(range(max_samples))
        print(f"✓ Limited to {len(data)} samples")
    
    # Convert to our format
    seeds = []
    for idx, item in enumerate(data):
        question = item.get('question', '').strip()
        if not question:
            continue
        
        risk_type = item.get('types_of_harm', 'Other')
        
        # Filter by risk type if specified
        if risk_types and risk_type not in risk_types:
            continue
        
        seed_obj = {
            'seed_seq': idx + 1,
            'Seed': question,
            'seed_hash': create_seed_hash(question),
            'Prompt': question,
            'Adversarial': '',
            'risk_type': risk_type,
            'risk_area': item.get('risk_area', ''),
            'risk_detail': item.get('specific_harms', ''),
            'dataset': 'do-not-answer'
        }
        
        seeds.append(seed_obj)
    
    # Save to JSONL
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open('w', encoding='utf-8') as f:
        for seed in seeds:
            f.write(json.dumps(seed) + '\n')
    
    print(f"\n✅ Saved {len(seeds)} seeds to {output_file}")
    
    # Statistics
    risk_type_counts = Counter(s['risk_type'] for s in seeds)
    risk_area_counts = Counter(s['risk_area'] for s in seeds if s['risk_area'])
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total seeds: {len(seeds)}")
    print(f"   Unique risk types: {len(risk_type_counts)}")
    
    print(f"\n   Top 10 Risk Types:")
    for risk_type, count in risk_type_counts.most_common(10):
        pct = (count / len(seeds)) * 100
        print(f"      {risk_type:<30} {count:4d} ({pct:5.1f}%)")
    
    if risk_area_counts:
        print(f"\n   Top 5 Risk Areas:")
        for risk_area, count in risk_area_counts.most_common(5):
            print(f"      {risk_area:<30} {count:4d}")
    
    # Sample
    if seeds:
        print(f"\n📝 Sample seed:")
        sample = seeds[0]
        print(f"   Seq: {sample['seed_seq']}")
        print(f"   Hash: {sample['seed_hash']}")
        print(f"   Type: {sample['risk_type']}")
        print(f"   Seed: {sample['Seed'][:80]}...")
    
    return len(seeds)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Download DNA dataset")
    parser.add_argument(
        '--output',
        default='/content/drive/MyDrive/llm-redteam-core/data/dna/seeds.jsonl',
        help='Output path for seeds.jsonl'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples (default: all)'
    )
    parser.add_argument(
        '--risk-types',
        nargs='+',
        default=None,
        help='Filter by risk types'
    )
    
    args = parser.parse_args()
    
    download_dna_dataset(
        output_path=args.output,
        max_samples=args.max_samples,
        risk_types=args.risk_types
    )


if __name__ == "__main__":
    main()
