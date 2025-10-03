#!/usr/bin/env python
# ELI5: This script downloads the 'Do-Not-Answer' dataset (DNA) and saves it
# as a simple text file (JSONL) where each line is one seed question.
# We include a stable seed_id: if the dataset already has an id, we use it;
# otherwise we make a short hash from the seed text.

from datasets import load_dataset            # library to download datasets
import json, hashlib, pathlib                # tools to save files and make ids

def make_id(seed_text: str) -> str:          # ELI5: make a short id if needed
    return hashlib.sha1(seed_text.encode('utf-8')).hexdigest()[:10]

def main():
    dna = load_dataset('LibrAI/do-not-answer', split='train')  # download DNA
    out_dir = pathlib.Path('data/dna')                         # where we save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'dna_seeds.jsonl'                     # one seed/line

    with out_path.open('w', encoding='utf-8') as f:            # open file to write
        for r in dna:                                          # go over rows
            seed_text = r['question']                          # original seed
            seed_id = r.get('id') or make_id(seed_text)        # keep dataset id or hash
            risk = r.get('types_of_harm') or r.get('risk_area') or 'Unknown'  # label
            f.write(json.dumps({                               # write one JSON line
                'seed_id': seed_id,                            # <-- include seed_id
                'seed': seed_text,                             # seed text
                'risk': risk,                                  # risk label
            }, ensure_ascii=False) + '\n')

    # ELI5: Print a friendly message with how many seeds we wrote
    print('Wrote:', out_path.as_posix(), 'rows:', sum(1 for _ in out_path.open('r', encoding='utf-8')))

if __name__ == '__main__':
    main()
