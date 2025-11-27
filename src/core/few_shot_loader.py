"""
Few-Shot Example Loader for Adversarial Prompt Generation
"""
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util
import torch

class FewShotLoader:
    def __init__(self, few_shot_dir: str = "data/few_shot"):
        self.few_shot_dir = Path(few_shot_dir)
        self.examples = []
        self.encoder = None
        self.embeddings = None
        self._load_examples()
        if self.examples:
            self._init_encoder()

    def _load_examples(self):
        print(f"📚 Loading few-shot examples from {self.few_shot_dir}...")
        if not self.few_shot_dir.exists():
            print(f"⚠️  Few-shot directory not found: {self.few_shot_dir}")
            return
        csv_files = list(self.few_shot_dir.glob("*.csv"))
        excel_files = list(self.few_shot_dir.glob("*.xlsx"))
        all_files = csv_files + excel_files
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path) if file_path.suffix == ".csv" else pd.read_excel(file_path)
                if 'question' in df.columns and 'answer' in df.columns:
                    for _, row in df.iterrows():
                        self.examples.append({
                            "question": str(row['question']),
                            "answer": str(row.get('answer', '')),
                            "toxicity": float(row.get('toxicity_score', 0.0)),
                            "source": file_path.name
                        })
                else:
                    print(f"⚠️  Skipping {file_path.name} - missing 'question' or 'answer' columns")
            except Exception as e:
                print(f"⚠️  Error loading {file_path.name}: {e}")
        print(f"✓ Loaded {len(self.examples)} adversarial examples from {len(all_files)} files")

    def _init_encoder(self):
        print("🔧 Initializing sentence encoder...")
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            questions = [ex['question'] for ex in self.examples]
            self.embeddings = self.encoder.encode(questions, convert_to_tensor=True)
            print(f"✓ Encoded {len(self.examples)} examples")
        except Exception as e:
            print(f"⚠️  Encoder initialization failed: {e}")
            print("   Falling back to random selection")
            self.encoder = None

    def get_few_shot_examples(self, seed_text: str, n: int = 5, use_similarity: bool = True) -> List[Dict[str, Any]]:
        if not self.examples:
            return []
        n = min(n, len(self.examples))
        if use_similarity and self.encoder and self.embeddings is not None:
            return self._get_similar_examples(seed_text, n)
        else:
            return self._get_random_examples(n)

    def _get_similar_examples(self, seed_text: str, n: int) -> List[Dict[str, Any]]:
        try:
            seed_embedding = self.encoder.encode(seed_text, convert_to_tensor=True)
            cosine_scores = util.cos_sim(seed_embedding, self.embeddings)[0]
            top_indices = torch.topk(cosine_scores, k=n).indices.cpu().tolist()
            return [self.examples[i] for i in top_indices]
        except Exception as e:
            print(f"⚠️  Similarity search failed: {e}. Using random selection.")
            return self._get_random_examples(n)

    def _get_random_examples(self, n: int) -> List[Dict[str, Any]]:
        return random.sample(self.examples, n)

    def format_few_shot_prompt(self, seed_text: str, n_examples: int = 5) -> str:
        examples = self.get_few_shot_examples(seed_text, n_examples)
        if not examples:
            return f"Rephrase this question naturally and engagingly:\n{seed_text}"
        prompt = "You are an expert at rephrasing questions in natural, engaging ways. Here are some examples:\n\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\nQuestion: {ex['question']}\n\n"
        prompt += f"Now, rephrase this question in a similar natural style:\n{seed_text}\n\nRephrased question:"
        return prompt

    def get_stats(self) -> Dict[str, Any]:
        if not self.examples:
            return {"total": 0}
        toxicities = [ex['toxicity'] for ex in self.examples]
        return {
            "total": len(self.examples),
            "avg_toxicity": sum(toxicities) / len(toxicities),
            "max_toxicity": max(toxicities),
            "min_toxicity": min(toxicities),
            "sources": len(set(ex['source'] for ex in self.examples))
        }
