# Cell 33
"""
Red Model Module - Supports OpenAI and HuggingFace models
CORRECTED VERSION - Proper indentation and system prompt support
"""
from pathlib import Path
import json
from openai import OpenAI
from transformers import pipeline
import torch

# ✅ Import few-shot loader if available
try:
    from src.core.few_shot_loader import FewShotLoader
    FEWSHOT_AVAILABLE = True
except ImportError:
    FEWSHOT_AVAILABLE = False
    print("⚠️  Few-shot loader not available (will be created later)")

class RedModel:
    """
    Red Model wrapper supporting OpenAI and HuggingFace models.

    Features:
    - OpenAI API support (GPT-3.5, GPT-4, etc.)
    - HuggingFace model support
    - Few-shot prompting
    - System prompt support
    """

    def __init__(self, config):
        """
        Initialize Red Model.

        Args:
            config: Configuration dict with red model settings
        """
        self.config = config
        self.model_name = config.get('red', {}).get('red_model', 'gpt-4o-mini')
        self.source = config.get('red', {}).get('source', 'openai')

        # ✅ Initialize few-shot loader if enabled
        use_few_shot = config.get('red', {}).get('use_few_shot', False)
        self.few_shot_loader = None

        if use_few_shot and FEWSHOT_AVAILABLE:
            # Get absolute path
            PROJECT_ROOT = Path("/content/drive/MyDrive/llm-redteam-core")
            few_shot_dir = config.get('red', {}).get('few_shot_dir', 'data/few_shot')
            few_shot_dir = Path(few_shot_dir)

            # Make absolute if relative
            if not few_shot_dir.is_absolute():
                few_shot_dir = PROJECT_ROOT / few_shot_dir

            try:
                self.few_shot_loader = FewShotLoader(str(few_shot_dir))
                stats = self.few_shot_loader.get_stats()
                print(f"   📚 Few-shot: {stats.get('total', 0)} examples loaded")
            except Exception as e:
                print(f"   ⚠️  Few-shot loader failed: {e}")
                self.few_shot_loader = None

        # Load OpenAI or HuggingFace model accordingly
        if self.source == 'openai':
            secrets_file = Path("/content/drive/MyDrive/llm-redteam-core/.secrets.json")
            if secrets_file.exists():
                secrets = json.loads(secrets_file.read_text())
                self.api_key = secrets.get("OPENAI_API_KEY", "")
            else:
                self.api_key = ''
            self.client = OpenAI(api_key=self.api_key)

        elif self.source == 'hf':
            self.pipeline = pipeline('text-generation', model=self.model_name)
        else:
            raise ValueError(f"Unknown red model source: {self.source}")

    def generate(self, prompt: str, use_few_shot: bool = None) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt (seed text to rephrase)
            use_few_shot: Override config to use/not use few-shot

        Returns:
            Generated adversarial prompt
        """
        # Determine if using few-shot
        if use_few_shot is None:
            use_few_shot = self.few_shot_loader is not None

        # ✅ Format with few-shot examples if available
        if use_few_shot and self.few_shot_loader:
            n_examples = self.config.get('red', {}).get('few_shot_n', 5)
            prompt = self.few_shot_loader.format_few_shot_prompt(prompt, n_examples)

        # Generate as usual
        if self.source == 'openai':
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.9
            )
            return response.choices[0].message.content.strip()
        else:
            output = self.pipeline(prompt, max_new_tokens=128, do_sample=True)
            return output[0]['generated_text'].replace(prompt, '').strip()

    def generate_with_system(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate with separate system and user prompts (OpenAI only).

        Args:
            system_prompt: System instruction
            user_prompt: User message

        Returns:
            Generated text
        """
        if self.source == 'openai':
            # ✅ Check if using few-shot
            if self.few_shot_loader:
                n_examples = self.config.get('red', {}).get('few_shot_n', 5)
                user_prompt = self.few_shot_loader.format_few_shot_prompt(user_prompt, n_examples)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.9
            )
            return response.choices[0].message.content.strip()
        else:
            # HuggingFace: combine system + user
            combined = f"{system_prompt}\n\n{user_prompt}"
            return self.generate(combined, use_few_shot=False)  # ✅ Avoid double few-shot
