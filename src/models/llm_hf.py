from __future__ import annotations
# ELI5: This is the "local model" helper (Hugging Face). You give it a prompt
# and it gives you text back.

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# We *optionally* use 4-bit to save GPU memory. If bitsandbytes isn't around, we fall back safely.
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False

class HFGenerator:
    def __init__(self,
                 model_name: str,
                 load_4bit: bool = True,
                 device_map: str = "auto",
                 max_new_tokens: int = 128,
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 auth: Optional[str] = None  # ELI5: accept token if user passes it (we ignore if unused)
                 ):
        # Save small knobs (how long, how random, etc.)
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        # ELI5: If we have bitsandbytes and a GPU, we can load in 4-bit (uses less memory).
        quant = None
        if load_4bit and _HAS_BNB and torch.cuda.is_available():
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        # ELI5: Some transformers versions accept token=..., older accept use_auth_token=...
        tok_kwargs = {}
        mdl_kwargs = {"device_map": device_map}
        if quant is not None:
            mdl_kwargs["quantization_config"] = quant
            mdl_kwargs["torch_dtype"] = torch.bfloat16
        else:
            mdl_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Try new 'token=' first, fall back to legacy 'use_auth_token=True'
        def _tok_load(name):
            if auth:
                try:
                    return AutoTokenizer.from_pretrained(name, token=auth)
                except TypeError:
                    return AutoTokenizer.from_pretrained(name, use_auth_token=True)
            return AutoTokenizer.from_pretrained(name)

        def _mdl_load(name, **kwargs):
            if auth:
                try:
                    return AutoModelForCausalLM.from_pretrained(name, token=auth, **kwargs)
                except TypeError:
                    return AutoModelForCausalLM.from_pretrained(name, use_auth_token=True, **kwargs)
            return AutoModelForCausalLM.from_pretrained(name, **kwargs)

        self.tok = _tok_load(model_name)
        self.model = _mdl_load(model_name, **mdl_kwargs)

        # ELI5: Build a text-generation pipeline (don't pass a 'device' when using device_map='auto').
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tok,
            device_map=device_map
        )

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        # ELI5: If you give a "system" hint, we just stick it before the prompt.
        full = f"{system}\n\n{prompt}" if system else prompt
        out = self.pipe(
            full,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        text = out[0]["generated_text"]
        # ELI5: Return only what the model added (not the prompt itself), if we can detect it.
        return text[len(full):].strip() if text.startswith(full) else text.strip()

    def __call__(self, prompt: str, system: Optional[str] = None) -> str:
        return self.generate(prompt, system)
