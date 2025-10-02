from __future__ import annotations
from typing import Optional
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class HFGenerator:
    def __init__(self, model_name: str, load_4bit: bool=True, device_map: str="auto",
                 max_new_tokens: int=128, temperature: float=0.7, top_p: float=0.95):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if load_4bit else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HUGGING_FACE_HUB_TOKEN", None))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if not load_4bit else None,
            device_map=device_map
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        text = f"<s>[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n[USER]\n{user_prompt}\n[/USER]\n[ASSISTANT]\n"
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "[ASSISTANT]" in gen:
            gen = gen.split("[ASSISTANT]")[-1].strip()
        return gen.strip()
