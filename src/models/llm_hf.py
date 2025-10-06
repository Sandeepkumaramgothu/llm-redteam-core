
from __future__ import annotations
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class HFGenerator:
 """FP16 HF text-generation; no 4-bit / bitsandbytes required."""
 def __init__(self,
              model_name: str,
              max_new_tokens: int = 256,
              temperature: float = 0.7,
              top_p: float = 0.95,
              device_map: str = "auto"):
     self.model_name = model_name
     self.max_new_tokens = max_new_tokens
     self.temperature = temperature
     self.top_p = top_p

     token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
     tok_kwargs = {"token": token} if token else {}

     self.tok = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
     self.model = AutoModelForCausalLM.from_pretrained(
         model_name,
         torch_dtype=torch.float16,
         device_map=device_map,
         **tok_kwargs
     )
     print(f"[HFGenerator] loaded {model_name} (fp16; device_map={device_map})")
     self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tok)

 def generate(self, prompt: str) -> str:
     out = self.pipe(
         prompt,
         max_new_tokens=self.max_new_tokens,
         do_sample=True,
         temperature=self.temperature,
         top_p=self.top_p,
         pad_token_id=self.tok.eos_token_id
     )[0]["generated_text"]
     return out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
