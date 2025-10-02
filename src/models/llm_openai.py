from __future__ import annotations
import os
from typing import Optional
from openai import OpenAI

class OpenAIChat:
    def __init__(self, model: str, max_new_tokens: int = 192, temperature: float = 0.7, top_p: float = 0.95):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        # Using the Responses API (works for GPT-4.* and o-series)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":user_prompt},
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return resp.choices[0].message.content.strip()
