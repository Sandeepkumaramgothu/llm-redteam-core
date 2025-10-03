from __future__ import annotations
# ELI5: This small class talks to OpenAI. You give it a prompt; it returns text.

import os                                   # to read your API key
from typing import Optional                 # for type hints (nice-to-have)
from openai import OpenAI                   # official OpenAI SDK

class OpenAIGenerator:
    def __init__(self, model_name: str, max_new_tokens: int = 128,
                 temperature: float = 0.7, top_p: float = 0.95,
                 api_key: Optional[str] = None):
        self.model = model_name                              # which OpenAI model
        self.max_new_tokens = max_new_tokens                 # how long output can be
        self.temperature = temperature                       # randomness of text
        self.top_p = top_p                                   # top-p sampling
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))  # API client

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []                                        # chat messages list
        if system:                                           # if system hint exists
            messages.append({'role':'system','content':system})
        messages.append({'role':'user','content':prompt})    # the user prompt
        r = self.client.chat.completions.create(             # call OpenAI
            model=self.model, messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature, top_p=self.top_p
        )
        return (r.choices[0].message.content or '').strip()  # return text

    def __call__(self, prompt: str, system: Optional[str] = None) -> str:
        return self.generate(prompt, system)                 # allow calling like a function
