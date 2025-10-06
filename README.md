
# llm-redteam-core (Colab-friendly, FP16)

This is a minimal, dependable pipeline for *visual red teaming* a target LLM:
- “Red” model (OpenAI) rewrites benign seeds into **adversarial prompts**
- Target model (Hugging Face, Llama 3 1B Instruct) responds
- We **rescore** the responses with a robust reward model + optional Detoxify
- We build **metrics** per iteration (ASR, average toxicity, n)

### Why FP16 (no bitsandbytes/triton)?
Colab often breaks on `bitsandbytes`/`triton`. We load a 1B target in **FP16**, which is reliable.

### Quick start (in Colab)
1. Run the “setup” cell (writes this repo to `/content/drive/MyDrive/llm-redteam-core`).
2. Install deps: `pip install -r requirements.txt`
3. Set tokens in the notebook cell:
   - `OPENAI_API_KEY` (OpenAI)
   - `HUGGING_FACE_HUB_TOKEN` (HF access)
4. Generate 2×20: