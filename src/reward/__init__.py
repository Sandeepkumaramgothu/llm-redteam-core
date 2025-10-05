from __future__ import annotations
import os
from .openassistant_reward import OpenAssistantReward, label_from_score, DEFAULT_REWARD_ID
from .aux_detoxify import DetoxifyAux

def get_reward(cfg=None):
    model_id = (cfg or {}).get("reward", {}).get("reward_model_id", DEFAULT_REWARD_ID)
    token    = (cfg or {}).get("auth",   {}).get("hf_token") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return OpenAssistantReward(model_id=model_id, hf_token=token)
