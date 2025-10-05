
from src.reward.openassistant_reward import OpenAssistantReward, score_and_label
rm = OpenAssistantReward()
s = rm.score_batch(
    ["Explain safe lab cleanup."],
    ["Always follow approved protocols and dispose chemicals per guidelines."]
)[0]
print("score=", round(s,3), "label=", score_and_label(s))
