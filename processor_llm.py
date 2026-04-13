import os
import re
from huggingface_hub import InferenceClient

# ── Config ─────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")          # Set as HuggingFace Space secret
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

VALID_CATEGORIES = ["Workflow Error", "Deprecation Warning"]

SYSTEM_PROMPT = (
    "You are an enterprise log classifier. "
    "Classify log messages into exactly one category. "
    "Return ONLY the category name — no explanation, no punctuation."
)

FEW_SHOT_EXAMPLES = [
    {
        "log": "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
        "label": "Workflow Error",
    },
    {
        "log": "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' instead.",
        "label": "Deprecation Warning",
    },
    {
        "log": "Invoice generation aborted for order ID 8910 due to invalid tax calculation module.",
        "label": "Workflow Error",
    },
]


def _build_messages(log_msg: str) -> list[dict]:
    categories_str = ", ".join(f'"{c}"' for c in VALID_CATEGORIES)

    user_content = (
        f"Classify the following log into one of these categories: {categories_str}.\n"
        "If none fits, return \"Unclassified\".\n\n"
    )

    # Add few-shot examples
    for ex in FEW_SHOT_EXAMPLES:
        user_content += f'Log: {ex["log"]}\nCategory: {ex["label"]}\n\n'

    user_content += f"Log: {log_msg}\nCategory:"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def classify_with_llm(log_msg: str) -> str:
    """
    Tier 3: LLM-based classifier using HuggingFace Inference API.
    Used for LegacyCRM logs where training data is insufficient for ML.
    Latency: 500–2000ms depending on model load.
    """
    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=_build_messages(log_msg),
            max_tokens=15,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] Inference error: {e}")
        return "Unclassified"

    # Normalize response to valid category
    for cat in VALID_CATEGORIES:
        if cat.lower() in raw.lower():
            return cat

    return "Unclassified"


if __name__ == "__main__":
    # Requires HF_TOKEN in environment
    test_logs = [
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
        "The 'ReportGenerator' module will be retired in version 4.0. Migrate to 'AdvancedAnalyticsSuite'.",
        "System reboot initiated by user 12345.",   # should be Unclassified
    ]
    for log in test_logs:
        result = classify_with_llm(log)
        print(f"{result:25s} | {log[:80]}")
