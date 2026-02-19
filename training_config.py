"""
Training configuration for Qwen3-0.6B customer support ticket router.
"""

# Model
MODEL_ID = "Qwen/Qwen3-0.6B"
FINETUNED_MODEL_ID = "tertiaryinfotech/qwen3-0.6b-ticket-router"

# Dataset
DATASET_NAME = "bitext/customer-support-llm-chatbot-training-dataset"
TEXT_COLUMN = "instruction"
LABEL_COLUMN = "intent"

# Data filtering
TOP_K_INTENTS = 12
TRAIN_ROWS = 5000
EVAL_SIZE = 60

# Training
MAX_SEQ_LENGTH = 512
TRAINING_METHOD = "SFT"

# Output schema
OUTPUT_SCHEMA = {
    "intent": "<string: one of the TOP_K intents>",
    "confidence": "<float: 0.0 to 1.0>",
    "reason": "<string: brief explanation>"
}

# System prompt used during training and inference
SYSTEM_PROMPT = "You are a customer support ticket classifier. Respond ONLY with valid JSON."

USER_PROMPT_TEMPLATE = 'Classify this customer support ticket:\n"{instruction}"'

ASSISTANT_TEMPLATE = '{{"intent": "{intent}", "confidence": {confidence}, "reason": "{reason}"}}'

# Supported intents (populated after dataset analysis)
# These are the top 12 by frequency in the first 5000 rows
INTENTS = []
