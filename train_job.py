#!/usr/bin/env python3
"""
Submit fine-tuning job to HuggingFace AutoTrain / Spaces.
Fine-tunes Qwen3-0.6B on customer support ticket classification.
"""

import os
import json
import time
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from collections import Counter

# ── Config ──
MODEL_ID = "Qwen/Qwen3-0.6B"
FINETUNED_MODEL_ID = "tertiaryinfotech/qwen3-0.6b-ticket-router"
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
TEXT_COLUMN = "instruction"
LABEL_COLUMN = "intent"
TOP_K_INTENTS = 12
TRAIN_ROWS = 5000
MAX_SEQ_LENGTH = 512

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def prepare_dataset():
    """Load dataset, filter top intents, format for SFT training."""
    print("📦 Loading dataset...")
    ds = load_dataset(DATASET_NAME, split="train")
    
    # Get top 12 intents from first 5000 rows
    subset = ds.select(range(min(TRAIN_ROWS, len(ds))))
    intent_counts = Counter(subset[LABEL_COLUMN])
    top_intents = [intent for intent, _ in intent_counts.most_common(TOP_K_INTENTS)]
    print(f"📊 Top {TOP_K_INTENTS} intents: {top_intents}")
    
    # Filter to only top intents
    filtered = subset.filter(lambda x: x[LABEL_COLUMN] in top_intents)
    print(f"📊 Filtered dataset: {len(filtered)} rows")
    
    # Split train/eval
    split = filtered.train_test_split(test_size=60, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    
    # Format as chat-style for SFT
    def format_for_sft(example):
        user_msg = f"Classify this customer support ticket and respond with JSON containing intent, confidence (0-1), and reason.\n\nTicket: {example[TEXT_COLUMN]}"
        assistant_msg = json.dumps({
            "intent": example[LABEL_COLUMN],
            "confidence": 0.95,
            "reason": f"Customer is requesting {example[LABEL_COLUMN].replace('_', ' ')}"
        })
        
        example["text"] = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        return example
    
    train_ds = train_ds.map(format_for_sft)
    eval_ds = eval_ds.map(format_for_sft)
    
    return train_ds, eval_ds, top_intents


def upload_dataset_to_hub(train_ds, eval_ds):
    """Upload formatted dataset to HuggingFace for training."""
    dataset_repo = "tertiaryinfotech/ticket-router-training-data"
    
    print(f"📤 Uploading dataset to {dataset_repo}...")
    api = HfApi()
    
    try:
        create_repo(dataset_repo, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
    except Exception as e:
        print(f"Repo creation: {e}")
    
    # Save locally first
    os.makedirs("/tmp/sft_data", exist_ok=True)
    train_ds.to_json("/tmp/sft_data/train.jsonl")
    eval_ds.to_json("/tmp/sft_data/eval.jsonl")
    
    # Upload
    api.upload_file(
        path_or_fileobj="/tmp/sft_data/train.jsonl",
        path_in_repo="train.jsonl",
        repo_id=dataset_repo,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    api.upload_file(
        path_or_fileobj="/tmp/sft_data/eval.jsonl",
        path_in_repo="eval.jsonl",
        repo_id=dataset_repo,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    
    print(f"✅ Dataset uploaded to https://huggingface.co/datasets/{dataset_repo}")
    return dataset_repo


def create_training_space():
    """Create a HuggingFace Space that runs the fine-tuning job."""
    space_id = "tertiaryinfotech/qwen3-finetune-job"
    api = HfApi()
    
    print(f"🚀 Creating training Space: {space_id}")
    
    try:
        create_repo(
            space_id,
            repo_type="space",
            space_sdk="docker",
            token=HF_TOKEN,
            exist_ok=True,
        )
    except Exception as e:
        print(f"Space creation: {e}")
    
    # Create Dockerfile for training
    dockerfile = '''FROM python:3.11-slim

RUN pip install --no-cache-dir \\
    torch --index-url https://download.pytorch.org/whl/cpu \\
    transformers \\
    datasets \\
    peft \\
    accelerate \\
    trl \\
    huggingface_hub \\
    bitsandbytes

COPY train.py /app/train.py
WORKDIR /app

CMD ["python", "train.py"]
'''
    
    # Create training script
    train_script = '''#!/usr/bin/env python3
"""SFT Fine-tuning of Qwen3-0.6B for ticket classification."""

import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from huggingface_hub import HfApi

# Config
MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_MODEL = "tertiaryinfotech/qwen3-0.6b-ticket-router"
DATASET_REPO = "tertiaryinfotech/ticket-router-training-data"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print("🧠 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    trust_remote_code=True,
)

# LoRA config
print("🔧 Applying LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
print("📦 Loading training data...")
train_ds = load_dataset(DATASET_REPO, data_files="train.jsonl", split="train", token=HF_TOKEN)
eval_ds = load_dataset(DATASET_REPO, data_files="eval.jsonl", split="train", token=HF_TOKEN)

print(f"📊 Train: {len(train_ds)} rows, Eval: {len(eval_ds)} rows")

# Training
print("🏋️ Starting SFT training...")
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    fp16=False,
    report_to="none",
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    max_seq_length=512,
    dataset_text_field="text",
)

trainer.train()

# Save and push
print("📤 Pushing model to HuggingFace...")
model.save_pretrained("./output/final")
tokenizer.save_pretrained("./output/final")

api = HfApi()
api.create_repo(OUTPUT_MODEL, exist_ok=True, token=HF_TOKEN)
api.upload_folder(
    folder_path="./output/final",
    repo_id=OUTPUT_MODEL,
    token=HF_TOKEN,
)

print(f"✅ Model pushed to https://huggingface.co/{OUTPUT_MODEL}")
print("🎉 Fine-tuning complete!")
'''
    
    # Upload files to Space
    os.makedirs("/tmp/hf_space", exist_ok=True)
    
    with open("/tmp/hf_space/Dockerfile", "w") as f:
        f.write(dockerfile)
    
    with open("/tmp/hf_space/train.py", "w") as f:
        f.write(train_script)
    
    api.upload_file(
        path_or_fileobj="/tmp/hf_space/Dockerfile",
        path_in_repo="Dockerfile",
        repo_id=space_id,
        repo_type="space",
        token=HF_TOKEN,
    )
    api.upload_file(
        path_or_fileobj="/tmp/hf_space/train.py",
        path_in_repo="train.py",
        repo_id=space_id,
        repo_type="space",
        token=HF_TOKEN,
    )
    
    print(f"✅ Training Space created: https://huggingface.co/spaces/{space_id}")
    print("⚠️  You need to set HF_TOKEN as a secret in the Space settings")
    print(f"   Go to: https://huggingface.co/spaces/{space_id}/settings")
    print("   Add secret: HF_TOKEN = your_token")
    print("   Then restart the Space to begin training")
    
    return space_id


if __name__ == "__main__":
    import sys
    
    if not HF_TOKEN:
        # Try loading from .env
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        HF_TOKEN = line.strip().split("=", 1)[1]
                        os.environ["HF_TOKEN"] = HF_TOKEN
    
    if not HF_TOKEN:
        print("❌ HF_TOKEN not set. Export it or add to .env")
        sys.exit(1)
    
    # Step 1: Prepare and upload dataset
    train_ds, eval_ds, top_intents = prepare_dataset()
    dataset_repo = upload_dataset_to_hub(train_ds, eval_ds)
    
    # Step 2: Create training Space
    space_id = create_training_space()
    
    print("\n" + "="*60)
    print("📋 NEXT STEPS:")
    print("="*60)
    print(f"1. Go to https://huggingface.co/spaces/{space_id}/settings")
    print(f"2. Add secret: HF_TOKEN")
    print(f"3. Select hardware: CPU basic (free) or GPU for faster training")
    print(f"4. Restart the Space")
    print(f"5. Monitor logs at https://huggingface.co/spaces/{space_id}")
    print(f"6. Model will be pushed to https://huggingface.co/{FINETUNED_MODEL_ID}")
    print("="*60)
