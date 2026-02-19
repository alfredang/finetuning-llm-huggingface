#!/usr/bin/env python3
"""
Evaluation script for Qwen3-0.6B customer support ticket router.

Usage:
    python evaluate.py                  # Run full evaluation (base + fine-tuned)
    python evaluate.py --base-only      # Only evaluate base model
    python evaluate.py --finetuned-only # Only evaluate fine-tuned model
    python evaluate.py --playground     # Interactive playground mode
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from training_config import (
    MODEL_ID, FINETUNED_MODEL_ID, DATASET_NAME,
    TEXT_COLUMN, LABEL_COLUMN, TOP_K_INTENTS,
    TRAIN_ROWS, EVAL_SIZE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
)


# ── Data Preparation ──────────────────────────────────────────────────────────

def load_and_prepare_data():
    """Load dataset, find top-K intents, return eval pool."""
    print("📦 Loading dataset...")
    ds = load_dataset(DATASET_NAME, split="train")
    subset = ds.select(range(min(TRAIN_ROWS, len(ds))))

    # Top K intents by frequency
    counter = Counter(subset[LABEL_COLUMN])
    top_intents = [intent for intent, _ in counter.most_common(TOP_K_INTENTS)]
    print(f"🏷️  Top {TOP_K_INTENTS} intents: {top_intents}")

    # Filter to top intents only
    filtered = subset.filter(lambda x: x[LABEL_COLUMN] in top_intents)

    # Stratified eval pool: EVAL_SIZE / TOP_K_INTENTS per intent
    per_intent = EVAL_SIZE // TOP_K_INTENTS
    eval_rows = []
    intent_counts = {i: 0 for i in top_intents}
    for row in filtered:
        intent = row[LABEL_COLUMN]
        if intent_counts[intent] < per_intent:
            eval_rows.append(row)
            intent_counts[intent] += 1
        if len(eval_rows) >= EVAL_SIZE:
            break

    print(f"✅ Eval pool: {len(eval_rows)} rows ({per_intent} per intent)")
    return eval_rows, top_intents


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_base_model():
    """Load base Qwen3-0.6B model and tokenizer."""
    print(f"🔄 Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model():
    """Load fine-tuned model with LoRA adapter."""
    print(f"🔄 Loading fine-tuned model: {FINETUNED_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_ID)
    model.eval()
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────

def classify_ticket(model, tokenizer, instruction: str, intents: list[str]) -> str:
    """Run a single classification inference."""
    intents_str = ", ".join(intents)
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\nValid intents: {intents_str}"},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(instruction=instruction)},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def parse_response(response: str) -> Optional[dict]:
    """Try to extract JSON from model response."""
    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON block
    match = re.search(r'\{[^{}]+\}', response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def validate_schema(parsed: dict, intents: list[str]) -> bool:
    """Check if parsed JSON matches expected schema."""
    if not isinstance(parsed, dict):
        return False
    if "intent" not in parsed or "confidence" not in parsed or "reason" not in parsed:
        return False
    if not isinstance(parsed.get("confidence"), (int, float)):
        return False
    if not (0 <= parsed["confidence"] <= 1):
        return False
    return True


def evaluate_model(model, tokenizer, eval_rows, intents, label="Model"):
    """Run evaluation and return metrics dict."""
    print(f"\n{'='*60}")
    print(f"📊 Evaluating: {label}")
    print(f"{'='*60}")

    total = len(eval_rows)
    correct = 0
    valid_json = 0
    schema_pass = 0
    correct_on_schema = 0

    for i, row in enumerate(eval_rows):
        instruction = row[TEXT_COLUMN]
        true_intent = row[LABEL_COLUMN]

        response = classify_ticket(model, tokenizer, instruction, intents)
        parsed = parse_response(response)

        is_valid = parsed is not None
        is_schema = is_valid and validate_schema(parsed, intents)
        predicted_intent = parsed.get("intent", "") if is_valid else ""
        is_correct = predicted_intent == true_intent

        if is_valid:
            valid_json += 1
        if is_schema:
            schema_pass += 1
        if is_correct:
            correct += 1
            if is_schema:
                correct_on_schema += 1

        status = "✅" if is_correct else "❌"
        print(f"  [{i+1:3d}/{total}] {status} true={true_intent:<30s} pred={predicted_intent:<30s}")

    metrics = {
        "accuracy": correct / total if total else 0,
        "valid_json_rate": valid_json / total if total else 0,
        "schema_pass_rate": schema_pass / total if total else 0,
        "accuracy_on_schema": correct_on_schema / schema_pass if schema_pass else 0,
        "total": total,
        "correct": correct,
        "valid_json": valid_json,
        "schema_pass": schema_pass,
        "correct_on_schema": correct_on_schema,
    }

    print(f"\n📈 Results for {label}:")
    print(f"   Accuracy:           {metrics['accuracy']:.1%} ({correct}/{total})")
    print(f"   Valid JSON rate:    {metrics['valid_json_rate']:.1%} ({valid_json}/{total})")
    print(f"   Schema pass rate:   {metrics['schema_pass_rate']:.1%} ({schema_pass}/{total})")
    print(f"   Accuracy on schema: {metrics['accuracy_on_schema']:.1%} ({correct_on_schema}/{schema_pass if schema_pass else 1})")

    return metrics


def print_comparison(base_metrics, ft_metrics):
    """Print before/after comparison table."""
    print(f"\n{'='*60}")
    print("📊 Before vs After Fine-Tuning")
    print(f"{'='*60}")
    print(f"{'Metric':<25s} {'Base':>10s} {'Fine-tuned':>12s} {'Delta':>10s}")
    print("-" * 60)

    for key, label in [
        ("accuracy", "Accuracy"),
        ("valid_json_rate", "Valid JSON Rate"),
        ("schema_pass_rate", "Schema Pass Rate"),
        ("accuracy_on_schema", "Acc. on Schema"),
    ]:
        b = base_metrics[key]
        f = ft_metrics[key]
        d = f - b
        sign = "+" if d >= 0 else ""
        print(f"  {label:<23s} {b:>9.1%} {f:>11.1%} {sign}{d:>8.1%}")

    print("-" * 60)


# ── Playground ────────────────────────────────────────────────────────────────

def playground(model, tokenizer, intents):
    """Interactive playground for testing individual tickets."""
    print(f"\n{'='*60}")
    print("🎮 Playground Mode — type a ticket, get a classification")
    print("   Type 'quit' to exit")
    print(f"{'='*60}")
    print(f"   Intents: {', '.join(intents)}\n")

    while True:
        try:
            ticket = input("🎫 Ticket> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not ticket or ticket.lower() in ("quit", "exit", "q"):
            break

        response = classify_ticket(model, tokenizer, ticket, intents)
        parsed = parse_response(response)

        if parsed and validate_schema(parsed, intents):
            print(f"   📋 Intent:     {parsed['intent']}")
            print(f"   📊 Confidence: {parsed['confidence']}")
            print(f"   💬 Reason:     {parsed['reason']}\n")
        else:
            print(f"   ⚠️  Raw response: {response}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ticket router models")
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--finetuned-only", action="store_true", help="Only evaluate fine-tuned model")
    parser.add_argument("--playground", action="store_true", help="Interactive playground")
    args = parser.parse_args()

    eval_rows, intents = load_and_prepare_data()

    if args.playground:
        try:
            model, tokenizer = load_finetuned_model()
            label = "fine-tuned"
        except Exception:
            print("⚠️  Fine-tuned model not found, using base model")
            model, tokenizer = load_base_model()
            label = "base"
        print(f"Using {label} model for playground")
        playground(model, tokenizer, intents)
        return

    base_metrics = None
    ft_metrics = None

    if not args.finetuned_only:
        model, tokenizer = load_base_model()
        base_metrics = evaluate_model(model, tokenizer, eval_rows, intents, "Base (Qwen3-0.6B)")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not args.base_only:
        try:
            model, tokenizer = load_finetuned_model()
            ft_metrics = evaluate_model(model, tokenizer, eval_rows, intents, "Fine-tuned (LoRA)")
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"\n⚠️  Could not load fine-tuned model: {e}")
            print("   Run training first, then re-evaluate with --finetuned-only")

    if base_metrics and ft_metrics:
        print_comparison(base_metrics, ft_metrics)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
