#!/usr/bin/env python3
"""
Demo script — compare base vs fine-tuned model on sample tickets.
"""

import torch
from training_config import MODEL_ID, FINETUNED_MODEL_ID
from evaluate import classify_ticket, parse_response, validate_schema, load_base_model, load_finetuned_model

SAMPLE_TICKETS = [
    "I want to cancel my order #12345, it hasn't shipped yet",
    "How do I change my shipping address for a pending delivery?",
    "I'd like to know what payment methods you accept",
    "Can you help me create a new account on your platform?",
    "I need to speak with a real person about my issue",
    "What is your refund policy for digital products?",
    "I want to file a complaint about rude customer service",
    "What delivery options do you have for international shipping?",
]

INTENTS = [
    "cancel_order", "complaint", "contact_customer_service",
    "contact_human_agent", "create_account", "change_order",
    "change_shipping_address", "check_cancellation_fee",
    "check_invoices", "check_payment_methods",
    "check_refund_policy", "delivery_options",
]


def run_demo(model, tokenizer, label: str):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    for ticket in SAMPLE_TICKETS:
        response = classify_ticket(model, tokenizer, ticket, INTENTS)
        parsed = parse_response(response)
        if parsed and validate_schema(parsed, INTENTS):
            intent = parsed["intent"]
            conf = parsed["confidence"]
            reason = parsed["reason"]
            print(f"\n  🎫 \"{ticket[:60]}...\"")
            print(f"     → {intent} (confidence: {conf}) — {reason}")
        else:
            print(f"\n  🎫 \"{ticket[:60]}...\"")
            print(f"     ⚠️  Raw: {response[:120]}")


def main():
    print("🚀 Customer Support Ticket Router — Demo\n")

    # Base model
    model, tokenizer = load_base_model()
    run_demo(model, tokenizer, "Base Model (Qwen3-0.6B)")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Fine-tuned
    try:
        model, tokenizer = load_finetuned_model()
        run_demo(model, tokenizer, "Fine-tuned Model (LoRA Adapter)")
    except Exception as e:
        print(f"\n⚠️  Fine-tuned model not available: {e}")
        print("   Train the model first using HuggingFace model trainer.")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
