#!/usr/bin/env python3
"""
FastAPI inference server for the Qwen3-0.6B customer support ticket router.

Usage:
    uvicorn inference_server:app --host 0.0.0.0 --port 8000
"""

import json
import os
import re
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from training_config import (
    MODEL_ID, FINETUNED_MODEL_ID, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
)

# ── Global State ──────────────────────────────────────────────────────────────

_model = None
_tokenizer = None

SUPPORTED_INTENTS = [
    "cancel_order", "complaint", "contact_customer_service",
    "contact_human_agent", "create_account", "change_order",
    "change_shipping_address", "check_cancellation_fee",
    "check_invoices", "check_payment_methods",
    "check_refund_policy", "delivery_options",
]


# ── Schemas ───────────────────────────────────────────────────────────────────

class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Customer support ticket text")

class BatchRequest(BaseModel):
    tickets: list[str] = Field(..., min_items=1, max_items=50, description="List of ticket texts")

class ClassificationResult(BaseModel):
    intent: str
    confidence: float
    reason: str

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model():
    global _model, _tokenizer
    print(f"🔄 Loading model: {FINETUNED_MODEL_ID}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    try:
        _model = PeftModel.from_pretrained(base, FINETUNED_MODEL_ID)
        print("✅ Loaded fine-tuned LoRA adapter")
    except Exception:
        print("⚠️  LoRA adapter not found, using base model")
        _model = base
    _model.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Support Ticket Router",
    description="Classify customer support tickets using fine-tuned Qwen3-0.6B",
    version="1.0.0",
    lifespan=lifespan,
)


def _classify(text: str) -> ClassificationResult:
    intents_str = ", ".join(SUPPORTED_INTENTS)
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\nValid intents: {intents_str}"},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(instruction=text)},
    ]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs, max_new_tokens=128, temperature=0.1, do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    response = _tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Parse JSON
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]+\}', response)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                parsed = None

    if parsed and all(k in parsed for k in ("intent", "confidence", "reason")):
        return ClassificationResult(
            intent=str(parsed["intent"]),
            confidence=float(parsed.get("confidence", 0.5)),
            reason=str(parsed.get("reason", "")),
        )

    return ClassificationResult(intent="unknown", confidence=0.0, reason=f"Could not parse: {response[:200]}")


@app.get("/health", response_model=HealthResponse)
async def health():
    device = str(next(_model.parameters()).device) if _model else "not loaded"
    return HealthResponse(status="ok", model=FINETUNED_MODEL_ID, device=device)


@app.get("/intents")
async def list_intents():
    return {"intents": SUPPORTED_INTENTS, "count": len(SUPPORTED_INTENTS)}


@app.post("/classify", response_model=ClassificationResult)
async def classify(req: TicketRequest):
    return _classify(req.text)


@app.post("/classify/batch")
async def classify_batch(req: BatchRequest):
    return {"results": [_classify(t) for t in req.tickets]}
