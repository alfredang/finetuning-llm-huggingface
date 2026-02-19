<div align="center">

# 🎫 Customer Support Ticket Router

### Fine-tuned Qwen3-0.6B for Intent Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Model-yellow)](https://huggingface.co/tertiaryinfotech/qwen3-0.6b-ticket-router)
[![Qwen](https://img.shields.io/badge/Qwen3-0.6B-purple)](https://huggingface.co/Qwen/Qwen3-0.6B)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

Fine-tune a 0.6B parameter LLM to classify customer support tickets into 12 intent categories — using HuggingFace's cloud training infrastructure for **$0.37**.

[Getting Started](#getting-started) · [Architecture](#architecture) · [Results](#results) · [API](#inference-server)

</div>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│                                                              │
│  bitext/customer-support     HuggingFace Jobs     HuggingFace│
│  -llm-chatbot-training  ──►  (SFT on Qwen3)  ──► Model Hub │
│  -dataset                    GPU Cloud             (LoRA)    │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                    Inference Pipeline                         │
│                                                              │
│  Customer    FastAPI     Qwen3-0.6B    JSON Response         │
│  Ticket  ──► Server ──► + LoRA     ──► {intent,             │
│              :8000       Adapter        confidence,           │
│                                         reason}              │
└─────────────────────────────────────────────────────────────┘
```

## Results

| Metric | Base Model | Fine-tuned | Delta |
|--------|-----------|------------|-------|
| Accuracy | ~5% | ~85%+ | +80% |
| Valid JSON Rate | ~10% | ~98% | +88% |
| Schema Pass Rate | ~8% | ~95% | +87% |
| Accuracy on Schema | ~15% | ~88% | +73% |

> *Metrics are estimates — run `python evaluate.py` after training for actual numbers.*

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Fine-tuning | SFT via HuggingFace Training Jobs |
| Adapter | LoRA (PEFT) |
| Dataset | [bitext/customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/customer-support-llm-chatbot-training-dataset) |
| Inference | FastAPI + Uvicorn |
| Training Cost | **$0.37** |

## Project Structure

```
finetuning-llm-huggingface/
├── README.md                 # This file
├── training_config.py        # All hyperparameters and config
├── evaluate.py               # Evaluation script (base & fine-tuned)
├── inference_server.py       # FastAPI inference server
├── demo.py                   # Demo: compare base vs fine-tuned
├── TRAINING_PROMPT.md        # Prompt for HF model trainer
├── HF_MCP_SETUP.md           # HuggingFace MCP setup guide
├── requirements.txt          # Python dependencies
├── setup.sh                  # Setup script
├── .env.example              # Environment variables template
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.10+
- HuggingFace account with API token
- ~2GB disk space for model weights

### Setup

```bash
# Clone the repo
git clone https://github.com/alfredang/finetuning-llm-huggingface.git
cd finetuning-llm-huggingface

# Set your HuggingFace token
cp .env.example .env
# Edit .env with your token

# Install dependencies
bash setup.sh
```

### Training

Follow `TRAINING_PROMPT.md` — paste the prompt into Claude Code with the HuggingFace model trainer skill. See `HF_MCP_SETUP.md` for detailed setup instructions.

**Cost: ~$0.37** on HuggingFace GPU cloud.

### Evaluation

```bash
# Evaluate base model only (before training)
python evaluate.py --base-only

# Evaluate both models (after training)
python evaluate.py

# Interactive playground
python evaluate.py --playground
```

### Inference Server

```bash
# Start the API server
uvicorn inference_server:app --host 0.0.0.0 --port 8000

# Test it
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I want to cancel my order"}'
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/classify` | Classify a single ticket |
| `POST` | `/classify/batch` | Classify multiple tickets |
| `GET` | `/health` | Health check |
| `GET` | `/intents` | List supported intents |

### Demo

```bash
python demo.py
```

## Cost Breakdown

| Item | Cost |
|------|------|
| HuggingFace Training Job (1x T4 GPU, ~15 min) | $0.37 |
| Dataset | Free |
| Base Model | Free |
| Inference (self-hosted) | Free |
| **Total** | **$0.37** |

## Supported Intents

1. `cancel_order`
2. `complaint`
3. `contact_customer_service`
4. `contact_human_agent`
5. `create_account`
6. `change_order`
7. `change_shipping_address`
8. `check_cancellation_fee`
9. `check_invoices`
10. `check_payment_methods`
11. `check_refund_policy`
12. `delivery_options`

---

<div align="center">

Built with 🤗 HuggingFace · Qwen3 · FastAPI

</div>
