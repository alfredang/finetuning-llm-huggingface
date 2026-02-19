# HuggingFace MCP Setup Guide

Step-by-step guide to set up HuggingFace model training with Claude Code.

## 1. Create HuggingFace Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with these permissions:
   - **Write** — push models/datasets
   - **Jobs** — create and manage training jobs
3. Copy the token

## 2. Install HuggingFace CLI

```bash
pip install "huggingface_hub[cli]"
```

## 3. Authenticate

```bash
huggingface-cli login --token YOUR_TOKEN
# Or set env var:
export HF_TOKEN=YOUR_TOKEN
```

## 4. Install HF MCP Server for Claude Code

Add to your Claude Code MCP config (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "huggingface": {
      "command": "npx",
      "args": ["-y", "@anthropic/huggingface-mcp"],
      "env": {
        "HF_TOKEN": "YOUR_TOKEN"
      }
    }
  }
}
```

## 5. Install the Model Trainer Skill

In Claude Code, run:
```
/install-skill huggingface-model-trainer
```

Or manually add the skill by following the [HuggingFace MCP documentation](https://huggingface.co/docs/mcp).

## 6. Run the Training Prompt

Open `TRAINING_PROMPT.md` and paste the prompt into Claude Code. The HF model trainer will:
1. Prepare the dataset (filter top 12 intents, format as chat)
2. Select hardware (typically Nvidia T4 or A10G)
3. Launch SFT training job on HuggingFace
4. Save the LoRA adapter to `tertiaryinfotech/qwen3-0.6b-ticket-router`

Expected cost: ~$0.37 for a single training run.

## 7. Verify

```bash
# Check the model exists
huggingface-cli repo info tertiaryinfotech/qwen3-0.6b-ticket-router

# Run evaluation
python evaluate.py
```
