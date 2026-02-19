#!/bin/bash
set -euo pipefail

echo "🔧 Setting up Qwen3-0.6B Ticket Router project..."

# Install dependencies
pip install -r requirements.txt

# Authenticate with HuggingFace
if [ -z "${HF_TOKEN:-}" ]; then
    echo "⚠️  HF_TOKEN not set. Set it in .env or export HF_TOKEN=your_token"
    echo "   Then run: huggingface-cli login --token \$HF_TOKEN"
else
    huggingface-cli login --token "$HF_TOKEN"
    echo "✅ Authenticated with HuggingFace"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run evaluation:  python evaluate.py --base-only"
echo "  2. Train model:     Use HuggingFace model trainer (see TRAINING_PROMPT.md)"
echo "  3. Evaluate both:   python evaluate.py"
echo "  4. Start server:    uvicorn inference_server:app --host 0.0.0.0 --port 8000"
