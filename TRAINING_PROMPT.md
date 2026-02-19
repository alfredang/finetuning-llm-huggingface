# Training Prompt

Paste this into Claude Code / Codex with the HuggingFace model trainer skill:

---

Fine-tune Qwen/Qwen3-0.6B into a customer support ticket router using bitext/Bitext-customer-support-llm-chatbot-training-dataset.

Task definition:
- Input column: instruction
- Target label column: intent
- Keep only top 12 intents by frequency in first 5000 train rows
- Training sample format (chat style):
  User: "Classify this customer support ticket: {instruction}"
  Assistant: {"intent": "<intent>", "confidence": <0-1>, "reason": "<short reason>"}

Training strategy: SFT
Max sequence length: 512
Hardware: Suggest best option and calculate cost
Save to: tertiaryinfotech/qwen3-0.6b-ticket-router

After training: run evals on held-out slice, report accuracy, valid JSON rate, schema pass rate, accuracy on schema.
