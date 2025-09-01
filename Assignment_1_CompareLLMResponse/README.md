# LLM-Comparison-Project

This project compares responses from two large language models: **OpenAI GPT-3.5** and **Claude 3 Sonnet (Anthropic)**.

---

## 🔍 Features

- Accepts a user prompt
- Queries both GPT-3.5 and Claude-3 Sonnet using their APIs
- Prints outputs from both models
- Generates a Markdown report (`llm_comparison.md`) including:
  - ✅ Prompt
  - ✅ GPT-3.5 Response
  - ✅ Claude Response
  - ✅ Comparison Notes

---

## ⚙️ Setup

1. Clone this repository or copy the project folder.

2. Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate         # On Windows
   source venv/bin/activate     # On macOS/Linux
