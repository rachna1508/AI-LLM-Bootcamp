#  Prompt-Based Summarization Tool (Offline - Transformers)

This is a command-line summarization tool that uses **open-source models** (e.g., `facebook/bart-large-cnn`) from the Hugging Face Transformers library to summarize news articles or other text files **without requiring the OpenAI API**.

No API key or internet access required after downloading the model  
Works completely offline   
No billing or quota issues 

---

##  Features

- Summarize news articles and long texts using HuggingFace models.
- Customize summary with `max_length`, `min_length`, and sampling.
- Save the summary to a `.txt` file.
- CLI-based — simple and fast.

---

## 🛠️ Requirements

Make sure Python 3.8+ is installed.

Install dependencies:

```bash
pip install -r requirements.txt
