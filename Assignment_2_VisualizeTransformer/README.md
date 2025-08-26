# Visualize Transformer Forward Pass on a Sample Sentence

## Objective
- Use HuggingFace's `bert-base-uncased` model to tokenize and pass a sentence through BERT.
- Extract hidden states and attention.
- Visualize attention heads using BertViz.
- Document tensor shapes and data flow.

## Setup
1. Install dependencies:
2. Use VS Code with the Jupyter extension or run `jupyter notebook` to open the notebook.

## Files
- `bert_visualize.ipynb`: Jupyter notebook with code and visualization.

## How to Run
- Open `bert_visualize.ipynb` in VS Code or Jupyter.
- Run cells step-by-step.
- The last cell will display an interactive attention visualization.

## Key Points
- Tokenized sentence: `"Transformers are amazing!"`
- Input shape: `(1, 5)` tokens including special tokens.
- Hidden states: 13 layers (embedding + 12 transformer layers), each `(1, 5, 768)`.
- Attention tensors: 12 layers × 12 heads, shape `(1, 12, 5, 5)`.

## Common Issues
- **NameError**: Make sure to import modules before using.
- **Widget errors**: Install `ipywidgets` and restart VS Code.
- **Visualization errors**: Pass attention tensors and token list to `head_view()`.

---

