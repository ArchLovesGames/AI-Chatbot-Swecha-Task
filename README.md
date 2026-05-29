# Tiny Local LLM

A small local language-model playground you can train and run.

It includes two models:

- `train_gpt.py` / `generate_gpt.py`: a tiny GPT-style neural Transformer.
- `tiny_llm.py`: a dependency-free n-gram fallback that runs with only Python.
- `rag_llm.py`: a local retrieval augmented generation layer.

## Quick Start

Create the environment and install PyTorch:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Train the neural model on the included sample corpus:

```bash
.venv/bin/python train_gpt.py --input data/tiny_corpus.txt --steps 400
```

Generate text:

```bash
.venv/bin/python generate_gpt.py --prompt "The model" --tokens 300
```

Use a larger text file for better results:

```bash
.venv/bin/python train_gpt.py --input path/to/book.txt --steps 3000 --output runs/book
.venv/bin/python generate_gpt.py --checkpoint runs/book/checkpoint.pt --prompt "Once upon" --tokens 800
```

## RAG Mode

Build a local document index:

```bash
.venv/bin/python rag_llm.py index --documents docs --index rag_index.json
```

Ask a grounded question:

```bash
.venv/bin/python rag_llm.py ask "Why is RAG useful?"
```

Inspect retrieved chunks:

```bash
.venv/bin/python rag_llm.py retrieve "Transformer attention"
```

Print a RAG prompt that can be passed to a stronger generator:

```bash
.venv/bin/python rag_llm.py ask "How does this project train?" --prompt
```

Index your own notes:

```bash
.venv/bin/python rag_llm.py index --documents path/to/notes --extensions .txt,.md,.py
```

## Dependency-Free Fallback

Train the n-gram model:

```bash
python3 tiny_llm.py train --input data/tiny_corpus.txt --model model.json
```

Generate text:

```bash
python3 tiny_llm.py generate --model model.json --prompt "The model" --tokens 300
```

## How It Works

The neural model is a compact character-level GPT:

- Token embedding + positional embedding.
- Causal multi-head self-attention.
- Transformer blocks with layer norm and MLPs.
- Cross-entropy next-character training.
- Temperature and top-k sampling for generation.

The fallback model is a character-level n-gram language model:

- It treats each character as a token.
- During training, it counts which characters tend to follow each context.
- During generation, it predicts the next character from the longest matching
  recent context, backing off to shorter contexts when needed.
- Temperature controls randomness. Lower values are more predictable; higher
  values are more chaotic.
- Top-k keeps sampling focused on the most likely next characters.

Both versions show the core language-model loop: tokenize, train next-token
prediction, save weights, and sample completions.

The RAG layer adds the retrieval loop: chunk documents, index them, retrieve
the most relevant chunks for a question, and answer with citations.

## Streamlit Document Chatbot

Run the upload-based chatbot:

```bash
streamlit run streamlit_rag_chatbot.py
```

Upload a PDF, TXT, or Markdown document, then either ask a question or leave the
question box blank to get a summary. The app uses the local TF-IDF retrieval
fallback by default. If Ollama is running locally, turn on "Use Ollama if
available" in the sidebar and choose an installed model such as `llama3.2`.
