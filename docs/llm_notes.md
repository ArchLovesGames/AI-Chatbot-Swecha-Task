# Local LLM Notes

This project contains a tiny GPT-style language model. The model is trained as
a next-character predictor, which means it learns to guess the next character
from the characters that came before it.

The neural model uses token embeddings, positional embeddings, causal
self-attention, Transformer blocks, layer normalization, and a linear language
model head. Training minimizes cross-entropy loss between predicted next
characters and the real next characters in the text.

Retrieval augmented generation, or RAG, adds a knowledge-retrieval step before
answer generation. Instead of asking the model to answer from its parameters
alone, the system searches a document index for relevant chunks and places those
chunks into the answer context.

RAG is useful because the knowledge base can be updated without retraining the
neural model. It also makes answers easier to inspect because each answer can
cite the retrieved source chunks that supported it.

This implementation uses local TF-IDF retrieval. It tokenizes documents into
words, splits them into overlapping chunks, computes inverse document frequency,
and ranks chunks with cosine similarity. The answer command returns an
extractive response with citations. The prompt mode prints a grounded prompt
that can be passed to a stronger generator.

