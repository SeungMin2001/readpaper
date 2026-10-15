"""Resolve common AI compounds locally, without asking the LLM to copy a span.

This deliberately conservative vocabulary contains English phrases, not translations.
Unknown phrases fall back to the clicked token rather than guessing a larger span.
"""

import hashlib
import json
import re
import unicodedata


GLOSS_PROMPT_VERSION = "summary-meaning-only-v2"

_COMPOUNDS = """
multi head attention
scaled dot product attention
dot product attention
self attention
cross attention
attention mechanism
attention mechanisms
attention weight
attention weights
reinforcement learning
supervised learning
unsupervised learning
self supervised learning
semi supervised learning
machine learning
deep learning
transfer learning
contrastive learning
in context learning
few shot learning
zero shot learning
large language model
large language models
masked language model
masked language modeling
language model
language models
neural network
neural networks
convolutional neural network
convolutional neural networks
recurrent neural network
recurrent neural networks
graph neural network
graph neural networks
positional encoding
positional encodings
positional embedding
positional embeddings
word embedding
word embeddings
residual connection
residual connections
skip connection
skip connections
gradient descent
stochastic gradient descent
gradient accumulation
gradient clipping
learning rate
batch normalization
layer normalization
loss function
loss functions
objective function
objective functions
cross entropy
cross entropy loss
policy gradient
policy optimization
proximal policy optimization
value function
reward model
reward function
experience replay
beam search
maximum likelihood
maximum likelihood estimation
knowledge distillation
retrieval augmented generation
low rank adaptation
mixture of experts
support vector machine
support vector machines
diffusion model
diffusion models
generative adversarial network
generative adversarial networks
natural language processing
computer vision
chain of thought
test time scaling
key value cache
kv cache
"""
_PHRASES = {tuple(line.split()) for line in _COMPOUNDS.splitlines() if line.strip()}
_MAX_PARTS = max(map(len, _PHRASES))


def clean_term(text: str) -> str:
    # Preserve case, internal punctuation and C++/C#; they can change a term's meaning.
    text = unicodedata.normalize("NFKC", text).strip()
    return re.sub(r"^[^\w]+|[^\w+#]+$", "", text)


def resolve_term(tokens: list[str], focus: int) -> dict:
    if not tokens or not 0 <= focus < len(tokens):
        raise ValueError("Invalid clicked word position")
    best = (focus, focus + 1)
    for start in range(max(0, focus - _MAX_PARTS + 1), focus + 1):
        for end in range(focus + 1, min(len(tokens), start + _MAX_PARTS) + 1):
            # Never group across punctuation that separates clauses or phrases.
            if any(re.search(r"[,;:!?.)\]]$", token) for token in tokens[start:end - 1]):
                continue
            text = " ".join(clean_term(token) for token in tokens[start:end])
            parts = tuple(re.findall(r"[^\W_]+", text.casefold()))
            if parts in _PHRASES and end - start > best[1] - best[0]:
                best = (start, end)
    start, end = best
    return {"start": start, "end": end,
            "term": " ".join(clean_term(token) for token in tokens[start:end])}


def gloss_cache_key(summary: str, term: str, *, model: str, endpoint: str) -> str:
    """Only reuse results with exactly the same meaning-bearing model inputs."""
    return hashlib.sha256(json.dumps(
        [GLOSS_PROMPT_VERSION, endpoint, model, summary, term], ensure_ascii=False,
    ).encode()).hexdigest()
