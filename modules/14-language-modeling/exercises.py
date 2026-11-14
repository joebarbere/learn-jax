"""Module 14 exercises: language models from counting to a small neural network.

Replace each `raise NotImplementedError` with your code, then run `make test N=14`.
Use jax.numpy (jnp) so everything runs on the GPU and works under jax.jit / jax.grad.
Token ids are integers in [0, vocab); a "text" is a 1-D integer array of ids.
"""
import jax  # noqa: F401  (jax.vmap, jax.lax.dynamic_slice, jax.random)
import jax.numpy as jnp  # noqa: F401


def bigram_counts(ids, vocab: int):
    """Count how often token b follows token a: a (vocab, vocab) int32 array C with
    C[a, b] = number of positions i where ids[i] == a and ids[i + 1] == b.

    No Python loops over positions: build the (previous, next) index arrays and use
    jnp.zeros(...).at[rows, cols].add(1). A text of length n has n - 1 pairs.
    """
    raise NotImplementedError


def bigram_nll(counts, ids, smoothing: float = 1.0):
    """Mean negative log-likelihood (nats per token) of `ids` under the bigram model in `counts`.

    Probabilities with add-k smoothing: P(b | a) = (C[a, b] + k) / (sum_b' C[a, b'] + k * vocab), so
    pairs never seen get a small probability instead of zero (log 0 = -inf). Average -log P(next | prev)
    over the n - 1 consecutive pairs of `ids`. Returns a scalar float.
    """
    raise NotImplementedError


def get_batch(key, data, batch_size: int, block_size: int):
    """Sample `batch_size` random windows from the 1-D token array `data`.

    Pick start indices uniformly from [0, len(data) - block_size - 1] (jax.random.randint; the upper
    bound you pass is exclusive). For each start s:
        x = data[s : s + block_size]          y = data[s + 1 : s + block_size + 1]
    so y is x shifted left by one — at every position, the target is the next token.
    Use jax.vmap over the starts with jax.lax.dynamic_slice(data, (s,), (block_size,)) — the slice
    *size* must be a static Python int, but the *start* may be a traced value. Return (x, y), each of
    shape (batch_size, block_size), same dtype as `data`.
    """
    raise NotImplementedError


def mlp_lm_init(key, vocab: int, context: int, emb_dim: int, hidden: int) -> dict:
    """Parameters for a Bengio-style MLP language model, as a dict:

        "emb": (vocab, emb_dim)            normal * 0.1   — one learned vector per token
        "w1":  (context * emb_dim, hidden) normal / sqrt(context * emb_dim)
        "b1":  (hidden,)                   zeros
        "w2":  (hidden, vocab)             normal / sqrt(hidden) * 0.1   (small → near-uniform first guesses)
        "b2":  (vocab,)                    zeros

    Split `key` so each random array gets its own key. All float32.
    """
    raise NotImplementedError


def mlp_lm_apply(params: dict, ctx):
    """Next-token logits from the previous `context` tokens.

    ctx: (batch, context) integer ids. Look up embeddings → (batch, context, emb_dim), flatten to
    (batch, context * emb_dim), hidden = tanh(flat @ w1 + b1), logits = hidden @ w2 + b2.
    Returns (batch, vocab) logits.
    """
    raise NotImplementedError


def sample(key, logits_fn, context, n: int, temperature: float = 1.0):
    """Generate `n` new tokens autoregressively and return them as a 1-D int array of length n.

    `context`: 1-D int array — the starting window (its length is the model's context size).
    `logits_fn(window)` takes a 1-D window of that length and returns (vocab,) logits for the next token.
    At each step: compute logits, divide by `temperature`, sample with jax.random.categorical (split the
    key every step), append the token, and slide the window (drop its first token, add the new one).
    temperature == 0 means greedy: take the argmax instead of sampling.
    A plain Python loop is fine here.
    """
    raise NotImplementedError
