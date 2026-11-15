"""Module 32 exercises: the pieces that make an encoder (BERT) different from a decoder (GPT).

Replace each `raise NotImplementedError` with your code, then run `make test N=32`.
The lesson trains its BERT with your `mask_tokens` and `mlm_loss` once they pass their checks.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def causal_mask(T: int):
    """(T, T) bool: query position t may attend to key position s iff s <= t (GPT, Module 15)."""
    raise NotImplementedError


def bidirectional_mask(T: int):
    """(T, T) bool: every position may attend to every position (BERT)."""
    raise NotImplementedError


def mask_tokens(key, ids, mask_id: int, vocab_size: int, rate: float = 0.15):
    """BERT's masking recipe. Returns (inputs, is_target).

    ids: int32 array (any shape). Pick each position independently with probability `rate`; those
    become prediction targets (`is_target` True). Of the picked positions:
      * 80% are replaced by `mask_id`,
      * 10% by a uniformly random token id in [0, vocab_size),
      * 10% are left unchanged.
    Unpicked positions are returned unchanged. Must work under jax.jit (no Python branching on values).
    Hint: split the key three ways; draw one uniform per position to pick targets and another to choose
    among mask / random / keep.
    """
    raise NotImplementedError


def mlm_loss(logits, targets, is_target):
    """Masked-LM loss: mean cross-entropy over target positions only.

    logits: (B, T, V) float; targets: (B, T) int (the ORIGINAL tokens); is_target: (B, T) bool.
    Return sum(cross_entropy * is_target) / max(number of targets, 1) as a float32 scalar.
    """
    raise NotImplementedError


def cls_pool(hidden):
    """BERT-style sentence vector: the hidden state at position 0 (the [CLS] token). (B, T, D) -> (B, D)."""
    raise NotImplementedError


def last_token_pool(hidden, lengths=None):
    """GPT-style sentence vector: the hidden state of the last real token. (B, T, D) -> (B, D).

    With `lengths` (B,) int, take position lengths[b] - 1 for row b; without it, take the final position.
    (In a causal model only the last token has seen the whole input.)
    """
    raise NotImplementedError


def contextual_similarity(vectors):
    """Cosine-similarity matrix of row vectors. (N, D) -> (N, N), diagonal 1.

    Used to compare the same word's vectors in different sentences.
    """
    raise NotImplementedError
