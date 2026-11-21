"""Module 45 exercises: a masked-diffusion language model, from noise to text.

Replace each `raise NotImplementedError` with your code, then run `make test N=45`. models.py trains the
denoiser with your masking and loss, and samples with your unmasking step and schedules.

Conventions: token ids x are (B, T) int; the extra token mask_id (8192 here) means "unknown". A masking rate
t ∈ (0, 1] is one number per sequence, shape (B,): the forward process hides each token with probability t.
"""
import jax  # noqa: F401  (for your code)
import jax.numpy as jnp  # noqa: F401


def mask_tokens(key, x, t, mask_id: int):
    """The forward (noising) process: replace each token of x (B, T) by mask_id independently with probability
    t (B,) — the same t for every position of a sequence. Returns z_t, same shape and dtype as x.
    """
    raise NotImplementedError


def mdlm_loss(logits, x, z, t, mask_id: int):
    """MDLM's training loss (Sahoo et al. 2024), for the linear schedule α_t = 1 − t:

        loss = mean over the batch of  (1 / t) · (1 / T) · Σ_{i where z_i = mask} −log p_θ(x_i | z)

    logits: (B, T, V) the denoiser's predictions from z; x: (B, T) the clean tokens; z: (B, T) the masked
    input; t: (B,). Only masked positions count (the model sees the others). The 1/t weight makes this an upper
    bound on the negative log-likelihood per token: few masks (small t) means each one counts more. A scalar.
    """
    raise NotImplementedError


def unmask_step(key, logits, z, n_reveal: int, mask_id: int, temperature: float = 1.0, by_confidence: bool = True):
    """One reverse (denoising) step: reveal n_reveal of the still-masked positions in each row.

    1. Sample a token at every position from softmax(logits / temperature), never mask_id itself (set its
       logit to −inf first).
    2. Score each position: the model's probability of the token it sampled there (by_confidence=True,
       LLaDA's "low-confidence remasking": reveal the surest first), or a uniform random number (False).
    3. Only positions where z == mask_id may be revealed; pick the n_reveal highest-scoring of them per row
       (jax.lax.top_k), and put the sampled tokens there. Everything else stays as in z.

    logits (B, T, V), z (B, T). Returns the new z. (models.py may pass z with entries that are neither tokens
    to keep nor masks, e.g. −1: leave those alone too.)
    """
    raise NotImplementedError


def reveal_schedule(n_masked: int, steps: int):
    """How many tokens to reveal at each of `steps` steps so that all n_masked are revealed, as evenly as
    possible, larger counts first: reveal_schedule(10, 4) == [3, 3, 2, 2]. A list of ints summing to n_masked.
    """
    raise NotImplementedError


def block_order(T: int, prompt_len: int, block: int):
    """Semi-autoregressive generation (LLaDA, block diffusion): fill the answer positions prompt_len … T−1 in
    consecutive blocks of `block` tokens, left to right. Return a list of (start, end) pairs, end exclusive;
    the last block may be shorter. block_order(10, 2, 3) == [(2, 5), (5, 8), (8, 10)].
    """
    raise NotImplementedError
