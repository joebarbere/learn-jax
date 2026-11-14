"""Module 20 exercises: LoRA, piece by piece.

Replace each `raise NotImplementedError` with your code, then run `make test N=20`.
The lesson uses your `lora_init` and `lora_apply` inside a real 360M-parameter model, so once they
work, the fine-tuning cells come alive.

Conventions (same as learnjax.llama): weights are [d_in, d_out] and a layer computes x @ W.
A LoRA adapter for W is a pair A [d_in, r], B [r, d_out]; the adapted layer computes
    x @ W + (alpha / r) * (x @ A) @ B
"""
import jax  # noqa: F401  (you'll want it)
import jax.numpy as jnp

# Matrices you can adapt in learnjax.llama, and their (d_in, d_out) as a function of the config.
TARGET_SHAPES = {
    "wq": lambda c: (c.d_model, c.n_heads * c.head_dim),
    "wk": lambda c: (c.d_model, c.n_kv_heads * c.head_dim),
    "wv": lambda c: (c.d_model, c.n_kv_heads * c.head_dim),
    "wo": lambda c: (c.n_heads * c.head_dim, c.d_model),
    "w_gate": lambda c: (c.d_model, c.d_ff),
    "w_up": lambda c: (c.d_model, c.d_ff),
    "w_down": lambda c: (c.d_ff, c.d_model),
}


def lora_init(key, d_in: int, d_out: int, r: int, dtype=jnp.float32):
    """Return (A, B) for one adapted matrix.

    A: shape [d_in, r], normal with standard deviation 1/sqrt(d_in) (so x @ A has unit-ish scale).
    B: shape [r, d_out], all zeros — so the adapted layer starts *exactly* equal to the base layer.
    Both in `dtype`.
    """
    raise NotImplementedError


def lora_apply(x, W, A, B, alpha: float, r: int):
    """x @ W + (alpha / r) * (x @ A) @ B, computed in x's dtype (cast A and B to x.dtype first).

    Never form the d_in × d_out product A @ B here — going through the rank-r bottleneck is the point.
    Works for any leading batch dims of x: x is [..., d_in], result is [..., d_out].
    """
    raise NotImplementedError


def lora_param_count(cfg, r: int, targets) -> int:
    """Trainable parameters when adapting `targets` (names from TARGET_SHAPES) in every layer.

    Each adapted matrix of shape (d_in, d_out) adds r * (d_in + d_out). Multiply by cfg.n_layers.
    """
    raise NotImplementedError


def merge_lora(W, A, B, alpha: float, r: int):
    """Fold the adapter into the weight: W + (alpha / r) * A @ B, returned in W's dtype.

    After merging, plain x @ W_merged gives the adapted output with zero extra inference cost.
    """
    raise NotImplementedError


def freeze_mask(params, trainable: tuple[str, ...]):
    """A pytree of bools with params' structure: True for leaves that should train.

    A leaf is trainable if its path, written as keys joined by "/" (e.g. "lora/wq/A" or
    "base/layers/wq"), contains any of the strings in `trainable`. Use
    jax.tree_util.tree_map_with_path; a path entry for a dict key `k` is a DictKey with `.key == k`.
    """
    raise NotImplementedError


def adapter_bytes(adapters) -> int:
    """Total bytes of every array in the adapter pytree (size × itemsize, summed over leaves)."""
    raise NotImplementedError
