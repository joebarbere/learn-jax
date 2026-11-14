"""Module 20 reference solutions. Try the exercises first — `make test N=20` grades yours."""
import jax
import jax.numpy as jnp

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
    A = (jax.random.normal(key, (d_in, r)) / jnp.sqrt(d_in)).astype(dtype)
    B = jnp.zeros((r, d_out), dtype)
    return A, B


def lora_apply(x, W, A, B, alpha: float, r: int):
    return x @ W + (alpha / r) * ((x @ A.astype(x.dtype)) @ B.astype(x.dtype))


def lora_param_count(cfg, r: int, targets) -> int:
    per_layer = 0
    for t in targets:
        d_in, d_out = TARGET_SHAPES[t](cfg)
        per_layer += r * (d_in + d_out)
    return per_layer * cfg.n_layers


def merge_lora(W, A, B, alpha: float, r: int):
    delta = (alpha / r) * (A.astype(jnp.float32) @ B.astype(jnp.float32))
    return (W.astype(jnp.float32) + delta).astype(W.dtype)


def freeze_mask(params, trainable: tuple[str, ...]):
    def label(path, _leaf):
        name = "/".join(str(getattr(p, "key", getattr(p, "idx", p))) for p in path)
        return any(t in name for t in trainable)

    return jax.tree_util.tree_map_with_path(label, params)


def adapter_bytes(adapters) -> int:
    return int(sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(adapters)))
