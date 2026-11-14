"""Module 14 reference solutions. Try the exercises first — `make test N=14` grades yours."""
import jax
import jax.numpy as jnp


def bigram_counts(ids, vocab: int):
    ids = jnp.asarray(ids)
    return jnp.zeros((vocab, vocab), jnp.int32).at[ids[:-1], ids[1:]].add(1)


def bigram_nll(counts, ids, smoothing: float = 1.0):
    counts = jnp.asarray(counts, jnp.float32)
    ids = jnp.asarray(ids)
    vocab = counts.shape[0]
    probs = (counts + smoothing) / (counts.sum(axis=1, keepdims=True) + smoothing * vocab)
    return -jnp.mean(jnp.log(probs[ids[:-1], ids[1:]]))


def get_batch(key, data, batch_size: int, block_size: int):
    data = jnp.asarray(data)
    starts = jax.random.randint(key, (batch_size,), 0, data.shape[0] - block_size)
    window = lambda s: jax.lax.dynamic_slice(data, (s,), (block_size + 1,))
    chunks = jax.vmap(window)(starts)
    return chunks[:, :-1], chunks[:, 1:]


def mlp_lm_init(key, vocab: int, context: int, emb_dim: int, hidden: int) -> dict:
    k_emb, k1, k2 = jax.random.split(key, 3)
    fan_in = context * emb_dim
    return {
        "emb": jax.random.normal(k_emb, (vocab, emb_dim)) * 0.1,
        "w1": jax.random.normal(k1, (fan_in, hidden)) / jnp.sqrt(fan_in),
        "b1": jnp.zeros(hidden),
        "w2": jax.random.normal(k2, (hidden, vocab)) / jnp.sqrt(hidden) * 0.1,
        "b2": jnp.zeros(vocab),
    }


def mlp_lm_apply(params: dict, ctx):
    emb = params["emb"][ctx]                       # (batch, context, emb_dim)
    flat = emb.reshape(emb.shape[0], -1)           # (batch, context * emb_dim)
    hidden = jnp.tanh(flat @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def sample(key, logits_fn, context, n: int, temperature: float = 1.0):
    window = jnp.asarray(context)
    out = []
    for _ in range(n):
        logits = logits_fn(window)
        if temperature == 0:
            tok = jnp.argmax(logits)
        else:
            key, sub = jax.random.split(key)
            tok = jax.random.categorical(sub, logits / temperature)
        out.append(tok)
        window = jnp.concatenate([window[1:], tok[None].astype(window.dtype)])
    return jnp.array(out, dtype=jnp.int32)
