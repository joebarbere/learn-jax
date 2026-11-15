"""Module 37 reference solutions. Try the exercises first — `make test N=37` grades yours."""
import jax
import jax.numpy as jnp
import optax


def attention_entropy(weights):
    w = jnp.asarray(weights, jnp.float32)
    return -jnp.sum(jnp.where(w > 0, w * jnp.log(jnp.where(w > 0, w, 1.0)), 0.0), axis=-1)


def previous_token_score(attn):
    a = jnp.asarray(attn)
    T = a.shape[-1]
    i = jnp.arange(1, T)
    return float(a[..., i, i - 1].mean())


def induction_score(attn, tokens):
    a = jnp.asarray(attn, jnp.float32)
    t = jnp.asarray(tokens)
    T = t.shape[-1]
    prev = jnp.concatenate([jnp.full_like(t[:, :1], -1), t[:, :-1]], axis=1)      # token before key j
    match = prev[:, None, :] == t[:, :, None]                                    # [B, i, j]
    ii, jj = jnp.meshgrid(jnp.arange(T), jnp.arange(T), indexing="ij")
    target = match & (jj >= 1) & (jj <= ii)
    mass = (a * target).sum(-1)                                                  # [B, T]
    has = target.any(-1)
    return float(jnp.where(has, mass, 0.0).sum() / jnp.maximum(has.sum(), 1))


def ablate_head(head_mask, layer: int, head: int):
    return jnp.asarray(head_mask).at[layer, head].set(0.0)


def _probe_logits(params, x):
    W, b = params
    return jnp.einsum("nd,dkc->nkc", x, W) + b


@jax.jit
def _probe_loss(params, x, y):
    lp = jax.nn.log_softmax(_probe_logits(params, x), -1)
    onehot = jax.nn.one_hot(y, lp.shape[-1])
    return -(lp * onehot).sum(-1).mean()


def train_linear_probe(acts, labels, n_classes: int, steps: int = 300, lr: float = 1e-2):
    x = jnp.asarray(acts, jnp.float32)
    y = jnp.asarray(labels, jnp.int32)
    d, K = x.shape[1], y.shape[1]
    params = (jnp.zeros((d, K, n_classes)), jnp.zeros((K, n_classes)))
    opt = optax.adam(lr)

    @jax.jit
    def fit(params, x, y):
        def body(carry, _):
            p, s = carry
            g = jax.grad(_probe_loss)(p, x, y)
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), None
        (p, _), _ = jax.lax.scan(body, (params, opt.init(params)), None, length=steps)
        return p

    return fit(params, x, y)


def probe_accuracy(W, b, acts, labels):
    pred = jnp.argmax(_probe_logits((W, b), jnp.asarray(acts, jnp.float32)), -1)
    return float((pred == jnp.asarray(labels)).mean())


def to_relative_board(abs_board, player):
    ab = jnp.asarray(abs_board)
    pl = jnp.asarray(player)[..., None]
    mine = (ab != 0) & (ab == pl)
    theirs = (ab != 0) & (ab != pl)
    return jnp.where(mine, 1, jnp.where(theirs, 2, 0))
