"""Rosetta entries for Part 12, frontier architectures (Module 40): LatentMoE and Engram's pieces."""
import numpy as np

from learnjax.rosetta.entry import Entry as E


def _latent_inputs(r: np.random.Generator) -> dict:
    N, d, ell, m, n_exp, k = 5, 8, 4, 3, 6, 2
    s = 1 / (1 + np.exp(-r.normal(size=(N, n_exp))))
    top = np.argsort(-s, -1)[:, :k]
    g = np.zeros_like(s)
    np.put_along_axis(g, top, np.take_along_axis(s, top, -1), -1)
    return {"x": r.normal(size=(N, d)), "gates": g / g.sum(-1, keepdims=True),
            "down": r.normal(size=(d, ell)) * 0.5, "up": r.normal(size=(ell, d)) * 0.5,
            "W1": r.normal(size=(n_exp, ell, m)) * 0.5, "W3": r.normal(size=(n_exp, ell, m)) * 0.5,
            "W2": r.normal(size=(n_exp, m, ell)) * 0.5}


ENTRIES: list[E] = [
    E("LatentMoE layer", "Attention and transformers",
      "Project each token down to a narrow latent, run its picked experts there, add their outputs weighted by "
      "the gates, and project the sum back up once. The router (which made the gates) saw the full-width token.",
      r"y = W_\uparrow \sum_{i} g_i\, E_i(W_\downarrow x),\quad "
      r"E_i(z) = \big(\mathrm{silu}(z W_1^{(i)}) \odot z W_3^{(i)}\big) W_2^{(i)},\quad W_\downarrow \in \mathbb{R}^{d \times \ell}",
      r"$\mathrm{MoE}(x)=W_\uparrow\big(\sum_{i\in\mathcal T_{K',N'}} p'_i E_i(W_\downarrow x;\ell)\big)+\sum_j E_j(x;d)$ "
      r"(Elango et al., eq. 2; the shared experts $E_j$ stay at width $d$)",
      "x (N, d), gates (N, E), down (d, ℓ), up (ℓ, d), W1/W3 (E, ℓ, m), W2 (E, m, ℓ) -> (N, d)",
      "latent_moe(x, gates, down, up, W1, W3, W2)", _latent_inputs,
      python="""
        import math

        def latent_moe(x, gates, down, up, W1, W3, W2):
            silu = lambda a: a / (1 + math.exp(-a))
            mat = lambda v, W: [sum(v[i] * W[i][j] for i in range(len(v))) for j in range(len(W[0]))]
            out = []
            for xn, gn in zip(x, gates):
                z = mat(xn, down)                                     # d -> ℓ
                acc = [0.0] * len(z)
                for e, g in enumerate(gn):
                    if g == 0:
                        continue                                      # an expert this token didn't pick
                    h = [silu(a) * b for a, b in zip(mat(z, W1[e]), mat(z, W3[e]))]
                    acc = [a + g * o for a, o in zip(acc, mat(h, W2[e]))]
                out.append(mat(acc, up))                              # ℓ -> d, once
            return out
      """,
      numpy="""
        def latent_moe(x, gates, down, up, W1, W3, W2):
            z = x @ down
            a, b = np.einsum("nl,elm->nem", z, W1), np.einsum("nl,elm->nem", z, W3)
            h = a / (1 + np.exp(-a)) * b
            return np.einsum("ne,nel->nl", gates, np.einsum("nem,eml->nel", h, W2)) @ up
      """,
      jax="""
        def latent_moe(x, gates, down, up, W1, W3, W2):
            z = x @ down                                              # every expert on every token (dense)
            h = jax.nn.silu(jnp.einsum("nl,elm->nem", z, W1)) * jnp.einsum("nl,elm->nem", z, W3)
            return jnp.einsum("ne,nel->nl", gates, jnp.einsum("nem,eml->nel", h, W2)) @ up
      """,
      torch="""
        def latent_moe(x, gates, down, up, W1, W3, W2):
            z = x @ down
            h = F.silu(torch.einsum("nl,elm->nem", z, W1)) * torch.einsum("nl,elm->nem", z, W3)
            return torch.einsum("ne,nel->nl", gates, torch.einsum("nem,eml->nel", h, W2)) @ up
      """,
      einsum="z_{nl} = x_{nd} D_{dl};\\ y_{nd} = \\big(\\sum_e g_{ne}\\, E_e(z)_{nl}\\big) U_{ld}",
      notes=("Dense compute, as in Module 17: unpicked experts run and are multiplied by 0. A real layer "
             "dispatches each token only to its picked experts (`jax.lax.ragged_dot`, or an all-to-all across "
             "devices) — that's where projecting down *before* dispatch saves bytes.",
             "The up-projection is linear, so applying it once after the weighted sum equals applying it to every "
             "expert's output: one ℓ×d matmul per token instead of k."),
      module="40", also=("17", "24")),
    E("n-gram hash (multiplicative XOR)", "Neural network layers",
      "Turn the last n token ids into one table row: multiply each id by its own odd constant, XOR the products, "
      "and take the remainder modulo a prime table size. Positions before the start read a padding id.",
      r"\mathrm{row}_t = \Big(\bigoplus_{j=0}^{n-1} x_{t-j}\, m_j\Big) \bmod P",
      r"$z_{t,n,k} = \varphi_{n,k}(g_{t,n})$, $\mathbf{e}_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}]$ (Engram eq. 1)",
      "ids (B, T) int, multipliers m (3,), primes P2, P3 -> rows (B, T, 2): one 2-gram and one 3-gram head",
      "ngram_hash(ids, mult, p2, p3)",
      lambda r: {"ids": r.integers(0, 50, (2, 7)), "mult": np.array([3, 5, 7]), "p2": 13, "p3": 17},
      python="""
        def ngram_hash(ids, mult, p2, p3, pad=0):
            out = []
            for row in ids:
                rows = []
                for t in range(len(row)):
                    tok = lambda j: int(row[t - j]) if t - j >= 0 else pad
                    mix2 = (tok(0) * int(mult[0])) ^ (tok(1) * int(mult[1]))
                    mix3 = mix2 ^ (tok(2) * int(mult[2]))
                    rows.append([mix2 % p2, mix3 % p3])
                out.append(rows)
            return out
      """,
      numpy="""
        def ngram_hash(ids, mult, p2, p3, pad=0):
            ids = ids.astype(np.int64)
            T = ids.shape[1]
            s1 = np.pad(ids, ((0, 0), (1, 0)), constant_values=pad)[:, :T]
            s2 = np.pad(ids, ((0, 0), (2, 0)), constant_values=pad)[:, :T]
            mix2 = (ids * mult[0]) ^ (s1 * mult[1])
            mix3 = mix2 ^ (s2 * mult[2])
            return np.stack([mix2 % p2, mix3 % p3], -1)
      """,
      jax="""
        def ngram_hash(ids, mult, p2, p3, pad=0):
            T = ids.shape[1]
            s1 = jnp.pad(ids, ((0, 0), (1, 0)), constant_values=pad)[:, :T]
            s2 = jnp.pad(ids, ((0, 0), (2, 0)), constant_values=pad)[:, :T]
            mix2 = jnp.bitwise_xor(ids * mult[0], s1 * mult[1])
            mix3 = jnp.bitwise_xor(mix2, s2 * mult[2])
            return jnp.stack([mix2 % p2, mix3 % p3], -1)
      """,
      torch="""
        def ngram_hash(ids, mult, p2, p3, pad=0):
            T = ids.shape[1]
            s1 = F.pad(ids, (1, 0), value=pad)[:, :T]
            s2 = F.pad(ids, (2, 0), value=pad)[:, :T]
            mix2 = torch.bitwise_xor(ids * mult[0], s1 * mult[1])
            mix3 = torch.bitwise_xor(mix2, s2 * mult[2])
            return torch.stack([mix2 % p2, mix3 % p3], -1)
      """,
      notes=("JAX uses 32-bit integers unless you turn on `jax_enable_x64`, so `id × multiplier` wraps around "
             "at 2³¹ — fine for these small numbers, not for DeepSeek's 64-bit multipliers. Engram computes its "
             "hashes on the host in NumPy int64 (they depend only on the token ids), which is also what makes "
             "the lookups prefetchable.",
             "Python's `%` and NumPy's / PyTorch's `%` agree for non-negative numbers; for negative ones C-style "
             "`torch.fmod` differs. The XOR of non-negative products stays non-negative."),
      module="40"),
    E("Engram context gate", "Attention and transformers",
      "Compare the looked-up memory with the current hidden state: if they agree (a large normalized dot "
      "product) the gate opens and the memory's value passes; if they disagree it closes.",
      r"k = e W_K,\ v = e W_V,\quad \alpha = \sigma\!\Big(\tfrac{1}{\sqrt d}\,"
      r"\mathrm{RMSNorm}(h)^\top \mathrm{RMSNorm}(k)\Big),\quad \tilde v = \alpha\, v",
      r"$\alpha_t=\sigma\big(\mathrm{RMSNorm}(\mathbf h_t)^\top\mathrm{RMSNorm}(\mathbf k_t)/\sqrt d\big)$ (Engram eq. 4)",
      "h (B, T, d), e (B, T, d_mem), Wk/Wv (d_mem, d), gh/gk (d,) -> (B, T, d)", "gate(h, e, Wk, Wv, gh, gk)",
      lambda r: {"h": r.normal(size=(2, 3, 8)), "e": r.normal(size=(2, 3, 6)), "Wk": r.normal(size=(6, 8)),
                 "Wv": r.normal(size=(6, 8)), "gh": 1 + 0.1 * r.normal(size=8), "gk": 1 + 0.1 * r.normal(size=8)},
      python="""
        import math

        def gate(h, e, Wk, Wv, gh, gk, eps=1e-6):
            d = len(Wk[0])
            mat = lambda v, W: [sum(v[i] * W[i][j] for i in range(len(v))) for j in range(d)]
            rms = lambda v, g: [a / math.sqrt(sum(b * b for b in v) / len(v) + eps) * w for a, w in zip(v, g)]
            out = []
            for hb, eb in zip(h, e):
                rows = []
                for ht, et in zip(hb, eb):
                    k, v = mat(et, Wk), mat(et, Wv)
                    s = sum(a * b for a, b in zip(rms(ht, gh), rms(k, gk))) / math.sqrt(d)
                    alpha = 1 / (1 + math.exp(-s))
                    rows.append([alpha * a for a in v])
                out.append(rows)
            return out
      """,
      numpy="""
        def gate(h, e, Wk, Wv, gh, gk, eps=1e-6):
            rms = lambda a, g: a / np.sqrt(np.mean(a * a, -1, keepdims=True) + eps) * g
            k, v = e @ Wk, e @ Wv
            s = (rms(h, gh) * rms(k, gk)).sum(-1) / np.sqrt(h.shape[-1])
            return 1 / (1 + np.exp(-s))[..., None] * v
      """,
      jax="""
        def gate(h, e, Wk, Wv, gh, gk, eps=1e-6):
            rms = lambda a, g: a * jax.lax.rsqrt(jnp.mean(a * a, -1, keepdims=True) + eps) * g
            k, v = e @ Wk, e @ Wv
            alpha = jax.nn.sigmoid((rms(h, gh) * rms(k, gk)).sum(-1) / jnp.sqrt(h.shape[-1]))
            return alpha[..., None] * v
      """,
      torch="""
        def gate(h, e, Wk, Wv, gh, gk, eps=1e-6):
            k, v = e @ Wk, e @ Wv
            d = h.shape[-1]
            s = (F.rms_norm(h, (d,), gh, eps) * F.rms_norm(k, (d,), gk, eps)).sum(-1) / d ** 0.5
            return torch.sigmoid(s)[..., None] * v
      """,
      notes=("DeepSeek's demo code passes the score through a signed square root, sign(s)·√|s|, before the "
             "sigmoid; the paper's equation doesn't. This course follows the paper.",
             "With hyper-connections (mHC), Engram has one key projection and one gate per residual branch and "
             "a shared value projection; this is the single-branch form."),
      module="40"),
    E("causal depthwise dilated convolution", "Neural network layers",
      "Each channel gets its own short filter that looks only backwards in time, skipping `dilation − 1` steps "
      "between taps: a cheap way to mix a few neighbouring positions without mixing channels.",
      r"y_{t,c} = \sum_{i=0}^{K-1} w_{i,c}\; x_{t-(K-1-i)\delta,\,c},\qquad x_{s<0} = 0",
      r"$\mathbf Y=\mathrm{SiLU}\big(\mathrm{Conv1D}(\mathrm{RMSNorm}(\tilde{\mathbf V}))\big)+\tilde{\mathbf V}$ "
      r"(Engram eq. 5: kernel 4, dilation = max n-gram order); Mamba's short conv is the same with $\delta = 1$",
      "x (B, T, C), w (K, C), dilation δ -> (B, T, C)", "dwconv(x, w, 2)",
      lambda r: {"x": r.normal(size=(2, 10, 3)), "w": r.normal(size=(4, 3))},
      python="""
        def dwconv(x, w, dil):
            K = len(w)
            out = []
            for xb in x:
                rows = []
                for t in range(len(xb)):
                    rows.append([sum(w[i][c] * xb[t - (K - 1 - i) * dil][c]
                                     for i in range(K) if t - (K - 1 - i) * dil >= 0)
                                 for c in range(len(xb[0]))])
                out.append(rows)
            return out
      """,
      numpy="""
        def dwconv(x, w, dil):
            K, T = w.shape[0], x.shape[1]
            xp = np.pad(x, ((0, 0), ((K - 1) * dil, 0), (0, 0)))
            return sum(w[i] * xp[:, i * dil:i * dil + T] for i in range(K))
      """,
      jax="""
        def dwconv(x, w, dil):
            K, C = w.shape
            return jax.lax.conv_general_dilated(
                x, w[:, None, :], window_strides=(1,), padding=[((K - 1) * dil, 0)], rhs_dilation=(dil,),
                dimension_numbers=("NWC", "WIO", "NWC"), feature_group_count=C)
      """,
      jax_by_hand="""
        def dwconv(x, w, dil):
            K, T = w.shape[0], x.shape[1]
            xp = jnp.pad(x, ((0, 0), ((K - 1) * dil, 0), (0, 0)))
            return sum(w[i] * xp[:, i * dil:i * dil + T] for i in range(K))
      """,
      torch="""
        def dwconv(x, w, dil):
            K, C = w.shape
            T = x.shape[1]
            y = F.conv1d(x.transpose(1, 2), w.T[:, None, :], padding=(K - 1) * dil, dilation=dil, groups=C)
            return y[..., :T].transpose(1, 2)                         # keep the first T outputs: causal
      """,
      notes=("All three libraries compute cross-correlation (no kernel flip), so w[K−1] meets the current step.",
             "PyTorch pads both ends; keeping the first T outputs throws away the ones that saw the future. JAX's "
             "`padding=[(left, right)]` pads only the left."),
      module="40", also=("36",)),
]
