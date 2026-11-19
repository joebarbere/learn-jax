"""Rosetta entries for Part 12, frontier architectures: LatentMoE and Engram's pieces (Module 40), linear
attention and the delta rule (Module 41), distillation (Module 42)."""
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


def _lin_inputs(r: np.random.Generator) -> dict:
    T, dk, dv = 7, 4, 3
    k = r.normal(size=(T, dk))
    return {"q": r.normal(size=(T, dk)), "k": k / np.linalg.norm(k, axis=-1, keepdims=True),
            "v": r.normal(size=(T, dv)), "log_a": -np.abs(r.normal(size=T)) * 0.3,
            "beta": 1 / (1 + np.exp(-r.normal(size=T)))}


ENTRIES += [
    E("decayed linear attention (recurrent form)", "Sequence and state-space models",
      "Keep one matrix as memory. Every step, fade it by a_t and add the outer product of the key and the value; "
      "read it with the query. With a_t = 1 this is linear attention; with a learned a_t it is Mamba-2's SSD "
      "(query = C, key = B, value = x).",
      r"S_t = a_t S_{t-1} + k_t v_t^\top,\quad o_t = S_t^\top q_t,\quad a_t = e^{\log a_t}",
      r"$h_t = A_t h_{t-1} + B_t x_t,\ y_t = C_t^\top h_t$ with $A_t = a_t I$ (Mamba-2); "
      r"$S_n = \gamma S_{n-1} + K_n^\top V_n$ (RetNet, fixed $\gamma$)",
      "q, k (T, dk), v (T, dv), log_a (T,) -> (T, dv)", "lin_attn(q, k, v, log_a)", _lin_inputs,
      python="""
        import math

        def lin_attn(q, k, v, log_a):
            dk, dv = len(k[0]), len(v[0])
            S = [[0.0] * dv for _ in range(dk)]
            out = []
            for qt, kt, vt, la in zip(q, k, v, log_a):
                a = math.exp(la)
                S = [[a * S[i][j] + kt[i] * vt[j] for j in range(dv)] for i in range(dk)]
                out.append([sum(qt[i] * S[i][j] for i in range(dk)) for j in range(dv)])
            return out
      """,
      numpy="""
        def lin_attn(q, k, v, log_a):
            S, out = np.zeros((k.shape[1], v.shape[1])), []
            for qt, kt, vt, la in zip(q, k, v, log_a):
                S = np.exp(la) * S + np.outer(kt, vt)
                out.append(qt @ S)
            return np.stack(out)
      """,
      jax="""
        def lin_attn(q, k, v, log_a):
            def step(S, inp):
                qt, kt, vt, la = inp
                S = jnp.exp(la) * S + jnp.outer(kt, vt)
                return S, qt @ S
            S0 = jnp.zeros((k.shape[1], v.shape[1]))
            return jax.lax.scan(step, S0, (q, k, v, log_a))[1]
      """,
      jax_by_hand="""
        def lin_attn(q, k, v, log_a):
            c = jnp.cumsum(log_a)                                     # the parallel form: masked attention
            L = jnp.where(jnp.tril(jnp.ones((len(c), len(c)), bool)), c[:, None] - c[None, :], -jnp.inf)
            return ((q @ k.T) * jnp.exp(L)) @ v
      """,
      torch="""
        def lin_attn(q, k, v, log_a):
            S, out = q.new_zeros(k.shape[1], v.shape[1]), []
            for qt, kt, vt, la in zip(q, k, v, log_a):
                S = torch.exp(la) * S + torch.outer(kt, vt)
                out.append(qt @ S)
            return torch.stack(out)
      """,
      einsum=r"o_{te} = \sum_{s \le t} \Big(\prod_{s < m \le t} a_m\Big) (q_{td} k_{sd})\, v_{se}",
      notes=("The recurrent and the parallel forms are the same function: the recurrence costs O(T·dk·dv) and "
             "O(dk·dv) memory, the parallel form O(T²·d) but is all matmuls. Mamba-2 and every modern linear "
             "layer use chunks: parallel inside a chunk, recurrent between chunks.",
             "Keep the decay in log space and build the mask from differences of a cumulative sum: multiplying "
             "many a_t < 1 directly underflows."),
      module="41", also=("36", "15")),
    E("gated delta rule", "Sequence and state-space models",
      "A fixed-size memory that overwrites instead of adding. Fade the memory, ask it what it currently stores "
      "for this key, and write back only the difference from the new value, scaled by a write strength β.",
      r"S_t = a_t S_{t-1} + \beta_t\, k_t \big(v_t - (a_t S_{t-1})^\top k_t\big)^\top"
      r" = a_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top,\quad o_t = S_t^\top q_t",
      r"$\mathbf S_t = \mathbf S_{t-1}\big(\alpha_t(\mathbf I - \beta_t \mathbf k_t \mathbf k_t^\top)\big)"
      r" + \beta_t \mathbf v_t \mathbf k_t^\top$ (Gated DeltaNet eq. 10; their $\mathbf S$ is our $S^\top$)",
      "q, k (T, dk) (unit-length keys), v (T, dv), beta (T,) in (0, 1), log_a (T,) -> (T, dv)",
      "gated_delta(q, k, v, beta, log_a)", _lin_inputs,
      python="""
        import math

        def gated_delta(q, k, v, beta, log_a):
            dk, dv = len(k[0]), len(v[0])
            S = [[0.0] * dv for _ in range(dk)]
            out = []
            for qt, kt, vt, b, la in zip(q, k, v, beta, log_a):
                a = math.exp(la)
                S = [[a * x for x in row] for row in S]
                pred = [sum(kt[i] * S[i][j] for i in range(dk)) for j in range(dv)]
                S = [[S[i][j] + b * kt[i] * (vt[j] - pred[j]) for j in range(dv)] for i in range(dk)]
                out.append([sum(qt[i] * S[i][j] for i in range(dk)) for j in range(dv)])
            return out
      """,
      numpy="""
        def gated_delta(q, k, v, beta, log_a):
            S, out = np.zeros((k.shape[1], v.shape[1])), []
            for qt, kt, vt, b, la in zip(q, k, v, beta, log_a):
                S = np.exp(la) * S
                S = S + b * np.outer(kt, vt - kt @ S)
                out.append(qt @ S)
            return np.stack(out)
      """,
      jax="""
        def gated_delta(q, k, v, beta, log_a):
            def step(S, inp):
                qt, kt, vt, b, la = inp
                S = jnp.exp(la) * S
                S = S + b * jnp.outer(kt, vt - kt @ S)
                return S, qt @ S
            S0 = jnp.zeros((k.shape[1], v.shape[1]))
            return jax.lax.scan(step, S0, (q, k, v, beta, log_a))[1]
      """,
      torch="""
        def gated_delta(q, k, v, beta, log_a):
            S, out = q.new_zeros(k.shape[1], v.shape[1]), []
            for qt, kt, vt, b, la in zip(q, k, v, beta, log_a):
                S = torch.exp(la) * S
                S = S + b * torch.outer(kt, vt - kt @ S)
                out.append(qt @ S)
            return torch.stack(out)
      """,
      notes=("This is one step of gradient descent on the regression loss ½‖Sᵀk − v‖² with learning rate β: "
             "the layer learns key→value pairs as it reads (Schlag et al. 2021: 'fast weights').",
             "Unit-length keys keep (I − βkkᵀ) a contraction for β in (0, 1); without the normalization the "
             "state can blow up.",
             "Training runs it chunkwise: the writes inside a chunk solve a small lower-triangular system "
             "(the WY / UT transform), so the work is matmuls (Module 41's models.py)."),
      module="41"),
]

ENTRIES += [
    E("triangular solve (forward substitution)", "Linear algebra",
      "Solve L x = b when L is lower-triangular: the first equation has one unknown, the second two (one already "
      "known), and so on — no matrix inverse needed. The chunked delta rule solves one per chunk.",
      r"x_i = \frac{1}{L_{ii}}\Big(b_i - \sum_{j < i} L_{ij}\, x_j\Big),\quad i = 1, \dots, n",
      r"$X = L^{-1} B$ for lower-triangular $L$; with a unit diagonal, $(I + A)^{-1}$ (the UT transform)",
      "L (n, n) lower-triangular, b (n, m) -> (n, m)", "tri_solve(L, b)",
      lambda r: {"L": np.tril(r.normal(size=(5, 5)) * 0.3) + 2 * np.eye(5), "b": r.normal(size=(5, 3))},
      python="""
        def tri_solve(L, b):
            n, m = len(L), len(b[0])
            x = [[0.0] * m for _ in range(n)]
            for i in range(n):
                for c in range(m):
                    s = b[i][c] - sum(L[i][j] * x[j][c] for j in range(i))
                    x[i][c] = s / L[i][i]
            return x
      """,
      numpy="""
        def tri_solve(L, b):
            x = np.zeros_like(b, dtype=float)
            for i in range(len(L)):
                x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
            return x
      """,
      jax="""
        def tri_solve(L, b):
            return jax.scipy.linalg.solve_triangular(L, b, lower=True)
      """,
      jax_by_hand="""
        def tri_solve(L, b):
            def row(x, i):
                xi = (b[i] - jnp.where(jnp.arange(len(L)) < i, L[i], 0.0) @ x) / L[i, i]
                return x.at[i].set(xi), None
            return jax.lax.scan(row, jnp.zeros_like(b), jnp.arange(len(L)))[0]
      """,
      torch="""
        def tri_solve(L, b):
            return torch.linalg.solve_triangular(L, b, upper=False)
      """,
      notes=("O(n²) per right-hand side, against O(n³) for a general solve; a unit diagonal "
             "(`unit_diagonal=True`) skips the division.",
             "PyTorch says `upper=False` where JAX and SciPy say `lower=True`."),
      module="41", apis=("jax.scipy.linalg.solve_triangular",)),
]


def _kd_inputs(r: np.random.Generator) -> dict:
    N, V, k = 4, 12, 5
    t = r.normal(size=(N, V)) * 2
    lp = t - np.log(np.exp(t).sum(-1, keepdims=True))
    ids = np.argsort(-lp, -1)[:, :k]
    return {"student": r.normal(size=(N, V)), "ids": ids, "logp": np.take_along_axis(lp, ids, -1), "T": 2.0}


ENTRIES += [
    E("distillation loss (teacher's top-k, temperature T)", "Probability and information",
      "Teach a small model with a big one's predictions instead of only the one true next token. Soften both "
      "with a temperature T, keep the teacher's k most likely tokens (renormalized), and minimize the KL "
      "divergence from teacher to student; multiply by T² so the gradient's size doesn't depend on T.",
      r"\mathcal L_{\mathrm{KD}} = T^2 \sum_{i \in \mathrm{top}_k} p_i \big(\log p_i - \log q_i\big),\quad "
      r"p = \mathrm{softmax}_{\mathrm{top}_k}(\ell^{\mathrm{teacher}} / T),\ q = \mathrm{softmax}(z^{\mathrm{student}} / T)",
      r"$\mathcal L = (1-\alpha)\,\mathrm{CE}(y, q) + \alpha\, T^2\, \mathrm{KL}(p_T \,\Vert\, q_T)$ "
      r"(Hinton et al. 2015; Gemma 2 uses $\alpha = 1$)",
      "student logits (N, V), teacher top ids (N, k), teacher top log-probs (N, k), T -> scalar",
      "kd(student, ids, logp, T)", _kd_inputs,
      python="""
        import math

        def kd(student, ids, logp, T):
            total = 0.0
            for z, row_ids, row_lp in zip(student, ids, logp):
                pt = [math.exp(l / T) for l in row_lp]
                pt = [x / sum(pt) for x in pt]                        # teacher: top k, renormalized
                m = max(z) / T
                lse = m + math.log(sum(math.exp(zi / T - m) for zi in z))
                total += sum(p * (math.log(p) - (z[i] / T - lse)) for p, i in zip(pt, row_ids))
            return T * T * total / len(student)
      """,
      numpy="""
        def kd(student, ids, logp, T):
            p = np.exp(logp / T - (logp / T).max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            z = student / T
            logq = z - z.max(-1, keepdims=True)
            logq -= np.log(np.exp(logq).sum(-1, keepdims=True))
            return T * T * np.mean(np.sum(p * (np.log(p) - np.take_along_axis(logq, ids, -1)), -1))
      """,
      jax="""
        def kd(student, ids, logp, T):
            p = jax.nn.softmax(logp / T, -1)
            logq = jnp.take_along_axis(jax.nn.log_softmax(student / T, -1), ids, -1)
            return T * T * jnp.mean(jnp.sum(p * (jnp.log(p) - logq), -1))
      """,
      torch="""
        def kd(student, ids, logp, T):
            p = F.softmax(logp / T, -1)
            logq = torch.gather(F.log_softmax(student / T, -1), -1, ids)
            return T * T * (p * (p.log() - logq)).sum(-1).mean()
      """,
      notes=("The teacher's log-probabilities are normalized over the whole vocabulary before the top-k cut; "
             "dividing them by T and renormalizing over the k kept tokens equals a softmax of the teacher's "
             "logits / T restricted to those tokens.",
             "PyTorch's `F.kl_div(input, target)` takes the *student's* log-probabilities first and the teacher's "
             "probabilities second, and its default reduction averages over every element, not over positions: "
             "use `reduction='batchmean'`."),
      module="42", also=("26",)),
]
