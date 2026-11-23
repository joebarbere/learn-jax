"""Rosetta entries for Part 12, frontier architectures: LatentMoE and Engram's pieces (Module 40), linear
attention and the delta rule (Module 41), distillation (Module 42), SigLIP's loss (Module 43), masked diffusion (Module 45), and setting, merging and
editing weights by hand (Module 46)."""
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
      notes=(("Dense compute, as in Module 17: unpicked experts run and are multiplied by 0. A real layer "
             "dispatches each token only to its picked experts (`jax.lax.ragged_dot`, or an all-to-all across "
             "devices) — that's where projecting down *before* dispatch saves bytes."),
             ("The up-projection is linear, so applying it once after the weighted sum equals applying it to every "
             "expert's output: one ℓ×d matmul per token instead of k.")),
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
      notes=(("JAX uses 32-bit integers unless you turn on `jax_enable_x64`, so `id × multiplier` wraps around "
             "at 2³¹ — fine for these small numbers, not for DeepSeek's 64-bit multipliers. Engram computes its "
             "hashes on the host in NumPy int64 (they depend only on the token ids), which is also what makes "
             "the lookups prefetchable."),
             ("Python's `%` and NumPy's / PyTorch's `%` agree for non-negative numbers; for negative ones C-style "
             "`torch.fmod` differs. The XOR of non-negative products stays non-negative.")),
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
      notes=(("DeepSeek's demo code passes the score through a signed square root, sign(s)·√|s|, before the "
             "sigmoid; the paper's equation doesn't. This course follows the paper."),
             ("With hyper-connections (mHC), Engram has one key projection and one gate per residual branch and "
             "a shared value projection; this is the single-branch form.")),
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
             ("PyTorch pads both ends; keeping the first T outputs throws away the ones that saw the future. JAX's "
             "`padding=[(left, right)]` pads only the left.")),
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
      notes=(("The recurrent and the parallel forms are the same function: the recurrence costs O(T·dk·dv) and "
             "O(dk·dv) memory, the parallel form O(T²·d) but is all matmuls. Mamba-2 and every modern linear "
             "layer use chunks: parallel inside a chunk, recurrent between chunks."),
             ("Keep the decay in log space and build the mask from differences of a cumulative sum: multiplying "
             "many a_t < 1 directly underflows.")),
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
      notes=(("This is one step of gradient descent on the regression loss ½‖Sᵀk − v‖² with learning rate β: "
             "the layer learns key→value pairs as it reads (Schlag et al. 2021: 'fast weights')."),
             ("Unit-length keys keep (I − βkkᵀ) a contraction for β in (0, 1); without the normalization the "
             "state can blow up."),
             ("Training runs it chunkwise: the writes inside a chunk solve a small lower-triangular system "
             "(the WY / UT transform), so the work is matmuls (Module 41's models.py).")),
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
      notes=(("O(n²) per right-hand side, against O(n³) for a general solve; a unit diagonal "
             "(`unit_diagonal=True`) skips the division."),
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
      notes=(("The teacher's log-probabilities are normalized over the whole vocabulary before the top-k cut; "
             "dividing them by T and renormalizing over the k kept tokens equals a softmax of the teacher's "
             "logits / T restricted to those tokens."),
             ("PyTorch's `F.kl_div(input, target)` takes the *student's* log-probabilities first and the teacher's "
             "probabilities second, and its default reduction averages over every element, not over positions: "
             "use `reduction='batchmean'`.")),
      module="42", also=("26",)),
]


def _mdlm_inputs(r: np.random.Generator) -> dict:
    B, T, V = 3, 6, 7
    return {"logits": r.normal(size=(B, T, V)), "x": r.integers(0, V, (B, T)),
            "masked": r.random((B, T)) < 0.5, "t": np.array([0.2, 0.5, 0.9])}


ENTRIES += [
    E("masked diffusion loss (MDLM)", "Probability and information",
      "Hide each token with probability t, ask the model to predict the hidden ones from everything it can see "
      "(both sides), and score only the hidden positions. Dividing by t makes it a bound on the negative "
      "log-likelihood: at a low masking rate each hidden token stands for many.",
      r"\mathcal L = \mathbb{E}_{t}\, \frac{1}{t\,T} \sum_{i:\, z_i = \texttt{[MASK]}} -\log p_\theta(x_i \mid z),"
      r"\quad z_i = \begin{cases}\texttt{[MASK]} & \text{with prob. } t\\ x_i & \text{otherwise}\end{cases}",
      r"$\mathcal L_{\mathrm{NELBO}} = \mathbb{E}_q \int_0^1 \frac{\alpha'_t}{1-\alpha_t} \sum_\ell "
      r"\log\langle x_\theta^\ell(z_t), x^\ell\rangle\, dt$ (Sahoo et al. 2024, eq. 11; $\alpha_t = 1 - t$ gives "
      r"the weight $1/t$)",
      "logits (B, T, V), x (B, T), masked (B, T) bool, t (B,) -> scalar",
      "mdlm(logits, x, masked, t)", _mdlm_inputs,
      python="""
        import math

        def mdlm(logits, x, masked, t):
            total = 0.0
            for row, xs, ms, tb in zip(logits, x, masked, t):
                s = 0.0
                for z, xi, m in zip(row, xs, ms):
                    if m:
                        mx = max(z)
                        lse = mx + math.log(sum(math.exp(zi - mx) for zi in z))
                        s += lse - z[xi]                              # -log softmax(z)[x_i]
                total += s / (tb * len(xs))
            return total / len(logits)
      """,
      numpy="""
        def mdlm(logits, x, masked, t):
            z = logits - logits.max(-1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
            nll = -np.take_along_axis(logp, x[..., None], -1)[..., 0]
            return np.mean((nll * masked).sum(-1) / (t * x.shape[-1]))
      """,
      jax="""
        def mdlm(logits, x, masked, t):
            nll = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), x[..., None], -1)[..., 0]
            return jnp.mean(jnp.sum(nll * masked, -1) / (t * x.shape[-1]))
      """,
      torch="""
        def mdlm(logits, x, masked, t):
            nll = F.cross_entropy(logits.transpose(1, 2), x, reduction="none")    # (B, T)
            return ((nll * masked).sum(-1) / (t * x.shape[-1])).mean()
      """,
      notes=(("Divide by T (all positions), not by the number of masked ones: the 1/t weight already accounts for "
             "how many were hidden on average. Dividing by the masked count instead gives BERT's loss, which is "
             "not a likelihood bound."),
             ("PyTorch's `F.cross_entropy` wants the class axis second, so (B, T, V) logits are transposed to "
             "(B, V, T); JAX and NumPy index the last axis.")),
      module="45"),
]


def _siglip_inputs(r: np.random.Generator) -> dict:
    n, d = 4, 6
    return {"img": r.normal(size=(n, d)), "txt": r.normal(size=(n, d)), "t": 10.0, "b": -5.0}


ENTRIES += [
    E("sigmoid image-text loss (SigLIP)", "Probability and information",
      "Score every image against every caption in the batch; each pair is its own yes/no question: matching "
      "pairs (the diagonal) should score high, all others low. A learned scale t and bias b map cosine "
      "similarity to a logit.",
      r"\mathcal L = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{n} \log \sigma\big(z_{ij}\,(t\,\hat x_i \cdot \hat y_j + b)\big),"
      r"\quad z_{ij} = \begin{cases}+1 & i = j\\ -1 & i \ne j\end{cases}",
      r"$-\frac{1}{|\mathcal B|}\sum_{i=1}^{|\mathcal B|}\sum_{j=1}^{|\mathcal B|}\log\frac{1}{1+e^{z_{ij}(-t\mathbf x_i\cdot\mathbf y_j+b)}}$ "
      r"(Zhai et al. 2023, eq. 2; $\mathbf x, \mathbf y$ already normalized; their $-t$ in the exponent is the "
      r"same sign convention)",
      "img (n, d), txt (n, d) matching row by row, t, b scalars -> scalar",
      "siglip(img, txt, t, b)", _siglip_inputs,
      python="""
        import math

        def siglip(img, txt, t, b):
            unit = lambda v: [a / math.sqrt(sum(x * x for x in v)) for a in v]
            img, txt = [unit(v) for v in img], [unit(v) for v in txt]
            total = 0.0
            for i, x in enumerate(img):
                for j, y in enumerate(txt):
                    logit = t * sum(a * c for a, c in zip(x, y)) + b
                    z = 1.0 if i == j else -1.0
                    total += math.log(1 + math.exp(-z * logit))      # -log sigmoid(z * logit)
            return total / len(img)
      """,
      numpy="""
        def siglip(img, txt, t, b):
            img = img / np.linalg.norm(img, axis=1, keepdims=True)
            txt = txt / np.linalg.norm(txt, axis=1, keepdims=True)
            z = 2 * np.eye(len(img)) - 1
            return np.logaddexp(0, -z * (t * img @ txt.T + b)).sum() / len(img)
      """,
      jax="""
        def siglip(img, txt, t, b):
            img = img / jnp.linalg.norm(img, axis=1, keepdims=True)
            txt = txt / jnp.linalg.norm(txt, axis=1, keepdims=True)
            z = 2 * jnp.eye(len(img)) - 1
            return -jax.nn.log_sigmoid(z * (t * img @ txt.T + b)).sum() / len(img)
      """,
      torch="""
        def siglip(img, txt, t, b):
            img, txt = F.normalize(img, dim=1), F.normalize(txt, dim=1)
            z = 2 * torch.eye(len(img), dtype=img.dtype) - 1
            return -F.logsigmoid(z * (t * img @ txt.T + b)).sum() / len(img)
      """,
      notes=(("`log(1 + exp(−x))` overflows for very negative x; NumPy's `np.logaddexp(0, −x)`, JAX's "
             "`jax.nn.log_sigmoid` and PyTorch's `F.logsigmoid` are the stable forms."),
             ("SigLIP learns log t (initialized at log 10) and b (at −10), so the many negative pairs start with a "
             "small loss. The loss sums over j and averages over i, not a mean over all n² pairs.")),
      module="43"),
]


def _resize_inputs(r: np.random.Generator) -> dict:
    return {"grid": r.normal(size=(3, 3, 2)), "new": 5}


ENTRIES += [
    E("bilinear resize of a grid (position-embedding interpolation)", "Linear algebra",
      "Stretch a g × g grid of vectors to g' × g': each new cell sits at a fractional position in the old "
      "grid and takes a weighted average of the four old cells around it, weights by distance. How a ViT's "
      "learned position embeddings are reused at a new image resolution.",
      r"y_{i'j'} = \sum_{a,b \in \{0,1\}} w_a^{(i')}\, w_b^{(j')}\, x_{\lfloor u\rfloor + a,\ \lfloor v\rfloor + b},\quad "
      r"u = \big(i' + \tfrac12\big)\tfrac{g}{g'} - \tfrac12,\ w_1 = u - \lfloor u\rfloor,\ w_0 = 1 - w_1",
      r"$u$ is clamped to $[0, g-1]$ at the edges (the half-pixel convention of `jax.image.resize` and "
      r"PyTorch's `align_corners=False`)",
      "grid (g, g, d), new size g' (upsampling) -> (g', g', d)",
      "resize(grid, new)", _resize_inputs,
      python="""
        import math

        def resize(grid, new):
            g = len(grid)

            def taps(i):
                u = min(max((i + 0.5) * g / new - 0.5, 0.0), g - 1.0)
                lo = min(int(math.floor(u)), g - 2)
                return lo, u - lo

            out = []
            for i in range(new):
                (r0, wr), row = taps(i), []
                for j in range(new):
                    c0, wc = taps(j)
                    row.append([(1 - wr) * (1 - wc) * grid[r0][c0][k] + (1 - wr) * wc * grid[r0][c0 + 1][k]
                                + wr * (1 - wc) * grid[r0 + 1][c0][k] + wr * wc * grid[r0 + 1][c0 + 1][k]
                                for k in range(len(grid[0][0]))])
                out.append(row)
            return out
      """,
      numpy="""
        def resize(grid, new):
            g = grid.shape[0]
            u = np.clip((np.arange(new) + 0.5) * g / new - 0.5, 0, g - 1)
            lo = np.minimum(np.floor(u).astype(int), g - 2)
            w = u - lo
            rows = grid[lo] * (1 - w)[:, None, None] + grid[lo + 1] * w[:, None, None]
            return rows[:, lo] * (1 - w)[None, :, None] + rows[:, lo + 1] * w[None, :, None]
      """,
      jax="""
        def resize(grid, new):
            return jax.image.resize(grid, (new, new, grid.shape[-1]), "bilinear")
      """,
      torch="""
        def resize(grid, new):
            x = grid.permute(2, 0, 1)[None]                          # (1, d, g, g): channels first
            y = F.interpolate(x, size=(new, new), mode="bilinear", align_corners=False)
            return y[0].permute(1, 2, 0)
      """,
      notes=(("PyTorch resizes the last two axes of an (N, C, H, W) tensor, so the grid is moved channels-first "
             "and back; JAX's `jax.image.resize` takes the full output shape and resizes every axis whose "
             "size changes."),
             ("For shrinking, `jax.image.resize` antialiases by default (a wider kernel) and PyTorch does not unless "
             "`antialias=True`; for enlarging, as here, they agree.")),
      apis=("jax.image.resize",), module="43"),
]


def _nb_inputs(r: np.random.Generator) -> dict:
    return {"rates": r.uniform(0.05, 0.95, (3, 5)), "prior": r.dirichlet(np.ones(3))}


def _ties_inputs(r: np.random.Generator) -> dict:
    return {"taus": r.normal(size=(3, 40)), "density": 0.25, "lam": 1.0}


def _rome_inputs(r: np.random.Generator) -> dict:
    A = r.normal(size=(6, 6))
    return {"W": r.normal(size=(6, 4)), "k": r.normal(size=6), "v": r.normal(size=4),
            "C_inv": np.linalg.inv(A @ A.T / 6 + np.eye(6))}


ENTRIES += [
    E("naive Bayes readout for binary features", "Probability and information",
      "Count how often each feature fires for each class, assume the features are independent given the class, "
      "and the log-probability of each class becomes a linear layer: one weight per feature and class (the "
      "log-odds of firing) plus a bias. Weights from counting, not from gradients.",
      r"W_{fc} = \log r_{cf} - \log(1 - r_{cf}),\quad b_c = \sum_f \log(1 - r_{cf}) + \log \pi_c,"
      r"\quad \log p(c \mid x) = x^\top W_{:,c} + b_c + \text{const}",
      r"$\log P(c)\prod_f r_{cf}^{x_f}(1-r_{cf})^{1-x_f}$ (Bernoulli naive Bayes)",
      "rates (C, F), prior (C,) -> {W: (F, C), b: (C,)}", "readout(rates, prior)", _nb_inputs,
      python="""
        import math

        def readout(rates, prior):
            C, F = len(rates), len(rates[0])
            W = [[math.log(rates[c][f]) - math.log(1 - rates[c][f]) for c in range(C)] for f in range(F)]
            b = [sum(math.log(1 - rates[c][f]) for f in range(F)) + math.log(prior[c]) for c in range(C)]
            return {"W": W, "b": b}
      """,
      numpy="""
        def readout(rates, prior):
            return {"W": (np.log(rates) - np.log1p(-rates)).T, "b": np.log1p(-rates).sum(1) + np.log(prior)}
      """,
      jax="""
        def readout(rates, prior):
            return {"W": (jnp.log(rates) - jnp.log1p(-rates)).T, "b": jnp.log1p(-rates).sum(1) + jnp.log(prior)}
      """,
      torch="""
        def readout(rates, prior):
            return {"W": (torch.log(rates) - torch.log1p(-rates)).T, "b": torch.log1p(-rates).sum(1) + torch.log(prior)}
      """,
      notes=(("Smooth the counts first (add 1 to each count and 2 to each total): a rate of exactly 0 or 1 gives an "
              "infinite weight."),
             ("The same formula is the first layer of a 1990s OCR system and the readout Toumbourou's LLM-written "
              "MNIST network ended up using (Module 46).")),
      module="46"),
    E("TIES merging of task vectors", "Optimization",
      "Merge several fine-tunes of one model: keep only each task vector's biggest changes, let the tasks vote on "
      "the sign of every weight, and average only the changes that agree with the vote. Small, conflicting "
      "changes stop cancelling the important ones.",
      r"\hat\tau_t = \mathrm{top}_k(\tau_t),\quad \gamma = \mathrm{sgn}\Big(\sum_t \hat\tau_t\Big),\quad "
      r"\tau_m = \lambda\,\frac{\sum_t \hat\tau_t\,[\mathrm{sgn}\,\hat\tau_t = \gamma]}"
      r"{\sum_t [\mathrm{sgn}\,\hat\tau_t = \gamma,\ \hat\tau_t \ne 0]}",
      r"Trim, Elect Sign, Disjoint Merge (Yadav et al. 2023, Alg. 1); task vector $\tau_t = \theta_t - \theta_{\mathrm{pre}}$",
      "taus (n, P), density, lam -> (P,)", "ties(taus, density, lam)", _ties_inputs,
      python="""
        def ties(taus, density, lam):
            n, P = len(taus), len(taus[0])
            k = max(1, round(density * P))
            trimmed = []
            for t in taus:
                kth = sorted((abs(x) for x in t), reverse=True)[k - 1]
                trimmed.append([x if abs(x) >= kth else 0.0 for x in t])
            out = []
            for j in range(P):
                col = [trimmed[i][j] for i in range(n)]
                s = sum(col)
                sign = (s > 0) - (s < 0)
                keep = [x for x in col if x != 0 and ((x > 0) - (x < 0)) == sign]
                out.append(lam * sum(keep) / len(keep) if keep else 0.0)
            return out
      """,
      numpy="""
        def ties(taus, density, lam):
            k = max(1, round(density * taus.shape[1]))
            kth = -np.sort(-np.abs(taus), axis=1)[:, k - 1:k]
            trimmed = np.where(np.abs(taus) >= kth, taus, 0.0)
            agree = (np.sign(trimmed) == np.sign(trimmed.sum(0))) & (trimmed != 0)
            n = agree.sum(0)
            return lam * np.where(n > 0, (trimmed * agree).sum(0) / np.maximum(n, 1), 0.0)
      """,
      jax="""
        def ties(taus, density, lam):
            k = max(1, round(density * taus.shape[1]))
            kth = -jnp.sort(-jnp.abs(taus), axis=1)[:, k - 1:k]
            trimmed = jnp.where(jnp.abs(taus) >= kth, taus, 0.0)
            agree = (jnp.sign(trimmed) == jnp.sign(trimmed.sum(0))) & (trimmed != 0)
            n = agree.sum(0)
            return lam * jnp.where(n > 0, (trimmed * agree).sum(0) / jnp.maximum(n, 1), 0.0)
      """,
      torch="""
        def ties(taus, density, lam):
            k = max(1, round(density * taus.shape[1]))
            kth = torch.topk(taus.abs(), k, dim=1).values[:, -1:]
            trimmed = torch.where(taus.abs() >= kth, taus, torch.zeros_like(taus))
            agree = (torch.sign(trimmed) == torch.sign(trimmed.sum(0))) & (trimmed != 0)
            n = agree.sum(0)
            return lam * torch.where(n > 0, (trimmed * agree).sum(0) / n.clamp(min=1), torch.zeros_like(n, dtype=taus.dtype))
      """,
      notes=(("TIES keeps the top 20% by default. Tools like mergekit apply it tensor by tensor; the paper trims "
              "each whole task vector at once. Module 46 does it tensor by tensor."),
             ("Plain task arithmetic is the special case with no trimming and no vote: "
              "$\\theta = \\theta_{\\mathrm{pre}} + \\lambda\\sum_t \\tau_t$. DARE (Yu et al. 2023) instead drops "
              "a random fraction p of each task vector and rescales the rest by $1/(1-p)$.")),
      module="46"),
    E("rank-one edit of a weight matrix (ROME)", "Linear algebra",
      "Make one input k produce a chosen output v by adding an outer product to W, the smallest change that "
      "does it when 'small' is measured on the inputs the layer usually sees (their covariance C): inputs unlike "
      "k barely move.",
      r"W' = W + \frac{(C^{-1}k)\,(v - k^\top W)}{(C^{-1}k)^\top k},\qquad k^\top W' = v",
      r"$\hat W = W + \Lambda (C^{-1}k_*)^\top$, $\Lambda = (v_* - W k_*) / ((C^{-1}k_*)^\top k_*)$ "
      r"(Meng et al. 2022, eq. 2; their W acts on column vectors, $W k$, so the outer product is transposed)",
      "W (d_in, d_out), k (d_in,), v (d_out,), C_inv (d_in, d_in) -> (d_in, d_out)", "edit(W, k, v, C_inv)",
      _rome_inputs,
      python="""
        def edit(W, k, v, C_inv):
            n, m = len(W), len(W[0])
            u = [sum(C_inv[i][j] * k[j] for j in range(n)) for i in range(n)]
            uk = sum(a * b for a, b in zip(u, k))
            r = [v[j] - sum(k[i] * W[i][j] for i in range(n)) for j in range(m)]   # v − kᵀW
            return [[W[i][j] + u[i] * r[j] / uk for j in range(m)] for i in range(n)]
      """,
      numpy="""
        def edit(W, k, v, C_inv):
            u = C_inv @ k
            return W + np.outer(u, v - k @ W) / (u @ k)
      """,
      jax="""
        def edit(W, k, v, C_inv):
            u = C_inv @ k
            return W + jnp.outer(u, v - k @ W) / (u @ k)
      """,
      torch="""
        def edit(W, k, v, C_inv):
            u = C_inv @ k
            return W + torch.outer(u, v - k @ W) / (u @ k)
      """,
      notes=(("With C = I this is the plain minimum-norm rank-one fix. C (the covariance of the layer's inputs on "
              "ordinary text) makes the change smallest where inputs usually lie, so unrelated facts move less."),
             ("ROME picks k as the MLP's hidden activation at the subject's last token and finds v with a few gradient "
              "steps on one vector; MEMIT spreads many such edits over several layers.")),
      apis=("jax.numpy.outer",), module="46"),
]


def _topk_inputs(r: np.random.Generator) -> dict:
    return {"pre": r.normal(size=(4, 12)), "k": 3}


def _fvu_inputs(r: np.random.Generator) -> dict:
    x = r.normal(size=(20, 5)) * 2 + 1
    return {"x": x, "x_hat": x + 0.3 * r.normal(size=x.shape)}


def _ablate_inputs(r: np.random.Generator) -> dict:
    u = r.normal(size=5)
    return {"W": r.normal(size=(7, 5)), "u": u / np.linalg.norm(u), "alpha": 1.0}


def _knn_inputs(r: np.random.Generator) -> dict:
    X = r.normal(size=(30, 6))
    return {"X": X, "Y": X @ r.normal(size=(6, 4)) + 0.5 * r.normal(size=(30, 4)), "k": 4}


ENTRIES += [
    E("TopK sparse code", "Neural network layers",
      "A sparse autoencoder's code: of all its latent units, keep only the k that respond most to this input "
      "(after a ReLU) and silence the rest. Sparsity is set exactly by k, not tuned with a penalty.",
      r"z = \operatorname{TopK}_k\big(\mathrm{ReLU}((x - b_{dec}) W_{enc} + b_{enc})\big),\quad "
      r"\hat x = z W_{dec} + b_{dec}",
      r"$z = \mathrm{TopK}(W_{enc}(x - b_{pre}))$ (Gao et al. 2024, eq. 3; ReLU after TopK, same result)",
      "pre (N, F), k: int -> (N, F)", "topk(pre, k)", _topk_inputs,
      python="""
        def topk(pre, k):
            out = []
            for row in pre:
                kth = sorted(row, reverse=True)[k - 1]
                out.append([max(v, 0.0) if v >= kth else 0.0 for v in row])
            return out
      """,
      numpy="""
        def topk(pre, k):
            kth = -np.sort(-pre, axis=-1)[..., k - 1:k]              # the k-th largest per row
            return np.where(pre >= kth, np.maximum(pre, 0), 0.0)
      """,
      jax="""
        def topk(pre, k):
            kth = jax.lax.top_k(pre, k)[0][..., -1:]
            return jnp.where(pre >= kth, jax.nn.relu(pre), 0.0)
      """,
      torch="""
        def topk(pre, k):
            kth = torch.topk(pre, k, dim=-1).values[..., -1:]
            return torch.where(pre >= kth, torch.relu(pre), torch.zeros_like(pre))
      """,
      notes=(("A threshold mask (keep values ≥ the k-th largest) instead of scattering top_k's indices: the same "
             "code unless two values tie, and gradients flow through the kept entries as usual."),
             ("The older recipe is a plain ReLU plus an L1 penalty on z; the penalty also shrinks the kept values, "
             "which TopK avoids.")),
      module="47"),
    E("fraction of variance unexplained", "Retrieval and evaluation",
      "How much of the data's spread a reconstruction misses: squared error divided by the squared distance "
      "to the mean. 0 is perfect; 1 is no better than predicting the average every time.",
      r"\mathrm{FVU} = \frac{\sum_n \lVert x_n - \hat x_n \rVert^2}{\sum_n \lVert x_n - \bar x \rVert^2}",
      r"$1 - R^2$, 'normalized MSE' (Gao et al. 2024)",
      "x (N, d), x_hat (N, d) -> ()", "fvu(x, x_hat)", _fvu_inputs,
      python="""
        def fvu(x, x_hat):
            n, d = len(x), len(x[0])
            mean = [sum(r[j] for r in x) / n for j in range(d)]
            err = sum((a - b) ** 2 for r, h in zip(x, x_hat) for a, b in zip(r, h))
            var = sum((a - m) ** 2 for r in x for a, m in zip(r, mean))
            return err / var
      """,
      numpy="""
        def fvu(x, x_hat):
            return ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
      """,
      jax="""
        def fvu(x, x_hat):
            return ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
      """,
      torch="""
        def fvu(x, x_hat):
            return ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
      """,
      notes=(("FVU alone can flatter an SAE: a few huge-norm tokens dominate the variance, so explaining them makes "
             "FVU small while ordinary tokens are reconstructed badly. Module 47 drops them first."),),
      module="47"),
    E("directional ablation of a weight matrix", "Linear algebra",
      "Stop a matrix from writing anything along direction u: subtract, from each of its outputs, that output's "
      "component along u. alpha = 1 removes it, alpha = 2 flips it, alpha = 0 leaves it alone.",
      r"W' = W - \alpha\,(W u)\,u^\top,\qquad (xW')\cdot u = (1-\alpha)\,(xW)\cdot u \quad (\lVert u \rVert = 1)",
      r"$W'_{out} = W_{out} - \hat r \hat r^\top W_{out}$ (Arditi et al. 2024; their W acts on column vectors)",
      "W (d_in, d), u (d,), alpha: float -> (d_in, d)", "ablate(W, u, alpha)", _ablate_inputs,
      python="""
        def ablate(W, u, alpha):
            out = []
            for row in W:
                c = sum(a * b for a, b in zip(row, u))                 # this row's component along u
                out.append([a - alpha * c * b for a, b in zip(row, u)])
            return out
      """,
      numpy="""
        def ablate(W, u, alpha):
            return W - alpha * np.outer(W @ u, u)
      """,
      jax="""
        def ablate(W, u, alpha):
            return W - alpha * jnp.outer(W @ u, u)
      """,
      torch="""
        def ablate(W, u, alpha):
            return W - alpha * torch.outer(W @ u, u)
      """,
      notes=(("Apply it to every matrix that writes to the residual stream (embeddings, attention outputs, MLP "
             "down-projections) and the stream never contains u: the same as projecting u out of the activations "
             "after every write, but with no runtime hook."),
             "With tied embeddings, untie the output head first, or the edit changes the unembedding too."),
      module="47", also=("46",)),
    E("mutual nearest-neighbour alignment", "Retrieval and evaluation",
      "Do two models agree about which inputs are alike? For each input, list its k nearest neighbours in each "
      "model's space and count how many the two lists share; average the overlap. It ignores scale and rotation, "
      "and works across different widths.",
      r"m_{kNN}(X, Y) = \frac{1}{N}\sum_i \frac{\lvert \mathcal N_k^X(i) \cap \mathcal N_k^Y(i)\rvert}{k}",
      r"$m_{NN}$ (Huh et al. 2024, eq. 2)",
      "X (N, d1), Y (N, d2), k: int -> ()", "mknn(X, Y, k)", _knn_inputs,
      python="""
        def mknn(X, Y, k):
            def nbrs(A):
                unit = [[a / sum(b * b for b in r) ** 0.5 for a in r] for r in A]
                out = []
                for i, r in enumerate(unit):
                    sims = [(sum(a * b for a, b in zip(r, s)), j) for j, s in enumerate(unit) if j != i]
                    out.append({j for _, j in sorted(sims, reverse=True)[:k]})
                return out
            return sum(len(a & b) for a, b in zip(nbrs(X), nbrs(Y))) / (k * len(X))
      """,
      numpy="""
        def mknn(X, Y, k):
            def nbrs(A):
                A = A / np.linalg.norm(A, axis=1, keepdims=True)
                S = A @ A.T
                np.fill_diagonal(S, -np.inf)
                return np.argsort(-S, axis=1)[:, :k]
            a, b = nbrs(X), nbrs(Y)
            return (a[:, :, None] == b[:, None, :]).any(-1).sum() / (k * len(X))
      """,
      jax="""
        def mknn(X, Y, k):
            def nbrs(A):
                A = A / jnp.linalg.norm(A, axis=1, keepdims=True)
                S = jnp.where(jnp.eye(len(A), dtype=bool), -jnp.inf, A @ A.T)
                return jax.lax.top_k(S, k)[1]
            a, b = nbrs(X), nbrs(Y)
            return (a[:, :, None] == b[:, None, :]).any(-1).sum() / (k * len(X))
      """,
      torch="""
        def mknn(X, Y, k):
            def nbrs(A):
                A = A / A.norm(dim=1, keepdim=True)
                S = (A @ A.T).fill_diagonal_(-float("inf"))
                return S.topk(k, dim=1).indices
            a, b = nbrs(X), nbrs(Y)
            return (a[:, :, None] == b[:, None, :]).any(-1).sum() / (k * len(X))
      """,
      notes=(("Chance is k/(N−1). Module 47 measures it between SmolLM2-135M and SmolLM2-360M layer by layer, "
             "against a shuffled pairing."),),
      module="47"),
]


def _atp_inputs(r: np.random.Generator) -> dict:
    return {"grad": r.normal(size=(4, 3, 6)), "clean": r.normal(size=(4, 3, 6)), "corrupt": r.normal(size=(4, 3, 6))}


def _influence_inputs(r: np.random.Generator) -> dict:
    A = np.triu(r.normal(size=(7, 7)), 1)  # row = target, column = source; upper triangular: a DAG
    w = np.zeros(7)
    w[0] = 0.7
    w[1] = 0.3
    return {"A": A, "w": w}


def _ld_inputs(r: np.random.Generator) -> dict:
    return {"logits": r.normal(size=(5, 4, 9)), "io": r.integers(0, 9, 5), "s": r.integers(0, 9, 5)}


ENTRIES += [
    E("logit difference", "Retrieval and evaluation",
      "How strongly the model prefers the right answer to one specific wrong answer, at the last position, averaged "
      "over prompts. Unlike a probability it is linear in the final residual stream, so contributions add up.",
      r"\mathrm{LD} = \frac{1}{B}\sum_b \big(z_{b,T,\,io_b} - z_{b,T,\,s_b}\big)",
      r"$\mathrm{logit}(IO) - \mathrm{logit}(S)$ (Wang et al. 2022)",
      "logits (B, T, V), io (B,), s (B,) -> ()", "logit_diff(logits, io, s)", _ld_inputs,
      python="""
        def logit_diff(logits, io, s):
            total = 0.0
            for b in range(len(logits)):
                last = logits[b][-1]
                total += last[io[b]] - last[s[b]]
            return total / len(logits)
      """,
      numpy="""
        def logit_diff(logits, io, s):
            last = logits[:, -1]
            b = np.arange(len(last))
            return (last[b, io] - last[b, s]).mean()
      """,
      jax="""
        def logit_diff(logits, io, s):
            last = logits[:, -1]
            pick = lambda ids: jnp.take_along_axis(last, ids[:, None], 1)[:, 0]
            return (pick(io) - pick(s)).mean()
      """,
      torch="""
        def logit_diff(logits, io, s):
            last = logits[:, -1]
            return (last.gather(1, io[:, None]) - last.gather(1, s[:, None])).mean()
      """,
      notes=(("Normalised against a clean and a corrupted run, (LD − LD_corrupt) / (LD_clean − LD_corrupt), it becomes "
              "the patching effect Module 48 plots: 0 = nothing restored, 1 = everything."),),
      module="48"),
    E("attribution patching", "Calculus and autodiff",
      "Estimate what pasting a clean activation into a corrupted run would do to a metric, for every activation at "
      "once: the gradient on the corrupted run times the difference between the two activations. One backward pass "
      "replaces thousands of forward passes.",
      r"\Delta m \approx \nabla_a m\,\big\rvert_{a_{\text{corr}}} \cdot (a_{\text{clean}} - a_{\text{corr}})",
      r"$\hat c_{\text{AtP}}(n) = (n(x^{\text{clean}}) - n(x^{\text{noise}}))^\top \frac{\partial \mathcal L}{\partial n}"
      r"\big\rvert_{n(x^{\text{noise}})}$ (Kramár et al. 2024)",
      "grad, clean, corrupt (..., d) -> (...)", "atp(grad, clean, corrupt)", _atp_inputs,
      python="""
        def atp(grad, clean, corrupt):
            def walk(g, c, r):
                if not isinstance(g[0], list):
                    return sum(gi * (ci - ri) for gi, ci, ri in zip(g, c, r))
                return [walk(*t) for t in zip(g, c, r)]
            return walk(grad, clean, corrupt)
      """,
      numpy="""
        def atp(grad, clean, corrupt):
            return (grad * (clean - corrupt)).sum(-1)
      """,
      jax="""
        def atp(grad, clean, corrupt):
            return (grad * (clean - corrupt)).sum(-1)
      """,
      torch="""
        def atp(grad, clean, corrupt):
            return (grad * (clean - corrupt)).sum(-1)
      """,
      notes=(("Get grad with jax.grad of the metric with respect to an additive offset at the activation (zero at "
              "the corrupted run): with jaxprobe, patch the tap with x + delta and differentiate in delta."),
             ("It is exact for a linear metric and wrong where attention saturates; Module 48 shows a head it "
              "underrates by 3×.")),
      module="48"),
    E("total influence in a graph", "Linear algebra",
      "How much each node matters to the output through every path. Normalise the graph so each node's incoming "
      "weights sum to 1, then add up direct effects, two-step effects, three-step effects… The series stops because "
      "a causal graph has no cycles.",
      r"\hat A_{ts} = \frac{\lvert A_{ts}\rvert}{\sum_{s'}\lvert A_{ts'}\rvert},\qquad x = w\hat A + w\hat A^2 + \dots = "
      r"w\hat A\,(I-\hat A)^{-1}",
      r"$B = (I - A)^{-1} - I$, influence $= \text{logit weights}\cdot B$ (Ameisen et al. 2025)",
      "A (n, n) row = target, w (n,) -> (n,)", "influence(A, w)", _influence_inputs,
      python="""
        def influence(A, w):
            n = len(A)
            An = []
            for row in A:
                s = sum(abs(a) for a in row)
                An.append([abs(a) / s if s else 0.0 for a in row])
            x, v = [0.0] * n, [sum(w[t] * An[t][s] for t in range(n)) for s in range(n)]
            for _ in range(n):
                x = [a + b for a, b in zip(x, v)]
                v = [sum(v[t] * An[t][s] for t in range(n)) for s in range(n)]
            return x
      """,
      numpy="""
        def influence(A, w):
            a = np.abs(A)
            s = a.sum(1, keepdims=True)
            An = np.divide(a, s, out=np.zeros_like(a), where=s > 0)
            return w @ An @ np.linalg.inv(np.eye(len(A)) - An)
      """,
      jax="""
        def influence(A, w):
            a = jnp.abs(A)
            s = a.sum(1, keepdims=True)
            An = jnp.where(s > 0, a / jnp.where(s > 0, s, 1.0), 0.0)
            return w @ An @ jnp.linalg.inv(jnp.eye(len(A)) - An)
      """,
      torch="""
        def influence(A, w):
            a = A.abs()
            s = a.sum(1, keepdim=True)
            An = torch.where(s > 0, a / s.clamp_min(1e-30), torch.zeros_like(a))
            return w @ An @ torch.linalg.inv(torch.eye(len(A), dtype=A.dtype) - An)
      """,
      notes=(("For a graph of 10,000 nodes the inverse is heavy; the series w Â, w Â², … needs only matrix-vector "
              "products and stops after the graph's depth (jaxprobe.trace.influence does that)."),
             "Influence decides what to keep: prune features until the kept ones carry, say, 80% of the total."),
      module="48"),
]


def _refusal_inputs(r: np.random.Generator) -> dict:
    return {"logits": r.normal(size=(4, 11)) * 2, "ids": np.array([2, 7, 9])}


def _dim_inputs(r: np.random.Generator) -> dict:
    return {"harmful": r.normal(size=(3, 6, 5)) + 0.5, "harmless": r.normal(size=(3, 8, 5))}


ENTRIES += [
    E("refusal score", "Probability and information",
      "How likely is the model to start its answer with a refusal? Add up the probability of the tokens that "
      "begin refusals (\"I\", \"Sorry\", \"As\") at the first answer position and turn it into log-odds: positive "
      "means a refusal is more likely than not, and the scale doesn't saturate the way a probability does.",
      r"p = \sum_{t \in \mathcal R} \mathrm{softmax}(z)_t,\qquad s = \log p - \log(1 - p)",
      r"$\log \frac{P_{\text{refusal}}}{1 - P_{\text{refusal}}}$ (Arditi et al. 2024, App. D; their $\mathcal R$ is a "
      r"hand-picked set of first tokens per model)",
      "logits (N, V), ids (k,) -> (N,)", "refusal_score(logits, ids)", _refusal_inputs,
      python="""
        import math

        def refusal_score(logits, ids):
            out = []
            for z in logits:
                m = max(z)
                e = [math.exp(v - m) for v in z]
                p = sum(e[i] for i in ids) / sum(e)
                out.append(math.log(p + 1e-8) - math.log(1 - p + 1e-8))
            return out
      """,
      numpy="""
        def refusal_score(logits, ids):
            z = logits - logits.max(-1, keepdims=True)
            q = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
            p = q[:, ids].sum(-1)
            return np.log(p + 1e-8) - np.log(1 - p + 1e-8)
      """,
      jax="""
        def refusal_score(logits, ids):
            p = jax.nn.softmax(logits, -1)[:, ids].sum(-1)
            return jnp.log(p + 1e-8) - jnp.log(1 - p + 1e-8)
      """,
      torch="""
        def refusal_score(logits, ids):
            p = torch.softmax(logits, -1)[:, ids].sum(-1)
            return torch.log(p + 1e-8) - torch.log(1 - p + 1e-8)
      """,
      notes=("Compute the softmax in float32: in bf16, p near 1 rounds to exactly 1 and log(1 − p) blows up.",
             ("It reads one position, so it's cheap enough to score 150 candidate directions; check what it misses by "
              "generating full answers for the one you pick.")),
      module="50"),
    E("difference-in-means direction", "Arrays and reductions",
      "Collect the residual stream on prompts with a behaviour and prompts without it; the direction from the "
      "second average to the first is the cheapest guess at where the behaviour lives. One per layer.",
      r"r^{(\ell)} = \frac{1}{\lvert H\rvert}\sum_{x \in H} h^{(\ell)}(x) - \frac{1}{\lvert B\rvert}\sum_{x \in B} h^{(\ell)}(x)",
      r"$r_i^{(l)} = \mu_i^{(l)} - \nu_i^{(l)}$ (Arditi et al. 2024, eq. 3; $i$ is a token position)",
      "harmful (L, N_h, d), harmless (L, N_b, d) -> (L, d)", "diff_in_means(harmful, harmless)", _dim_inputs,
      python="""
        def diff_in_means(harmful, harmless):
            def mean(rows):
                return [sum(c) / len(rows) for c in zip(*rows)]
            return [[a - b for a, b in zip(mean(h), mean(n))] for h, n in zip(harmful, harmless)]
      """,
      numpy="""
        def diff_in_means(harmful, harmless):
            return harmful.mean(1) - harmless.mean(1)
      """,
      jax="""
        def diff_in_means(harmful, harmless):
            return harmful.mean(1) - harmless.mean(1)
      """,
      torch="""
        def diff_in_means(harmful, harmless):
            return harmful.mean(1) - harmless.mean(1)
      """,
      notes=(("Average in float32: the residual stream of a bf16 model has a few large dimensions, and a bf16 sum "
              "of hundreds of rows loses the small differences that make up the direction."),
             ("The two sets can differ in more than the behaviour (length, topic); Arditi et al. use the last few "
              "template tokens after the request, so the content itself is out of view.")),
      module="50", also=("47",)),
]


def _ewc_inputs(r: np.random.Generator) -> dict:
    return {"params": r.normal(size=(5, 4)), "anchor": r.normal(size=(5, 4)),
            "fisher": np.abs(r.normal(size=(5, 4)))}


def _tkl_inputs(r: np.random.Generator) -> dict:
    return {"t": r.normal(size=(3, 4, 9)) * 2, "s": r.normal(size=(3, 4, 9)) * 2}


ENTRIES += [
    E("elastic weight consolidation penalty", "Optimization",
      "Let each weight move, but charge for moving the ones the old task relied on: a spring back to the old value, "
      "as stiff as that weight's Fisher information (how much the old task's loss bends when the weight moves).",
      r"\Omega(\theta) = \sum_i F_i\,(\theta_i - \theta^*_i)^2",
      r"$\mathcal L(\theta) = \mathcal L_B(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta^*_{A,i})^2$ "
      r"(Kirkpatrick et al. 2017, eq. 3)",
      "params, anchor, fisher (same shapes) -> scalar", "ewc(params, anchor, fisher)", _ewc_inputs,
      python="""
        def ewc(params, anchor, fisher):
            return sum(f * (p - a) ** 2 for pr, ar, fr in zip(params, anchor, fisher)
                       for p, a, f in zip(pr, ar, fr))
      """,
      numpy="""
        def ewc(params, anchor, fisher):
            return np.sum(fisher * (params - anchor) ** 2)
      """,
      jax="""
        def ewc(params, anchor, fisher):   # params etc. can be whole pytrees
            terms = jax.tree.map(lambda p, a, f: jnp.sum(f * (p - a) ** 2), params, anchor, fisher)
            return sum(jax.tree.leaves(terms))
      """,
      torch="""
        def ewc(params, anchor, fisher):
            return (fisher * (params - anchor) ** 2).sum()
      """,
      notes=(("The diagonal Fisher is usually estimated as the mean *squared gradient* of the old task's loss (the "
              "empirical Fisher); Kirkpatrick et al. sample labels from the model instead. Both are cheap: one "
              "backward pass per batch."),
             ("For a model, `params` is a pytree: `jax.tree.map` pairs the three trees leaf by leaf. In PyTorch you "
              "loop over `named_parameters()` and keep dicts of anchors and Fishers by name.")),
      module="49"),
    E("per-token KL: forward, reverse, Jensen-Shannon", "Probability and information",
      "Distillation compares the teacher's and the student's whole next-token distributions at every position. "
      "Forward KL(teacher ‖ student) punishes the student for missing anything the teacher thinks likely "
      "(mass-covering); reverse KL(student ‖ teacher) punishes it for saying anything the teacher thinks unlikely "
      "(mode-seeking); Jensen-Shannon sits between and is bounded by log 2.",
      r"\mathrm{KL}(p\Vert q) = \sum_v p_v(\log p_v - \log q_v),\quad "
      r"\mathrm{JSD} = \tfrac12\mathrm{KL}(p\Vert m) + \tfrac12\mathrm{KL}(q\Vert m),\ m = \tfrac{p+q}{2}",
      r"$D_{JSD(\beta)}$ with $\beta = 0.5$ (Agarwal et al. 2023, GKD); SDFT's analytic per-token estimator "
      r"(Shenfeld et al. 2026, App. A.1)",
      "teacher logits (..., V), student logits (..., V) -> three arrays (...)", "tkl(t, s)", _tkl_inputs,
      python="""
        import math

        def tkl(t, s):
            def logsm(z):
                m = max(z)
                lse = m + math.log(sum(math.exp(v - m) for v in z))
                return [v - lse for v in z]

            def one(zt, zs):
                lt, ls = logsm(zt), logsm(zs)
                p, q = [math.exp(v) for v in lt], [math.exp(v) for v in ls]
                lm = [math.log((a + b) / 2) for a, b in zip(p, q)]
                fwd = sum(a * (x - y) for a, x, y in zip(p, lt, ls))
                rev = sum(b * (y - x) for b, x, y in zip(q, lt, ls))
                jsd = 0.5 * sum(a * (x - m) for a, x, m in zip(p, lt, lm)) + \\
                    0.5 * sum(b * (y - m) for b, y, m in zip(q, ls, lm))
                return fwd, rev, jsd

            out = [[one(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(t, s)]
            return tuple([[[c[k] for c in row] for row in out] for k in range(3)])
      """,
      numpy="""
        def tkl(t, s):
            def logsm(z):
                z = z - z.max(-1, keepdims=True)
                return z - np.log(np.exp(z).sum(-1, keepdims=True))
            lt, ls = logsm(t), logsm(s)
            p, q = np.exp(lt), np.exp(ls)
            lm = np.log((p + q) / 2)
            fwd = (p * (lt - ls)).sum(-1)
            rev = (q * (ls - lt)).sum(-1)
            return fwd, rev, 0.5 * (p * (lt - lm)).sum(-1) + 0.5 * (q * (ls - lm)).sum(-1)
      """,
      jax="""
        def tkl(t, s):
            lt, ls = jax.nn.log_softmax(t, -1), jax.nn.log_softmax(s, -1)
            lm = jnp.logaddexp(lt, ls) - jnp.log(2.0)          # log of the average, without leaving log space
            fwd = jnp.sum(jnp.exp(lt) * (lt - ls), -1)
            rev = jnp.sum(jnp.exp(ls) * (ls - lt), -1)
            return fwd, rev, 0.5 * jnp.sum(jnp.exp(lt) * (lt - lm), -1) + 0.5 * jnp.sum(jnp.exp(ls) * (ls - lm), -1)
      """,
      torch="""
        def tkl(t, s):
            lt, ls = F.log_softmax(t, -1), F.log_softmax(s, -1)
            lm = torch.logaddexp(lt, ls) - math.log(2.0)
            fwd = (lt.exp() * (lt - ls)).sum(-1)
            rev = (ls.exp() * (ls - lt)).sum(-1)
            return fwd, rev, 0.5 * (lt.exp() * (lt - lm)).sum(-1) + 0.5 * (ls.exp() * (ls - lm)).sum(-1)
      """,
      notes=(("Work in log space: with a 152,000-token vocabulary most probabilities underflow in bf16, and "
              "`jnp.logaddexp` gives log((p + q)/2) without exponentiating."),
             ("Which direction is \"forward\" depends on the paper. Here forward = KL(teacher ‖ student), as in GKD "
              "and SDFT; `optax.losses.kl_divergence(log_q, p)` and `F.kl_div(log_q, p)` both compute it with the "
              "student's log-probabilities first.")),
      module="49", apis=("jax.numpy.logaddexp",)),
]
