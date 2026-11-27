"""Rosetta entries for layers, attention and inference (Modules 12–21, 33): convolution, pooling, embeddings,
attention and its relatives, sampling filters, quantization, LoRA, perplexity."""
import numpy as np

from learnjax.rosetta.entry import Entry as E


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


ENTRIES: list[E] = [
    # ---------------------------------------------------------------- Neural network layers
    E("2-D convolution", "Neural network layers",
      "Slide a small kernel over the image; at every position, multiply the patch under it by the kernel and "
      "add everything up. One output channel per kernel.",
      r"y_{n,i,j,o} = \sum_{\Delta i,\, \Delta j,\, c} x_{n,\, i+\Delta i,\, j+\Delta j,\, c}\; "
      r"k_{\Delta i,\, \Delta j,\, c,\, o}",
      r"$y = x * k$ or $x \star k$; $W * x + b$ in CNN papers (written as convolution, computed as "
      r"cross-correlation)",
      "x (N, H, W, C_in), kernel (KH, KW, C_in, C_out) -> (N, H, W, C_out) with SAME padding, stride 1",
      "conv2d(x, k)", lambda r: {"x": r.normal(size=(1, 5, 5, 2)), "k": r.normal(size=(3, 3, 2, 3))},
      python="""
        def conv2d(x, k):  # x: [n][h][w][c], k: [kh][kw][c][o]; stride 1, zero "SAME" padding
            KH, KW, C, O = len(k), len(k[0]), len(k[0][0]), len(k[0][0][0])
            out = []
            for img in x:
                H, W = len(img), len(img[0])
                out.append([[[sum(img[i + di - KH // 2][j + dj - KW // 2][c] * k[di][dj][c][o]
                                  for di in range(KH) for dj in range(KW) for c in range(C)
                                  if 0 <= i + di - KH // 2 < H and 0 <= j + dj - KW // 2 < W)
                              for o in range(O)] for j in range(W)] for i in range(H)])
            return out
      """,
      numpy="""
        def conv2d(x, k):
            KH, KW = k.shape[:2]
            H, W = x.shape[1:3]
            xp = np.pad(x, ((0, 0), (KH // 2, KH // 2), (KW // 2, KW // 2), (0, 0)))
            return sum(np.einsum("nhwc,co->nhwo", xp[:, di:di + H, dj:dj + W], k[di, dj])
                       for di in range(KH) for dj in range(KW))
      """,
      jax="""
        def conv2d(x, k):
            return jax.lax.conv_general_dilated(x, k, window_strides=(1, 1), padding="SAME",
                                                dimension_numbers=("NHWC", "HWIO", "NHWC"))
      """,
      jax_by_hand="""
        def conv2d(x, k):  # one shifted matmul per kernel offset
            KH, KW = k.shape[:2]
            H, W = x.shape[1:3]
            xp = jnp.pad(x, ((0, 0), (KH // 2, KH // 2), (KW // 2, KW // 2), (0, 0)))
            return sum(jnp.einsum("nhwc,co->nhwo", xp[:, di:di + H, dj:dj + W], k[di, dj])
                       for di in range(KH) for dj in range(KW))
      """,
      torch="""
        def conv2d(x, k):
            y = F.conv2d(x.permute(0, 3, 1, 2), k.permute(3, 2, 0, 1), padding="same")  # NCHW, OIHW
            return y.permute(0, 2, 3, 1)
      """,
      einsum='"nhwc,co->nhwo" per kernel offset',
      notes=(("Every library computes cross-correlation: the kernel is *not* flipped (the loop version above, "
             "with no flip, matches all of them). Papers still call it convolution."),
             ("Layouts: the course and Flax use NHWC images and HWIO kernels; PyTorch uses NCHW and stores "
             "`Conv2d.weight` as (out, in, kh, kw). Permute both when porting."),
             ("`flax.nnx.Conv(c_in, c_out, (3, 3), use_bias=False)` holds the same HWIO kernel and pads "
             "\"SAME\" by default; `torch.nn.Conv2d` pads 0 by default.")),
      apis=("jax.lax.conv_general_dilated", "flax.nnx.Conv"),
      module="12", also=("33",)),
    E("max pooling", "Neural network layers",
      "Shrink a feature map by keeping only the largest value in each small window.",
      r"y_{n,i,j,c} = \max_{0 \le a, b < s} x_{n,\, s i + a,\, s j + b,\, c}", "",
      "(N, H, W, C) -> (N, H // s, W // s, C), window s, stride s", "max_pool(x)",
      lambda r: {"x": r.normal(size=(1, 5, 5, 2))},
      python="""
        def max_pool(x, size=2):
            out = []
            for img in x:
                H, W, C = len(img), len(img[0]), len(img[0][0])
                out.append([[[max(img[i + a][j + b][c] for a in range(size) for b in range(size))
                              for c in range(C)]
                             for j in range(0, W - size + 1, size)]
                            for i in range(0, H - size + 1, size)])
            return out
      """,
      numpy="""
        def max_pool(x, size=2):
            N, H, W, C = x.shape
            x = x[:, :H // size * size, :W // size * size]  # drop the leftover row/column
            return x.reshape(N, H // size, size, W // size, size, C).max(axis=(2, 4))
      """,
      jax="""
        def max_pool(x, size=2):
            window = (1, size, size, 1)
            return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, window, window, "VALID")
      """,
      torch="""
        def max_pool(x, size=2):
            return F.max_pool2d(x.permute(0, 3, 1, 2), size).permute(0, 2, 3, 1)
      """,
      notes=(("`reduce_window` takes the reduction's identity (`-inf` for max) and a binary op (`jax.lax.max`); "
             "average pooling is the same call with `0.0`, `jax.lax.add`, then divide by the window size."),
             ("With \"VALID\" a 5×5 input gives 2×2: the leftover row and column are dropped, like PyTorch's "
             "default `ceil_mode=False`.")),
      apis=("jax.lax.reduce_window", "jax.lax.max"),
      module="12"),
    E("transposed convolution", "Neural network layers",
      "Upsample: every input pixel stamps a scaled copy of the kernel onto a larger output, s pixels apart; "
      "where stamps overlap they add.",
      r"y_{n,\, s i + \Delta i,\, s j + \Delta j,\, o} \mathrel{+}= x_{n,i,j,c}\; k_{\Delta i,\, \Delta j,\, c,\, o}",
      r"\"deconvolution\" (a misnomer), fractionally strided convolution, $\mathrm{ConvT}$",
      "(N, H, W, C_in), kernel (KH, KW, C_in, C_out) -> (N, (H-1)s + KH, (W-1)s + KW, C_out), VALID padding",
      "conv_transpose(x, k)", lambda r: {"x": r.normal(size=(1, 3, 3, 2)), "k": r.normal(size=(3, 3, 2, 1))},
      python="""
        def conv_transpose(x, k, s=2):
            KH, KW, C, O = len(k), len(k[0]), len(k[0][0]), len(k[0][0][0])
            out = []
            for img in x:
                H, W = len(img), len(img[0])
                y = [[[0.0] * O for _ in range((W - 1) * s + KW)] for _ in range((H - 1) * s + KH)]
                for i in range(H):
                    for j in range(W):
                        for di in range(KH):
                            for dj in range(KW):
                                for c in range(C):
                                    for o in range(O):
                                        y[i * s + di][j * s + dj][o] += img[i][j][c] * k[di][dj][c][o]
                out.append(y)
            return out
      """,
      numpy="""
        def conv_transpose(x, k, s=2):
            (N, H, W, _), (KH, KW, _, O) = x.shape, k.shape
            y = np.zeros((N, (H - 1) * s + KH, (W - 1) * s + KW, O))
            for i in range(H):
                for j in range(W):
                    y[:, i * s:i * s + KH, j * s:j * s + KW] += np.einsum("nc,hwco->nhwo", x[:, i, j], k)
            return y
      """,
      jax="""
        from flax import nnx

        def conv_transpose(x, k, s=2):
            layer = nnx.ConvTranspose(k.shape[2], k.shape[3], k.shape[:2], strides=(s, s), padding="VALID",
                                      use_bias=False, rngs=nnx.Rngs(0))
            layer.kernel[...] = k[::-1, ::-1]  # Flax's default kernel is the spatially flipped one
            return layer(x)
      """,
      jax_by_hand="""
        def conv_transpose(x, k, s=2):
            (N, H, W, _), (KH, KW, _, O) = x.shape, k.shape
            y = jnp.zeros((N, (H - 1) * s + KH, (W - 1) * s + KW, O))
            for i in range(H):
                for j in range(W):
                    stamp = jnp.einsum("nc,hwco->nhwo", x[:, i, j], k)
                    y = y.at[:, i * s:i * s + KH, j * s:j * s + KW].add(stamp)
            return y
      """,
      torch="""
        def conv_transpose(x, k, s=2):
            y = F.conv_transpose2d(x.permute(0, 3, 1, 2), k.permute(2, 3, 0, 1), stride=s)  # weight (in, out, kh, kw)
            return y.permute(0, 2, 3, 1)
      """,
      notes=(("Kernel-flip trap: for the same weights, Flax's `ConvTranspose` (default `transpose_kernel=False`) "
             "equals PyTorch's `conv_transpose2d` only with the kernel flipped in both spatial axes, "
             "`k[::-1, ::-1]`. Flip when porting weights either way."),
             ("PyTorch stores `ConvTranspose2d.weight` as (in, out, kh, kw), the reverse of `Conv2d`'s "
             "(out, in, kh, kw)."),
             ("Module 33's decoders use Flax's default padding \"SAME\", which gives exactly H·s (7 → 14); "
             "\"VALID\" gives (H−1)·s + KH.")),
      apis=("flax.nnx.ConvTranspose",),
      module="33"),
    E("embedding lookup", "Neural network layers",
      "Turn token ids into vectors by reading row `id` of a learned table: a one-hot vector times the table, "
      "without the multiply.",
      r"e_{t} = E_{x_t,:} = \mathrm{onehot}(x_t)^\top E,\quad E \in \mathbb{R}^{V \times d}",
      r"$W_e$, $E[x]$, $x W_E$ (one-hot rows times the embedding matrix)", "ids (B, T) int, table (V, d) -> (B, T, d)",
      "embed(ids, table)", lambda r: {"ids": np.array([[0, 3, 3, 5], [1, 0, 4, 2]]), "table": r.normal(size=(6, 3))},
      python="""
        def embed(ids, table):
            return [[list(table[i]) for i in row] for row in ids]
      """,
      numpy="""
        def embed(ids, table):
            return table[ids]
      """,
      jax="""
        from flax import nnx

        def embed(ids, table):
            layer = nnx.Embed(num_embeddings=table.shape[0], features=table.shape[1], rngs=nnx.Rngs(0))
            layer.embedding[...] = table
            return layer(ids)
      """,
      jax_by_hand="""
        def embed(ids, table):
            return jnp.take(table, ids, axis=0)  # or table[ids]
      """,
      torch="""
        def embed(ids, table):
            return F.embedding(ids, table)  # nn.Embedding(V, d).weight is the (V, d) table
      """,
      notes=(("The gradient of a lookup is a scatter-add into the table: a repeated id gets the sum of its "
             "gradients (see \"scatter-add (update at indices)\")."),
             ("Out-of-range ids don't raise in JAX: `nnx.Embed` and `jnp.take` return NaN rows, plain `table[ids]` "
             "silently clamps to the last row; PyTorch raises `IndexError`. Check ids against the vocabulary "
             "size yourself.")),
      apis=("flax.nnx.Embed",),
      module="14", also=("13", "16", "33")),
    E("dynamic slice (a window at a traced position)", "Neural network layers",
      "Read a fixed-size window starting at a position that is only known at run time: the size is part of "
      "the program, the start is data.",
      r"y_{i} = x_{s + i},\quad 0 \le i < n", "", "(L, ...), start s, size n -> (n, ...)", "window(x, start)",
      lambda r: {"x": r.normal(size=(6, 4)), "start": 2},
      python="""
        def window(x, start, size=3):
            return [list(row) for row in x[start:start + size]]
      """,
      numpy="""
        def window(x, start, size=3):
            return x[start:start + size]
      """,
      jax="""
        def window(x, start, size=3):
            return jax.lax.dynamic_slice(x, (start, 0), (size, x.shape[1]))
      """,
      torch="""
        def window(x, start, size=3):
            return x.narrow(0, start, size)
      """,
      notes=(("Under `jit`, `x[start:start + 3]` with a traced `start` fails (slice bounds must be static); "
             "`dynamic_slice` works because only the start is dynamic, the size (3) is fixed."),
             ("Out-of-range starts are clamped, not errors: on a length-6 axis, `dynamic_slice(x, (5,), (3,))` "
             "returns rows 3, 4, 5 — a full window, but not the one you asked for.")),
      apis=("jax.lax.dynamic_slice",),
      module="12", also=("16", "18")),
    # ---------------------------------------------------------------- Attention and transformers
    E("scaled dot-product attention (causal)", "Attention and transformers",
      "Each position asks a question (query), compares it with every earlier position's label (key), and "
      "takes the softmax-weighted average of their contents (values). Causal: no looking ahead.",
      r"\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\Big(\frac{Q K^\top}{\sqrt{d_k}} + M\Big) V,\quad "
      r"M_{ts} = \begin{cases} 0 & s \le t \\ -\infty & s > t \end{cases}",
      r"Vaswani et al. 2017's $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ per head; $\alpha_{ts}$ for the weights",
      "q, k, v (B, T, heads, d) -> (B, T, heads, d)", "attention(q, k, v)",
      lambda r: {"q": r.normal(size=(1, 4, 2, 3)), "k": r.normal(size=(1, 4, 2, 3)),
                 "v": r.normal(size=(1, 4, 2, 3))},
      python="""
        def attention(q, k, v):  # [batch][time][head][dim]
            out = []
            for qb, kb, vb in zip(q, k, v):
                T, N, D = len(qb), len(qb[0]), len(qb[0][0])
                ob = [[[0.0] * D for _ in range(N)] for _ in range(T)]
                for n in range(N):
                    for t in range(T):
                        scores = [sum(qb[t][n][d] * kb[s][n][d] for d in range(D)) / math.sqrt(D)
                                  for s in range(t + 1)]  # only s <= t: causal
                        m = max(scores)
                        w = [math.exp(sc - m) for sc in scores]
                        z = sum(w)
                        for d in range(D):
                            ob[t][n][d] = sum(w[s] * vb[s][n][d] for s in range(t + 1)) / z
                out.append(ob)
            return out
      """,
      numpy="""
        def attention(q, k, v):
            T, D = q.shape[1], q.shape[-1]
            s = np.einsum("btnd,bsnd->bnts", q, k) / np.sqrt(D)
            s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
            w = np.exp(s - s.max(-1, keepdims=True))
            w = w / w.sum(-1, keepdims=True)
            return np.einsum("bnts,bsnd->btnd", w, v)
      """,
      jax="""
        def attention(q, k, v):
            return jax.nn.dot_product_attention(q, k, v, is_causal=True)
      """,
      jax_by_hand="""
        def attention(q, k, v):
            T, D = q.shape[1], q.shape[-1]
            s = jnp.einsum("btnd,bsnd->bnts", q, k) / jnp.sqrt(D)
            s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -jnp.inf)
            return jnp.einsum("bnts,bsnd->btnd", jax.nn.softmax(s, axis=-1), v)
      """,
      torch="""
        def attention(q, k, v):  # PyTorch wants (B, heads, T, d)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               is_causal=True)
            return o.transpose(1, 2)
      """,
      einsum='"btnd,bsnd->bnts" (scores), then "bnts,bsnd->btnd" (mix values)',
      notes=(("Layout trap: `jax.nn.dot_product_attention` takes (batch, time, heads, dim); "
             "`F.scaled_dot_product_attention` takes (batch, heads, time, dim). Same math, transposed axes."),
             ("Both default to the $1/\\sqrt{d}$ scale. Mask with `-inf` (or a large negative) *before* the "
             "softmax, never by zeroing weights after it (the rows would no longer sum to 1)."),
             "The causal mask is the lower triangle: row t may see columns 0..t."),
      apis=("jax.nn.dot_product_attention",),
      module="15", also=("16", "17")),
    E("grouped-query attention", "Attention and transformers",
      "Several query heads share one key/value head, so the KV cache is smaller. Query head h reads KV head "
      "h // (query heads per KV head).",
      r"\mathrm{head}_h = \mathrm{Attention}\big(Q_h,\; K_{g(h)},\; V_{g(h)}\big),\quad g(h) = \lfloor h / "
      r"(H_q / H_{kv}) \rfloor",
      r"GQA (Ainslie et al. 2023); MQA when $H_{kv} = 1$; MHA when $H_{kv} = H_q$",
      "q (B, T, H_q, d), k and v (B, T, H_kv, d) -> (B, T, H_q, d)", "gqa(q, k, v)",
      lambda r: {"q": r.normal(size=(1, 4, 4, 2)), "k": r.normal(size=(1, 4, 2, 2)),
                 "v": r.normal(size=(1, 4, 2, 2))},
      python="""
        def gqa(q, k, v):  # causal; [batch][time][head][dim]
            out = []
            for qb, kb, vb in zip(q, k, v):
                T, Hq, D, Hkv = len(qb), len(qb[0]), len(qb[0][0]), len(kb[0])
                ob = [[[0.0] * D for _ in range(Hq)] for _ in range(T)]
                for h in range(Hq):
                    g = h // (Hq // Hkv)  # the KV head this query head shares
                    for t in range(T):
                        sc = [sum(qb[t][h][d] * kb[s][g][d] for d in range(D)) / math.sqrt(D) for s in range(t + 1)]
                        m = max(sc)
                        w = [math.exp(x - m) for x in sc]
                        for d in range(D):
                            ob[t][h][d] = sum(w[s] * vb[s][g][d] for s in range(t + 1)) / sum(w)
                out.append(ob)
            return out
      """,
      numpy="""
        def gqa(q, k, v):
            rep = q.shape[2] // k.shape[2]
            k, v = np.repeat(k, rep, axis=2), np.repeat(v, rep, axis=2)  # KV head g -> query heads g*rep..
            T, D = q.shape[1], q.shape[-1]
            s = np.einsum("btnd,bsnd->bnts", q, k) / np.sqrt(D)
            s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
            w = np.exp(s - s.max(-1, keepdims=True))
            return np.einsum("bnts,bsnd->btnd", w / w.sum(-1, keepdims=True), v)
      """,
      jax="""
        def gqa(q, k, v):
            return jax.nn.dot_product_attention(q, k, v, is_causal=True)  # accepts fewer KV heads directly
      """,
      jax_by_hand="""
        def gqa(q, k, v):
            rep = q.shape[2] // k.shape[2]
            k, v = jnp.repeat(k, rep, axis=2), jnp.repeat(v, rep, axis=2)
            T, D = q.shape[1], q.shape[-1]
            s = jnp.einsum("btnd,bsnd->bnts", q, k) / jnp.sqrt(D)
            s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -jnp.inf)
            return jnp.einsum("bnts,bsnd->btnd", jax.nn.softmax(s, axis=-1), v)
      """,
      torch="""
        def gqa(q, k, v):
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               is_causal=True, enable_gqa=True)
            return o.transpose(1, 2)
      """,
      notes=(("`repeat`, not `tile`: `jnp.repeat(k, 2, axis=2)` gives KV heads 0,0,1,1 (query heads 0 and 1 "
             "share KV head 0), which is what GQA means; `jnp.tile` gives 0,1,0,1 and silently pairs the wrong "
             "heads."),
             ("Repeating is only for clarity: it copies the cache. `jax.nn.dot_product_attention` and PyTorch's "
             "`enable_gqa=True` handle the grouping without the copy (Module 17's `repeat_kv` repeats).")),
      module="17", also=("18", "19")),
    E("rotary position embedding (RoPE)", "Attention and transformers",
      "Encode position by rotating each pair of query/key features by an angle proportional to the position. "
      "A dot product of two rotated vectors then depends only on how far apart they are.",
      r"\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos p\theta_i & -\sin p\theta_i \\ "
      r"\sin p\theta_i & \cos p\theta_i \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix},\quad "
      r"\theta_i = 10000^{-2i/d}",
      r"RoFormer (Su et al. 2021): $f(q, m) = q\, e^{i m \theta}$ in complex form, $R^d_{\Theta, m} q$ in matrix "
      r"form",
      "x (B, T, heads, d), positions (T,) -> same shape as x", "rope(x, positions)",
      lambda r: {"x": r.normal(size=(1, 4, 1, 6)), "positions": np.arange(4)},
      python="""
        def rope(x, positions, theta=10000.0):  # pairs (0, 1), (2, 3), ...: the interleaved convention
            D = len(x[0][0][0])
            out = []
            for xb in x:
                ob = []
                for t, xt in enumerate(xb):
                    oh = []
                    for vec in xt:
                        o = list(vec)
                        for i in range(D // 2):
                            ang = positions[t] * theta ** (-2 * i / D)
                            a, b = vec[2 * i], vec[2 * i + 1]
                            o[2 * i] = a * math.cos(ang) - b * math.sin(ang)
                            o[2 * i + 1] = a * math.sin(ang) + b * math.cos(ang)
                        oh.append(o)
                    ob.append(oh)
                out.append(ob)
            return out
      """,
      numpy="""
        def rope(x, positions, theta=10000.0):
            D = x.shape[-1]
            ang = (positions[:, None] * theta ** (-np.arange(0, D, 2) / D))[None, :, None, :]  # (1, T, 1, D/2)
            a, b = x[..., 0::2], x[..., 1::2]
            out = np.empty_like(x)
            out[..., 0::2] = a * np.cos(ang) - b * np.sin(ang)
            out[..., 1::2] = a * np.sin(ang) + b * np.cos(ang)
            return out
      """,
      jax="""
        def rope(x, positions, theta=10000.0):  # Module 17's rope_complex
            D = x.shape[-1]
            ang = (positions[:, None] * theta ** (-jnp.arange(0, D, 2) / D))[None, :, None, :]
            xc = jax.lax.complex(x[..., 0::2], x[..., 1::2])  # pair (a, b) -> a + ib
            rot = xc * jnp.exp(1j * ang)                        # rotating = multiplying by e^{i angle}
            return jnp.stack([rot.real, rot.imag], axis=-1).reshape(x.shape)
      """,
      jax_by_hand="""
        def rope(x, positions, theta=10000.0):
            D = x.shape[-1]
            ang = (positions[:, None] * theta ** (-jnp.arange(0, D, 2) / D))[None, :, None, :]
            a, b = x[..., 0::2], x[..., 1::2]
            pairs = [a * jnp.cos(ang) - b * jnp.sin(ang), a * jnp.sin(ang) + b * jnp.cos(ang)]
            return jnp.stack(pairs, axis=-1).reshape(x.shape)  # re-interleave
      """,
      torch="""
        def rope(x, positions, theta=10000.0):  # the complex form of Meta's original Llama code
            D = x.shape[-1]
            ang = positions[:, None] * theta ** (-torch.arange(0, D, 2) / D)
            rot = torch.polar(torch.ones_like(ang), ang)[None, :, None, :]
            xc = torch.view_as_complex(x.reshape(*x.shape[:-1], D // 2, 2).contiguous())
            return torch.view_as_real(xc * rot).reshape(x.shape)
      """,
      notes=(("Two pairing conventions: *interleaved* pairs (0,1), (2,3), … (RoFormer, Meta's original Llama "
             "code, Module 17's `rope_complex`, shown here) and *rotate-half* pairs (i, i + d/2) (Hugging Face "
             "Llama, `learnjax/llama.py`, deepseek-v3-jax's `apply_rope`). They give different numbers on the "
             "same vector, and agree after permuting the head dims to evens-then-odds — which is why Hugging "
             "Face's Llama conversion permutes the q/k weight rows."),
             ("Rotate queries and keys, never values. Positions are absolute indices into the cache, so during "
             "decoding pass the token's real position, not 0.")),
      apis=("jax.lax.complex",),
      module="17", also=("18", "19")),
    E("SwiGLU feed-forward", "Attention and transformers",
      "The transformer's per-token MLP with a gate: one projection decides how much of another projection "
      "passes through, then a third projects back.",
      r"\mathrm{FFN}(x) = \big(\mathrm{SiLU}(x W_1) \odot x W_3\big) W_2,\quad \mathrm{SiLU}(z) = z\,\sigma(z)",
      r"Shazeer 2020: $\mathrm{FFN}_\mathrm{SwiGLU}(x) = (\mathrm{Swish}_1(xW) \otimes xV) W_2$",
      "x (N, d), W1 and W3 (d, d_ff), W2 (d_ff, d) -> (N, d)", "swiglu(x, w1, w3, w2)",
      lambda r: {"x": r.normal(size=(3, 4)), "w1": r.normal(size=(4, 6)), "w3": r.normal(size=(4, 6)),
                 "w2": r.normal(size=(6, 4))},
      python="""
        def swiglu(x, w1, w3, w2):
            def matmul(a, b):
                return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
                        for i in range(len(a))]
            gate, up = matmul(x, w1), matmul(x, w3)
            h = [[g / (1 + math.exp(-g)) * u for g, u in zip(gr, ur)] for gr, ur in zip(gate, up)]
            return matmul(h, w2)
      """,
      numpy="""
        def swiglu(x, w1, w3, w2):
            g = x @ w1
            return (g / (1 + np.exp(-g)) * (x @ w3)) @ w2
      """,
      jax="""
        def swiglu(x, w1, w3, w2):
            return (jax.nn.silu(x @ w1) * (x @ w3)) @ w2
      """,
      torch="""
        def swiglu(x, w1, w3, w2):
            return (F.silu(x @ w1) * (x @ w3)) @ w2
      """,
      notes=(("Names differ by codebase: Meta's Llama calls them `w1` (gate), `w3` (up), `w2` (down); Hugging "
             "Face calls them `gate_proj`, `up_proj`, `down_proj`. Check which one gets the SiLU."),
             ("Three matrices instead of two, so models shrink $d_\\mathrm{ff}$ to about $\\tfrac{8}{3}d$: "
             "$3 \\cdot d \\cdot \\tfrac{8}{3}d = 8d^2$, the same parameters as a $4d$ two-matrix MLP.")),
      module="17", also=("16", "19")),
    E("mixture-of-experts routing (top-k with grouped matmul)", "Attention and transformers",
      "A router scores every expert for every token; each token goes to its top k experts only, and their "
      "outputs are mixed by the (renormalized) scores.",
      r"\mathcal{T}_n = \mathrm{TopK}_e(s_{n,e} + b_e,\, k),\quad g_{n,e} = \frac{s_{n,e}\,[e \in \mathcal{T}_n]}"
      r"{\sum_{e' \in \mathcal{T}_n} s_{n,e'}},\quad y_n = \sum_e g_{n,e}\; x_n W_e",
      r"DeepSeek-V3: $g_{i,t}$, $s_{i,t}$, $\mathrm{Topk}(\{s_{j,t} + b_j\}, K_r)$; Mixtral: "
      r"$\mathrm{softmax}(\mathrm{TopK}(x W_g))$",
      "x (N, d), scores (N, E), bias (E,), W (E, d, h), k -> gates (N, E), y (N, h)", "moe(x, scores, bias, W, k)",
      lambda r: {"x": r.normal(size=(4, 3)), "scores": _sigmoid(r.normal(size=(4, 4))),
                 "bias": np.array([0.0, 0.5, 0.0, 0.0]), "W": r.normal(size=(4, 3, 2)), "k": 2},
      python="""
        def moe(x, scores, bias, W, k):
            gates, y = [], []
            for xn, sn in zip(x, scores):
                E = len(sn)
                picked = sorted(range(E), key=lambda e: sn[e] + bias[e], reverse=True)[:k]  # bias picks who...
                total = sum(sn[e] for e in picked)
                g = [sn[e] / total if e in picked else 0.0 for e in range(E)]          # ...raw scores weigh
                out = [0.0] * len(W[0][0])
                for e in picked:  # run only the picked experts
                    for h in range(len(out)):
                        out[h] += g[e] * sum(xn[d] * W[e][d][h] for d in range(len(xn)))
                gates.append(g)
                y.append(out)
            return {"gates": gates, "y": y}
      """,
      numpy="""
        def moe(x, scores, bias, W, k):
            idx = np.argsort(-(scores + bias), axis=-1)[:, :k]
            picked = np.zeros(scores.shape, bool)
            np.put_along_axis(picked, idx, True, axis=-1)
            gates = np.where(picked, scores, 0.0)
            gates = gates / gates.sum(-1, keepdims=True)
            y = np.einsum("ne,nd,edh->nh", gates, x, W)  # dense: every expert on every token
            return {"gates": gates, "y": y}
      """,
      jax="""
        def moe(x, scores, bias, W, k):
            _, idx = jax.lax.top_k(scores + bias, k)                     # (N, k) expert ids
            picked = jax.nn.one_hot(idx, scores.shape[-1]).sum(-2) > 0
            gates = jnp.where(picked, scores, 0.0)
            gates = gates / gates.sum(-1, keepdims=True)
            token = jnp.repeat(jnp.arange(x.shape[0]), k)                # dispatch: one row per (token, expert)
            expert = idx.reshape(-1)
            order = jnp.argsort(expert)                                  # ragged_dot wants rows grouped by expert
            sizes = jnp.bincount(expert, length=W.shape[0]).astype(jnp.int32)
            out = jax.lax.ragged_dot(x[token[order]], W, sizes)          # each row times its own expert
            w = gates[token[order], expert[order]][:, None]
            y = jnp.zeros((x.shape[0], W.shape[2])).at[token[order]].add(w * out)  # combine back per token
            return {"gates": gates, "y": y}
      """,
      jax_by_hand="""
        def moe(x, scores, bias, W, k):  # dense, like deepseek-v3-jax
            _, idx = jax.lax.top_k(scores + bias, k)
            picked = jax.nn.one_hot(idx, scores.shape[-1]).sum(-2) > 0
            gates = jnp.where(picked, scores, 0.0)
            gates = gates / gates.sum(-1, keepdims=True)
            return {"gates": gates, "y": jnp.einsum("ne,nd,edh->nh", gates, x, W)}
      """,
      torch="""
        def moe(x, scores, bias, W, k):
            idx = torch.topk(scores + bias, k, dim=-1).indices
            picked = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, idx, True)
            gates = torch.where(picked, scores, 0.0)
            gates = gates / gates.sum(-1, keepdim=True)
            y = torch.zeros(x.shape[0], W.shape[2])
            for e in range(W.shape[0]):  # loop over experts, each on its own tokens
                rows = picked[:, e].nonzero().squeeze(-1)
                y[rows] += gates[rows, e, None] * (x[rows] @ W[e])
            return {"gates": gates, "y": y}
      """,
      einsum='"ne,nd,edh->nh" (dense form)',
      notes=(("The bias only decides *which* experts are picked; the gate values use the raw scores "
             "(DeepSeek-V3's auxiliary-loss-free balancing, Module 17)."),
             "`jax.lax.top_k` returns (values, indices), like `torch.topk`; NumPy has no top-k, so sort.",
             ("`ragged_dot` needs the rows sorted by expert and `group_sizes` that add up to the row count. The "
             "dense einsum is simpler and jit-friendly but wastes (E−k)/E of the expert FLOPs."),
             ("Softmax over all experts then renormalizing the top k equals a softmax over just the top-k "
             "logits, so Mixtral-style routers can be written either way.")),
      apis=("jax.lax.ragged_dot", "jax.lax.top_k"),
      module="17", also=("24",)),
    # ---------------------------------------------------------------- Sampling and inference
    E("KV cache write", "Sampling and inference",
      "Store this step's keys (or values) into a preallocated cache at the current position, so later tokens "
      "can attend to them without recomputing.",
      r"C_{:, p + t} \leftarrow K^{\text{new}}_{:, t},\quad 0 \le t < T_\text{new}", "",
      "cache (B, S, heads, d), new (B, T_new, heads, d), position p -> cache (B, S, heads, d)",
      "write_cache(cache, new, pos)",
      lambda r: {"cache": r.normal(size=(1, 6, 2, 2)), "new": r.normal(size=(1, 2, 2, 2)), "pos": 3},
      python="""
        def write_cache(cache, new, pos):
            out = [[[list(head) for head in step] for step in seq] for seq in cache]  # copy
            for b in range(len(new)):
                for t in range(len(new[b])):
                    out[b][pos + t] = [list(head) for head in new[b][t]]
            return out
      """,
      numpy="""
        def write_cache(cache, new, pos):
            out = cache.copy()
            out[:, pos:pos + new.shape[1]] = new
            return out
      """,
      jax="""
        def write_cache(cache, new, pos):
            return jax.lax.dynamic_update_slice(cache, new, (0, pos, 0, 0))
      """,
      torch="""
        def write_cache(cache, new, pos):
            out = cache.clone()
            out[:, pos:pos + new.shape[1]] = new
            return out
      """,
      notes=(("JAX returns a new cache instead of writing in place; under `jit` with the cache donated "
             "(`donate_argnums`), XLA can reuse the buffer, so it costs no copy."),
             ("Clamping trap: if `pos + T_new` runs past the end, the start is moved back so the write fits — "
             "writing 3 rows at position 5 of a length-6 cache lands on rows 3–5 and overwrites older entries. "
             "Check the length before writing.")),
      apis=("jax.lax.dynamic_update_slice",),
      module="18", also=("17", "19")),
    E("temperature and top-k filter", "Sampling and inference",
      "Before sampling the next token: divide the scores by a temperature (below 1 sharpens, above 1 "
      "flattens), keep only the k most likely tokens, and renormalize.",
      r"p_i \propto \begin{cases} e^{z_i / \tau} & z_i \ge z_{(k)} \\ 0 & \text{otherwise} \end{cases},\quad "
      r"z_{(k)} = k\text{-th largest logit}", r"$\mathrm{softmax}(z / T)$; top-$k$ sampling (Fan et al. 2018)",
      "logits (B, V), k, temperature -> probabilities (B, V)", "top_k_probs(logits, k, temperature)",
      lambda r: {"logits": 2 * r.normal(size=(2, 6)), "k": 3, "temperature": 0.7},
      python="""
        def top_k_probs(logits, k, temperature):
            out = []
            for row in logits:
                z = [v / temperature for v in row]
                kth = sorted(z, reverse=True)[k - 1]
                kept = [v if v >= kth else -math.inf for v in z]
                m = max(kept)
                e = [math.exp(v - m) for v in kept]  # exp(-inf) = 0
                out.append([v / sum(e) for v in e])
            return out
      """,
      numpy="""
        def top_k_probs(logits, k, temperature):
            z = logits / temperature
            kth = np.sort(z, axis=-1)[..., -k, None]
            z = np.where(z < kth, -np.inf, z)
            e = np.exp(z - z.max(-1, keepdims=True))
            return e / e.sum(-1, keepdims=True)
      """,
      jax="""
        def top_k_probs(logits, k, temperature):  # Module 18's top_k_filter, then softmax
            z = logits / temperature
            kth = jnp.sort(z, axis=-1)[..., -k, None]
            return jax.nn.softmax(jnp.where(z < kth, -jnp.inf, z), axis=-1)
      """,
      torch="""
        def top_k_probs(logits, k, temperature):
            z = logits / temperature
            kth = torch.topk(z, k, dim=-1).values[..., -1:]
            return torch.softmax(z.masked_fill(z < kth, float("-inf")), dim=-1)
      """,
      notes=(("Temperature never changes *which* k tokens survive (dividing by a positive number keeps the "
             "order), only their probabilities; it does change which tokens survive top-p."),
             ("Comparing against the k-th value keeps every token tied with it, so a tie can leave more than k; "
             "`torch.topk` alone returns exactly k indices."),
             "τ → 0 is greedy (argmax); τ = 1 leaves the model's distribution unchanged."),
      module="18", also=("16", "19")),
    E("top-p (nucleus) filter", "Sampling and inference",
      "Keep the smallest set of most-likely tokens whose probabilities add up to at least p, drop the long "
      "tail, renormalize.",
      r"V_p = \text{smallest top set with } \sum_{i \in V_p} p_i \ge p,\quad "
      r"p'_i = \frac{p_i\, [i \in V_p]}{\sum_{j \in V_p} p_j}",
      r"nucleus sampling (Holtzman et al. 2019), $V^{(p)}$", "logits (B, V), p -> probabilities (B, V)",
      "top_p_probs(logits, p)", lambda r: {"logits": 2 * r.normal(size=(2, 6)), "p": 0.8},
      python="""
        def top_p_probs(logits, p):
            out = []
            for row in logits:
                m = max(row)
                e = [math.exp(v - m) for v in row]
                probs = [v / sum(e) for v in e]
                keep, mass = set(), 0.0
                for i in sorted(range(len(row)), key=lambda i: -probs[i]):
                    if mass >= p:  # the tokens above already reach p: stop
                        break
                    keep.add(i)
                    mass += probs[i]
                total = sum(probs[i] for i in keep)
                out.append([probs[i] / total if i in keep else 0.0 for i in range(len(row))])
            return out
      """,
      numpy="""
        def top_p_probs(logits, p):
            e = np.exp(logits - logits.max(-1, keepdims=True))
            probs = e / e.sum(-1, keepdims=True)
            order = np.argsort(-probs, axis=-1)
            sorted_p = np.take_along_axis(probs, order, -1)
            keep = np.zeros(probs.shape, bool)
            np.put_along_axis(keep, order, np.cumsum(sorted_p, -1) - sorted_p < p, -1)  # mass above < p
            q = np.where(keep, probs, 0.0)
            return q / q.sum(-1, keepdims=True)
      """,
      jax="""
        def top_p_probs(logits, p):  # Module 18's top_p_filter, then softmax
            order = jnp.argsort(-logits, axis=-1)
            probs = jax.nn.softmax(jnp.take_along_axis(logits, order, -1), axis=-1)
            keep_sorted = jnp.cumsum(probs, -1) - probs < p          # mass strictly above this token < p
            keep = jnp.take_along_axis(keep_sorted, jnp.argsort(order, axis=-1), -1)  # undo the sort
            return jax.nn.softmax(jnp.where(keep, logits, -jnp.inf), axis=-1)
      """,
      torch="""
        def top_p_probs(logits, p):
            probs = torch.softmax(logits, dim=-1)
            sorted_p, order = torch.sort(probs, dim=-1, descending=True)
            keep = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, order, torch.cumsum(sorted_p, -1) - sorted_p < p)
            q = torch.where(keep, probs, 0.0)
            return q / q.sum(-1, keepdim=True)
      """,
      notes=(("The test is on the mass *above* a token (`cumsum - p_i < p`), so the token that crosses p is "
             "kept and the top token is always kept, even when it alone exceeds p."),
             "Apply temperature first: it changes the probabilities, so it changes the nucleus.",
             ("`argsort` of the sort order is its inverse permutation: that's how the JAX version maps the "
             "keep-mask back to vocabulary order (`scatter` does it in PyTorch, `put_along_axis` in NumPy).")),
      module="18", also=("19", "27")),
    E("int8 absmax quantization", "Sampling and inference",
      "Store weights as small integers plus one float scale per column: divide by the column's largest "
      "magnitude, scale to ±127, round. Multiply back to use them.",
      r"s_j = \frac{\max_i |W_{ij}|}{127},\quad q_{ij} = \mathrm{clip}\big(\mathrm{round}(W_{ij} / s_j), -127, "
      r"127\big),\quad \hat W_{ij} = q_{ij}\, s_j",
      r"LLM.int8() (Dettmers et al. 2022) absmax; $\Delta$ for the step size",
      "W (d_in, d_out) -> q int8 (d_in, d_out), s (d_out,), dequantized (d_in, d_out)", "quantize_int8(w)",
      lambda r: {"w": r.normal(size=(5, 4))},
      python="""
        def quantize_int8(w):
            cols = range(len(w[0]))
            s = [max(abs(row[j]) for row in w) / 127 or 1.0 for j in cols]  # all-zero column: scale 1
            q = [[max(-127, min(127, round(row[j] / s[j]))) for j in cols] for row in w]
            return {"q": q, "s": s, "deq": [[qi[j] * s[j] for j in cols] for qi in q]}
      """,
      numpy="""
        def quantize_int8(w):
            s = np.abs(w).max(axis=0) / 127
            s = np.where(s == 0, 1.0, s)
            q = np.clip(np.round(w / s), -127, 127).astype(np.int8)
            return {"q": q, "s": s, "deq": q * s}
      """,
      jax="""
        def quantize_int8(w):  # Module 18's quantize_int8 / dequantize_int8
            s = jnp.max(jnp.abs(w), axis=0) / 127.0
            s = jnp.where(s == 0, 1.0, s)
            q = jnp.clip(jnp.round(w / s), -127, 127).astype(jnp.int8)
            return {"q": q, "s": s, "deq": q.astype(jnp.float32) * s}
      """,
      torch="""
        def quantize_int8(w):
            s = w.abs().amax(dim=0) / 127
            s = torch.where(s == 0, 1.0, s)
            q = torch.clamp(torch.round(w / s), -127, 127).to(torch.int8)
            return {"q": q, "s": s, "deq": q.float() * s}
      """,
      notes=(("All four round halves to even (`round(2.5) == 2` in Python, NumPy, JAX and PyTorch), not away "
             "from zero."),
             ("One scale per *output* column for the course's (d_in, d_out) layout; PyTorch's `nn.Linear` "
             "weights are (d_out, d_in), so the same per-output scale is per *row* there."),
             "The range is symmetric, ±127 (not −128), so 0 stays exactly 0 and the sign is symmetric."),
      module="18", also=("19", "27")),
    E("importance-weighted quantization", "Sampling and inference",
      "Round each column to a few levels, but choose its scale to minimise the error that matters: each weight's "
      "squared error weighted by how large its input usually is (the importance, mean squared activation). "
      "Try a grid of candidate scales below absmax and keep the best.",
      r"\hat W_{:,j} = s_j^* q_{:,j},\quad q_{ij} = \mathrm{clip}\big(\mathrm{round}(W_{ij}/s), -m, m\big),\quad "
      r"s_j^* = \arg\min_{s \in S_j} \sum_i a_i \big(W_{ij} - s\, q_{ij}\big)^2,\quad "
      r"S_j = \Big\{ \tfrac{\max_i |W_{ij}|}{m} \cdot f \Big\}_{f \in \{0.5, 0.55, \dots, 1\}}",
      r"llama.cpp's imatrix (2024): $a_i = \mathbb{E}[x_i^2]$ over calibration text; GPTQ (Frantar et al. 2022) "
      r"uses the full second moment $H = \mathbb{E}[x x^\top]$, not just its diagonal",
      "W (d_in, d_out), a (d_in,) -> dequantized (d_in, d_out)", "quantize_weighted(w, a, m)",
      lambda r: {"w": r.normal(size=(6, 3)), "a": r.uniform(0.1, 3.0, size=6), "m": 3},
      python="""
        def quantize_weighted(w, a, m):
            fs = [0.5 + 0.05 * k for k in range(11)]
            out = [[0.0] * len(w[0]) for _ in w]
            for j in range(len(w[0])):
                col = [row[j] for row in w]
                amax = max(abs(v) for v in col) or 1.0
                best = None
                for f in fs:
                    s = amax / m * f
                    deq = [max(-m, min(m, round(v / s))) * s for v in col]
                    err = sum(ai * (v - d) ** 2 for ai, v, d in zip(a, col, deq))
                    if best is None or err < best[0]:
                        best = (err, deq)
                for i, d in enumerate(best[1]):
                    out[i][j] = d
            return out
      """,
      numpy="""
        def quantize_weighted(w, a, m):
            fs = np.linspace(0.5, 1.0, 11)
            amax = np.abs(w).max(axis=0)
            s = np.where(amax == 0, 1.0, amax)[None, :] / m * fs[:, None]     # (K, d_out) candidates
            deq = np.clip(np.round(w[None] / s[:, None]), -m, m) * s[:, None]  # (K, d_in, d_out)
            err = (a[None, :, None] * (w[None] - deq) ** 2).sum(axis=1)        # (K, d_out)
            k = err.argmin(axis=0)
            return deq[k, :, np.arange(w.shape[1])].T
      """,
      jax="""
        def quantize_weighted(w, a, m):
            fs = jnp.linspace(0.5, 1.0, 11)
            amax = jnp.abs(w).max(axis=0)
            s = jnp.where(amax == 0, 1.0, amax)[None, :] / m * fs[:, None]
            deq = jnp.clip(jnp.round(w[None] / s[:, None]), -m, m) * s[:, None]
            err = (a[None, :, None] * (w[None] - deq) ** 2).sum(axis=1)
            k = err.argmin(axis=0)
            return jnp.take_along_axis(deq, k[None, None, :], axis=0)[0]
      """,
      torch="""
        def quantize_weighted(w, a, m):
            fs = torch.linspace(0.5, 1.0, 11, dtype=w.dtype)
            amax = w.abs().amax(dim=0)
            s = torch.where(amax == 0, 1.0, amax)[None, :] / m * fs[:, None]
            deq = torch.clamp(torch.round(w[None] / s[:, None]), -m, m) * s[:, None]
            err = (a[None, :, None] * (w[None] - deq) ** 2).sum(dim=1)
            k = err.argmin(dim=0)
            return torch.gather(deq, 0, k[None, None, :].expand(1, *w.shape))[0]
      """,
      notes=(("With every a_i equal this is plain round-to-nearest with a searched scale; the importances only "
              "change *which* scale wins, so the file format and size are unchanged."),
             ("A smaller scale than absmax clips the largest weight but rounds the rest more finely; whether that "
              "pays depends on which inputs are large, which is what a_i records."),
             ("llama.cpp searches finer grids and also an offset for asymmetric types; this keeps the idea: "
              "the same rounding, a different choice of scale.")),
      module="27"),
    E("LoRA forward", "Neural network layers",
      "Fine-tune a frozen weight by adding a trainable low-rank update: two thin matrices whose product is "
      "the change to W.",
      r"y = x W + \frac{\alpha}{r}\, (x A) B,\quad A \in \mathbb{R}^{d_\text{in} \times r},\; B \in "
      r"\mathbb{R}^{r \times d_\text{out}}",
      r"Hu et al. 2021: $h = W_0 x + \Delta W x = W_0 x + BAx$ (column vectors, so their B and A are the "
      r"transposes of these)",
      "x (N, d_in), W (d_in, d_out), A (d_in, r), B (r, d_out) -> (N, d_out)", "lora(x, W, A, B, alpha, r)",
      lambda r: {"x": r.normal(size=(3, 4)), "W": r.normal(size=(4, 5)), "A": r.normal(size=(4, 2)),
                 "B": 0.1 * r.normal(size=(2, 5)), "alpha": 4.0, "r": 2},
      python="""
        def lora(x, W, A, B, alpha, r):
            def matmul(a, b):
                return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
                        for i in range(len(a))]
            base, low = matmul(x, W), matmul(matmul(x, A), B)
            return [[b + alpha / r * u for b, u in zip(br, ur)] for br, ur in zip(base, low)]
      """,
      numpy="""
        def lora(x, W, A, B, alpha, r):
            return x @ W + (alpha / r) * ((x @ A) @ B)
      """,
      jax="""
        def lora(x, W, A, B, alpha, r):  # Module 20's lora_apply
            return x @ W + (alpha / r) * ((x @ A) @ B)
      """,
      torch="""
        def lora(x, W, A, B, alpha, r):
            return x @ W + (alpha / r) * ((x @ A) @ B)
      """,
      notes=(("B starts at zero (A random), so at step 0 the model is exactly the base model and only the "
             "update trains."),
             ("Compute `(x @ A) @ B`, not `x @ (A @ B)`: the first never builds the full d_in × d_out matrix. "
             "For serving, merge once: `W + (alpha / r) * A @ B`."),
             ("PyTorch `nn.Linear` layers store (out, in), so a PyTorch LoRA's A and B weights are the "
             "transposes of the ones here.")),
      module="20", also=("26",)),
    # ---------------------------------------------------------------- Probability and information
    E("perplexity", "Probability and information",
      "The exponential of the average cross-entropy: how many equally likely choices the model is "
      "effectively picking between at each token. Lower is better; a uniform guess over V tokens scores V.",
      r"\mathrm{PPL} = \exp\Big(-\frac{1}{N} \sum_{n=1}^{N} \log p(y_n \mid \text{context}_n)\Big)",
      r"$2^{H}$ when the entropy $H$ is in bits; $e^{\mathcal{L}}$ with $\mathcal{L}$ the mean NLL",
      "logits (N, V), targets (N,) int -> ()", "perplexity(logits, y)",
      lambda r: {"logits": 2 * r.normal(size=(6, 5)), "y": np.array([0, 4, 1, 1, 3, 2])},
      python="""
        def perplexity(logits, y):
            nll = 0.0
            for z, label in zip(logits, y):
                m = max(z)
                nll += m + math.log(sum(math.exp(v - m) for v in z)) - z[label]
            return math.exp(nll / len(y))
      """,
      numpy="""
        def perplexity(logits, y):
            m = logits.max(-1, keepdims=True)
            logp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
            return np.exp(-logp[np.arange(len(y)), y].mean())
      """,
      jax="""
        def perplexity(logits, y):
            return jnp.exp(optax.softmax_cross_entropy_with_integer_labels(logits, y).mean())
      """,
      torch="""
        def perplexity(logits, y):
            return torch.exp(F.cross_entropy(logits, y))
      """,
      notes=(("Average the log-losses first, then exponentiate: the mean of per-token perplexities is a "
             "different (larger) number."),
             ("Bits per token = mean NLL / ln 2. Perplexities are only comparable with the same tokenizer: a "
             "model with a bigger vocabulary predicts fewer, harder tokens.")),
      module="04", also=("14", "16", "21")),
]
