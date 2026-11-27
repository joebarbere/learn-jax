"""Rosetta entries for Part 10–11 milestones (Modules 26, 29–38): RL and preferences, sequence models,
generative models, retrieval."""
import numpy as np

from learnjax.rosetta.entry import Entry as E


def _hopfield_inputs(r: np.random.Generator) -> dict:
    patterns = np.where(r.normal(size=(2, 8)) > 0, 1.0, -1.0)
    probe = patterns[0].copy()
    probe[[1, 5]] *= -1  # two flipped bits
    return {"patterns": patterns, "probe": probe}


def _rank_inputs(r: np.random.Generator) -> dict:
    ranked = np.stack([r.permutation(10)[:4] for _ in range(5)])
    relevant = np.array([ranked[0, 0], ranked[1, 2], ranked[2, 3], 99, ranked[4, 1]])  # rank 1, 3, 4, absent, 2
    return {"ranked": ranked, "relevant": relevant, "k": 3}


ENTRIES: list[E] = [
    # ---------------------------------------------------------------- Reinforcement learning (Module 26)
    E("GRPO group-relative advantage", "Reinforcement learning and preferences",
      "Score each sampled answer against the other answers to the same prompt: above the group's average is "
      "a positive advantage, below is negative. No value network needed.",
      r"A_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G) + \epsilon}",
      r"$\hat A_{i,t} = \tilde r_i$ (Shao et al., 2024, DeepSeekMath)", "rewards (groups * G,), G -> (groups * G,)",
      "group_adv(r, 4)", lambda r: {"r": np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])},
      python="""
        def group_adv(r, group_size, eps=1e-4):
            out = []
            for g in range(0, len(r), group_size):
                group = r[g:g + group_size]
                mean = sum(group) / len(group)
                std = math.sqrt(sum((x - mean) ** 2 for x in group) / len(group))
                out += [(x - mean) / (std + eps) for x in group]
            return out
      """,
      numpy="""
        def group_adv(r, group_size, eps=1e-4):
            g = r.reshape(-1, group_size)
            return ((g - g.mean(1, keepdims=True)) / (g.std(1, keepdims=True) + eps)).reshape(-1)
      """,
      jax="""
        def group_adv(r, group_size, eps=1e-4):
            g = r.reshape(-1, group_size)
            return ((g - g.mean(1, keepdims=True)) / (g.std(1, keepdims=True) + eps)).reshape(-1)
      """,
      torch="""
        def group_adv(r, group_size, eps=1e-4):
            g = r.view(-1, group_size)
            std = g.std(1, keepdim=True, correction=0)  # torch's default is the unbiased (n - 1) std
            return ((g - g.mean(1, keepdim=True)) / (std + eps)).view(-1)
      """,
      notes=(("NumPy and JAX `std` divide by n; `torch.std` divides by n - 1 unless you pass `correction=0`. "
             "With G = 4 that makes every advantage about 13% smaller."),
             ("A group where every answer gets the same reward (the second one here) has zero advantage: it "
             "teaches nothing, and the epsilon keeps it from being 0/0.")),
      module="26", also=("35",)),
    # ---------------------------------------------------------------- Perceptron and Hopfield (Module 29)
    E("perceptron update", "Neural network layers",
      "Go through the examples one at a time; whenever one is on the wrong side of the line, nudge the line "
      "toward it. Rosenblatt's 1958 learning rule.",
      r"\text{if } y_n (w \cdot x_n + b) \le 0:\quad w \leftarrow w + \eta\, y_n x_n,\;\; b \leftarrow b + \eta\, y_n",
      r"$w \leftarrow w + \eta\,(y - \hat y)\,x$ with $y, \hat y \in \{0, 1\}$ (the common textbook form; "
      r"Rosenblatt, 1958)",
      "X (n, d), y (n,) in {-1, +1} -> w (d,), b", "perceptron_epoch(X, y)",
      lambda r: {"X": r.integers(-3, 4, size=(6, 2)).astype(float), "y": np.array([1.0, -1, 1, -1, 1, -1])},
      python="""
        def perceptron_epoch(X, y, lr=1.0):
            w, b = [0.0] * len(X[0]), 0.0
            for x, t in zip(X, y):
                if t * (sum(wi * xi for wi, xi in zip(w, x)) + b) <= 0:  # mistake (or on the line)
                    w = [wi + lr * t * xi for wi, xi in zip(w, x)]
                    b += lr * t
            return {"w": w, "b": b}
      """,
      numpy="""
        def perceptron_epoch(X, y, lr=1.0):
            w, b = np.zeros(X.shape[1]), 0.0
            for x, t in zip(X, y):  # sequential: each update changes the next prediction
                if t * (w @ x + b) <= 0:
                    w, b = w + lr * t * x, b + lr * t
            return {"w": w, "b": b}
      """,
      jax="""
        def perceptron_epoch(X, y, lr=1.0):
            def step(carry, xt):
                w, b = carry
                x, t = xt
                mistake = t * (w @ x + b) <= 0
                return (jnp.where(mistake, w + lr * t * x, w), jnp.where(mistake, b + lr * t, b)), None

            (w, b), _ = jax.lax.scan(step, (jnp.zeros(X.shape[1]), jnp.zeros(())), (X, y))
            return {"w": w, "b": b}
      """,
      torch="""
        def perceptron_epoch(X, y, lr=1.0):
            w, b = torch.zeros(X.shape[1]), torch.tensor(0.0)
            for x, t in zip(X, y):
                if t * (w @ x + b) <= 0:
                    w, b = w + lr * t * x, b + lr * t
            return {"w": w, "b": b}
      """,
      notes=(("The loop is sequential, so JAX writes it as `lax.scan` (a Python `if` on a traced value fails "
             "under `jit`); the branch becomes `jnp.where`."),
             ("Labels are ±1 here. With 0/1 labels the rule is usually written with the error $y - \\hat y$; it is "
             "the same rule up to a factor of 2 in the step.")),
      module="29"),
    E("Hopfield network: energy and update", "Neural network layers",
      "Store patterns in a weight matrix by Hebb's rule; then flip each unit to agree with its weighted input. "
      "Every flip lowers the energy, so a corrupted pattern slides down to the stored one.",
      r"W = \tfrac{1}{n} \textstyle\sum_\mu \xi^\mu {\xi^\mu}^\top,\; W_{ii} = 0;\quad "
      r"E(s) = -\tfrac{1}{2} s^\top W s;\quad s_i \leftarrow \mathrm{sign}\big(\textstyle\sum_j W_{ij} s_j\big)",
      r"$E = -\tfrac12 \sum_{i \ne j} T_{ij} V_i V_j$ with $V_i \in \{0, 1\}$ (Hopfield, 1982)",
      "patterns (p, n) in ±1, probe (n,) -> energies and recalled state (n,)", "hopfield(patterns, probe)",
      _hopfield_inputs,
      python="""
        def hopfield(patterns, probe):
            n = len(probe)
            W = [[0.0 if i == j else sum(p[i] * p[j] for p in patterns) / n for j in range(n)] for i in range(n)]

            def energy(s):
                return -0.5 * sum(s[i] * W[i][j] * s[j] for i in range(n) for j in range(n))

            s = list(probe)
            before = energy(s)
            for i in range(n):  # one sweep, in order; a zero input leaves the unit alone
                h = sum(W[i][j] * s[j] for j in range(n))
                if h != 0:
                    s[i] = 1.0 if h > 0 else -1.0
            return {"energy_before": before, "state": s, "energy_after": energy(s)}
      """,
      numpy="""
        def hopfield(patterns, probe):
            n = probe.shape[0]
            W = patterns.T @ patterns / n * (1 - np.eye(n))
            s = probe.copy()
            before = -0.5 * s @ W @ s
            for i in range(n):
                h = W[i] @ s
                s[i] = np.where(h > 0, 1.0, np.where(h < 0, -1.0, s[i]))
            return {"energy_before": before, "state": s, "energy_after": -0.5 * s @ W @ s}
      """,
      jax="""
        def hopfield(patterns, probe):
            n = probe.shape[0]
            W = patterns.T @ patterns / n * (1 - jnp.eye(n))

            def body(i, s):
                h = W[i] @ s
                return s.at[i].set(jnp.where(h > 0, 1.0, jnp.where(h < 0, -1.0, s[i])))

            s = jax.lax.fori_loop(0, n, body, probe)
            return {"energy_before": -0.5 * probe @ W @ probe, "state": s, "energy_after": -0.5 * s @ W @ s}
      """,
      torch="""
        def hopfield(patterns, probe):
            n = probe.shape[0]
            W = patterns.T @ patterns / n * (1 - torch.eye(n))
            s = probe.clone()
            before = -0.5 * s @ W @ s
            for i in range(n):
                h = W[i] @ s
                if h != 0:
                    s[i] = torch.sign(h)  # in place: fine in PyTorch, an error on a JAX array
            return {"energy_before": before, "state": s, "energy_after": -0.5 * s @ W @ s}
      """,
      notes=(("Updating units one at a time (asynchronously) is what guarantees the energy never rises; "
             "updating all at once can oscillate between two states."),
             ("Module 29 picks the unit at random each step (`jax.random.randint` + `fold_in`); this entry sweeps "
             "in order so every library gives the same answer."),
             "`sign(0)` is 0 in all three libraries, which would zero a unit; the update keeps $s_i$ instead."),
      module="29"),
    # ---------------------------------------------------------------- word2vec (Module 30)
    E("skip-gram negative-sampling loss", "Probability and information",
      "Make a word's vector agree with the words around it and disagree with a few random words. A cheap "
      "stand-in for a softmax over the whole vocabulary.",
      r"\mathcal{L} = -\log \sigma(u_o^\top v_c) - \sum_{k=1}^{K} \log \sigma(-u_k^\top v_c)",
      r"$\log \sigma({v'_{w_O}}^\top v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} "
      r"\log \sigma(-{v'_{w_i}}^\top v_{w_I})$ (Mikolov et al., 2013)",
      "center (B,), context (B,), negatives (B, K) int; emb_in, emb_out (V, D) -> ()",
      "sgns_loss(center, context, negatives, emb_in, emb_out)",
      lambda r: {"center": r.integers(0, 10, 3), "context": r.integers(0, 10, 3),
                 "negatives": r.integers(0, 10, (3, 2)), "emb_in": r.normal(size=(10, 4)),
                 "emb_out": r.normal(size=(10, 4))},
      python="""
        def sgns_loss(center, context, negatives, emb_in, emb_out):
            def log_sigmoid(x):  # stable for large |x|
                return -math.log1p(math.exp(-x)) if x >= 0 else x - math.log1p(math.exp(x))

            def dot(a, b):
                return sum(p * q for p, q in zip(a, b))

            total = 0.0
            for c, o, negs in zip(center, context, negatives):
                v = emb_in[c]
                total += log_sigmoid(dot(v, emb_out[o]))
                for k in negs:
                    total += log_sigmoid(-dot(v, emb_out[k]))
            return -total / len(center)
      """,
      numpy="""
        def sgns_loss(center, context, negatives, emb_in, emb_out):
            v = emb_in[center]                                            # (B, D)
            pos = -np.logaddexp(0, -np.sum(v * emb_out[context], -1))     # log sigmoid(x) = -log(1 + e^-x)
            neg = -np.logaddexp(0, np.einsum("bd,bkd->bk", v, emb_out[negatives])).sum(-1)
            return -np.mean(pos + neg)
      """,
      jax="""
        def sgns_loss(center, context, negatives, emb_in, emb_out):
            v = emb_in[center]
            pos = jax.nn.log_sigmoid(jnp.sum(v * emb_out[context], -1))
            neg = jax.nn.log_sigmoid(-jnp.einsum("bd,bkd->bk", v, emb_out[negatives])).sum(-1)
            return -jnp.mean(pos + neg)
      """,
      torch="""
        def sgns_loss(center, context, negatives, emb_in, emb_out):
            v = emb_in[center]
            pos = F.logsigmoid(torch.sum(v * emb_out[context], -1))
            neg = F.logsigmoid(-torch.einsum("bd,bkd->bk", v, emb_out[negatives])).sum(-1)
            return -torch.mean(pos + neg)
      """,
      einsum='"bd,bkd->bk"',
      notes=(("Use `log_sigmoid`, never `log(sigmoid(x))`: in float32 `sigmoid(-200)` rounds to 0 and the log "
             "is -inf, while `log_sigmoid(-200)` is -200."),
             ("Two embedding tables: `emb_in` for center words, `emb_out` for context words. The paper's $v'$ "
             "is `emb_out`; Module 30 keeps `emb_in` as the word vectors.")),
      module="30", also=("38",)),
    # ---------------------------------------------------------------- LSTM (Module 31)
    E("LSTM cell", "Sequence and state-space models",
      "One step of an LSTM: four gates computed from the input and the previous output decide what to forget, "
      "what to write and what to show. The cell state c carries memory with only multiplications by f.",
      r"[i, f, g, o] = [\sigma, \sigma, \tanh, \sigma](x W_x + h_{t-1} W_h + b);\quad "
      r"c_t = f \odot c_{t-1} + i \odot g;\quad h_t = o \odot \tanh(c_t)",
      r"$c_t = f_t \circ c_{t-1} + i_t \circ \tilde c_t$ (Hochreiter & Schmidhuber, 1997; forget gate: Gers "
      r"et al., 2000)",
      "x (d,), h (H,), c (H,); Wx (d, 4H), Wh (H, 4H), b (4H,) -> h, c (H,)", "lstm_cell(Wx, Wh, b, h, c, x)",
      lambda r: {"Wx": r.normal(size=(3, 8)), "Wh": r.normal(size=(2, 8)), "b": r.normal(size=8),
                 "h": r.normal(size=2), "c": r.normal(size=2), "x": r.normal(size=3)},
      python="""
        def lstm_cell(Wx, Wh, b, h, c, x):
            H = len(h)

            def sigmoid(v):
                return 1 / (1 + math.exp(-v))

            z = [sum(x[k] * Wx[k][j] for k in range(len(x))) + sum(h[k] * Wh[k][j] for k in range(H)) + b[j]
                 for j in range(4 * H)]
            h_new, c_new = [], []
            for j in range(H):  # gate order i, f, g, o
                i, f, g, o = sigmoid(z[j]), sigmoid(z[H + j]), math.tanh(z[2 * H + j]), sigmoid(z[3 * H + j])
                c_new.append(f * c[j] + i * g)
                h_new.append(o * math.tanh(c_new[j]))
            return {"h": h_new, "c": c_new}
      """,
      numpy="""
        def lstm_cell(Wx, Wh, b, h, c, x):
            def sigmoid(v):
                return 1 / (1 + np.exp(-v))

            zi, zf, zg, zo = np.split(x @ Wx + h @ Wh + b, 4)
            c = sigmoid(zf) * c + sigmoid(zi) * np.tanh(zg)
            return {"h": sigmoid(zo) * np.tanh(c), "c": c}
      """,
      jax="""
        def lstm_cell(Wx, Wh, b, h, c, x):
            zi, zf, zg, zo = jnp.split(x @ Wx + h @ Wh + b, 4)
            c = jax.nn.sigmoid(zf) * c + jax.nn.sigmoid(zi) * jnp.tanh(zg)
            return {"h": jax.nn.sigmoid(zo) * jnp.tanh(c), "c": c}
      """,
      torch="""
        def lstm_cell(Wx, Wh, b, h, c, x):
            cell = torch.nn.LSTMCell(Wx.shape[0], h.shape[0])
            with torch.no_grad():  # PyTorch stores (4H, d) weights and two biases
                cell.weight_ih.copy_(Wx.T)
                cell.weight_hh.copy_(Wh.T)
                cell.bias_ih.copy_(b)
                cell.bias_hh.zero_()
                h_new, c_new = cell(x[None], (h[None], c[None]))
            return {"h": h_new[0], "c": c_new[0]}
      """,
      notes=(("PyTorch packs the gates in the same order as the course (i, f, g, o) but stores weights "
             "transposed, `weight_ih` (4H, d), and has two biases, `bias_ih` + `bias_hh`; only their sum "
             "matters. Porting weights: transpose, and put the bias in one of them."),
             ("The cell state is updated by multiply-and-add, not squashed through a nonlinearity each step: "
             "that additive path is why gradients survive long sequences.")),
      module="31"),
    # ---------------------------------------------------------------- Generative models (Module 33)
    E("VAE reparameterization and Gaussian KL", "Generative models",
      "Sample a latent as mean plus noise times standard deviation, so the randomness is an input and "
      "gradients flow through the mean and variance. The KL term pulls each posterior toward a standard normal.",
      r"z = \mu + \sigma \odot \epsilon,\; \epsilon \sim \mathcal{N}(0, I);\quad "
      r"D_{\mathrm{KL}}\big(\mathcal{N}(\mu, \sigma^2) \,\Vert\, \mathcal{N}(0, I)\big) = "
      r"-\tfrac{1}{2} \textstyle\sum_j \big(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\big)",
      r"$-D_{KL} = \tfrac12 \sum_j (1 + \log((\sigma_j)^2) - (\mu_j)^2 - (\sigma_j)^2)$ (Kingma & Welling, 2013)",
      "mu, logvar, eps (B, d) -> z (B, d), kl (B,)", "vae_sample(mu, logvar, eps)",
      lambda r: {"mu": r.normal(size=(3, 4)), "logvar": r.normal(size=(3, 4)), "eps": r.normal(size=(3, 4))},
      python="""
        def vae_sample(mu, logvar, eps):
            z, kl = [], []
            for m_row, lv_row, e_row in zip(mu, logvar, eps):
                z.append([m + math.exp(0.5 * lv) * e for m, lv, e in zip(m_row, lv_row, e_row)])
                kl.append(-0.5 * sum(1 + lv - m * m - math.exp(lv) for m, lv in zip(m_row, lv_row)))
            return {"z": z, "kl": kl}
      """,
      numpy="""
        def vae_sample(mu, logvar, eps):
            z = mu + np.exp(0.5 * logvar) * eps
            kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar), axis=-1)
            return {"z": z, "kl": kl}
      """,
      jax="""
        def vae_sample(mu, logvar, eps):  # in a model: eps = jax.random.normal(key, mu.shape)
            z = mu + jnp.exp(0.5 * logvar) * eps
            kl = -0.5 * jnp.sum(1 + logvar - mu ** 2 - jnp.exp(logvar), axis=-1)
            return {"z": z, "kl": kl}
      """,
      torch="""
        def vae_sample(mu, logvar, eps):
            q = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
            kl = torch.distributions.kl_divergence(q, p).sum(-1)
            return {"z": mu + q.stddev * eps, "kl": kl}  # q.rsample() draws its own eps
      """,
      notes=(("Encoders usually output log-variance, not σ: it can be any real number. Then "
             "$\\sigma = e^{\\frac12 \\log \\sigma^2}$, which is where the 0.5 comes from."),
             "`torch.distributions.Normal` takes the standard deviation, not the variance.",
             "JAX has no distributions module in the core library; the closed-form KL is two lines anyway."),
      module="33"),
    E("diffusion forward noising", "Generative models",
      "Jump straight to step t of the noising process: shrink the clean image and add Gaussian noise, with "
      "the mix set by the cumulative product of the noise schedule.",
      r"x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon,\quad "
      r"\bar\alpha_t = \textstyle\prod_{s \le t} (1 - \beta_s),\; \beta \text{ linear from } 10^{-4} \text{ to } 0.02",
      r"$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t) I)$ (Ho et al., 2020)",
      "x0 (B, d), t (B,) int in [0, T), noise (B, d) -> x_t (B, d)", "q_sample(x0, t, noise)",
      lambda r: {"x0": r.normal(size=(3, 5)), "t": np.array([0, 499, 999]), "noise": r.normal(size=(3, 5))},
      python="""
        def q_sample(x0, t, noise, T=1000, beta_start=1e-4, beta_end=0.02):
            abar, prod = [], 1.0
            for s in range(T):
                prod *= 1 - (beta_start + (beta_end - beta_start) * s / (T - 1))
                abar.append(prod)
            return [[math.sqrt(abar[ti]) * x + math.sqrt(1 - abar[ti]) * e for x, e in zip(row, n_row)]
                    for row, ti, n_row in zip(x0, t, noise)]
      """,
      numpy="""
        def q_sample(x0, t, noise, T=1000, beta_start=1e-4, beta_end=0.02):
            abar = np.cumprod(1 - np.linspace(beta_start, beta_end, T))[t][:, None]
            return np.sqrt(abar) * x0 + np.sqrt(1 - abar) * noise
      """,
      jax="""
        def q_sample(x0, t, noise, T=1000, beta_start=1e-4, beta_end=0.02):
            abar = jnp.cumprod(1 - jnp.linspace(beta_start, beta_end, T))[t][:, None]
            return jnp.sqrt(abar) * x0 + jnp.sqrt(1 - abar) * noise
      """,
      torch="""
        def q_sample(x0, t, noise, T=1000, beta_start=1e-4, beta_end=0.02):
            abar = torch.cumprod(1 - torch.linspace(beta_start, beta_end, T), 0)[t][:, None]
            return torch.sqrt(abar) * x0 + torch.sqrt(1 - abar) * noise
      """,
      notes=("The paper counts steps 1..T; code indexes 0..T-1, so the paper's $t$ is the code's `t + 1`.",
             "At t = 999, $\\bar\\alpha \\approx 4 \\times 10^{-5}$: almost pure noise, which is the point.",
             "`torch.cumprod` requires the dimension argument; `np.cumprod` / `jnp.cumprod` flatten without one."),
      module="33"),
    E("contrastive (CLIP / InfoNCE) loss", "Generative models",
      "Score every image against every caption; train so each image's own caption wins, and each caption's "
      "own image wins. A classification where the classes are the other items in the batch.",
      r"S_{ij} = \frac{\hat u_i \cdot \hat v_j}{\tau};\quad \mathcal{L} = \tfrac{1}{2}\Big(-\tfrac1N \sum_i "
      r"\log \mathrm{softmax}_j(S)_{ii} - \tfrac1N \sum_j \log \mathrm{softmax}_i(S)_{jj}\Big)",
      r"InfoNCE (van den Oord et al., 2018); symmetric form in CLIP (Radford et al., 2021)",
      "img (N, d), txt (N, d), temperature -> ()", "clip_loss(img, txt)",
      lambda r: {"img": r.normal(size=(4, 6)), "txt": r.normal(size=(4, 6))},
      python="""
        def clip_loss(img, txt, temperature=0.07):
            def unit(v):
                n = math.sqrt(sum(a * a for a in v))
                return [a / n for a in v]

            def log_softmax_at(row, i):
                m = max(row)
                return row[i] - m - math.log(sum(math.exp(x - m) for x in row))

            img, txt = [unit(v) for v in img], [unit(v) for v in txt]
            N = len(img)
            S = [[sum(a * b for a, b in zip(img[i], txt[j])) / temperature for j in range(N)] for i in range(N)]
            i2t = -sum(log_softmax_at(S[i], i) for i in range(N)) / N              # each row: its caption
            t2i = -sum(log_softmax_at([S[i][j] for i in range(N)], j) for j in range(N)) / N  # each column
            return 0.5 * (i2t + t2i)
      """,
      numpy="""
        def clip_loss(img, txt, temperature=0.07):
            def log_softmax(x, axis):
                x = x - x.max(axis, keepdims=True)
                return x - np.log(np.exp(x).sum(axis, keepdims=True))

            img = img / np.linalg.norm(img, axis=-1, keepdims=True)
            txt = txt / np.linalg.norm(txt, axis=-1, keepdims=True)
            S = img @ txt.T / temperature
            return -0.5 * (np.diag(log_softmax(S, 1)).mean() + np.diag(log_softmax(S, 0)).mean())
      """,
      jax="""
        def clip_loss(img, txt, temperature=0.07):
            img = img / jnp.linalg.norm(img, axis=-1, keepdims=True)
            txt = txt / jnp.linalg.norm(txt, axis=-1, keepdims=True)
            S = img @ txt.T / temperature
            labels = jnp.arange(S.shape[0])
            i2t = optax.softmax_cross_entropy_with_integer_labels(S, labels).mean()
            t2i = optax.softmax_cross_entropy_with_integer_labels(S.T, labels).mean()
            return 0.5 * (i2t + t2i)
      """,
      torch="""
        def clip_loss(img, txt, temperature=0.07):
            S = F.normalize(img, dim=-1) @ F.normalize(txt, dim=-1).T / temperature
            labels = torch.arange(S.shape[0])
            return 0.5 * (F.cross_entropy(S, labels) + F.cross_entropy(S.T, labels))
      """,
      einsum='"id,jd->ij"',
      notes=(("The labels are just `arange(N)`: item i's match is at position i. The batch is the set of "
             "negatives, so bigger batches make a harder task."),
             ("CLIP learns the temperature (as a log scale); Module 33 fixes τ = 0.07 and Module 38's retriever "
             "uses τ = 0.05 with only the query-to-document direction.")),
      module="33", also=("38",)),
    # ---------------------------------------------------------------- Deep RL (Module 34)
    E("DQN target and Huber loss", "Reinforcement learning and preferences",
      "Regress the Q-value of the action taken toward reward plus the discounted best next value from a frozen "
      "target network; Huber loss keeps one surprising transition from producing a huge gradient.",
      r"y = r + \gamma (1 - d) \max_{a'} Q_{\bar\theta}(s', a');\quad \mathcal{L} = \tfrac1B \textstyle\sum "
      r"\mathrm{Huber}_1\big(Q_\theta(s, a) - y\big),\; \mathrm{Huber}_1(e) = \begin{cases} \tfrac12 e^2 & "
      r"|e| \le 1 \\ |e| - \tfrac12 & \text{else} \end{cases}",
      r"$L_i(\theta_i) = \mathbb{E}\big[(r + \gamma \max_{a'} Q(s', a'; \theta_i^-) - Q(s, a; \theta_i))^2\big]$ "
      r"with the error clipped to [-1, 1] (Mnih et al., 2015)",
      "q (B, A), a (B,) int, r (B,), done (B,), q_next_target (B, A) -> target (B,), loss ()",
      "dqn(q, a, r, done, q_next_target)",
      lambda r: {"q": 2 * r.normal(size=(4, 3)), "a": np.array([0, 2, 1, 1]), "r": r.normal(size=4),
                 "done": np.array([0.0, 0.0, 1.0, 0.0]), "q_next_target": 2 * r.normal(size=(4, 3))},
      python="""
        def dqn(q, a, r, done, q_next_target, gamma=0.99):
            target, loss = [], 0.0
            for qs, ai, ri, di, qn in zip(q, a, r, done, q_next_target):
                y = ri + gamma * (1 - di) * max(qn)
                e = qs[ai] - y
                loss += 0.5 * e * e if abs(e) <= 1 else abs(e) - 0.5
                target.append(y)
            return {"target": target, "loss": loss / len(q)}
      """,
      numpy="""
        def dqn(q, a, r, done, q_next_target, gamma=0.99):
            y = r + gamma * (1 - done) * q_next_target.max(-1)
            e = q[np.arange(len(a)), a] - y
            return {"target": y, "loss": np.mean(np.where(np.abs(e) <= 1, 0.5 * e ** 2, np.abs(e) - 0.5))}
      """,
      jax="""
        def dqn(q, a, r, done, q_next_target, gamma=0.99):
            y = r + gamma * (1 - done) * q_next_target.max(-1)
            q_a = jnp.take_along_axis(q, a[:, None], axis=-1)[:, 0]
            loss = optax.huber_loss(q_a, jax.lax.stop_gradient(y), delta=1.0).mean()
            return {"target": y, "loss": loss}
      """,
      jax_by_hand="""
        def dqn(q, a, r, done, q_next_target, gamma=0.99):
            y = jax.lax.stop_gradient(r + gamma * (1 - done) * q_next_target.max(-1))
            e = jnp.take_along_axis(q, a[:, None], axis=-1)[:, 0] - y
            return {"target": y, "loss": jnp.mean(jnp.where(jnp.abs(e) <= 1, 0.5 * e ** 2, jnp.abs(e) - 0.5))}
      """,
      torch="""
        def dqn(q, a, r, done, q_next_target, gamma=0.99):
            with torch.no_grad():
                y = r + gamma * (1 - done) * q_next_target.max(-1).values
            q_a = q.gather(1, a[:, None])[:, 0]
            return {"target": y, "loss": F.smooth_l1_loss(q_a, y)}  # = Huber with delta 1
      """,
      notes=(("The target must not be differentiated: `jax.lax.stop_gradient` in JAX, `torch.no_grad()` (or "
             "`.detach()`) in PyTorch. Otherwise the loss also drags the target toward the prediction."),
             "`F.smooth_l1_loss(beta=β)` is `F.huber_loss(delta=β)` divided by β; they agree only at 1.",
             ("`max` along an axis returns values and indices in PyTorch (`.values`); in NumPy/JAX it is just "
             "the values."), "`done` zeroes the bootstrap: there is no next state after the episode ends."),
      module="34"),
    E("PUCT score (AlphaZero search)", "Reinforcement learning and preferences",
      "Pick which move to explore next: its average value so far, plus a bonus that is large for moves the "
      "policy network likes and that have been tried rarely.",
      r"a^* = \arg\max_a \Big(Q(s, a) + c\, P(s, a) \frac{\sqrt{N(s)}}{1 + N(s, a)}\Big)",
      r"$U(s, a) = c_{\text{puct}} P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$ (Silver et al., 2017)",
      "q, prior, n_child (A,); n_parent, c -> scores (A,), choice", "puct(q, prior, n_child, 20.0)",
      lambda r: {"q": r.uniform(-1, 1, 5), "prior": np.array([0.4, 0.3, 0.15, 0.1, 0.05]),
                 "n_child": np.array([8.0, 6.0, 4.0, 2.0, 0.0])},
      python="""
        def puct(q, prior, n_child, n_parent, c=1.5):
            scores = [qa + c * p * math.sqrt(n_parent) / (1 + n) for qa, p, n in zip(q, prior, n_child)]
            return {"scores": scores, "choice": max(range(len(scores)), key=lambda i: scores[i])}
      """,
      numpy="""
        def puct(q, prior, n_child, n_parent, c=1.5):
            scores = q + c * prior * np.sqrt(n_parent) / (1 + n_child)
            return {"scores": scores, "choice": np.argmax(scores)}
      """,
      jax="""
        def puct(q, prior, n_child, n_parent, c=1.5):
            scores = q + c * prior * jnp.sqrt(n_parent) / (1 + n_child)
            return {"scores": scores, "choice": jnp.argmax(scores)}
      """,
      torch="""
        def puct(q, prior, n_child, n_parent, c=1.5):
            scores = q + c * prior * torch.sqrt(torch.tensor(n_parent)) / (1 + n_child)
            return {"scores": scores, "choice": torch.argmax(scores)}
      """,
      notes=(("An unvisited move (N = 0) gets the full $c P \\sqrt{N(s)}$ bonus, so a strong prior gets tried "
             "early even with no value estimate yet."),
             ("DeepMind's `mctx` (Module 34) grows c with the parent's visits in `muzero_policy`: "
             "$c = c_1 + \\log\\frac{N + c_2 + 1}{c_2}$. Module 34's AlphaZero agent uses "
             "`gumbel_muzero_policy`, which replaces PUCT at the root with Gumbel sampling (Danihelka et al., "
             "2022)."),
             "`torch.sqrt` needs a tensor; `math.sqrt` works for a plain count."),
      module="34"),
    # ---------------------------------------------------------------- Preferences (Module 35)
    E("Bradley–Terry reward-model loss", "Reinforcement learning and preferences",
      "Train a reward model from pairs a human ranked: the probability that the chosen answer beats the "
      "rejected one is the sigmoid of their reward difference; maximize it.",
      r"\mathcal{L} = -\frac1N \sum_n \log \sigma\big(r(x_n, y_n^{w}) - r(x_n, y_n^{l})\big)",
      r"$P(y_w \succ y_l) = \frac{e^{r(y_w)}}{e^{r(y_w)} + e^{r(y_l)}}$ (Bradley & Terry, 1952; "
      r"Ouyang et al., 2022)",
      "r_chosen (N,), r_rejected (N,) -> ()", "bt_loss(r_chosen, r_rejected)",
      lambda r: {"r_chosen": r.normal(size=6), "r_rejected": r.normal(size=6)},
      python="""
        def bt_loss(r_chosen, r_rejected):
            total = 0.0
            for rc, rr in zip(r_chosen, r_rejected):
                d = rc - rr
                total += math.log1p(math.exp(-d)) if d >= 0 else -d + math.log1p(math.exp(d))  # -log sigmoid(d)
            return total / len(r_chosen)
      """,
      numpy="""
        def bt_loss(r_chosen, r_rejected):
            return np.mean(np.logaddexp(0, -(r_chosen - r_rejected)))
      """,
      jax="""
        def bt_loss(r_chosen, r_rejected):
            return jnp.mean(-jax.nn.log_sigmoid(r_chosen - r_rejected))
      """,
      torch="""
        def bt_loss(r_chosen, r_rejected):
            return -F.logsigmoid(r_chosen - r_rejected).mean()
      """,
      notes=(("It is logistic regression on the reward difference with label 1: "
             "`F.binary_cross_entropy_with_logits(rc - rr, ones)` gives the same number."),
             ("Only differences matter, so adding a constant to every reward changes nothing; reward models "
             "are often re-centered after training.")),
      module="35"),
    E("DPO loss", "Reinforcement learning and preferences",
      "Skip the reward model: treat β times how much more the policy likes an answer than the frozen "
      "reference does as its implicit reward, and apply the Bradley–Terry loss to chosen vs rejected.",
      r"\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w \mid x)}"
      r"{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\Big)",
      r"$\hat r_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ (Rafailov et al., 2023)",
      "sequence log-probs pi_c, pi_r, ref_c, ref_r (N,), beta -> loss (), margin (N,)",
      "dpo(pi_c, pi_r, ref_c, ref_r)",
      lambda r: {"pi_c": -20 + 3 * r.normal(size=4), "pi_r": -20 + 3 * r.normal(size=4),
                 "ref_c": -20 + 3 * r.normal(size=4), "ref_r": -20 + 3 * r.normal(size=4)},
      python="""
        def dpo(pi_c, pi_r, ref_c, ref_r, beta=0.1):
            margins, total = [], 0.0
            for a, b, c, d in zip(pi_c, pi_r, ref_c, ref_r):
                m = beta * (a - c) - beta * (b - d)  # implicit reward, chosen minus rejected
                total += math.log1p(math.exp(-m)) if m >= 0 else -m + math.log1p(math.exp(m))
                margins.append(m)
            return {"loss": total / len(margins), "margin": margins}
      """,
      numpy="""
        def dpo(pi_c, pi_r, ref_c, ref_r, beta=0.1):
            m = beta * (pi_c - ref_c) - beta * (pi_r - ref_r)
            return {"loss": np.mean(np.logaddexp(0, -m)), "margin": m}
      """,
      jax="""
        def dpo(pi_c, pi_r, ref_c, ref_r, beta=0.1):
            m = beta * (pi_c - ref_c) - beta * (pi_r - ref_r)
            return {"loss": jnp.mean(-jax.nn.log_sigmoid(m)), "margin": m}
      """,
      torch="""
        def dpo(pi_c, pi_r, ref_c, ref_r, beta=0.1):
            m = beta * (pi_c - ref_c) - beta * (pi_r - ref_r)
            return {"loss": -F.logsigmoid(m).mean(), "margin": m}
      """,
      notes=(("The inputs are whole-sequence log-probabilities: the sum of the response tokens' log-probs "
             "(prompt tokens masked out), not the mean. Module 35's `sequence_logprob` sums."),
             "The reference log-probs are constants (computed once, no gradient); only $\\pi_\\theta$ trains.",
             "β plays the KL-penalty role of RLHF: small β lets the policy drift further from the reference."),
      module="35", also=("26",)),
    # ---------------------------------------------------------------- State-space models (Module 36)
    E("2-D RNN sweep (MDRNN)", "Sequence and state-space models",
      "A recurrent cell on a grid: each cell mixes its input with the hidden states of the cell above and the "
      "cell to its left, so information flows down and right from one corner. Four sweeps, one from each "
      "corner, let every cell see the whole image.",
      r"h_{i,j} = \tanh\big(x_{i,j} W_x + h_{i-1,j} W_u + h_{i,j-1} W_l + b\big),\quad h_{-1,j} = h_{i,-1} = 0",
      r"Graves, Fernández & Schmidhuber 2007: one recurrent connection per dimension, $2^d$ sweeps for $d$ "
      r"dimensions; MDLSTM adds one forget gate per dimension",
      "x (R, C, F), Wx (F, H), Wu (H, H), Wl (H, H), b (H,) -> (R, C, H)", "sweep(x, wx, wu, wl, b)",
      lambda r: {"x": r.normal(size=(3, 4, 2)), "wx": r.normal(size=(2, 3)) * 0.7,
                 "wu": r.normal(size=(3, 3)) * 0.4, "wl": r.normal(size=(3, 3)) * 0.4, "b": r.normal(size=3) * 0.1},
      python="""
        def sweep(x, wx, wu, wl, b):
            R, C, H = len(x), len(x[0]), len(b)
            def mv(v, w):  # row vector times matrix
                return [sum(v[k] * w[k][j] for k in range(len(v))) for j in range(len(w[0]))]
            h = [[[0.0] * H for _ in range(C)] for _ in range(R)]
            for i in range(R):
                for j in range(C):
                    up = h[i - 1][j] if i else [0.0] * H
                    left = h[i][j - 1] if j else [0.0] * H
                    a, u, l = mv(x[i][j], wx), mv(up, wu), mv(left, wl)
                    h[i][j] = [math.tanh(a[k] + u[k] + l[k] + b[k]) for k in range(H)]
            return h
      """,
      numpy="""
        def sweep(x, wx, wu, wl, b):
            R, C, _ = x.shape
            H = len(b)
            h = np.zeros((R, C, H))
            for i in range(R):
                for j in range(C):
                    up = h[i - 1, j] if i else np.zeros(H)
                    left = h[i, j - 1] if j else np.zeros(H)
                    h[i, j] = np.tanh(x[i, j] @ wx + up @ wu + left @ wl + b)
            return h
      """,
      jax="""
        def sweep(x, wx, wu, wl, b):  # the diagonal wavefront: skew, then scan over R + C - 1 columns
            R, C, _ = x.shape
            rows = jnp.arange(R)[:, None]
            cols = rows + jnp.arange(C)[None, :]                        # row i shifted right by i
            z = jnp.zeros((R, R + C - 1, len(b))).at[rows, cols].set(x @ wx + b)
            mask = jnp.zeros((R, R + C - 1)).at[rows, cols].set(1.0)    # which skewed cells are real

            def step(h_prev, col):
                zc, m = col
                up = jnp.concatenate([jnp.zeros_like(h_prev[:1]), h_prev[:-1]])
                h = jnp.tanh(zc + up @ wu + h_prev @ wl) * m[:, None]
                return h, h

            hs = jax.lax.scan(step, jnp.zeros((R, len(b))), (jnp.swapaxes(z, 0, 1), mask.T))[1]
            return jnp.swapaxes(hs, 0, 1)[rows, cols]                  # unskew
      """,
      torch="""
        def sweep(x, wx, wu, wl, b):
            R, C, _ = x.shape
            h = torch.zeros(R, C, len(b), dtype=x.dtype)
            for i in range(R):
                for j in range(C):
                    s = x[i, j] @ wx + b
                    if i:
                        s = s + h[i - 1, j] @ wu
                    if j:
                        s = s + h[i, j - 1] @ wl
                    h[i, j] = torch.tanh(s)
            return h
      """,
      jax_by_hand="""
        def sweep(x, wx, wu, wl, b):  # cell by cell; JAX arrays are immutable, so build rows in lists
            R, C, _ = x.shape
            rows, prev = [], jnp.zeros((C, len(b)))
            for i in range(R):
                left, row = jnp.zeros(len(b)), []
                for j in range(C):
                    left = jnp.tanh(x[i, j] @ wx + prev[j] @ wu + left @ wl + b)
                    row.append(left)
                prev = jnp.stack(row)
                rows.append(prev)
            return jnp.stack(rows)
      """,
      notes=(("Cells on one anti-diagonal (i + j constant) don't depend on each other, so the fast version computes "
              "them together: R + C - 1 sequential steps instead of R * C. Skewing row i right by i turns "
              "anti-diagonals into columns (PixelRNN's Diagonal LSTM uses the same trick)."),
             ("The skew pads with zeros; mask the padded cells after the tanh, or tanh(b) leaks into the next "
              "column as a fake neighbour."),
             ("Other corners are the same function on a flipped grid: flip rows for bottom-left, columns for "
              "top-right, both for bottom-right, then flip the result back.")),
      module="31b"),
    E("linear recurrence as a parallel scan", "Sequence and state-space models",
      "The heart of an SSM layer: each state is the previous one scaled plus a new input. It looks sequential, "
      "but pairs of steps compose associatively, so it can run as a tree in log(L) parallel rounds.",
      r"h_t = a_t\, h_{t-1} + b_t,\; h_0 = 0;\quad (a_1, b_1) \bullet (a_2, b_2) = (a_1 a_2,\; a_2 b_1 + b_2)",
      r"$h_t = \bar A h_{t-1} + \bar B x_t$ (Gu et al., 2021; Gu & Dao, 2023); scan: Blelloch, 1990",
      "a (L,), b (L,) -> h (L,)", "recurrence(a, b)",
      lambda r: {"a": r.uniform(0.5, 1.0, 8), "b": r.normal(size=8)},
      python="""
        def recurrence(a, b):
            h, out = 0.0, []
            for at, bt in zip(a, b):
                h = at * h + bt
                out.append(h)
            return out
      """,
      numpy="""
        def recurrence(a, b):
            h, out = 0.0, np.empty_like(b)
            for t in range(len(b)):
                h = a[t] * h + b[t]
                out[t] = h
            return out
      """,
      jax="""
        def recurrence(a, b):
            def combine(left, right):  # compose step "left" then step "right"
                a1, b1 = left
                a2, b2 = right
                return a1 * a2, a2 * b1 + b2

            _, h = jax.lax.associative_scan(combine, (a, b))
            return h
      """,
      jax_by_hand="""
        def recurrence(a, b):  # the sequential version: L dependent steps
            def step(h, ab):
                h = ab[0] * h + ab[1]
                return h, h

            _, h = jax.lax.scan(step, jnp.zeros(()), (a, b))
            return h
      """,
      torch="""
        def recurrence(a, b):
            h, out = torch.tensor(0.0), []
            for at, bt in zip(a, b):  # no public parallel scan in PyTorch
                h = at * h + bt
                out.append(h)
            return torch.stack(out)
      """,
      notes=(("`combine` must be associative but need not be commutative: swapping `left` and `right` gives a "
             "wrong answer. `associative_scan` calls it on batches of pairs, so write it elementwise."),
             ("The parallel version does more total work (about 2L combines vs L steps) but in log₂ L rounds; "
             "on a GPU, rounds are what cost time."),
             "PyTorch has no public associative scan; Mamba ships a custom CUDA kernel for it."),
      apis=("jax.lax.associative_scan",),
      module="36"),
    E("long convolution via FFT", "Sequence and state-space models",
      "Convolving a length-L signal with a length-L kernel costs L² multiplies directly, L log L through the "
      "FFT: transform both, multiply, transform back. Zero-pad first or the end wraps around to the start.",
      r"y_t = \sum_{s=0}^{t} k_s\, x_{t-s} = \Big[\mathcal{F}^{-1}\big(\mathcal{F}(k) \cdot \mathcal{F}(x)\big)"
      r"\Big]_t \text{ (zero-padded to } 2L)",
      r"$y = x * \bar K$ (Gu et al., 2021, S4; Poli et al., 2023, Hyena)", "kernel (L,), x (L,) -> y (L,)",
      "causal_conv(k, x)", lambda r: {"k": r.normal(size=8), "x": r.normal(size=8)},
      python="""
        def causal_conv(k, x):  # the definition, O(L^2)
            return [sum(k[s] * x[t - s] for s in range(t + 1)) for t in range(len(x))]
      """,
      numpy="""
        def causal_conv(k, x):
            n = 2 * len(x)  # pad so the circular convolution doesn't wrap
            return np.fft.irfft(np.fft.rfft(k, n) * np.fft.rfft(x, n), n)[:len(x)]
      """,
      jax="""
        def causal_conv(k, x):
            n = 2 * x.shape[0]
            return jnp.fft.irfft(jnp.fft.rfft(k, n) * jnp.fft.rfft(x, n), n)[:x.shape[0]]
      """,
      torch="""
        def causal_conv(k, x):
            n = 2 * x.shape[0]
            return torch.fft.irfft(torch.fft.rfft(k, n=n) * torch.fft.rfft(x, n=n), n=n)[:x.shape[0]]
      """,
      notes=(("Without padding (n = L) you get circular convolution: late inputs leak into early outputs, "
             "which breaks causality."),
             ("`np.convolve(k, x)[:L]` is the same thing directly. PyTorch's `F.conv1d` is cross-correlation "
             "(it does not flip the kernel): pass `k.flip(0)` and left-pad by L - 1 to match."),
             "`irfft` needs `n` to know the output length; leave it out and an odd length comes back one short."),
      apis=("jax.numpy.fft.rfft", "jax.numpy.fft.irfft"),
      module="36"),
    # ---------------------------------------------------------------- Retrieval (Module 38)
    E("segment sum (group-by add)", "Arrays and reductions",
      "Add up rows that share a label: one output row per group. Module 38 uses it to sum the vectors "
      "assigned to each cluster in k-means.",
      r"S_g = \sum_{i \,:\, \mathrm{seg}_i = g} x_i", "", "data (n, d), segment_ids (n,) int, num_segments -> (G, d)",
      "segment_sum(data, ids, 4)", lambda r: {"data": r.normal(size=(6, 3)), "ids": np.array([0, 2, 2, 1, 0, 2])},
      python="""
        def segment_sum(data, ids, num_segments):
            out = [[0.0] * len(data[0]) for _ in range(num_segments)]
            for row, g in zip(data, ids):
                out[g] = [o + v for o, v in zip(out[g], row)]
            return out
      """,
      numpy="""
        def segment_sum(data, ids, num_segments):
            out = np.zeros((num_segments, data.shape[1]))
            np.add.at(out, ids, data)
            return out
      """,
      jax="""
        def segment_sum(data, ids, num_segments):
            return jax.ops.segment_sum(data, ids, num_segments=num_segments)
      """,
      jax_by_hand="""
        def segment_sum(data, ids, num_segments):
            return jnp.zeros((num_segments, data.shape[1])).at[ids].add(data)
      """,
      torch="""
        def segment_sum(data, ids, num_segments):
            return torch.zeros(num_segments, data.shape[1]).index_add(0, ids, data)
      """,
      notes=(("Pass `num_segments` as a Python int: without it JAX infers `max(ids) + 1`, which needs concrete "
             "values and fails under `jit`."), "Empty groups (3 here) come out as zeros."),
      apis=("jax.ops.segment_sum",),
      module="38"),
    E("BM25 scoring as a sparse matrix product", "Retrieval and evaluation",
      "Weight each (document, term) count by how rare the term is and how long the document is, with "
      "diminishing returns for repeats; a query's score for every document is then one sparse matrix-vector "
      "product.",
      r"\mathrm{score}(d, q) = \sum_{t \in q} \mathrm{IDF}(t)\, \frac{f_{t,d}\,(k_1 + 1)}{f_{t,d} + k_1 "
      r"\big(1 - b + b\, \frac{|d|}{\mathrm{avgdl}}\big)},\quad \mathrm{IDF}(t) = \ln\!\Big(1 + "
      r"\frac{N - n_t + 0.5}{n_t + 0.5}\Big)",
      r"Okapi BM25 (Robertson et al., 1994); $k_1 = 1.5$, $b = 0.75$ here",
      "term counts tf (N, V), query (V,) 0/1 -> scores (N,)", "bm25(tf, query)",
      lambda r: {"tf": np.vstack([r.integers(0, 3, (4, 6)), [1, 0, 0, 0, 0, 0]]),
                 "query": np.array([1.0, 0, 1, 0, 0, 1])},
      python="""
        def bm25(tf, query, k1=1.5, b=0.75):
            N, V = len(tf), len(tf[0])
            lengths = [sum(row) for row in tf]
            avg = sum(lengths) / N
            df = [sum(1 for row in tf if row[t] > 0) for t in range(V)]
            scores = []
            for d in range(N):
                s = 0.0
                for t in range(V):
                    if tf[d][t] and query[t]:  # only nonzero entries: what a sparse matrix stores
                        idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                        s += idf * tf[d][t] * (k1 + 1) / (tf[d][t] + k1 * (1 - b + b * lengths[d] / avg))
                scores.append(s)
            return scores
      """,
      numpy="""
        def bm25(tf, query, k1=1.5, b=0.75):
            N = tf.shape[0]
            lengths = tf.sum(1, keepdims=True)
            idf = np.log1p((N - (tf > 0).sum(0) + 0.5) / ((tf > 0).sum(0) + 0.5))
            W = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths / lengths.mean()))
            return W @ query
      """,
      jax="""
        from jax.experimental import sparse

        def bm25(tf, query, k1=1.5, b=0.75):
            tf = tf.astype(jnp.float32)
            N, df = tf.shape[0], (tf > 0).sum(0)
            lengths = tf.sum(1, keepdims=True)
            W = jnp.log1p((N - df + 0.5) / (df + 0.5)) * tf * (k1 + 1) / (
                tf + k1 * (1 - b + b * lengths / lengths.mean()))
            return sparse.BCOO.fromdense(W) @ query  # stores only the nonzero weights
      """,
      torch="""
        def bm25(tf, query, k1=1.5, b=0.75):
            tf = tf.float()
            N, df = tf.shape[0], (tf > 0).sum(0)
            lengths = tf.sum(1, keepdim=True)
            W = torch.log1p((N - df + 0.5) / (df + 0.5)) * tf * (k1 + 1) / (
                tf + k1 * (1 - b + b * lengths / lengths.mean()))
            return torch.sparse.mm(W.to_sparse(), query[:, None])[:, 0]
      """,
      notes=(("Module 38 builds the BCOO matrix straight from (value, (doc, term)) lists over 50k abstracts; "
             "`fromdense` is for small examples, and under `jit` it needs `nse=` (the number of stored "
             "entries), because the output size can't depend on the data."),
             ("The original Robertson IDF, $\\ln\\frac{N - n_t + 0.5}{n_t + 0.5}$, goes negative for terms in "
             "more than half the documents; the $\\ln(1 + \\cdot)$ form (Lucene's, and the course's) never "
             "does.")),
      apis=("jax.experimental.sparse.BCOO",),
      module="38"),
    E("reciprocal rank fusion", "Retrieval and evaluation",
      "Merge several ranked lists by giving each document 1/(k + rank) from every list it appears in. Uses only "
      "positions, so scores on different scales (BM25, cosine) never have to be compared.",
      r"\mathrm{RRF}(d) = \sum_{r \in R} \frac{1}{k + \mathrm{rank}_r(d)},\quad k = 60,\; \mathrm{rank} \in "
      r"\{1, 2, \dots\}",
      r"Cormack, Clarke & Büttcher, 2009", "rankings (R, n) doc ids, n_docs -> fused score per doc (n_docs,)",
      "rrf(rankings, 7)", lambda r: {"rankings": np.stack([r.permutation(7)[:5], r.permutation(7)[:5]])},
      python="""
        def rrf(rankings, n_docs, k=60):
            score = [0.0] * n_docs
            for ranking in rankings:
                for rank, doc in enumerate(ranking, start=1):  # ranks start at 1
                    score[doc] += 1 / (k + rank)
            return score
      """,
      numpy="""
        def rrf(rankings, n_docs, k=60):
            w = np.broadcast_to(1 / (k + np.arange(1, rankings.shape[1] + 1)), rankings.shape)
            score = np.zeros(n_docs)
            np.add.at(score, rankings, w)
            return score
      """,
      jax="""
        def rrf(rankings, n_docs, k=60):
            w = jnp.broadcast_to(1 / (k + jnp.arange(1, rankings.shape[1] + 1)), rankings.shape)
            return jnp.zeros(n_docs).at[rankings].add(w)
      """,
      torch="""
        def rrf(rankings, n_docs, k=60):
            w = (1 / (k + torch.arange(1, rankings.shape[1] + 1))).expand(rankings.shape)
            return torch.zeros(n_docs).index_add(0, rankings.flatten(), w.flatten())
      """,
      notes=(("Ranks are 1-based in the paper. `enumerate` starts at 0, so pass `start=1`; off by one, the top "
             "document gets 1/60 instead of 1/61."),
             ("A document missing from a list just gets nothing from it; documents in no list score 0. Sort by "
             "score (descending) to get the fused ranking.")),
      module="38"),
    E("MRR and nDCG@k", "Retrieval and evaluation",
      "Two ways to score a ranking when each query has one right answer: MRR averages 1/rank; nDCG discounts "
      "by log₂(rank + 1), so rank 2 still earns 63%. Both give 0 if the answer isn't in the top k.",
      r"\mathrm{MRR@}k = \frac1Q \sum_q \frac{[\mathrm{rank}_q \le k]}{\mathrm{rank}_q};\quad "
      r"\mathrm{nDCG@}k = \frac1Q \sum_q \frac{[\mathrm{rank}_q \le k]}{\log_2(\mathrm{rank}_q + 1)}",
      r"$\mathrm{DCG}_k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i + 1)}$, nDCG = DCG / ideal DCG (Järvelin & "
      r"Kekäläinen, 2002)", "ranked (Q, n) doc ids, relevant (Q,) doc id, k -> mrr, ndcg",
      "rank_metrics(ranked, relevant, k)", _rank_inputs,
      python="""
        def rank_metrics(ranked, relevant, k):
            rr, gain = [], []
            for row, rel in zip(ranked, relevant):
                rank = row.index(rel) + 1 if rel in row else 0  # 1-based; 0 = not retrieved
                hit = 0 < rank <= k
                rr.append(1 / rank if hit else 0.0)
                gain.append(1 / math.log2(rank + 1) if hit else 0.0)
            return {"mrr": sum(rr) / len(rr), "ndcg": sum(gain) / len(gain)}
      """,
      numpy="""
        def rank_metrics(ranked, relevant, k):
            hit = ranked == relevant[:, None]
            rank = np.where(hit.any(1), hit.argmax(1) + 1, 0)
            ok = (rank > 0) & (rank <= k)
            safe = np.maximum(rank, 1)
            return {"mrr": np.where(ok, 1 / safe, 0).mean(), "ndcg": np.where(ok, 1 / np.log2(safe + 1), 0).mean()}
      """,
      jax="""
        def rank_metrics(ranked, relevant, k):
            hit = ranked == relevant[:, None]
            rank = jnp.where(hit.any(1), hit.argmax(1) + 1, 0)
            ok = (rank > 0) & (rank <= k)
            safe = jnp.maximum(rank, 1)
            return {"mrr": jnp.where(ok, 1 / safe, 0).mean(),
                    "ndcg": jnp.where(ok, 1 / jnp.log2(safe + 1), 0).mean()}
      """,
      torch="""
        def rank_metrics(ranked, relevant, k):
            hit = ranked == relevant[:, None]
            rank = torch.where(hit.any(1), hit.int().argmax(1) + 1, 0)
            ok = (rank > 0) & (rank <= k)
            safe = rank.clamp(min=1).float()
            return {"mrr": torch.where(ok, 1 / safe, 0.0).mean(),
                    "ndcg": torch.where(ok, 1 / torch.log2(safe + 1), 0.0).mean()}
      """,
      notes=("With one relevant document the ideal DCG is 1, so nDCG is just the discounted gain.",
             "Use 1-based ranks: a 0-based rank makes the top hit $1/\\log_2 1 = 1/0$.",
             ("`where` evaluates both branches, so divide by a safe rank (`maximum(rank, 1)`) even for misses, or "
             "you get inf/NaN warnings (and NaN gradients in JAX)."),
             "NumPy and JAX take `argmax` of a bool array; `torch.argmax` raises on bool, so cast to int first."),
      module="38", also=("21",)),
]
