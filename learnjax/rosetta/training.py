"""Rosetta entries for training: optimizers and schedules, activations, losses, normalization layers, dropout."""
import numpy as np

from learnjax.rosetta.entry import Entry as E


def _logits_labels(r: np.random.Generator) -> dict:
    return {"z": 3 * r.normal(size=(4, 3)), "y": (r.random((4, 3)) < 0.5).astype(np.float64)}


ENTRIES: list[E] = [
    # ---------------------------------------------------------------- Optimization (Module 05)
    E("SGD with momentum", "Optimization",
      "Keep a running sum of past gradients and step along it: a heavy ball that keeps rolling through small "
      "bumps and builds speed down long, gentle slopes.",
      r"v_t = \mu\, v_{t-1} + g_t,\quad \theta_t = \theta_{t-1} - \eta\, v_t",
      r"$v_t = \beta v_{t-1} + (1-\beta) g_t$ (an average instead of a sum: same direction, step $1/(1-\beta)$ "
      r"times smaller)",
      "params (n,), gradients (steps, n) -> params (n,)", "sgd_momentum(p, grads)",
      lambda r: {"p": r.normal(size=4), "grads": r.normal(size=(3, 4))},
      python="""
        def sgd_momentum(p, grads, lr=0.1, mu=0.9):
            p, v = list(p), [0.0] * len(p)
            for g in grads:
                for i in range(len(p)):
                    v[i] = mu * v[i] + g[i]
                    p[i] -= lr * v[i]
            return p
      """,
      numpy="""
        def sgd_momentum(p, grads, lr=0.1, mu=0.9):
            v = np.zeros_like(p)
            for g in grads:
                v = mu * v + g
                p = p - lr * v
            return p
      """,
      jax="""
        def sgd_momentum(p, grads, lr=0.1, mu=0.9):
            opt = optax.sgd(lr, momentum=mu)
            state = opt.init(p)
            for g in grads:
                updates, state = opt.update(g, state, p)
                p = optax.apply_updates(p, updates)
            return p
      """,
      torch="""
        def sgd_momentum(p, grads, lr=0.1, mu=0.9):
            p = p.clone().requires_grad_(True)
            opt = torch.optim.SGD([p], lr=lr, momentum=mu)  # dampening=0, nesterov=False
            for g in grads:
                p.grad = g.clone()
                opt.step()
            return p.detach()
      """,
      notes=(("optax and PyTorch both keep the *sum* form $v = \\mu v + g$ (no $1-\\mu$ factor), so the same lr "
             "and momentum give the same steps. Papers that write the averaged form need a larger lr."),
             ("PyTorch's `dampening` (default 0) would scale the new gradient by $1-$dampening; optax has no "
             "such knob."),
             ("`optax.sgd` without `momentum=` is plain gradient descent; so is `torch.optim.SGD` with its "
             "default `momentum=0`.")),
      apis=("optax.sgd",),
      module="05", also=("10",)),
    E("RMSProp", "Optimization",
      "Divide each parameter's step by a running average of its recent gradient size, so steep directions "
      "take small steps and flat ones take big steps.",
      r"s_t = \rho\, s_{t-1} + (1-\rho)\, g_t^2,\quad \theta_t = \theta_{t-1} - \eta\, \frac{g_t}{\sqrt{s_t} + "
      r"\epsilon}", r"$E[g^2]_t$ for the running mean square; $\gamma$ or $\alpha$ for the decay",
      "params (n,), gradients (steps, n) -> params (n,)", "rmsprop(p, grads)",
      lambda r: {"p": r.normal(size=4), "grads": r.normal(size=(3, 4))},
      python="""
        def rmsprop(p, grads, lr=0.01, rho=0.99, eps=1e-8):
            p, s = list(p), [0.0] * len(p)
            for g in grads:
                for i in range(len(p)):
                    s[i] = rho * s[i] + (1 - rho) * g[i] ** 2
                    p[i] -= lr * g[i] / (math.sqrt(s[i]) + eps)
            return p
      """,
      numpy="""
        def rmsprop(p, grads, lr=0.01, rho=0.99, eps=1e-8):
            s = np.zeros_like(p)
            for g in grads:
                s = rho * s + (1 - rho) * g ** 2
                p = p - lr * g / (np.sqrt(s) + eps)
            return p
      """,
      jax="""
        def rmsprop(p, grads, lr=0.01, rho=0.99, eps=1e-8):
            opt = optax.rmsprop(lr, decay=rho, eps=eps, eps_in_sqrt=False)
            state = opt.init(p)
            for g in grads:
                updates, state = opt.update(g, state, p)
                p = optax.apply_updates(p, updates)
            return p
      """,
      torch="""
        def rmsprop(p, grads, lr=0.01, rho=0.99, eps=1e-8):
            p = p.clone().requires_grad_(True)
            opt = torch.optim.RMSprop([p], lr=lr, alpha=rho, eps=eps)
            for g in grads:
                p.grad = g.clone()
                opt.step()
            return p.detach()
      """,
      notes=(("Defaults differ: optax `decay=0.9` with eps *inside* the square root ($\\sqrt{s+\\epsilon}$, "
             "`eps_in_sqrt=True`); PyTorch `alpha=0.99` with eps outside ($\\sqrt{s}+\\epsilon$). The snippets "
             "set both explicitly so all versions agree."),
             ("No bias correction: the first steps are larger than later ones because $s$ starts at 0. Adam "
             "adds the correction (and momentum).")),
      apis=("optax.rmsprop",),
      module="05"),
    E("AdamW (decoupled weight decay)", "Optimization",
      "Adam, plus shrink every weight a little toward zero each step — applied directly to the weights, not "
      "mixed into the gradient where Adam's scaling would distort it.",
      r"\theta_t = \theta_{t-1} - \eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\, "
      r"\theta_{t-1}\Big)",
      r"Loshchilov & Hutter write the decay with its own schedule multiplier, $\eta_t \lambda \theta$",
      "params (n,), gradients (steps, n) -> params (n,)", "adamw(p, grads)",
      lambda r: {"p": r.normal(size=4), "grads": r.normal(size=(3, 4))},
      python="""
        def adamw(p, grads, lr=0.1, wd=0.01, b1=0.9, b2=0.999, eps=1e-8):
            p, m, v = list(p), [0.0] * len(p), [0.0] * len(p)
            for t, g in enumerate(grads, start=1):
                for i in range(len(p)):
                    m[i] = b1 * m[i] + (1 - b1) * g[i]
                    v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
                    m_hat, v_hat = m[i] / (1 - b1 ** t), v[i] / (1 - b2 ** t)
                    p[i] -= lr * (m_hat / (math.sqrt(v_hat) + eps) + wd * p[i])
            return p
      """,
      numpy="""
        def adamw(p, grads, lr=0.1, wd=0.01, b1=0.9, b2=0.999, eps=1e-8):
            m, v = np.zeros_like(p), np.zeros_like(p)
            for t, g in enumerate(grads, start=1):
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g ** 2
                adam = (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
                p = p - lr * (adam + wd * p)
            return p
      """,
      jax="""
        def adamw(p, grads, lr=0.1, wd=0.01):
            opt = optax.adamw(lr, weight_decay=wd)
            state = opt.init(p)
            for g in grads:
                updates, state = opt.update(g, state, p)  # adamw needs the params, for the decay term
                p = optax.apply_updates(p, updates)
            return p
      """,
      torch="""
        def adamw(p, grads, lr=0.1, wd=0.01):
            p = p.clone().requires_grad_(True)
            opt = torch.optim.AdamW([p], lr=lr, weight_decay=wd)
            for g in grads:
                p.grad = g.clone()
                opt.step()
            return p.detach()
      """,
      notes=(("Default weight decay differs 100×: `optax.adamw` uses 1e-4, `torch.optim.AdamW` 0.01. Set it "
             "explicitly when porting a recipe."),
             ("Both multiply the decay by the learning rate ($\\eta\\lambda\\theta$), so a schedule shrinks "
             "the decay too."),
             ("Decay usually skips biases and norm gains: `optax.adamw(..., mask=...)` in JAX; separate "
             "parameter groups with `weight_decay=0` in PyTorch (Module 05 §8, Module 11)."),
             ("\"Adam + L2\" (adding $\\lambda\\theta$ to the gradient) is *not* AdamW: Adam then divides the "
             "decay by $\\sqrt{\\hat v}$.")),
      apis=("optax.adamw",),
      module="05", also=("11", "16", "25")),
    E("learning-rate schedules (warmup, cosine, linear)", "Optimization",
      "Change the step size over training: ramp it up from small (warmup), then let it glide down along half a "
      "cosine wave, or along a straight line.",
      r"\eta_t = \begin{cases} \eta_0 + (\eta_{\max} - \eta_0)\,\frac{t}{T_w} & t < T_w \\ \eta_\text{end} + "
      r"\tfrac12(\eta_{\max} - \eta_\text{end})\big(1 + \cos\frac{\pi (t - T_w)}{T - T_w}\big) & T_w \le t "
      r"\le T \end{cases}",
      r"SGDR's $\eta_t = \eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min})(1 + \cos(\pi T_\text{cur}/T_i))$",
      "steps -> one learning rate per step (two schedules)", "schedules(steps)", lambda r: {"steps": 12},
      python="""
        def schedules(steps, init=0.1, peak=1.0, warmup=3, total=10, end=0.05):
            cosine, linear = [], []
            for t in range(steps):
                if t < warmup:
                    lr = init + (peak - init) * t / warmup
                else:
                    frac = min(t - warmup, total - warmup) / (total - warmup)
                    lr = end + 0.5 * (peak - end) * (1 + math.cos(math.pi * frac))
                cosine.append(lr)
                linear.append(peak + (end - peak) * min(t, total) / total)
            return {"warmup_cosine": cosine, "linear": linear}
      """,
      numpy="""
        def schedules(steps, init=0.1, peak=1.0, warmup=3, total=10, end=0.05):
            t = np.arange(steps)
            frac = np.clip((t - warmup) / (total - warmup), 0, 1)
            cosine = np.where(t < warmup, init + (peak - init) * t / warmup,
                              end + 0.5 * (peak - end) * (1 + np.cos(np.pi * frac)))
            linear = peak + (end - peak) * np.minimum(t, total) / total
            return {"warmup_cosine": cosine, "linear": linear}
      """,
      jax="""
        def schedules(steps, init=0.1, peak=1.0, warmup=3, total=10, end=0.05):
            cosine = optax.warmup_cosine_decay_schedule(init, peak, warmup_steps=warmup,
                                                        decay_steps=total, end_value=end)
            linear = optax.linear_schedule(peak, end, transition_steps=total)
            t = jnp.arange(steps)
            return {"warmup_cosine": jax.vmap(cosine)(t), "linear": jax.vmap(linear)(t)}
      """,
      torch="""
        def schedules(steps, init=0.1, peak=1.0, warmup=3, total=10, end=0.05):
            def run(make):
                opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=peak)
                sched, lrs = make(opt), []
                for t in range(steps):
                    lrs.append(opt.param_groups[0]["lr"])
                    opt.step()
                    if t < total:  # past T_max, CosineAnnealingLR would climb back up
                        sched.step()
                return torch.tensor(lrs)
            sl = torch.optim.lr_scheduler
            cosine = run(lambda opt: sl.SequentialLR(opt, [
                sl.LinearLR(opt, start_factor=init / peak, end_factor=1.0, total_iters=warmup),
                sl.CosineAnnealingLR(opt, T_max=total - warmup, eta_min=end)], milestones=[warmup]))
            linear = run(lambda opt: sl.LinearLR(opt, start_factor=1.0, end_factor=end / peak,
                                                 total_iters=total))
            return {"warmup_cosine": cosine, "linear": linear}
      """,
      notes=(("An optax schedule is a pure function of the step number: pass it as the learning rate "
             "(`optax.adamw(schedule)`) and the optimizer state counts steps. A PyTorch scheduler mutates the "
             "optimizer's `lr`, and you call `sched.step()` once per step, after `opt.step()`."),
             ("`decay_steps` in `optax.warmup_cosine_decay_schedule` is the *total* length, warmup included; "
             "PyTorch's `CosineAnnealingLR(T_max=...)` counts only the cosine part."),
             ("PyTorch's `LinearLR` works in *factors* of the base lr and requires `start_factor > 0`: warming "
             "up from exactly 0 needs `LambdaLR` instead."),
             ("After the schedule ends, optax holds the end value; PyTorch's `CosineAnnealingLR` is periodic and "
             "climbs back up (here 0.05 → 0.097 at step 11) unless you stop calling `sched.step()`, as the "
             "PyTorch tab does."),
             "Some papers count warmup as 1 to $T_w$ instead of 0 to $T_w - 1$; check which step gets the peak."),
      apis=("optax.warmup_cosine_decay_schedule", "optax.linear_schedule"),
      module="05", also=("11", "16", "34")),
    E("Muon: orthogonalized update (Newton–Schulz)", "Optimization",
      "Keep the gradient matrix's directions but give every direction an equal-sized step: replace "
      "$G = USV^\\top$ by roughly $UV^\\top$, computed with five rounds of matrix multiplies instead of an SVD.",
      r"X_0 = \frac{G}{\lVert G\rVert_F},\quad X_{k+1} = a X_k + (b A_k + c A_k^2) X_k,\; A_k = X_k X_k^\top,"
      r"\quad \Delta\theta = -\eta \sqrt{\max(1, d_\text{out}/d_\text{in})}\; X_5",
      r"$\mathrm{msign}(G)$ or $\mathrm{Ortho}(G)$ for the polar factor; $(a, b, c) = (3.4445, -4.7750, 2.0315)$",
      "gradient (d_in, d_out) -> first update (d_in, d_out)", "muon_step(G)",
      lambda r: {"G": r.normal(size=(3, 5))},
      python="""
        def muon_step(G, lr=0.1, steps=5, a=3.4445, b=-4.7750, c=2.0315, eps=1e-8):
            def matmul(A, B):
                return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
                        for i in range(len(A))]
            def T(A):
                return [list(col) for col in zip(*A)]
            d_in, d_out = len(G), len(G[0])
            size = math.sqrt(sum(v * v for row in G for v in row)) + eps
            X = [[v / size for v in row] for row in G]
            if d_in > d_out:  # iterate on the wide orientation: X X^T is then the small Gram matrix
                X = T(X)
            for _ in range(steps):
                A = matmul(X, T(X))
                AA = matmul(A, A)
                B = [[b * A[i][j] + c * AA[i][j] for j in range(len(A))] for i in range(len(A))]
                BX = matmul(B, X)
                X = [[a * X[i][j] + BX[i][j] for j in range(len(X[0]))] for i in range(len(X))]
            if d_in > d_out:
                X = T(X)
            scale = -lr * math.sqrt(max(1, d_out / d_in))
            return [[scale * v for v in row] for row in X]
      """,
      numpy="""
        def muon_step(G, lr=0.1, steps=5, a=3.4445, b=-4.7750, c=2.0315, eps=1e-8):
            X = G / (np.linalg.norm(G) + eps)
            wide = X.shape[0] <= X.shape[1]
            X = X if wide else X.T
            for _ in range(steps):
                A = X @ X.T
                X = a * X + (b * A + c * A @ A) @ X
            X = X if wide else X.T
            d_in, d_out = G.shape
            return -lr * np.sqrt(max(1, d_out / d_in)) * X
      """,
      jax="""
        def muon_step(G, lr=0.1):  # the first update from a fresh state
            opt = optax.contrib.muon(lr, nesterov=False)
            updates, _ = opt.update(G, opt.init(G), G)
            return updates
      """,
      jax_by_hand="""
        def muon_step(G, lr=0.1, steps=5, a=3.4445, b=-4.7750, c=2.0315, eps=1e-8):
            X = G / (jnp.linalg.norm(G) + eps)
            wide = X.shape[0] <= X.shape[1]
            X = X if wide else X.T
            for _ in range(steps):
                A = X @ X.T
                X = a * X + (b * A + c * A @ A) @ X
            X = X if wide else X.T
            d_in, d_out = G.shape
            return -lr * jnp.sqrt(max(1, d_out / d_in)) * X
      """,
      torch="""
        def muon_step(G, lr=0.1, steps=5, a=3.4445, b=-4.7750, c=2.0315, eps=1e-8):
            X = G / (torch.linalg.matrix_norm(G) + eps)
            wide = X.shape[0] <= X.shape[1]
            X = X if wide else X.T
            for _ in range(steps):
                A = X @ X.T
                X = a * X + (b * A + c * A @ A) @ X
            X = X if wide else X.T
            d_in, d_out = G.shape
            return -lr * (max(1, d_out / d_in) ** 0.5) * X
      """,
      notes=(("Only the *first* step is shown: momentum starts at zero, and the Frobenius normalization cancels "
             "its $(1-\\beta)$ factor, so step 1 is just the orthogonalized gradient. Later steps "
             "orthogonalize the momentum (Nesterov-style by default in both libraries)."),
             ("`torch.optim.Muon` exists (PyTorch 2.9+) but runs Newton–Schulz in bfloat16 and reads a weight as "
             "(d_out, d_in), so its shape factor uses the other axis; its numbers match this formula only to "
             "about 10%. The PyTorch tab is the float32 formula by hand."),
             ("optax treats axis 0 as the input (fan-in) axis, matching Flax's (d_in, d_out) kernels. "
             "Muon is for hidden 2-D weights only; embeddings, the output head, biases and norm gains stay "
             "on AdamW (Module 05 §11)."),
             ("Module 05's `newton_schulz` exercise builds the orthogonalization step; try it before reading "
             "these tabs.")),
      apis=("optax.contrib.muon",),
      module="05", also=("25",)),
    # ---------------------------------------------------------------- Activations
    E("ReLU and leaky ReLU", "Neural network layers",
      "Keep positive values, zero out negatives (ReLU), or let a small fraction of negatives through (leaky "
      "ReLU) so those units still get a gradient.",
      r"\mathrm{ReLU}(x) = \max(0, x),\quad \mathrm{LeakyReLU}(x) = \begin{cases} x & x \ge 0 \\ \alpha x & x "
      r"< 0 \end{cases}", r"$[x]_+$, $(x)^+$ for ReLU", "(...) -> (...) each", "relus(x)",
      lambda r: {"x": r.normal(size=(2, 5))},
      python="""
        def relus(x, slope=0.01):
            return {"relu": [[max(0.0, v) for v in row] for row in x],
                    "leaky": [[v if v >= 0 else slope * v for v in row] for row in x]}
      """,
      numpy="""
        def relus(x, slope=0.01):
            return {"relu": np.maximum(0, x), "leaky": np.where(x >= 0, x, slope * x)}
      """,
      jax="""
        def relus(x, slope=0.01):
            return {"relu": jax.nn.relu(x), "leaky": jax.nn.leaky_relu(x, negative_slope=slope)}
      """,
      torch="""
        def relus(x, slope=0.01):
            return {"relu": F.relu(x), "leaky": F.leaky_relu(x, negative_slope=slope)}
      """,
      notes=(("`flax.nnx.relu`, `nnx.leaky_relu` and `nnx.silu` are the same functions as `jax.nn.relu` and "
             "friends, re-exported; there are no activation layer objects to construct, unlike `nn.ReLU()` in "
             "PyTorch."), "Both libraries default the leaky slope to 0.01."),
      apis=("jax.nn.relu", "flax.nnx.relu", "flax.nnx.leaky_relu"),
      module="10", also=("05", "12", "33")),
    E("GELU", "Neural network layers",
      "A smooth ReLU: scale x by the probability that a standard normal variable is below x, so small negatives "
      "are damped rather than cut off. The default activation in GPT-2 and BERT.",
      r"\mathrm{GELU}(x) = x\,\Phi(x) = \frac{x}{2}\Big(1 + \mathrm{erf}\big(x/\sqrt{2}\big)\Big)",
      r"tanh approximation: $\frac{x}{2}\big(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715 x^3))\big)$",
      "(...) -> (...)", "gelu(x)", lambda r: {"x": 2 * r.normal(size=(2, 5))},
      python="""
        def gelu(x):
            return [[v / 2 * (1 + math.erf(v / math.sqrt(2))) for v in row] for row in x]
      """,
      numpy="""
        from scipy.special import erf  # NumPy has no erf

        def gelu(x):
            return x / 2 * (1 + erf(x / np.sqrt(2)))
      """,
      jax="""
        def gelu(x):
            return jax.nn.gelu(x, approximate=False)  # the default, approximate=True, is the tanh formula
      """,
      jax_by_hand="""
        def gelu(x):
            return x / 2 * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))
      """,
      torch="""
        def gelu(x):
            return F.gelu(x)  # exact by default; approximate="tanh" for the GPT-2 formula
      """,
      notes=(("Defaults are opposite: `jax.nn.gelu` uses the tanh approximation unless `approximate=False`; "
             "PyTorch's `F.gelu` is exact unless `approximate=\"tanh\"`. The two differ by up to about 3e-4, "
             "enough to fail a strict check when porting weights."),
             "GPT-2 was trained with the tanh form; match it when loading GPT-2 weights."),
      apis=("jax.nn.gelu",),
      module="10", also=("08", "16", "32")),
    E("sigmoid, log-sigmoid and softplus", "Neural network layers",
      "Sigmoid squashes any number into (0, 1), a probability. Log-sigmoid is its log, and softplus is a "
      "smooth ReLU; both are computed in stable forms so large inputs don't overflow.",
      r"\sigma(x) = \frac{1}{1 + e^{-x}},\quad \log\sigma(x) = -\mathrm{softplus}(-x),\quad "
      r"\mathrm{softplus}(x) = \log(1 + e^{x})",
      r"$\sigma$ also names softmax in some papers; $\zeta(x)$ for softplus", "(...) -> (...) each",
      "sigmoids(x)", lambda r: {"x": np.array([-50.0, -3.0, -0.5, 0.0, 0.5, 3.0, 50.0])},
      python="""
        def sigmoids(x):
            def softplus(v):  # log(1 + e^v) without overflow: max(v, 0) + log(1 + e^-|v|)
                return max(v, 0.0) + math.log1p(math.exp(-abs(v)))
            def sigmoid(v):
                return 1 / (1 + math.exp(-v)) if v >= 0 else math.exp(v) / (1 + math.exp(v))
            return {"sigmoid": [sigmoid(v) for v in x],
                    "log_sigmoid": [-softplus(-v) for v in x],
                    "softplus": [softplus(v) for v in x]}
      """,
      numpy="""
        def sigmoids(x):
            softplus = lambda v: np.maximum(v, 0) + np.log1p(np.exp(-np.abs(v)))
            return {"sigmoid": np.exp(-softplus(-x)),
                    "log_sigmoid": -softplus(-x),
                    "softplus": softplus(x)}
      """,
      jax="""
        def sigmoids(x):
            return {"sigmoid": jax.nn.sigmoid(x),
                    "log_sigmoid": jax.nn.log_sigmoid(x),
                    "softplus": jax.nn.softplus(x)}
      """,
      torch="""
        def sigmoids(x):
            return {"sigmoid": torch.sigmoid(x),
                    "log_sigmoid": F.logsigmoid(x),
                    "softplus": F.softplus(x)}
      """,
      notes=(("Never compute `log(sigmoid(x))`: for x = -50, sigmoid underflows to 0 in float32 and the log "
             "is -inf. `log_sigmoid` gives -50 (the test inputs include ±50 for this reason)."),
             ("PyTorch's `F.softplus` has `beta` and `threshold` arguments (above `threshold=20` it returns x "
             "unchanged); `jax.nn.softplus` has neither.")),
      apis=("jax.nn.sigmoid", "jax.nn.log_sigmoid", "jax.nn.softplus"),
      module="09", also=("30", "33", "35")),
    E("SiLU (swish)", "Neural network layers",
      "Multiply x by its own sigmoid: like ReLU for large |x|, but smooth and slightly negative just below "
      "zero. The gate inside SwiGLU (Module 17).",
      r"\mathrm{SiLU}(x) = x\,\sigma(x) = \frac{x}{1 + e^{-x}}", r"$\mathrm{Swish}_1(x)$; $\mathrm{Swish}_\beta"
      r"(x) = x\,\sigma(\beta x)$", "(...) -> (...)", "silu(x)", lambda r: {"x": 3 * r.normal(size=(2, 5))},
      python="""
        def silu(x):
            return [[v / (1 + math.exp(-v)) for v in row] for row in x]
      """,
      numpy="""
        def silu(x):
            return x / (1 + np.exp(-x))
      """,
      jax="""
        def silu(x):
            return jax.nn.silu(x)  # jax.nn.swish is the same function
      """,
      torch="""
        def silu(x):
            return F.silu(x)
      """,
      notes=("SwiGLU is `silu(x @ W_gate) * (x @ W_up)`: SiLU is the gate, not the whole feed-forward block.",
             "`flax.nnx.silu` is `jax.nn.silu`, re-exported."),
      apis=("jax.nn.silu", "flax.nnx.silu"),
      module="17", also=("33",)),
    # ---------------------------------------------------------------- Losses
    E("log-softmax", "Probability and information",
      "The log of softmax, computed directly: each score minus the log-sum-exp of all scores. What every "
      "cross-entropy loss uses inside.",
      r"\log \mathrm{softmax}(x)_i = x_i - \log \sum_j e^{x_j}", r"$\log p_\theta(y \mid x)$",
      "(..., n) -> (..., n), along the last axis", "log_softmax(x)", lambda r: {"x": 30 * r.normal(size=(3, 5))},
      python="""
        def log_softmax(x):
            out = []
            for row in x:
                m = max(row)
                lse = m + math.log(sum(math.exp(v - m) for v in row))
                out.append([v - lse for v in row])
            return out
      """,
      numpy="""
        def log_softmax(x):
            m = x.max(axis=-1, keepdims=True)
            return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))
      """,
      jax="""
        def log_softmax(x):
            return jax.nn.log_softmax(x, axis=-1)
      """,
      jax_by_hand="""
        def log_softmax(x):
            return x - jax.nn.logsumexp(x, axis=-1, keepdims=True)
      """,
      torch="""
        def log_softmax(x):
            return F.log_softmax(x, dim=-1)
      """,
      notes=(("`log(softmax(x))` loses precision: tiny probabilities round to 0 and the log becomes -inf. "
             "The test inputs (scale 30) make that happen in float32."),
             ("PyTorch's `F.nll_loss` expects log-softmax outputs; `F.cross_entropy` takes raw logits and "
             "applies it for you.")),
      apis=("jax.nn.log_softmax",),
      module="04", also=("14", "16", "26")),
    E("one-hot encoding", "Probability and information",
      "Turn a class index into a vector of zeros with a single 1 at that index: the target distribution for "
      "cross-entropy, and the input that an embedding lookup replaces.",
      r"\mathrm{onehot}(y)_c = [\,c = y\,] = \begin{cases} 1 & c = y \\ 0 & \text{otherwise} \end{cases}",
      r"$e_y$ (the $y$-th standard basis vector); $\mathbb{1}[c = y]$", "(N,) int, C -> (N, C)",
      "one_hot(y, C)", lambda r: {"y": np.array([0, 2, 1, 2]), "C": 3},
      python="""
        def one_hot(y, C):
            return [[1.0 if c == label else 0.0 for c in range(C)] for label in y]
      """,
      numpy="""
        def one_hot(y, C):
            return np.eye(C)[y]
      """,
      jax="""
        def one_hot(y, C):
            return jax.nn.one_hot(y, C)  # float32
      """,
      torch="""
        def one_hot(y, C):
            return F.one_hot(y, C).float()  # needs an int64 index tensor; returns int64
      """,
      notes=(("`jax.nn.one_hot` returns floats and accepts any integer dtype; `F.one_hot` needs int64 input "
             "and returns int64, so add `.float()` before multiplying by probabilities."),
             ("Out-of-range labels: JAX quietly returns an all-zero row (handy for padding with -1); PyTorch "
             "raises an error."),
             ("`one_hot(y) @ E` equals `E[y]`: an embedding lookup is a one-hot matmul without the zeros "
             "(Module 13).")),
      apis=("jax.nn.one_hot",),
      module="01", also=("11", "13", "14")),
    E("binary cross-entropy from logits", "Probability and information",
      "The loss for yes/no predictions: minus the log probability the model gave the right answer, with the "
      "probability coming from a sigmoid. Computed from logits in a form that never takes log(0).",
      r"\ell(z, y) = -\big[y \log\sigma(z) + (1-y)\log(1-\sigma(z))\big] = \max(z, 0) - z y + \log(1 + "
      r"e^{-|z|})", r"logistic loss; $\mathrm{BCE}(\hat y, y)$", "(N, K) logits, (N, K) labels in [0, 1] -> ()",
      "bce(z, y)", _logits_labels,
      python="""
        def bce(z, y):
            total, count = 0.0, 0
            for zrow, yrow in zip(z, y):
                for zi, yi in zip(zrow, yrow):
                    total += max(zi, 0.0) - zi * yi + math.log1p(math.exp(-abs(zi)))
                    count += 1
            return total / count
      """,
      numpy="""
        def bce(z, y):
            return np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))
      """,
      jax="""
        def bce(z, y):
            return optax.sigmoid_binary_cross_entropy(z, y).mean()
      """,
      jax_by_hand="""
        def bce(z, y):
            return -jnp.mean(y * jax.nn.log_sigmoid(z) + (1 - y) * jax.nn.log_sigmoid(-z))
      """,
      torch="""
        def bce(z, y):
            return F.binary_cross_entropy_with_logits(z, y)  # mean by default
      """,
      notes=(("optax returns one loss per element, so you call `.mean()`; PyTorch averages by default "
             "(`reduction=\"mean\"`)."),
             ("PyTorch's `F.binary_cross_entropy` (no `_with_logits`) takes probabilities and is the unstable "
             "one; prefer the logits version, as optax does."),
             "Uses $1 - \\sigma(z) = \\sigma(-z)$: the second term is `log_sigmoid(-z)`."),
      apis=("optax.sigmoid_binary_cross_entropy",),
      module="09", also=("33", "35")),
    # ---------------------------------------------------------------- Normalization layers and dropout
    E("LayerNorm", "Neural network layers",
      "Rescale each example's feature vector to mean 0 and variance 1, then apply a learned scale and shift. "
      "Statistics come from one example's features, so batch size doesn't matter.",
      r"\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,\quad \mu = "
      r"\frac{1}{d}\sum_i x_i,\; \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2",
      r"$g$ and $b$ for the gain and bias", "(batch, d), (d,), (d,) -> (batch, d)", "layer_norm(x, gamma, beta)",
      lambda r: {"x": 2 + 3 * r.normal(size=(3, 4)), "gamma": r.normal(size=4), "beta": r.normal(size=4)},
      python="""
        def layer_norm(x, gamma, beta, eps=1e-5):
            out = []
            for row in x:
                mu = sum(row) / len(row)
                var = sum((v - mu) ** 2 for v in row) / len(row)  # biased: divide by d, not d - 1
                out.append([g * (v - mu) / math.sqrt(var + eps) + b for v, g, b in zip(row, gamma, beta)])
            return out
      """,
      numpy="""
        def layer_norm(x, gamma, beta, eps=1e-5):
            mu = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)
            return gamma * (x - mu) / np.sqrt(var + eps) + beta
      """,
      jax="""
        from flax import nnx

        def layer_norm(x, gamma, beta, eps=1e-5):
            ln = nnx.LayerNorm(x.shape[-1], epsilon=eps, rngs=nnx.Rngs(0))
            ln.scale[...] = gamma  # copy the test weights into the layer
            ln.bias[...] = beta
            return ln(x)
      """,
      jax_by_hand="""
        def layer_norm(x, gamma, beta, eps=1e-5):
            mu = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)
            return gamma * (x - mu) / jnp.sqrt(var + eps) + beta
      """,
      torch="""
        def layer_norm(x, gamma, beta, eps=1e-5):
            return F.layer_norm(x, (x.shape[-1],), gamma, beta, eps)
      """,
      notes=(("Default epsilon differs: Flax `nnx.LayerNorm` uses 1e-6, PyTorch `nn.LayerNorm` 1e-5. The "
             "snippets set 1e-5."),
             ("Both use the biased variance (divide by d). Flax's layer names the gain `scale`; PyTorch calls "
             "it `weight`."),
             ("Normalizes over the last axis only. PyTorch's `normalized_shape` can cover several trailing "
             "axes; in Flax use `reduction_axes`.")),
      apis=("flax.nnx.LayerNorm",),
      module="11", also=("15", "16", "32")),
    E("RMSNorm", "Neural network layers",
      "LayerNorm without the mean: divide each vector by its root-mean-square size and apply a learned scale. "
      "Cheaper, and what Llama-style models use (Module 17).",
      r"\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}",
      r"$\bar a_i = \frac{a_i}{\mathrm{RMS}(a)} g_i$ (Zhang & Sennrich)", "(batch, d), (d,) -> (batch, d)",
      "rms_norm(x, gamma)", lambda r: {"x": 3 * r.normal(size=(3, 4)), "gamma": r.normal(size=4)},
      python="""
        def rms_norm(x, gamma, eps=1e-5):
            out = []
            for row in x:
                rms = math.sqrt(sum(v * v for v in row) / len(row) + eps)
                out.append([g * v / rms for v, g in zip(row, gamma)])
            return out
      """,
      numpy="""
        def rms_norm(x, gamma, eps=1e-5):
            return gamma * x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
      """,
      jax="""
        from flax import nnx

        def rms_norm(x, gamma, eps=1e-5):
            norm = nnx.RMSNorm(x.shape[-1], epsilon=eps, rngs=nnx.Rngs(0))
            norm.scale[...] = gamma
            return norm(x)
      """,
      jax_by_hand="""
        def rms_norm(x, gamma, eps=1e-5):
            return gamma * x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
      """,
      torch="""
        def rms_norm(x, gamma, eps=1e-5):
            return F.rms_norm(x, (x.shape[-1],), gamma, eps)
      """,
      notes=("`jax.lax.rsqrt(v)` is $1/\\sqrt{v}$ as one fused operation; the course's Llama code uses it.",
             ("Default epsilon: Flax 1e-6; PyTorch's `nn.RMSNorm` uses the dtype's machine epsilon when `eps` is "
             "left as None (about 1.2e-7 for float32). Hugging Face Llama checkpoints specify their own "
             "(often 1e-5 or 1e-6) in `rms_norm_eps`: copy it."),
             "Some implementations (Gemma) scale by $1 + \\gamma$ instead of $\\gamma$: check before porting."),
      apis=("flax.nnx.RMSNorm", "jax.lax.rsqrt"),
      module="11", also=("17", "27")),
    E("BatchNorm (training mode)", "Neural network layers",
      "Normalize each feature using the mean and variance across the batch, then scale and shift; also keep "
      "running averages of those statistics to use at inference time.",
      r"y = \gamma \odot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta,\quad \hat\mu \leftarrow (1 - "
      r"m)\,\hat\mu + m\, \mu_B",
      r"$\mathrm{BN}_{\gamma,\beta}(x_i)$; the running-average weight is called momentum with opposite meanings",
      "(batch, C), (C,), (C,) -> y (batch, C), running mean (C,)", "batch_norm(x, gamma, beta)",
      lambda r: {"x": 1 + 2 * r.normal(size=(8, 3)), "gamma": r.normal(size=3), "beta": r.normal(size=3)},
      python="""
        def batch_norm(x, gamma, beta, eps=1e-5, m=0.1):  # m: weight of the new batch in the running mean
            n, C = len(x), len(x[0])
            mu = [sum(x[i][c] for i in range(n)) / n for c in range(C)]
            var = [sum((x[i][c] - mu[c]) ** 2 for i in range(n)) / n for c in range(C)]
            y = [[gamma[c] * (x[i][c] - mu[c]) / math.sqrt(var[c] + eps) + beta[c] for c in range(C)]
                 for i in range(n)]
            running_mean = [(1 - m) * 0.0 + m * mu[c] for c in range(C)]  # starts at 0
            return {"y": y, "running_mean": running_mean}
      """,
      numpy="""
        def batch_norm(x, gamma, beta, eps=1e-5, m=0.1):
            mu, var = x.mean(axis=0), x.var(axis=0)
            y = gamma * (x - mu) / np.sqrt(var + eps) + beta
            return {"y": y, "running_mean": m * mu}  # (1 - m) * 0 + m * mu
      """,
      jax="""
        from flax import nnx

        def batch_norm(x, gamma, beta, eps=1e-5, m=0.1):
            bn = nnx.BatchNorm(x.shape[-1], momentum=1 - m, epsilon=eps, rngs=nnx.Rngs(0))
            bn.scale[...] = gamma
            bn.bias[...] = beta
            y = bn(x)  # training mode: batch statistics, and the running averages update
            return {"y": y, "running_mean": bn.mean[...]}
      """,
      torch="""
        def batch_norm(x, gamma, beta, eps=1e-5, m=0.1):
            bn = torch.nn.BatchNorm1d(x.shape[-1], eps=eps, momentum=m)  # train mode by default
            with torch.no_grad():
                bn.weight.copy_(gamma)
                bn.bias.copy_(beta)
            y = bn(x)
            return {"y": y, "running_mean": bn.running_mean}
      """,
      notes=(("\"Momentum\" means opposite things: Flax `momentum=0.99` is the weight on the *old* running "
             "average; PyTorch `momentum=0.1` is the weight on the *new* batch. Flax 0.99 ≈ PyTorch 0.01, "
             "not 0.1."),
             ("The running *variance* differs too: Flax averages the biased batch variance, PyTorch the "
             "unbiased one (divide by n - 1). Normalization itself uses the biased variance in both."),
             "Flax is channels-last (axis -1); PyTorch's `BatchNorm2d` expects (N, C, H, W).",
             ("Inference uses the running averages: `use_running_average=True` (or `model.eval()`) in Flax, "
             "`model.eval()` in PyTorch.")),
      apis=("flax.nnx.BatchNorm",),
      module="11", also=("12",)),
    E("GroupNorm", "Neural network layers",
      "Split the channels into groups and normalize each group over its channels and all spatial positions, "
      "per example. Like LayerNorm for images, and independent of batch size.",
      r"y_{n,h,w,c} = \gamma_c\, \frac{x_{n,h,w,c} - \mu_{n,g}}{\sqrt{\sigma^2_{n,g} + \epsilon}} + \beta_c,\quad "
      r"\mu_{n,g}, \sigma^2_{n,g} \text{ over } (h, w, c \in g)", r"$\mathrm{GN}$; LayerNorm is $G = 1$, "
      r"InstanceNorm is $G = C$", "(N, H, W, C) channels-last, G groups -> (N, H, W, C)",
      "group_norm(x, gamma, beta, G)",
      lambda r: {"x": r.normal(size=(2, 3, 3, 6)), "gamma": r.normal(size=6), "beta": r.normal(size=6), "G": 3},
      python="""
        def group_norm(x, gamma, beta, G, eps=1e-5):
            N, H, W, C = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
            size = C // G
            out = [[[[0.0] * C for _ in range(W)] for _ in range(H)] for _ in range(N)]
            for n in range(N):
                for g in range(G):
                    chans = range(g * size, (g + 1) * size)
                    vals = [x[n][h][w][c] for h in range(H) for w in range(W) for c in chans]
                    mu = sum(vals) / len(vals)
                    var = sum((v - mu) ** 2 for v in vals) / len(vals)
                    for h in range(H):
                        for w in range(W):
                            for c in chans:
                                z = (x[n][h][w][c] - mu) / math.sqrt(var + eps)
                                out[n][h][w][c] = gamma[c] * z + beta[c]
            return out
      """,
      numpy="""
        def group_norm(x, gamma, beta, G, eps=1e-5):
            N, H, W, C = x.shape
            xg = x.reshape(N, H, W, G, C // G)
            mu = xg.mean(axis=(1, 2, 4), keepdims=True)
            var = xg.var(axis=(1, 2, 4), keepdims=True)
            return gamma * ((xg - mu) / np.sqrt(var + eps)).reshape(N, H, W, C) + beta
      """,
      jax="""
        from flax import nnx

        def group_norm(x, gamma, beta, G, eps=1e-5):
            gn = nnx.GroupNorm(x.shape[-1], num_groups=G, epsilon=eps, rngs=nnx.Rngs(0))
            gn.scale[...] = gamma
            gn.bias[...] = beta
            return gn(x)
      """,
      torch="""
        def group_norm(x, gamma, beta, G, eps=1e-5):
            x = x.permute(0, 3, 1, 2)  # PyTorch wants channels first: (N, C, H, W)
            return F.group_norm(x, G, gamma, beta, eps).permute(0, 2, 3, 1)
      """,
      notes=(("Layout trap: Flax normalizes channels-last (N, H, W, C); PyTorch's `F.group_norm` and "
             "`nn.GroupNorm` expect channels-first (N, C, H, W). Permute when porting."),
             ("Defaults: Flax `num_groups=32` and epsilon 1e-6; PyTorch has no default group count and uses "
             "epsilon 1e-5. The snippets pass both.")),
      apis=("flax.nnx.GroupNorm",),
      module="33"),
    E("dropout (inverted)", "Neural network layers",
      "During training, zero each activation with probability p and scale the survivors by 1/(1 - p), so the "
      "expected value is unchanged and inference needs no correction.",
      r"y_i = \frac{m_i\, x_i}{1 - p},\quad m_i \sim \mathrm{Bernoulli}(1 - p)",
      r"$\tilde{y} = r \ast y$ with $r_j \sim \mathrm{Bernoulli}(p)$, where Srivastava et al.'s $p$ is the "
      r"*keep* probability", "(batch, d), keep-mask (batch, d) -> (batch, d)", "dropout(x, keep, p)",
      lambda r: {"x": r.normal(size=(3, 5)), "keep": (r.random((3, 5)) >= 0.25).astype(np.float64), "p": 0.25},
      python="""
        def dropout(x, keep, p):
            return [[v * k / (1 - p) for v, k in zip(xr, kr)] for xr, kr in zip(x, keep)]
      """,
      numpy="""
        def dropout(x, keep, p):
            return x * keep / (1 - p)
      """,
      jax="""
        def dropout(x, keep, p):  # what nnx.Dropout(rate=p) computes, with the mask passed in
            return jnp.where(keep > 0, x / (1 - p), 0.0)
      """,
      torch="""
        def dropout(x, keep, p):  # what F.dropout(x, p) computes, with the mask passed in
            return x * keep / (1 - p)
      """,
      notes=(("The library layers draw the mask themselves: `nnx.Dropout(rate=p, rngs=nnx.Rngs(dropout=key))` "
             "and `F.dropout(x, p, training=True)`. Two libraries never draw the same mask, so the tabs take "
             "it as an input to show the arithmetic."),
             "Both libraries' p is the *drop* rate; the original paper's p is the *keep* probability.",
             ("At inference dropout is the identity: `model.eval()` (or `deterministic=True`) in Flax, "
             "`model.eval()` in PyTorch. Forgetting it makes outputs random.")),
      apis=("flax.nnx.Dropout",),
      module="11", also=("16",)),
]
