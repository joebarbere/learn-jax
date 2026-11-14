import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="04 · Probability")


@app.cell
def _():
    import math

    import marimo as mo
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    return ex, jax, jnp, lj, math, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 04 · Probability, softmax, cross-entropy and JAX random keys

    **Goal:** understand the loss function that trains every language model. A language model outputs a
    *probability distribution* over the next token; training pushes probability onto the token that
    actually came next. The measure of "how much probability did you give the truth" is
    **cross-entropy** — the `val loss 3.51` you watched fall during the deepseek-v3-jax run.

    **You will:** sample distributions and compare with their density · use JAX's explicit random keys
    (and see the bug they prevent) · watch Monte Carlo error shrink like $1/\sqrt{n}$ · turn scores into
    probabilities with softmax and a temperature · make softmax numerically safe · measure surprise with
    entropy, cross-entropy and KL divergence · translate a loss into perplexity · sample with the
    Gumbel-max trick.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Random variables and distributions

    A **random variable** is a quantity whose value comes from a random process; its **distribution**
    says how likely each value is. For a continuous variable that's a **probability density function**
    (pdf): the probability of landing in a small interval is the area under the curve there.

    Draw samples and histogram them (normalized to area 1), and the histogram approaches the pdf as the
    sample count grows. The standard normal $\mathcal{N}(0, 1)$ has pdf
    $\tfrac{1}{\sqrt{2\pi}} e^{-x^2/2}$ — it's how neural network weights are usually initialized.
    """)
    return


@app.cell
def _(mo):
    n_samples = mo.ui.slider(steps=[10, 100, 1000, 10_000, 100_000], value=1000, label="samples",
                             show_value=True)
    n_samples
    return (n_samples,)


@app.cell
def _(jax, n_samples, np, plt):
    _x = np.asarray(jax.random.normal(jax.random.key(42), (n_samples.value,)))
    _grid = np.linspace(-4, 4, 200)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.0))
    _ax.hist(_x, bins=40, range=(-4, 4), density=True, alpha=0.6, label=f"{n_samples.value:,} samples")
    _ax.plot(_grid, np.exp(-_grid**2 / 2) / np.sqrt(2 * np.pi), label="pdf of N(0, 1)")
    _ax.set_title(f"sample mean {_x.mean():+.3f} (true 0), sample std {_x.std():.3f} (true 1)", fontsize=9)
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · JAX random keys: no hidden global state

    NumPy and PyTorch keep a hidden global random state that every call advances. JAX doesn't: you pass
    a **key** explicitly, and the same key always produces the same numbers. That makes results
    reproducible and lets `jit` and `vmap` transform random code safely (a hidden state is a side effect,
    and JAX functions must be pure — Module 07).

    The price: **you** must never use a key twice for things that should be independent. To get new
    randomness, derive new keys:

    * `k1, k2 = jax.random.split(key)` — two independent keys from one.
    * `jax.random.fold_in(key, step)` — a distinct key per step number, handy in training loops.
    """)
    return


@app.cell
def _(jax, mo):
    _key = jax.random.key(0)
    _a = jax.random.normal(_key, (3,))
    _b = jax.random.normal(_key, (3,))
    _k1, _k2 = jax.random.split(_key)
    _c = jax.random.normal(_k1, (3,))
    _d = jax.random.normal(_k2, (3,))
    _e = jax.random.normal(jax.random.fold_in(_key, 7), (3,))

    def _f(v):
        return "[" + ", ".join(f"{float(t):+.3f}" for t in v) + "]"

    mo.md(f"""
    | call | 3 normal samples |
    |---|---|
    | `normal(key)` | `{_f(_a)}` |
    | `normal(key)` again — **identical** | `{_f(_b)}` |
    | `normal(k1)` after `k1, k2 = split(key)` | `{_f(_c)}` |
    | `normal(k2)` | `{_f(_d)}` |
    | `normal(fold_in(key, 7))` | `{_f(_e)}` |
    """)
    return


@app.cell
def _(jax, np, plt):
    _key = jax.random.key(1)
    _init_w = np.asarray(jax.random.normal(_key, (2000,)))
    _noise_bug = np.asarray(jax.random.normal(_key, (2000,)))
    _ka, _kb = jax.random.split(_key)
    _init_ok = np.asarray(jax.random.normal(_ka, (2000,)))
    _noise_ok = np.asarray(jax.random.normal(_kb, (2000,)))
    _fig, _axes = plt.subplots(1, 2, figsize=(7, 3.2), sharex=True, sharey=True)
    _axes[0].scatter(_init_w, _noise_bug, s=2)
    _axes[0].set_title(f"same key reused: corr = {np.corrcoef(_init_w, _noise_bug)[0, 1]:.2f}", fontsize=9)
    _axes[1].scatter(_init_ok, _noise_ok, s=2)
    _axes[1].set_title(f"split keys: corr = {np.corrcoef(_init_ok, _noise_ok)[0, 1]:+.2f}", fontsize=9)
    for _ax in _axes:
        _ax.set_xlabel("weight init")
    _axes[0].set_ylabel("'independent' noise")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The classic bug, drawn: weights initialized with a key, then "random" noise drawn with the **same**
    key. They aren't independent — they're *identical* (left). Nothing crashes; the model just trains
    worse, which makes this bug hard to spot. Split the key and the correlation disappears (right).

    ## 3 · Expectation, and how fast Monte Carlo converges

    The **expectation** $\mathbb{E}[f(X)]$ is the probability-weighted average of $f(X)$; the
    **variance** is $\mathbb{E}[(X - \mathbb{E}X)^2]$. When you can't compute an expectation exactly, you
    can estimate it by averaging samples — **Monte Carlo**. A training loss on a mini-batch is exactly
    this: a Monte Carlo estimate of the loss over all the data.

    The error of a Monte Carlo average shrinks like $\sigma/\sqrt{n}$: **4× the samples buys 2× the
    accuracy**. Below: estimating $\mathbb{E}[X^2] = 1$ for $X \sim \mathcal{N}(0,1)$, error averaged over
    50 independent trials per sample size (all trials at once on the GPU with `vmap`).
    """)
    return


@app.cell
def _(jax, jnp, np, plt):
    _ns = [10, 30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000, 300_000]
    _keys = jax.random.split(jax.random.key(3), 50)
    _errs = []
    for _n in _ns:
        _est = jax.vmap(lambda k, n=_n: jnp.mean(jax.random.normal(k, (n,)) ** 2))(_keys)
        _errs.append(float(jnp.mean(jnp.abs(_est - 1.0))))
    _fig, _ax = plt.subplots(figsize=(6.0, 3.2))
    _ax.loglog(_ns, _errs, "o-", label="mean |estimate − 1|")
    _ax.loglog(_ns, np.sqrt(2 / np.pi) * np.sqrt(2) / np.sqrt(_ns), "--", label=r"theory: $\propto 1/\sqrt{n}$")
    _ax.set_xlabel("samples n")
    _ax.set_ylabel("error")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3, which="both")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Softmax: from scores to probabilities

    A classifier or language model produces one real-valued score (a **logit**) per class. **Softmax**
    turns them into a probability distribution — positive, summing to 1, preserving the order:

    $$\text{softmax}(z)_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

    $T$ is the **temperature**. $T \to 0$ puts all the probability on the top score (greedy, boring);
    large $T$ flattens toward uniform (creative, then incoherent). It's the `--temp 0.8` you passed to
    `sample.py` in deepseek-v3-jax. Below: a model's logits for the word after *"The fog came in off
    the …"*.
    """)
    return


@app.cell
def _(mo):
    temperature = mo.ui.slider(0.1, 3.0, step=0.05, value=1.0, label="temperature T", show_value=True)
    temperature
    return (temperature,)


@app.cell
def _(jax, jnp, plt, temperature):
    words = ["bay", "river", "sea", "street", "harbor", "lake", "roof", "dame"]
    logits = jnp.array([3.1, 2.4, 2.2, 1.0, 1.9, 0.6, -0.5, -1.5])
    _p = jax.nn.softmax(logits / temperature.value)
    _h = float(-jnp.sum(_p * jnp.log(_p)))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.0))
    _ax.bar(words, _p)
    _ax.set_ylim(0, 1)
    _ax.set_title(f"T = {temperature.value:.2f}: top word p = {float(_p.max()):.2f}, entropy = {_h:.2f} nats "
                  f"(max {jnp.log(len(words)):.2f})", fontsize=9)
    _fig.tight_layout()
    _fig
    return logits, words


@app.cell
def _(jnp, mo):
    _x = jnp.array([1000.0, 1001.0, 1002.0])
    _naive = jnp.exp(_x) / jnp.sum(jnp.exp(_x))
    mo.md(f"""
    ## 5 · Making softmax numerically safe

    `exp` overflows float32 above about 88.7. Logits that large happen, and then the naive formula gives
    `exp(1000) / sum(...)` = inf / inf = **`{_naive.tolist()}`**.

    The fix uses an identity: softmax doesn't change if you subtract the same constant from every logit
    (it cancels between numerator and denominator). Subtract the **max**, and the largest exponent is
    $e^0 = 1$: no overflow, and at least one term in the sum is 1, so no division by zero.

    For the loss you want $\\log \\text{{softmax}}$, and taking the log of a probability that underflowed
    to 0 gives $-\\infty$. Compute it directly instead:
    $\\log\\text{{softmax}}(z)_i = z_i - \\log\\sum_j e^{{z_j}}$, where the **log-sum-exp** is computed as
    $m + \\log\\sum_j e^{{z_j - m}}$ with $m = \\max_j z_j$.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo):
    _x = jnp.array([1000.0, 1001.0, 1002.0])
    _s = lj.attempt(ex.stable_softmax, _x)
    _y = jnp.array([-200.0, 0.0, 200.0])
    _l = lj.attempt(ex.log_softmax, _y)
    mo.vstack([
        mo.md("**Your turn — `stable_softmax([1000, 1001, 1002])`** (expect [0.090, 0.245, 0.665]):"),
        _s.view(mo),
        mo.md("**`log_softmax([-200, 0, 200])`** (expect [−400, −200, 0] — finite, not −inf):"),
        _l.view(mo),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Entropy, cross-entropy and KL divergence

    Information theory measures **surprise**: seeing an event of probability $p$ carries $-\log p$ nats
    of surprise. Certain events: zero. Rare events: a lot.

    * **Entropy** $H(p) = -\sum_i p_i \log p_i$: the expected surprise under $p$ itself — how uncertain the
      distribution is. Uniform over $k$ outcomes: $\log k$ (maximum). One certain outcome: 0.
    * **Cross-entropy** $H(p, q) = -\sum_i p_i \log q_i$: expected surprise when reality follows $p$ but
      you *believe* $q$. Always $\ge H(p)$.
    * **KL divergence** $\mathrm{KL}(p\,\|\,q) = H(p, q) - H(p) = \sum_i p_i \log \frac{p_i}{q_i}$: the
      *extra* surprise from believing $q$. Zero only when $q = p$. Not symmetric.

    Move the slider to make the model's belief $q$ drift away from reality $p$:
    """)
    return


@app.cell
def _(mo):
    drift = mo.ui.slider(0.0, 1.0, step=0.05, value=0.3, label="how wrong is q?", show_value=True)
    drift
    return (drift,)


@app.cell
def _(drift, ex, jnp, lj, mo, np, plt):
    _p = jnp.array([0.5, 0.25, 0.15, 0.10])
    _r = jnp.array([0.05, 0.15, 0.30, 0.50])
    _q = (1 - drift.value) * _p + drift.value * _r
    _H = float(-jnp.sum(_p * jnp.log(_p)))
    _CE = float(-jnp.sum(_p * jnp.log(_q)))
    _yours_h = lj.attempt(ex.entropy, _p)
    _yours_kl = lj.attempt(ex.kl_divergence, _p, _q)
    _fig, _ax = plt.subplots(figsize=(5.5, 2.6))
    _idx = np.arange(4)
    _ax.bar(_idx - 0.2, _p, 0.4, label="p (reality)")
    _ax.bar(_idx + 0.2, _q, 0.4, label="q (model)")
    _ax.set_xticks(_idx, ["A", "B", "C", "D"])
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _yh = f"{float(_yours_h.value):.4f}" if _yours_h.ok else "*your turn: `entropy`*"
    _ykl = f"{float(_yours_kl.value):.4f}" if _yours_kl.ok else "*your turn: `kl_divergence`*"
    mo.hstack([_fig, mo.md(f"""
    | | reference | yours |
    |---|---|---|
    | $H(p)$ | {_H:.4f} | {_yh} |
    | $H(p, q)$ | {_CE:.4f} | |
    | $\\mathrm{{KL}}(p\\|q) = H(p,q) - H(p)$ | {_CE - _H:.4f} | {_ykl} |
    """)], widths=[1.2, 1])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Why training a language model *is* minimizing cross-entropy

    Predicting the next token is classification with one class per vocabulary entry. For one training
    example the "reality" $p$ puts probability 1 on the token that actually came next, so
    $H(p, q) = -\log q_{\text{true token}}$: the loss is just *minus the log-probability the model gave the
    right answer*. Average that over a batch and you have the training loss. Since $H(p)$ is fixed by the
    data, minimizing cross-entropy is the same as minimizing KL from the data to the model.

    **Perplexity** $= e^{\text{cross-entropy}}$ turns the loss into an intuitive number: the model is as
    uncertain as if it were choosing uniformly among that many tokens.
    """)
    return


@app.cell
def _(ex, jnp, lj, math, mo):
    _logits = jnp.array([[2.0, 0.5, -1.0, 0.0], [0.1, 0.2, 3.0, -2.0], [1.0, 1.0, 1.0, 1.0]])
    _labels = jnp.array([0, 1, 3])
    _ce = lj.attempt(ex.cross_entropy, _logits, _labels)
    _ppl_init = lj.attempt(ex.perplexity, math.log(8192))
    _ppl_final = lj.attempt(ex.perplexity, 3.507)
    if _ce.ok and _ppl_final.ok:
        _out = mo.md(f"""
    Your `cross_entropy` on a batch of 3 examples with 4 classes: **{float(_ce.value):.4f}** nats.

    Now the deepseek-v3-jax run you trained, with its 8,192-token vocabulary:

    | moment | loss (nats) | perplexity |
    |---|---|---|
    | untrained model ≈ uniform guess, $\\log 8192$ | {math.log(8192):.3f} | **{float(_ppl_init.value):,.0f}** |
    | step 1 of your run (main loss, from the log) | 9.088 | {math.exp(9.088):,.0f} |
    | after 12,000 steps (published val loss; your ROCm run reached 3.527) | 3.507 | **{float(_ppl_final.value):.0f}** |

    The very first loss is almost exactly $\\log 8192$: a freshly initialized model spreads probability
    nearly evenly. Training took it from "one of 8,192" to "one of about 33" — still uncertain (English
    is), but it has learned which ~33 tokens are plausible next.
    """)
    else:
        _out = mo.vstack([mo.md("**Your turn — `cross_entropy` and `perplexity`:**"), _ce.view(mo),
                          _ppl_final.view(mo)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Sampling a token: the Gumbel-max trick

    To *generate* text, draw the next token from $\text{softmax}(z)$. A neat way that needs no cumulative
    sums: add independent **Gumbel** noise $g_i = -\log(-\log u_i)$, $u_i \sim \text{Uniform}(0,1)$, to
    each logit and take the argmax. The winner is distributed exactly as $\text{softmax}(z)$ — and it
    vectorizes perfectly on a GPU. `jax.random.categorical` does exactly this.

    Your `sample_categorical` below draws 20,000 next words from the logits in section 4 (at $T = 1$); the
    bars should match the true probabilities.
    """)
    return


@app.cell
def _(ex, jax, lj, logits, mo, np, plt, words):
    _a = lj.attempt(ex.sample_categorical, jax.random.key(7), logits, 20_000)
    if _a.ok:
        _freq = np.bincount(np.asarray(_a.value), minlength=len(words)) / 20_000
        _p = np.asarray(jax.nn.softmax(logits))
        _fig, _ax = plt.subplots(figsize=(6.5, 2.8))
        _i = np.arange(len(words))
        _ax.bar(_i - 0.2, _p, 0.4, label="softmax(logits)")
        _ax.bar(_i + 0.2, _freq, 0.4, label="your samples")
        _ax.set_xticks(_i, words)
        _ax.legend(fontsize=8)
        _ax.set_title(f"max |difference| = {np.abs(_freq - _p).max():.3f}", fontsize=9)
        _fig.tight_layout()
        _out = _fig
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "The model is, on average, about as unsure as picking uniformly among 33 tokens": "a",
            "The model gets 1 in 33 next tokens right": "b",
            "The loss can't go below 3.5": "c",
        },
        label="**Check yourself:** validation cross-entropy 3.5 nats means perplexity ≈ 33. What does that say?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _feedback = {
        "a": mo.callout(mo.md("Right. Perplexity is the effective branching factor: $e^{3.5} \\approx 33$ "
                              "equally likely choices' worth of uncertainty, on average."), kind="success"),
        "b": mo.callout(mo.md("Not quite — it's not an accuracy. The model may rank the right token first "
                              "far more often; perplexity measures how much probability it spread around."),
                        kind="warn"),
        "c": mo.callout(mo.md("No — a bigger model or more training would go lower. The floor is the "
                              "entropy of language itself, which nobody knows exactly."), kind="warn"),
    }
    _feedback.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * JAX docs, [Pseudorandom numbers](https://docs.jax.dev/en/latest/random-numbers.html) — the key
      design and why it fits `jit`/`vmap`.
    * Chris Olah, [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) —
      entropy, cross-entropy and KL drawn as pictures.
    * Andrej Karpathy, [The spelled-out intro to language modeling: building
      makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — cross-entropy and sampling on a real
      character model (Module 14 builds on it).
    * Gregory Gundersen, [The Gumbel-Max Trick](https://gregorygundersen.com/blog/2019/12/10/gumbel-max-trick/)
      — why adding noise and taking the argmax samples from softmax.

    **Next:** Module 05 — optimization: using gradients (Module 03) to minimize losses like these.
    """)
    return


if __name__ == "__main__":
    app.run()
