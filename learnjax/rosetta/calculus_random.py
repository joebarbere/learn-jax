"""Rosetta entries: determinants and least squares, autodiff beyond `grad`, and random sampling."""
import numpy as np

from learnjax.rosetta.entry import Entry as E

_SAMPLES = {"n": 20000, "seed": 0}

ENTRIES: list[E] = [
    # ---------------------------------------------------------------- Linear algebra
    E("determinant", "Linear algebra",
      "How much a matrix scales volume: 2 means areas (or volumes) double, 0 means it squashes space flat, and a "
      "negative sign means it flips orientation.",
      r"\det A = \prod_i u_{ii} \cdot (-1)^{\#\text{row swaps}},\quad PA = LU",
      r"$|A|$, $\det(A)$", "(n, n) -> ()", "det(A)", lambda r: {"A": r.normal(size=(4, 4))},
      python="""
        def det(A):  # eliminate to upper-triangular; the determinant is the product of the pivots
            n = len(A)
            M = [list(row) for row in A]
            sign = 1.0
            for c in range(n):
                p = max(range(c, n), key=lambda r: abs(M[r][c]))
                if M[p][c] == 0:
                    return 0.0
                if p != c:
                    M[c], M[p] = M[p], M[c]
                    sign = -sign  # every row swap flips the sign
                for r in range(c + 1, n):
                    f = M[r][c] / M[c][c]
                    M[r] = [mr - f * mc for mr, mc in zip(M[r], M[c])]
            out = sign
            for i in range(n):
                out *= M[i][i]
            return out
      """,
      numpy="""
        def det(A):
            return np.linalg.det(A)
      """,
      jax="""
        def det(A):
            return jnp.linalg.det(A)
      """,
      torch="""
        def det(A):
            return torch.linalg.det(A)
      """,
      notes=(("For big matrices the determinant over- or underflows quickly; `slogdet` (sign and log of the "
             "absolute value) exists in all three libraries for that reason."),
             ("To check whether a system is solvable, don't test `det(A) == 0`; look at the condition number or "
             "the smallest singular value.")),
      apis=("jax.numpy.linalg.det",),
      module="02"),
    E("least squares", "Linear algebra",
      "The best-fitting line (or plane) through more points than unknowns: the w that makes Xw as close to y as "
      "possible, in squared distance.",
      r"\hat w = \arg\min_w \lVert X w - y \rVert_2^2 = (X^\top X)^{-1} X^\top y",
      r"normal equations $X^\top X w = X^\top y$; $X^{+} y$ (pseudo-inverse)", "(N, d), (N,) -> (d,)",
      "lstsq(X, y)", lambda r: {"X": r.normal(size=(8, 3)), "y": r.normal(size=8)},
      python="""
        def lstsq(X, y):  # build the normal equations, then Gaussian elimination
            d = len(X[0])
            A = [[sum(row[i] * row[j] for row in X) for j in range(d)] for i in range(d)]  # X^T X
            b = [sum(row[i] * yi for row, yi in zip(X, y)) for i in range(d)]  # X^T y
            M = [A[i] + [b[i]] for i in range(d)]
            for c in range(d):
                p = max(range(c, d), key=lambda r: abs(M[r][c]))
                M[c], M[p] = M[p], M[c]
                for r in range(c + 1, d):
                    f = M[r][c] / M[c][c]
                    M[r] = [mr - f * mc for mr, mc in zip(M[r], M[c])]
            w = [0.0] * d
            for r in reversed(range(d)):
                w[r] = (M[r][d] - sum(M[r][k] * w[k] for k in range(r + 1, d))) / M[r][r]
            return w
      """,
      numpy="""
        def lstsq(X, y):
            w, residuals, rank, singular_values = np.linalg.lstsq(X, y)
            return w
      """,
      jax="""
        def lstsq(X, y):
            w, residuals, rank, singular_values = jnp.linalg.lstsq(X, y)
            return w
      """,
      torch="""
        def lstsq(X, y):
            return torch.linalg.lstsq(X, y).solution  # a named tuple, not a 4-tuple
      """,
      notes=(("NumPy and JAX return a 4-tuple (solution, residuals, rank, singular values) and solve with the "
             "SVD; PyTorch returns a named tuple, so take `.solution`."),
             ("PyTorch's default driver is `gelsy` (QR with pivoting) on the CPU, but on a GPU the only driver is "
             "`gels`, which assumes X has full rank."),
             ("The normal equations square the condition number, so the loop version loses accuracy on nearly "
             "collinear features; the library calls don't form $X^\\top X$.")),
      apis=("jax.numpy.linalg.lstsq",),
      module="09", also=("02",)),
    # ---------------------------------------------------------------- Calculus and autodiff
    E("value and gradient together", "Calculus and autodiff",
      "Get the loss and its gradient from one pass: training needs both, and computing the gradient already "
      "computes the value on the way.",
      r"\big(L(w),\ \nabla_w L(w)\big),\quad L(w) = \frac{1}{N}\lVert X w - y \rVert^2,\quad "
      r"\nabla_w L = \frac{2}{N} X^\top (X w - y)",
      r"$\mathcal{L}$, $\nabla_\theta \mathcal{L}$", "(d,), (N, d), (N,) -> {value: (), grad: (d,)}",
      "value_and_grad(w, X, y)",
      lambda r: {"w": r.normal(size=3), "X": r.normal(size=(8, 3)), "y": r.normal(size=8)},
      python="""
        def value_and_grad(w, X, y):  # no autodiff in plain Python: the gradient formula, by hand
            N = len(y)
            err = [sum(xi * wi for xi, wi in zip(row, w)) - yi for row, yi in zip(X, y)]
            value = sum(e * e for e in err) / N
            grad = [2 / N * sum(row[j] * e for row, e in zip(X, err)) for j in range(len(w))]
            return {"value": value, "grad": grad}
      """,
      numpy="""
        def value_and_grad(w, X, y):  # NumPy has no autodiff: derive it by hand
            err = X @ w - y
            return {"value": np.mean(err ** 2), "grad": 2 / len(y) * X.T @ err}
      """,
      jax="""
        def loss(w, X, y):
            return jnp.mean((X @ w - y) ** 2)

        def value_and_grad(w, X, y):
            value, grad = jax.value_and_grad(loss)(w, X, y)  # gradient w.r.t. the first argument
            return {"value": value, "grad": grad}
      """,
      torch="""
        def value_and_grad(w, X, y):
            w = w.detach().requires_grad_(True)
            value = torch.mean((X @ w - y) ** 2)
            value.backward()
            return {"value": value.detach(), "grad": w.grad}
      """,
      notes=(("`jax.value_and_grad(f)` differentiates with respect to the first argument; pass `argnums=(0, 1)` "
             "for more. PyTorch differentiates whatever tensors have `requires_grad=True`."),
             ("With Flax NNX models, `nnx.value_and_grad(loss_fn)(model)` does the same and returns the gradient "
             "as an `nnx.State` shaped like the model's parameters (Module 10)."),
             "`has_aux=True` lets the loss also return extras (accuracy, logits) without differentiating them."),
      apis=("jax.value_and_grad", "flax.nnx.value_and_grad"),
      module="03", also=("05", "10")),
    E("Jacobian-vector product (forward mode)", "Calculus and autodiff",
      "How the outputs move when the input moves a little in direction v: the Jacobian times v, computed in "
      "one forward pass without building the Jacobian.",
      r"J v = \lim_{h \to 0} \frac{f(x + h v) - f(x)}{h},\quad f(x) = \tanh(W x)",
      r"$\partial f(x)[v]$, $Df(x)\,v$, a directional derivative", "(m, n), (n,), (n,) -> {value: (m,), jvp: (m,)}",
      "jvp(W, x, v)", lambda r: {"W": r.normal(size=(3, 4)), "x": r.normal(size=4), "v": r.normal(size=4)},
      python="""
        def jvp(W, x, v, h=1e-6):  # central difference along the direction v
            def f(x):
                return [math.tanh(sum(w * xi for w, xi in zip(row, x))) for row in W]
            up = f([xi + h * vi for xi, vi in zip(x, v)])
            down = f([xi - h * vi for xi, vi in zip(x, v)])
            return {"value": f(x), "jvp": [(a - b) / (2 * h) for a, b in zip(up, down)]}
      """,
      numpy="""
        def jvp(W, x, v):  # chain rule by hand: diag(1 - tanh^2) W v
            t = np.tanh(W @ x)
            return {"value": t, "jvp": (1 - t ** 2) * (W @ v)}
      """,
      jax="""
        def jvp(W, x, v):
            value, tangent = jax.jvp(lambda x: jnp.tanh(W @ x), (x,), (v,))
            return {"value": value, "jvp": tangent}
      """,
      torch="""
        def jvp(W, x, v):
            value, tangent = torch.func.jvp(lambda x: torch.tanh(W @ x), (x,), (v,))
            return {"value": value, "jvp": tangent}
      """,
      notes=(("Primals and tangents go in as tuples, one entry per argument: `jax.jvp(f, (x,), (v,))`. "
             "`torch.func.jvp` copies the JAX signature."),
             ("One JVP costs about one forward pass, whatever the size of J: forward mode is cheap when there are "
             "few inputs. `jax.jacfwd` is a JVP per input column.")),
      apis=("jax.jvp",),
      module="03"),
    E("vector-Jacobian product (reverse mode)", "Calculus and autodiff",
      "Push a weighting u of the outputs back to the inputs: u times the Jacobian, in one backward pass. "
      "Backpropagation is exactly this, with u = 1 for a scalar loss.",
      r"u^\top J = \nabla_x \big(u^\top f(x)\big),\quad f(x) = \tanh(W x)",
      r"$\bar{x} = J^\top \bar{y}$ (adjoints, \"bar\" notation); $\delta$ in backprop papers",
      "(m, n), (n,), (m,) -> {value: (m,), vjp: (n,)}", "vjp(W, x, u)",
      lambda r: {"W": r.normal(size=(3, 4)), "x": r.normal(size=4), "u": r.normal(size=3)},
      python="""
        def vjp(W, x, u, h=1e-6):  # gradient of the scalar u . f(x), one input at a time
            def f(x):
                return [math.tanh(sum(w * xi for w, xi in zip(row, x))) for row in W]
            def g(x):
                return sum(ui * fi for ui, fi in zip(u, f(x)))
            out = []
            for j in range(len(x)):
                up, down = list(x), list(x)
                up[j] += h
                down[j] -= h
                out.append((g(up) - g(down)) / (2 * h))
            return {"value": f(x), "vjp": out}
      """,
      numpy="""
        def vjp(W, x, u):  # chain rule by hand, backwards: W^T (u * (1 - tanh^2))
            t = np.tanh(W @ x)
            return {"value": t, "vjp": W.T @ (u * (1 - t ** 2))}
      """,
      jax="""
        def vjp(W, x, u):
            value, pullback = jax.vjp(lambda x: jnp.tanh(W @ x), x)
            (x_bar,) = pullback(u)  # one cotangent per input, as a tuple
            return {"value": value, "vjp": x_bar}
      """,
      torch="""
        def vjp(W, x, u):
            value, pullback = torch.func.vjp(lambda x: torch.tanh(W @ x), x)
            (x_bar,) = pullback(u)
            return {"value": value, "vjp": x_bar}
      """,
      notes=(("`jax.vjp` returns the value and a *function* (the pullback) you can call with many different u; "
             "`torch.func.vjp` matches it. Classic PyTorch spells it `y.backward(u)` or "
             "`torch.autograd.grad(y, x, grad_outputs=u)`."),
             ("One VJP costs a small multiple of one forward pass, whatever the number of inputs: why reverse mode "
             "trains networks with millions of parameters and one loss.")),
      apis=("jax.vjp",),
      module="03", also=("05",)),
    E("Hessian", "Calculus and autodiff",
      "Every second derivative at once: how the gradient itself changes. Its curvature decides how big a step "
      "gradient descent can take.",
      r"H_{ij} = \frac{\partial^2 f}{\partial x_i\, \partial x_j},\quad f(x) = \log \textstyle\sum_k e^{x_k}"
      r"\;\Rightarrow\; H = \mathrm{diag}(p) - p p^\top,\ p = \mathrm{softmax}(x)",
      r"$\nabla^2 f$, $\nabla\nabla^\top f$, $\mathbf{H}$", "(n,) -> (n, n)", "hessian(x)",
      lambda r: {"x": r.normal(size=4)},
      python="""
        def hessian(x, h=1e-4):  # second central differences
            def f(x):
                m = max(x)
                return m + math.log(sum(math.exp(v - m) for v in x))
            def shifted(i, di, j, dj):
                y = list(x)
                y[i] += di
                y[j] += dj
                return f(y)
            n = len(x)
            return [[(shifted(i, h, j, h) - shifted(i, h, j, -h) - shifted(i, -h, j, h) + shifted(i, -h, j, -h))
                     / (4 * h * h) for j in range(n)] for i in range(n)]
      """,
      numpy="""
        def hessian(x):  # by hand: the Hessian of log-sum-exp
            p = np.exp(x - x.max())
            p = p / p.sum()
            return np.diag(p) - np.outer(p, p)
      """,
      jax="""
        def hessian(x):
            return jax.hessian(jax.nn.logsumexp)(x)
      """,
      torch="""
        def hessian(x):
            return torch.func.hessian(lambda x: torch.logsumexp(x, dim=0))(x)
      """,
      notes=(("`jax.hessian` is `jacfwd(jacrev(f))`: reverse mode for the gradient, forward mode on top. It "
             "builds the full n x n matrix, so it is only practical for small n; for big models use "
             "Hessian-vector products (`jax.jvp` of `jax.grad`)."),
             ("`torch.func.hessian` mirrors JAX; the older `torch.autograd.functional.hessian` gives the same "
             "matrix.")),
      apis=("jax.hessian",),
      module="03", also=("05",)),
    E("stop-gradient", "Calculus and autodiff",
      "Use a value in the forward pass but treat it as a constant in the backward pass: gradients don't flow "
      "through it.",
      r"f(x) = \textstyle\sum_i x_i\, \mathrm{sg}(x_i) \;\Rightarrow\; \frac{\partial f}{\partial x_i} = "
      r"\mathrm{sg}(x_i) = x_i \quad (\text{not } 2x_i)",
      r"$\mathrm{sg}[\cdot]$ (VQ-VAE, BYOL), \"detach\", a frozen target $y$", "(n,) -> (n,)", "grad_f(x)",
      lambda r: {"x": r.normal(size=5)},
      python="""
        def grad_f(x):  # by hand: the stopped factor is a constant c, and d(x * c)/dx = c
            c = list(x)
            return c
      """,
      numpy="""
        def grad_f(x):  # NumPy has no autodiff: the derivative with the second factor held constant
            return x.copy()
      """,
      jax="""
        def f(x):
            return jnp.sum(x * jax.lax.stop_gradient(x))

        grad_f = jax.grad(f)
      """,
      torch="""
        def grad_f(x):
            x = x.detach().requires_grad_(True)
            torch.sum(x * x.detach()).backward()
            return x.grad
      """,
      notes=("PyTorch's `.detach()` is the same operation. Without it the gradient here would be `2 * x`.",
             ("Uses in the course: DQN's target network values (Module 34), the straight-through trick and "
             "auxiliary losses that must not move part of a model (Module 17).")),
      apis=("jax.lax.stop_gradient",),
      module="17", also=("34",)),
    # ---------------------------------------------------------------- Random numbers
    E("normal samples", "Random numbers",
      "Draws from the bell curve with mean 0 and standard deviation 1; scale and shift them for any other "
      "normal.",
      r"z \sim \mathcal{N}(0, 1):\quad z = \sqrt{-2 \ln u_1}\, \cos(2\pi u_2),\ u_1, u_2 \sim U(0, 1)",
      r"$\epsilon \sim \mathcal{N}(0, I)$, $x = \mu + \sigma \epsilon$", "n, seed -> (n,)", "normal(n, seed)",
      lambda r: dict(_SAMPLES),
      python="""
        def normal(n, seed):  # Box-Muller: two uniforms become two independent normals
            rng = random.Random(seed)
            out = []
            while len(out) < n:
                u1, u2 = 1.0 - rng.random(), rng.random()  # u1 in (0, 1], so log(u1) is finite
                r = math.sqrt(-2 * math.log(u1))
                out += [r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)]
            return out[:n]
      """,
      numpy="""
        def normal(n, seed):
            return np.random.default_rng(seed).normal(size=n)
      """,
      jax="""
        def normal(n, seed):
            return jax.random.normal(jax.random.key(seed), (n,))
      """,
      torch="""
        def normal(n, seed):
            return torch.randn(n, generator=torch.Generator().manual_seed(seed))
      """,
      notes=(("The same seed gives different numbers in every library; these versions agree in distribution "
             "(the test compares mean and standard deviation), never number for number."),
             ("JAX has no hidden global generator: you pass a key, and reusing a key repeats the numbers. Split "
             "it (`jax.random.split`) for each new draw (Module 04)."),
             "JAX samples float32 by default; NumPy float64."),
      apis=("jax.random.normal",), check="stats",
      module="04", also=("01", "33")),
    E("uniform samples", "Random numbers",
      "Every number in an interval equally likely: the raw material most other random draws are built from.",
      r"u \sim U(a, b):\quad p(u) = \frac{1}{b - a} \text{ for } a \le u < b",
      r"$u \sim \mathcal{U}[a, b)$", "n, a, b, seed -> (n,)", "uniform(n, -1.0, 3.0, seed)",
      lambda r: dict(_SAMPLES),
      python="""
        def uniform(n, a, b, seed):
            rng = random.Random(seed)
            return [a + (b - a) * rng.random() for _ in range(n)]
      """,
      numpy="""
        def uniform(n, a, b, seed):
            return np.random.default_rng(seed).uniform(a, b, size=n)
      """,
      jax="""
        def uniform(n, a, b, seed):
            return jax.random.uniform(jax.random.key(seed), (n,), minval=a, maxval=b)
      """,
      torch="""
        def uniform(n, a, b, seed):
            return torch.empty(n).uniform_(a, b, generator=torch.Generator().manual_seed(seed))
      """,
      notes=("All four include a and exclude b.",
             "`jax.random.uniform` defaults to [0, 1); the bounds are keyword arguments `minval` and `maxval`."),
      apis=("jax.random.uniform",), check="stats",
      module="04"),
    E("random integers", "Random numbers",
      "Whole numbers from low up to, but not including, high, each equally likely.",
      r"k \sim \mathrm{Uniform}\{\ell, \ell + 1, \dots, h - 1\}", "", "n, low, high, seed -> (n,) int",
      "randint(n, 0, 6, seed)", lambda r: dict(_SAMPLES),
      python="""
        def randint(n, low, high, seed):
            rng = random.Random(seed)
            return [rng.randrange(low, high) for _ in range(n)]  # randrange excludes high
      """,
      numpy="""
        def randint(n, low, high, seed):
            return np.random.default_rng(seed).integers(low, high, size=n)
      """,
      jax="""
        def randint(n, low, high, seed):
            return jax.random.randint(jax.random.key(seed), (n,), low, high)
      """,
      torch="""
        def randint(n, low, high, seed):
            return torch.randint(low, high, (n,), generator=torch.Generator().manual_seed(seed))
      """,
      notes=(("Trap: Python's `random.randint(a, b)` *includes* b; `randrange`, NumPy's `integers`, "
             "`jax.random.randint` and `torch.randint` all exclude the upper bound."),
             ("Argument order differs: JAX takes the shape before the bounds, PyTorch the bounds before the "
             "shape.")),
      apis=("jax.random.randint",), check="stats",
      module="05", also=("14",)),
    E("categorical sampling", "Random numbers",
      "Pick one of k classes with the probabilities a softmax gives: how a language model samples its next "
      "token.",
      r"i \sim \mathrm{Cat}\big(\mathrm{softmax}(z)\big):\quad i = \arg\max_k\,(z_k + g_k),\ "
      r"g_k = -\log(-\log u_k)",
      r"$y \sim p_\theta(\cdot \mid x)$, $x_{t+1} \sim \mathrm{softmax}(z_t / T)$", "(k,) logits, n, seed -> (k,) "
      "frequencies", "[(lambda s: [s.count(c) / len(s) for c in range(4)])(list(map(int, sample(logits, 20000, "
      "seed))))][0]",
      lambda r: {"logits": np.array([1.0, 0.0, -1.0, 0.5]), "seed": 0},
      python="""
        def sample(logits, n, seed):  # inverse CDF: walk the cumulative probabilities
            rng = random.Random(seed)
            m = max(logits)
            p = [math.exp(z - m) for z in logits]
            p = [v / sum(p) for v in p]
            out = []
            for _ in range(n):
                u, total = rng.random(), 0.0
                for i, pi in enumerate(p):
                    total += pi
                    if u < total:
                        break
                out.append(i)
            return out
      """,
      numpy="""
        def sample(logits, n, seed):
            p = np.exp(logits - logits.max())
            return np.random.default_rng(seed).choice(len(p), size=n, p=p / p.sum())
      """,
      jax="""
        def sample(logits, n, seed):
            return jax.random.categorical(jax.random.key(seed), logits, shape=(n,))  # takes logits
      """,
      jax_by_hand="""
        def sample(logits, n, seed):  # the Gumbel-max trick (Module 04 section 8)
            g = jax.random.gumbel(jax.random.key(seed), (n, logits.shape[-1]))
            return jnp.argmax(logits + g, axis=-1)
      """,
      torch="""
        def sample(logits, n, seed):
            probs = torch.softmax(logits, dim=-1)  # multinomial needs probabilities, not logits
            return torch.multinomial(probs, n, replacement=True,
                                     generator=torch.Generator().manual_seed(seed))
      """,
      notes=(("Input trap: `jax.random.categorical` takes *logits*; `torch.multinomial` and NumPy's `choice` take "
             "probabilities (negative logits make `torch.multinomial` raise an error)."),
             ("`torch.multinomial` defaults to `replacement=False`, which can't draw more samples than classes; "
             "pass `replacement=True` to sample a token stream. `torch.distributions.Categorical(logits=z)` "
             "takes logits like JAX."),
             "The test compares how often each class comes up (within 10%)."),
      apis=("jax.random.categorical",), tol=0.1,
      module="04", also=("14", "18")),
    E("Bernoulli samples", "Random numbers",
      "A biased coin: true with probability p, false otherwise. Dropout keeps each unit with such a coin.",
      r"b \sim \mathrm{Bernoulli}(p):\quad P(b = 1) = p,\ P(b = 0) = 1 - p;\quad b = [u < p]",
      r"$m \sim \mathrm{Bern}(1 - p_\text{drop})$ (a dropout mask)", "p, n, seed -> (n,) bool",
      "bernoulli(0.3, n, seed)", lambda r: dict(_SAMPLES),
      python="""
        def bernoulli(p, n, seed):
            rng = random.Random(seed)
            return [rng.random() < p for _ in range(n)]
      """,
      numpy="""
        def bernoulli(p, n, seed):
            return np.random.default_rng(seed).random(n) < p
      """,
      jax="""
        def bernoulli(p, n, seed):
            return jax.random.bernoulli(jax.random.key(seed), p, (n,))  # bool array
      """,
      torch="""
        def bernoulli(p, n, seed):
            return torch.bernoulli(torch.full((n,), p), generator=torch.Generator().manual_seed(seed))
      """,
      notes=(("`jax.random.bernoulli` returns booleans; `torch.bernoulli` returns floats (0.0 and 1.0) and takes a "
             "tensor of probabilities, one per draw."),
             ("Dropout's p is usually the probability of *dropping*; the mask keeps with probability 1 - p "
             "(Module 11).")),
      apis=("jax.random.bernoulli",), check="stats",
      module="11", also=("32",)),
    E("shuffle (random permutation)", "Random numbers",
      "Put the items in a random order, every order equally likely: how each training epoch reorders the data.",
      r"\pi \sim \mathrm{Uniform}(S_n):\quad P(\pi) = \frac{1}{n!};\ \text{Fisher–Yates: for } i = n{-}1, \dots, 1,"
      r"\ \text{swap } x_i \text{ with } x_j,\ j \sim U\{0, \dots, i\}",
      r"$\pi \in S_n$, $x_{\pi(1)}, \dots, x_{\pi(n)}$", "(n,), trials, seed -> (trials, n)",
      "(lambda d: {'first': [int(s[0]) for s in d], "
      "'is_permutation': [sorted(map(int, s)) == list(range(len(s))) for s in d]})(shuffles(x, 20000, seed))",
      lambda r: {"x": np.arange(8), "seed": 0},
      python="""
        def shuffles(x, trials, seed):  # Fisher-Yates, once per trial
            rng = random.Random(seed)
            out = []
            for _ in range(trials):
                y = list(x)
                for i in reversed(range(1, len(y))):
                    j = rng.randrange(i + 1)
                    y[i], y[j] = y[j], y[i]
                out.append(y)
            return out
      """,
      numpy="""
        def shuffles(x, trials, seed):
            rng = np.random.default_rng(seed)
            return np.stack([rng.permutation(x) for _ in range(trials)])
      """,
      jax="""
        def shuffles(x, trials, seed):
            keys = jax.random.split(jax.random.key(seed), trials)
            return jax.vmap(lambda k: jax.random.permutation(k, x))(keys)
      """,
      torch="""
        def shuffles(x, trials, seed):
            g = torch.Generator().manual_seed(seed)
            return torch.stack([x[torch.randperm(len(x), generator=g)] for _ in range(trials)])
      """,
      notes=(("`jax.random.permutation(key, x)` shuffles an array (along axis 0); `permutation(key, n)` returns a "
             "shuffled `arange(n)`. PyTorch only has `torch.randperm(n)`: index with it."),
             ("The test checks two things: every output is a true permutation, and the item landing first is "
             "uniform over all n (its mean and spread match)."),
             "NumPy's `rng.shuffle(x)` shuffles in place; JAX can't, so it returns a new array."),
      apis=("jax.random.permutation",), check="stats", tol=0.1,
      module="09", also=("10",)),
    E("sample without replacement", "Random numbers",
      "Draw k different items out of n, like dealing cards: once picked, an item can't come up again.",
      r"(i_1, \dots, i_k):\ i_j \ne i_l,\quad P = \frac{(n - k)!}{n!}",
      r"$S \subset \{1, \dots, n\},\ |S| = k$", "n, k, trials, seed -> (trials, k)",
      "{'values': (d := draws(10, 3, 10000, seed)), 'distinct': [len(set(map(int, row))) for row in d]}",
      lambda r: {"seed": 0},
      python="""
        def draws(n, k, trials, seed):
            rng = random.Random(seed)
            return [rng.sample(range(n), k) for _ in range(trials)]  # random.choices would allow repeats
      """,
      numpy="""
        def draws(n, k, trials, seed):
            rng = np.random.default_rng(seed)
            return np.stack([rng.choice(n, size=k, replace=False) for _ in range(trials)])
      """,
      jax="""
        def draws(n, k, trials, seed):
            keys = jax.random.split(jax.random.key(seed), trials)
            return jax.vmap(lambda key: jax.random.choice(key, n, (k,), replace=False))(keys)
      """,
      torch="""
        def draws(n, k, trials, seed):
            g = torch.Generator().manual_seed(seed)
            return torch.stack([torch.randperm(n, generator=g)[:k] for _ in range(trials)])
      """,
      notes=(("Default trap: `jax.random.choice` and NumPy's `choice` default to `replace=True` (repeats "
             "allowed); `torch.multinomial` defaults to `replacement=False`."),
             ("In PyTorch, `torch.randperm(n)[:k]` (uniform) or `torch.multinomial(weights, k)` (weighted) draw "
             "without replacement."),
             "The test checks that every draw has k distinct items and that the values are uniform over 0..n-1."),
      apis=("jax.random.choice",), check="stats", tol=0.08,
      module="34", also=("38",)),
    E("He (Kaiming) normal initialization", "Random numbers",
      "Start each weight random with variance 2 / fan_in, so signals keep their size through many ReLU layers "
      "instead of shrinking or exploding.",
      r"W_{ij} \sim \mathcal{N}\!\Big(0, \frac{2}{n_\text{in}}\Big)",
      r"$\mathrm{Var}[W] = 2 / n_\text{in}$ (He et al., 2015); Xavier/Glorot uses $2 / (n_\text{in} + n_\text{out})$",
      "fan_in, fan_out, seed -> (fan_in, fan_out)", "he_normal(256, 128, seed)", lambda r: {"seed": 0},
      python="""
        def he_normal(fan_in, fan_out, seed):  # Box-Muller normals, scaled
            rng = random.Random(seed)
            std = math.sqrt(2 / fan_in)
            def z():
                return math.sqrt(-2 * math.log(1.0 - rng.random())) * math.cos(2 * math.pi * rng.random())
            return [[std * z() for _ in range(fan_out)] for _ in range(fan_in)]
      """,
      numpy="""
        def he_normal(fan_in, fan_out, seed):
            return np.random.default_rng(seed).normal(0, np.sqrt(2 / fan_in), size=(fan_in, fan_out))
      """,
      jax="""
        def he_normal(fan_in, fan_out, seed):
            init = jax.nn.initializers.he_normal()  # the same function as flax.nnx.initializers.he_normal
            return init(jax.random.key(seed), (fan_in, fan_out))  # kernel layout (in, out)
      """,
      torch="""
        def he_normal(fan_in, fan_out, seed):
            W = torch.empty(fan_out, fan_in)  # nn.Linear layout (out, in)
            torch.nn.init.kaiming_normal_(W, generator=torch.Generator().manual_seed(seed))
            return W.T
      """,
      notes=(("Flax's `he_normal` draws from a *truncated* normal (nothing beyond about 2.3 standard deviations), "
             "rescaled so the variance is still 2 / fan_in; PyTorch's `kaiming_normal_` uses a plain normal."),
             ("Fan-in comes from the layout: Flax kernels are (in, out), so fan_in = `shape[-2]`; PyTorch weights "
             "are (out, in), so fan_in = `shape[1]`. Pass the wrong layout and you get the fan_out variance."),
             ("PyTorch's `nn.Linear` does *not* use this by default (it uses a uniform with variance "
             "1 / (3 fan_in)); call `kaiming_normal_` yourself (Module 11).")),
      apis=("flax.nnx.initializers.he_normal",), check="stats", tol=0.003,
      module="10", also=("11",)),
]
