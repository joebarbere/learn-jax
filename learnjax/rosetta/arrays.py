"""Rosetta entries for arrays, elementwise math and control flow (Modules 00b–08, plus later array helpers)."""
import numpy as np

from learnjax.rosetta.entry import Entry as E

ENTRIES: list[E] = [
    # ---------------------------------------------------------------- Elementwise math
    E("exp, log and powers", "Elementwise math",
      "The functions applied to every entry on its own: exponentials and logs (which turn products into sums), "
      "square roots and squares, and plain addition.",
      r"e^{x},\quad \ln x,\quad \log_{10} x,\quad \sqrt{x},\quad x^2,\quad x + y",
      r"$\exp(x)$; $\log$ in ML papers almost always means $\ln$ (base $e$)", "(n,), (n,) -> (n,) each",
      "powers(x, y)", lambda r: {"x": r.uniform(0.5, 3.0, size=5), "y": r.normal(size=5)},
      python="""
        def powers(x, y):
            return {"exp": [math.exp(v) for v in x], "log": [math.log(v) for v in x],
                    "log10": [math.log10(v) for v in x], "sqrt": [math.sqrt(v) for v in x],
                    "square": [v * v for v in x], "add": [a + b for a, b in zip(x, y)]}
      """,
      numpy="""
        def powers(x, y):
            return {"exp": np.exp(x), "log": np.log(x), "log10": np.log10(x), "sqrt": np.sqrt(x),
                    "square": np.square(x), "add": np.add(x, y)}  # np.add(x, y) is x + y
      """,
      jax="""
        def powers(x, y):
            return {"exp": jnp.exp(x), "log": jnp.log(x), "log10": jnp.log10(x), "sqrt": jnp.sqrt(x),
                    "square": jnp.square(x), "add": jnp.add(x, y)}  # jnp.add(x, y) is x + y
      """,
      torch="""
        def powers(x, y):
            return {"exp": torch.exp(x), "log": torch.log(x), "log10": torch.log10(x), "sqrt": torch.sqrt(x),
                    "square": torch.square(x), "add": torch.add(x, y)}
      """,
      notes=("`log` is the natural log in all of these; papers that mean base 2 (bits) or 10 say so.",
             ("Outside the domain, NumPy and JAX return NaN (`log(-1)`, `sqrt(-1)`) and `-inf` for `log(0)` "
              "instead of raising, as Python's `math` does; Module 11's `jax_debug_nans` finds where."),
             ("`jnp.add` exists so you can pass addition as a function (e.g. to `jax.tree.map(jnp.add, a, b)` "
              "in Module 26).")),
      apis=("jax.numpy.exp", "jax.numpy.log", "jax.numpy.log10", "jax.numpy.sqrt", "jax.numpy.square",
            "jax.numpy.add"),
      module="04", also=("02", "26")),
    E("trig and sign functions", "Elementwise math",
      "More functions applied entry by entry: sine and cosine (angles, waves, positional encodings), tanh (a "
      "squashing curve), absolute value, sign, rounding, and degrees to radians.",
      r"\sin x,\ \cos x,\ \tanh x = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}},\ |x|,\ \operatorname{sgn}(x),\ "
      r"\theta_\text{rad} = \theta_\text{deg} \cdot \tfrac{\pi}{180}",
      r"$\lfloor x \rceil$ for rounding", "(n,) -> (n,) each", "elementwise(x)",
      lambda r: {"x": np.array([-2.5, -1.2, 0.0, 0.5, 1.5, 2.7])},
      python="""
        def elementwise(x):
            return {"sin": [math.sin(v) for v in x], "cos": [math.cos(v) for v in x],
                    "tanh": [math.tanh(v) for v in x], "abs": [abs(v) for v in x],
                    "sign": [(v > 0) - (v < 0) for v in x], "round": [round(v) for v in x],
                    "deg2rad": [math.radians(v) for v in x]}
      """,
      numpy="""
        def elementwise(x):
            return {"sin": np.sin(x), "cos": np.cos(x), "tanh": np.tanh(x), "abs": np.abs(x),
                    "sign": np.sign(x), "round": np.round(x), "deg2rad": np.deg2rad(x)}
      """,
      jax="""
        def elementwise(x):
            return {"sin": jnp.sin(x), "cos": jnp.cos(x), "tanh": jnp.tanh(x), "abs": jnp.abs(x),
                    "sign": jnp.sign(x), "round": jnp.round(x), "deg2rad": jnp.deg2rad(x)}
      """,
      torch="""
        def elementwise(x):
            return {"sin": torch.sin(x), "cos": torch.cos(x), "tanh": torch.tanh(x), "abs": torch.abs(x),
                    "sign": torch.sign(x), "round": torch.round(x), "deg2rad": torch.deg2rad(x)}
      """,
      notes=(("Rounding goes half to even everywhere, Python's `round` included: `round(0.5) = 0`, "
             "`round(2.5) = 2`, `round(1.5) = 2`."),
             "Trig functions take radians in every library.",
             ("`sign(0) = 0` in all of them, but the gradient of `abs` at exactly 0 differs: 1 in JAX, 0 in "
              "PyTorch.")),
      apis=("jax.numpy.sin", "jax.numpy.cos", "jax.numpy.tanh", "jax.numpy.abs", "jax.numpy.sign",
            "jax.numpy.round", "jax.numpy.deg2rad"),
      module="01", also=("02", "03", "17")),
    E("log1p and expm1 (accurate near zero)", "Elementwise math",
      "`log(1 + x)` and `e^x - 1` computed without first rounding 1 + x: when x is tiny, the plain formulas lose "
      "every digit of x.",
      r"\operatorname{log1p}(x) = \ln(1 + x),\quad \operatorname{expm1}(x) = e^{x} - 1", "",
      "(n,) -> (n,) each", "near_zero(x)", lambda r: {"x": np.array([1e-8, 1e-5, 1e-3, 0.1, 2.0])},
      python="""
        def near_zero(x):
            return {"log1p": [math.log1p(v) for v in x], "expm1": [math.expm1(v) for v in x]}
      """,
      numpy="""
        def near_zero(x):
            return {"log1p": np.log1p(x), "expm1": np.expm1(x)}
      """,
      jax="""
        def near_zero(x):
            return {"log1p": jnp.log1p(x), "expm1": jnp.expm1(x)}
      """,
      torch="""
        def near_zero(x):
            return {"log1p": torch.log1p(x), "expm1": torch.expm1(x)}
      """,
      notes=(("In float32, `1 + 1e-8` rounds to exactly 1, so `log(1 + x)` returns 0 there; `log1p` returns "
             "1e-8."),
             ("They appear inside stable formulas: softplus $\\log(1+e^x)$ (Module 33), logistic losses "
              "(Module 09), and Mamba's discretization (Module 36).")),
      apis=("jax.numpy.log1p", "jax.numpy.expm1"),
      module="09", also=("33", "36")),
    E("clip, maximum and minimum (elementwise)", "Elementwise math",
      "Compare two arrays (or an array and a number) entry by entry and keep the larger or smaller; clip "
      "squeezes every value into a range.",
      r"\operatorname{clip}(x, a, b)_i = \min(\max(x_i, a), b),\quad \max(x, 0) = \operatorname{ReLU}(x)",
      r"$[x]_+$ or $(x)_+$ for $\max(x, 0)$", "(n,), (n,) -> (n,) each", "bounds(x, y)",
      lambda r: {"x": 2 * r.normal(size=6), "y": r.normal(size=6)},
      python="""
        def bounds(x, y):
            return {"clip": [min(max(v, -1.0), 1.0) for v in x], "relu": [max(v, 0.0) for v in x],
                    "smaller": [min(a, b) for a, b in zip(x, y)]}
      """,
      numpy="""
        def bounds(x, y):
            return {"clip": np.clip(x, -1.0, 1.0), "relu": np.maximum(x, 0.0), "smaller": np.minimum(x, y)}
      """,
      jax="""
        def bounds(x, y):
            return {"clip": jnp.clip(x, -1.0, 1.0), "relu": jnp.maximum(x, 0.0), "smaller": jnp.minimum(x, y)}
      """,
      torch="""
        def bounds(x, y):
            return {"clip": torch.clamp(x, -1.0, 1.0), "relu": torch.clamp(x, min=0.0),
                    "smaller": torch.minimum(x, y)}
      """,
      notes=(("Trap: `np.max(x, 0)` / `jnp.max(x, 0)` is a *reduction* along axis 0 (one number for a vector), "
             "not ReLU; the elementwise one is `maximum(x, 0)`."),
             ("PyTorch overloads `torch.max`: `torch.max(x, y)` with two tensors is elementwise, "
              "`torch.max(x, 0)` reduces along dim 0 and returns (values, indices). `torch.maximum` needs two "
              "tensors; use `torch.clamp(x, min=0)` against a number."),
             "`torch.clip` is an alias of `torch.clamp`."),
      apis=("jax.numpy.clip", "jax.numpy.maximum", "jax.numpy.minimum"),
      module="02", also=("05", "06", "10")),
    # ---------------------------------------------------------------- Arrays and reductions
    E("max and argmax (reductions)", "Arrays and reductions",
      "Along an axis, the largest value and where it is. argmax is how a classifier turns scores into a "
      "prediction and how greedy decoding picks a token.",
      r"m_i = \max_j x_{ij},\quad a_i = \arg\max_j x_{ij}", r"$\hat y = \arg\max_c\, p(c \mid x)$",
      "(m, n) -> (m,), (m,) int", "row_max(x)", lambda r: {"x": r.normal(size=(3, 4))},
      python="""
        def row_max(x):
            values = [max(row) for row in x]
            indices = [max(range(len(row)), key=lambda j: row[j]) for row in x]  # first of any ties
            return {"values": values, "indices": indices}
      """,
      numpy="""
        def row_max(x):
            return {"values": np.max(x, axis=1), "indices": np.argmax(x, axis=1)}
      """,
      jax="""
        def row_max(x):
            return {"values": jnp.max(x, axis=1), "indices": jnp.argmax(x, axis=1)}
      """,
      torch="""
        def row_max(x):
            values, indices = torch.max(x, dim=1)  # one call returns both
            return {"values": values, "indices": indices}
      """,
      notes=("Ties go to the first index in all of these.",
             ("argmax has no useful gradient (it is flat almost everywhere); training uses softmax and "
             "cross-entropy instead, and sampling uses the Gumbel-max trick (Module 04).")),
      apis=("jax.numpy.max", "jax.numpy.argmax"),
      module="04", also=("10", "18")),
    E("mean, standard deviation and median", "Arrays and reductions",
      "The centre and the spread of the numbers along an axis. The median is the middle value once sorted "
      "(the average of the two middle ones for an even count).",
      r"\mu = \frac{1}{n}\sum_i x_i,\quad \sigma = \sqrt{\frac{1}{n}\sum_i (x_i - \mu)^2}",
      r"$\bar x$, $\mathbb{E}[x]$; $s$ with $\frac{1}{n-1}$ is the unbiased sample version",
      "(m, n) -> (m,) each", "summary(x)", lambda r: {"x": r.normal(size=(2, 6))},
      python="""
        def summary(x):
            out = {"mean": [], "std": [], "median": []}
            for row in x:
                n = len(row)
                mu = sum(row) / n
                out["mean"].append(mu)
                out["std"].append(math.sqrt(sum((v - mu) ** 2 for v in row) / n))
                s = sorted(row)
                out["median"].append(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)
            return out
      """,
      numpy="""
        def summary(x):
            return {"mean": np.mean(x, axis=-1), "std": np.std(x, axis=-1), "median": np.median(x, axis=-1)}
      """,
      jax="""
        def summary(x):
            return {"mean": jnp.mean(x, axis=-1), "std": jnp.std(x, axis=-1), "median": jnp.median(x, axis=-1)}
      """,
      torch="""
        def summary(x):
            return {"mean": torch.mean(x, dim=-1),
                    "std": torch.std(x, dim=-1, correction=0),  # default is correction=1
                    "median": torch.quantile(x, 0.5, dim=-1)}  # not torch.median: see notes
      """,
      notes=(("Trap: `torch.std` divides by $n-1$ by default (the unbiased estimate); NumPy and JAX divide by "
             "$n$ (`ddof=0`). Pass `correction=0` in PyTorch or `ddof=1` in NumPy/JAX to match."),
             ("Trap: for an even count `torch.median` returns the *lower* of the two middle values "
              "(`[1, 2, 3, 4]` gives 2), while NumPy and JAX average them (2.5); `torch.quantile(x, 0.5)` "
              "averages."),
             "LayerNorm and BatchNorm (Module 11) use the divide-by-$n$ variance."),
      apis=("jax.numpy.mean", "jax.numpy.std", "jax.numpy.median"),
      module="01", also=("03", "07", "11", "16")),
    E("boolean masks: where, any and all", "Arrays and reductions",
      "Choose between two values entry by entry with a condition, and ask whether any or all entries pass a "
      "test. `where` is how masking works on a GPU: compute both, keep one.",
      r"y_i = \begin{cases} x_i & c_i \\ 0 & \text{otherwise} \end{cases},\quad "
      r"\exists j: x_{ij} > 1,\quad \forall j: x_{ij} > -1",
      r"$\mathbb{1}[x > 0] \cdot x$ (an indicator times x)", "(m, n) -> (m, n), (m,) bool, (m,) bool",
      "masks(x)", lambda r: {"x": r.normal(size=(3, 4))},
      python="""
        def masks(x):
            return {"where": [[v if v > 0 else 0.0 for v in row] for row in x],
                    "any": [any(v > 1 for v in row) for row in x],
                    "all": [all(v > -1 for v in row) for row in x]}
      """,
      numpy="""
        def masks(x):
            return {"where": np.where(x > 0, x, 0.0), "any": np.any(x > 1, axis=1),
                    "all": np.all(x > -1, axis=1)}
      """,
      jax="""
        def masks(x):
            return {"where": jnp.where(x > 0, x, 0.0), "any": jnp.any(x > 1, axis=1),
                    "all": jnp.all(x > -1, axis=1)}
      """,
      torch="""
        def masks(x):
            return {"where": torch.where(x > 0, x, 0.0), "any": torch.any(x > 1, dim=1),
                    "all": torch.all(x > -1, dim=1)}
      """,
      notes=(("Under `jit`, shapes must be known in advance, so `x[x > 0]` (raises "
              "`NonConcreteBooleanIndexError`) and one-argument `jnp.where(x > 0)` (raises "
              "`ConcretizationTypeError`) fail: their output size depends on the data. Use three-argument "
              "`where`, or `jnp.where(cond, size=k, fill_value=-1)`."),
             ("Both branches of `where` are computed, and so are both gradients: `where(x > 0, log(x), 0)` "
              "has the right value at x = 0 but a NaN gradient (0 times the infinite slope of log). Make the "
              "input safe too: `where(x > 0, log(where(x > 0, x, 1)), 0)`.")),
      apis=("jax.numpy.where", "jax.numpy.any", "jax.numpy.all"),
      module="01", also=("06", "15", "29", "34")),
    E("ranges and grids", "Arrays and reductions",
      "Evenly spaced numbers: by step (arange), by count with both ends included (linspace), evenly spaced in "
      "the exponent (logspace), and every (x, y) pair of two ranges (meshgrid).",
      r"a_k = a_0 + k\,\Delta,\quad \ell_k = a + k\,\tfrac{b - a}{n - 1},\quad g_k = 10^{\ell_k},\quad "
      r"X_{ij} = x_j,\ Y_{ij} = y_i",
      "", "() -> (4,), (5,), (4,), (2, 3), (2, 3)", "grids()", lambda r: {},
      python="""
        def grids():
            xs, ys = [0.0, 1.0, 2.0], [0.0, 1.0]
            return {"arange": [k * 0.25 for k in range(4)],  # start 0, step 0.25, stop before 1
                    "linspace": [k / 4 for k in range(5)],  # 5 points from 0 to 1, both ends
                    "logspace": [10.0 ** (-3 + k) for k in range(4)],  # exponents -3..0
                    "X": [[x for x in xs] for y in ys], "Y": [[y for x in xs] for y in ys]}
      """,
      numpy="""
        def grids():
            X, Y = np.meshgrid(np.arange(3.0), np.arange(2.0))  # indexing="xy": shape (len(y), len(x))
            return {"arange": np.arange(0, 1, 0.25), "linspace": np.linspace(0, 1, 5),
                    "logspace": np.logspace(-3, 0, 4), "X": X, "Y": Y}
      """,
      jax="""
        def grids():
            X, Y = jnp.meshgrid(jnp.arange(3.0), jnp.arange(2.0))
            return {"arange": jnp.arange(0, 1, 0.25), "linspace": jnp.linspace(0, 1, 5),
                    "logspace": jnp.logspace(-3, 0, 4), "X": X, "Y": Y}
      """,
      torch="""
        def grids():
            X, Y = torch.meshgrid(torch.arange(3.0), torch.arange(2.0), indexing="xy")
            return {"arange": torch.arange(0, 1, 0.25), "linspace": torch.linspace(0, 1, 5),
                    "logspace": torch.logspace(-3, 0, 4), "X": X, "Y": Y}
      """,
      notes=(("Trap: `meshgrid` defaults to `indexing=\"xy\"` in NumPy and JAX (shape `(len(y), len(x))`, "
              "image convention) but to `\"ij\"` in PyTorch (shape `(len(x), len(y))`, with a warning). Always "
              "pass `indexing=`."),
             ("`arange` excludes the end and a float step can land just short of it or just past it after "
              "rounding; when you want n points with both ends, use `linspace`.")),
      apis=("jax.numpy.arange", "jax.numpy.linspace", "jax.numpy.logspace", "jax.numpy.meshgrid"),
      module="01", also=("05", "08", "37")),
    E("stack, concatenate and split", "Arrays and reductions",
      "Join arrays along a new axis (stack) or an existing one (concatenate), and cut one back into equal "
      "pieces (split).",
      r"\operatorname{stack}(a, b)_{0} = a,\ \operatorname{stack}(a, b)_{1} = b;\quad "
      r"\operatorname{concat}(a, b) = \begin{bmatrix} a \\ b \end{bmatrix}",
      r"$[a; b]$ (vertical) or $[a, b]$ / $a \oplus b$ (side by side)",
      "(2, 3), (2, 3) -> (2, 2, 3), (4, 3), 2 x (2, 3)", "join(a, b)",
      lambda r: {"a": r.normal(size=(2, 3)), "b": r.normal(size=(2, 3))},
      python="""
        def join(a, b):
            c = a + b  # list + list concatenates rows
            return {"stack": [a, b], "concat": c, "first": c[:2], "second": c[2:]}
      """,
      numpy="""
        def join(a, b):
            c = np.concatenate([a, b], axis=0)
            first, second = np.split(c, 2, axis=0)  # 2 = number of equal pieces
            return {"stack": np.stack([a, b], axis=0), "concat": c, "first": first, "second": second}
      """,
      jax="""
        def join(a, b):
            c = jnp.concatenate([a, b], axis=0)
            first, second = jnp.split(c, 2, axis=0)
            return {"stack": jnp.stack([a, b], axis=0), "concat": c, "first": first, "second": second}
      """,
      torch="""
        def join(a, b):
            c = torch.cat([a, b], dim=0)
            first, second = torch.chunk(c, 2, dim=0)  # 2 pieces; torch.split(c, 2) means pieces of SIZE 2
            return {"stack": torch.stack([a, b], dim=0), "concat": c, "first": first, "second": second}
      """,
      notes=(("Trap: `np.split(x, 3)` / `jnp.split` cut into 3 equal pieces, but `torch.split(x, 3)` cuts "
              "into pieces of size 3. PyTorch's equivalent of `np.split(x, n)` is `torch.chunk(x, n)`."),
             ("`np.split` raises if the pieces can't be equal; `np.array_split` and `torch.chunk` allow a "
              "short last piece."),
             "`torch.cat` is PyTorch's name for concatenate (`torch.concatenate` is an alias)."),
      apis=("jax.numpy.stack", "jax.numpy.concatenate", "jax.numpy.split"),
      module="01", also=("16", "17")),
    E("moving axes: transpose, moveaxis and swapaxes", "Arrays and reductions",
      "Reorder the axes of an array without changing its numbers: swap two (e.g. a matrix transpose on the last "
      "two), move one to a new place (e.g. channels last), or add a leading axis to a vector.",
      r"(A^\top)_{ij} = A_{ji},\quad Y_{b,j,i} = X_{b,i,j}",
      r"$A^\top$, $A^{T}$, $A'$", "(2, 3, 4) -> (2, 4, 3), (3, 4, 2); (3,) -> (1, 3)", "axes(x, v)",
      lambda r: {"x": r.normal(size=(2, 3, 4)), "v": r.normal(size=3)},
      python="""
        def axes(x, v):
            B, I, J = len(x), len(x[0]), len(x[0][0])
            swapped = [[[x[b][i][j] for i in range(I)] for j in range(J)] for b in range(B)]
            moved = [[[x[b][i][j] for b in range(B)] for j in range(J)] for i in range(I)]
            return {"swapaxes": swapped, "moveaxis": moved, "atleast_2d": [list(v)]}
      """,
      numpy="""
        def axes(x, v):
            return {"swapaxes": np.swapaxes(x, 1, 2), "moveaxis": np.moveaxis(x, 0, -1),
                    "atleast_2d": np.atleast_2d(v)}
      """,
      jax="""
        def axes(x, v):
            return {"swapaxes": jnp.swapaxes(x, 1, 2), "moveaxis": jnp.moveaxis(x, 0, -1),
                    "atleast_2d": jnp.atleast_2d(v)}
      """,
      torch="""
        def axes(x, v):
            return {"swapaxes": x.transpose(1, 2), "moveaxis": torch.movedim(x, 0, -1),
                    "atleast_2d": torch.atleast_2d(v)}
      """,
      notes=(("Trap: `x.transpose(1, 2)` swaps two axes in PyTorch, but in NumPy/JAX `x.transpose(...)` wants "
              "the full new order (`x.transpose(0, 2, 1)`) and `transpose(1, 2)` on a 3-D array raises."),
             ("`x.T` reverses *all* axes in NumPy/JAX ((2, 3, 4) -> (4, 3, 2)); for the batched matrix "
              "transpose use `jnp.swapaxes(x, -1, -2)` / `jnp.matrix_transpose(x)`, or `x.mT` in PyTorch."),
             "PyTorch calls moveaxis `torch.movedim` (`torch.moveaxis` is an alias)."),
      apis=("jax.numpy.moveaxis", "jax.numpy.swapaxes", "jax.numpy.atleast_2d"),
      module="01", also=("03", "15", "31")),
    E("repeat, tile and broadcast_to", "Arrays and reductions",
      "Make copies: repeat each element in place, tile the whole array end to end, or broadcast a smaller array "
      "to a bigger shape without copying memory.",
      r"\operatorname{repeat}(x, 2) = (x_1, x_1, x_2, x_2, \dots),\quad "
      r"\operatorname{tile}(x, 2) = (x_1, x_2, \dots, x_1, x_2, \dots)",
      "", "(3,) -> (6,), (6,), (2, 3)", "copies(x)", lambda r: {"x": np.array([1.0, 2.0, 3.0])},
      python="""
        def copies(x):
            return {"repeat": [v for v in x for _ in range(2)], "tile": list(x) * 2,
                    "broadcast": [list(x) for _ in range(2)]}
      """,
      numpy="""
        def copies(x):
            return {"repeat": np.repeat(x, 2), "tile": np.tile(x, 2), "broadcast": np.broadcast_to(x, (2, 3))}
      """,
      jax="""
        def copies(x):
            return {"repeat": jnp.repeat(x, 2), "tile": jnp.tile(x, 2), "broadcast": jnp.broadcast_to(x, (2, 3))}
      """,
      torch="""
        def copies(x):
            return {"repeat": x.repeat_interleave(2), "tile": x.repeat(2),  # torch's repeat is np.tile!
                    "broadcast": x.broadcast_to((2, 3))}  # or x.expand(2, 3)
      """,
      notes=("Trap: `torch.Tensor.repeat` behaves like `np.tile`; PyTorch's `np.repeat` is `repeat_interleave`.",
             ("Grouped-query attention (Module 17) shares each key/value head across several query heads with "
              "`jnp.repeat(k, n_rep, axis=head_axis)` — `repeat`, not `tile`, so neighbouring query heads share "
              "a KV head."),
             "Under `jit`, `broadcast_to` costs nothing until something needs the full array."),
      apis=("jax.numpy.repeat", "jax.numpy.tile", "jax.numpy.broadcast_to"),
      module="17", also=("18",)),
    E("pad and roll", "Arrays and reductions",
      "Pad adds a border of zeros (for convolutions, or to make lengths equal); roll shifts every entry along "
      "an axis and wraps the ones that fall off around to the other end.",
      r"\operatorname{roll}(x, s)_i = x_{(i - s) \bmod n}", "", "(2, 3) -> (2, 6), (2, 3)", "border(x)",
      lambda r: {"x": r.normal(size=(2, 3))},
      python="""
        def border(x):
            n = len(x[0])
            return {"pad": [[0.0] + list(row) + [0.0, 0.0] for row in x],
                    "roll": [[row[(i - 1) % n] for i in range(n)] for row in x]}
      """,
      numpy="""
        def border(x):
            return {"pad": np.pad(x, ((0, 0), (1, 2))),  # (before, after) per axis, first axis first
                    "roll": np.roll(x, 1, axis=-1)}
      """,
      jax="""
        def border(x):
            return {"pad": jnp.pad(x, ((0, 0), (1, 2))), "roll": jnp.roll(x, 1, axis=-1)}
      """,
      torch="""
        def border(x):
            return {"pad": F.pad(x, (1, 2)),  # (left, right) of the LAST axis, then the one before...
                    "roll": torch.roll(x, 1, dims=-1)}
      """,
      notes=(("Trap: NumPy/JAX `pad` take one (before, after) pair per axis, first axis first; PyTorch's "
              "`F.pad` takes a flat tuple starting from the *last* axis: `(left, right, top, bottom, ...)`."),
             "Without `axis` / `dims`, `roll` flattens the array, shifts, and reshapes back, in all three."),
      apis=("jax.numpy.pad", "jax.numpy.roll"),
      module="06", also=("12", "36")),
    E("sort and argsort", "Arrays and reductions",
      "Sort the values, or get the order that would sort them (argsort) so you can reorder something else the "
      "same way. Top-p sampling sorts token probabilities.",
      r"x_{\pi(1)} \le x_{\pi(2)} \le \dots \le x_{\pi(n)},\quad \operatorname{argsort}(x) = \pi",
      r"$x_{(1)} \le \dots \le x_{(n)}$ (order statistics)", "(n,) -> (n,), (n,) int", "order(x)",
      lambda r: {"x": r.normal(size=6)},
      python="""
        def order(x):
            idx = sorted(range(len(x)), key=lambda i: x[i])
            return {"values": [x[i] for i in idx], "indices": idx}
      """,
      numpy="""
        def order(x):
            return {"values": np.sort(x), "indices": np.argsort(x)}
      """,
      jax="""
        def order(x):
            return {"values": jnp.sort(x), "indices": jnp.argsort(x)}
      """,
      torch="""
        def order(x):
            values, indices = torch.sort(x)  # returns both
            return {"values": values, "indices": indices}
      """,
      notes=(("Ascending by default. For descending, JAX has `descending=True` on `sort`/`argsort`, PyTorch has "
             "`descending=True`; NumPy has neither (use `np.argsort(-x)` or reverse with `[::-1]`)."),
             ("For only the k largest, `jax.lax.top_k` / `torch.topk` are faster than a full sort "
              "(Module 16)."),
             "JAX's sorts are stable (equal values keep their order); in PyTorch pass `stable=True`."),
      apis=("jax.numpy.sort", "jax.numpy.argsort"),
      module="18", also=("21", "38")),
    E("cumulative product", "Arrays and reductions",
      "A running product: entry i is the product of everything up to and including i. Diffusion's noise "
      "schedule is one.",
      r"c_i = \prod_{k \le i} x_k", r"$\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$ (diffusion, Module 33)",
      "(n,) -> (n,)", "running_product(x)", lambda r: {"x": r.uniform(0.9, 1.0, size=6)},
      python="""
        def running_product(x):
            out, total = [], 1.0
            for v in x:
                total *= v
                out.append(total)
            return out
      """,
      numpy="""
        def running_product(x):
            return np.cumprod(x)
      """,
      jax="""
        def running_product(x):
            return jnp.cumprod(x)
      """,
      torch="""
        def running_product(x):
            return torch.cumprod(x, dim=0)
      """,
      notes=("`torch.cumprod` needs `dim`.",
             ("Long products of numbers below 1 underflow to 0 (200 factors of 0.01 is $10^{-400}$, below even "
              "float64's range); keep such quantities as a running sum of logs, `cumsum(log x)`, and never "
              "exponentiate.")),
      apis=("jax.numpy.cumprod",),
      module="18", also=("33",)),
    E("identity, diagonal and triangular matrices", "Arrays and reductions",
      "Special matrices: the identity (ones on the diagonal), a diagonal matrix built from a vector, the "
      "diagonal read out of a matrix, and the lower triangle — which is the causal mask in attention.",
      r"I_{ij} = [i = j],\quad \operatorname{diag}(v)_{ij} = v_i [i = j],\quad "
      r"\operatorname{tril}(A)_{ij} = A_{ij} [j \le i]",
      r"$I_n$, $\operatorname{diag}(v)$, $\Lambda$; $M_{ij} = -\infty$ for $j > i$ (causal mask)",
      "() -> (3, 3); (3,) -> (3, 3); (3, 3) -> (3,), (3, 3)", "special(v, A)",
      lambda r: {"v": r.normal(size=3), "A": r.normal(size=(3, 3))},
      python="""
        def special(v, A):
            n = len(v)
            return {"eye": [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)],
                    "diag": [[v[i] if i == j else 0.0 for j in range(n)] for i in range(n)],
                    "diagonal": [A[i][i] for i in range(n)],
                    "tril": [[A[i][j] if j <= i else 0.0 for j in range(n)] for i in range(n)]}
      """,
      numpy="""
        def special(v, A):
            return {"eye": np.eye(3), "diag": np.diag(v), "diagonal": np.diagonal(A), "tril": np.tril(A)}
      """,
      jax="""
        def special(v, A):
            return {"eye": jnp.eye(3), "diag": jnp.diag(v), "diagonal": jnp.diagonal(A), "tril": jnp.tril(A)}
      """,
      torch="""
        def special(v, A):
            return {"eye": torch.eye(3), "diag": torch.diag(v), "diagonal": torch.diagonal(A),
                    "tril": torch.tril(A)}
      """,
      notes=(("`diag` does two different things: on a vector it builds a matrix, on a matrix it reads the "
             "diagonal. `diagonal` only reads, and works on batches of matrices (last two axes with "
             "`axis1=-2, axis2=-1`)."),
             ("The causal mask of a decoder (Module 15) is `tril(ones((T, T)))`: position i may look at j only "
              "when j <= i; masked scores become -inf before the softmax.")),
      apis=("jax.numpy.eye", "jax.numpy.diag", "jax.numpy.diagonal", "jax.numpy.tril"),
      module="02", also=("03", "15", "38")),
    # ---------------------------------------------------------------- Control flow and transforms
    E("apply a function to every row: vmap and lax.map", "Control flow and transforms",
      "Write a function for one example, then run it on a whole batch. `vmap` rewrites it into batched array "
      "operations; `lax.map` runs it one example at a time in a compiled loop.",
      r"y_i = f(x_i, w) \;\text{ for every row } i, \quad f(x, w) = \tanh(w \cdot x)",
      r"$\{f(x_i)\}_{i=1}^{N}$", "(N, d), (d,) -> (N,)", "per_row(X, w)",
      lambda r: {"X": r.normal(size=(4, 3)), "w": r.normal(size=3)},
      python="""
        def per_row(X, w):
            def f(x, w):
                return math.tanh(sum(a * b for a, b in zip(x, w)))
            return [f(x, w) for x in X]
      """,
      numpy="""
        def per_row(X, w):  # no vmap: rewrite f for a batch by hand
            return np.tanh(X @ w)
      """,
      jax="""
        def f(x, w):
            return jnp.tanh(x @ w)

        def per_row(X, w):
            return jax.vmap(f, in_axes=(0, None))(X, w)  # batch X's axis 0; share w
            # jax.lax.map(lambda x: f(x, w), X) gives the same, one row at a time
      """,
      torch="""
        def f(x, w):
            return torch.tanh(x @ w)

        def per_row(X, w):
            return torch.vmap(f, in_dims=(0, None))(X, w)
      """,
      notes=(("`in_axes` (JAX) and `in_dims` (PyTorch) say which axis of each argument is the batch; `None` "
             "means the same value for every example."),
             ("`lax.map` is a `scan` under the hood: sequential, so it needs memory for one example at a time. "
              "Use it when the vmapped version won't fit; `batch_size=` runs a few examples per step."),
             ("vmap composes with `grad`: `jax.vmap(jax.grad(loss), in_axes=(None, 0))` gives per-example "
              "gradients (Module 07).")),
      apis=("jax.vmap", "jax.lax.map"),
      module="07", also=("00b", "03", "17")),
    E("loop with carried state: lax.scan", "Control flow and transforms",
      "Walk along a sequence carrying a state: each step reads the state and one input, and returns a new state "
      "and an output. RNNs, Mamba, optimizers over steps, and token-by-token decoding all have this shape.",
      r"m_t = \beta\, m_{t-1} + (1 - \beta)\, x_t,\quad m_0 = 0 \;\;\text{(an exponential moving average)}",
      r"$h_t = f(h_{t-1}, x_t)$ (a recurrence)", "(T,), () -> (T,)", "ema(xs, beta)",
      lambda r: {"xs": r.normal(size=6), "beta": 0.9},
      python="""
        def ema(xs, beta):
            m, out = 0.0, []
            for x in xs:
                m = beta * m + (1 - beta) * x
                out.append(m)
            return out
      """,
      numpy="""
        def ema(xs, beta):
            m, out = 0.0, []
            for x in xs:
                m = beta * m + (1 - beta) * x
                out.append(m)
            return np.array(out)
      """,
      jax="""
        def ema(xs, beta):
            def step(m, x):
                m = beta * m + (1 - beta) * x
                return m, m  # (new carry, this step's output)
            final, ms = jax.lax.scan(step, jnp.zeros_like(xs[0]), xs)
            return ms
      """,
      torch="""
        def ema(xs, beta):
            m, out = torch.zeros(()), []
            for x in xs:
                m = beta * m + (1 - beta) * x
                out.append(m)
            return torch.stack(out)
      """,
      notes=(("`scan` compiles the step once and loops on the device. A Python `for` loop under `jit` is "
              "unrolled instead: T copies of the step in the program, slow to compile for long sequences "
              "(Module 08)."),
             ("The carry must keep the same shape and dtype every step; starting from a Python `0.0` while the "
              "step returns float32 arrays is a common error — start from `jnp.zeros_like(...)`."),
             "PyTorch and NumPy have no scan; a Python loop is the equivalent."),
      apis=("jax.lax.scan",),
      module="08", also=("05", "14", "31", "36")),
    E("fixed and open-ended loops: fori_loop and while_loop", "Control flow and transforms",
      "Repeat an update a fixed number of times (fori_loop) or until a condition is met (while_loop). Here: "
      "Newton's method for square roots.",
      r"y_{k+1} = \tfrac{1}{2}\Big(y_k + \frac{a}{y_k}\Big),\; y_0 = a \;\to\; \sqrt{a}",
      "", "(n,) -> (n,), (n,)", "newton_sqrt(a)", lambda r: {"a": r.uniform(1.0, 10.0, size=4)},
      python="""
        def newton_sqrt(a):
            fixed, until = [], []
            for v in a:
                y = v
                for _ in range(20):
                    y = (y + v / y) / 2
                fixed.append(y)
                y = v
                while abs(y * y - v) > 1e-5:
                    y = (y + v / y) / 2
                until.append(y)
            return {"fixed": fixed, "until_converged": until}
      """,
      numpy="""
        def newton_sqrt(a):
            y = a.copy()
            for _ in range(20):
                y = (y + a / y) / 2
            z = a.copy()
            while np.max(np.abs(z * z - a)) > 1e-5:
                z = (z + a / z) / 2
            return {"fixed": y, "until_converged": z}
      """,
      jax="""
        def newton_sqrt(a):
            def step(y):
                return (y + a / y) / 2
            fixed = jax.lax.fori_loop(0, 20, lambda i, y: step(y), a)
            until = jax.lax.while_loop(lambda y: jnp.max(jnp.abs(y * y - a)) > 1e-5, step, a)
            return {"fixed": fixed, "until_converged": until}
      """,
      torch="""
        def newton_sqrt(a):
            y = a.clone()
            for _ in range(20):
                y = (y + a / y) / 2
            z = a.clone()
            while torch.max(torch.abs(z * z - a)) > 1e-5:
                z = (z + a / z) / 2
            return {"fixed": y, "until_converged": z}
      """,
      notes=(("`jax.grad` works through `fori_loop` with a fixed trip count (it becomes a `scan`), but not "
              "through `while_loop`: reverse mode needs to know how many steps to store, and raises "
              "`ValueError`."),
             ("A Python `while` on a traced value fails under `jit` (`TracerBoolConversionError`): the "
              "condition must be computed on the device, which is what `while_loop` does.")),
      apis=("jax.lax.fori_loop", "jax.lax.while_loop"),
      module="08", also=("06",)),
    E("branches: lax.cond and lax.switch", "Control flow and transforms",
      "Pick which computation to run from a value known only at run time: one of two (cond) or one of many by "
      "index (switch).",
      r"f(x) = \begin{cases} x^2 & x < 0 \\ x & x \ge 0 \end{cases},\quad "
      r"g_k(x) \in \{\sin x,\ \cos x,\ x^2\}",
      "", "(n,), (n,) int -> (n,), (n,)", "branches(xs, ks)",
      lambda r: {"xs": np.array([-2.0, -0.5, 0.0, 1.0, 3.0]), "ks": np.array([0, 1, 2, 1, 0])},
      python="""
        def branches(xs, ks):
            piece = [x * x if x < 0 else x for x in xs]
            fns = [math.sin, math.cos, lambda x: x * x]
            pick = [fns[k](x) for k, x in zip(ks, xs)]
            return {"cond": piece, "switch": pick}
      """,
      numpy="""
        def branches(xs, ks):
            return {"cond": np.where(xs < 0, xs ** 2, xs),
                    "switch": np.choose(ks, [np.sin(xs), np.cos(xs), xs ** 2])}
      """,
      jax="""
        def branches(xs, ks):
            def piece(x):
                return jax.lax.cond(x < 0, lambda x: x ** 2, lambda x: x, x)
            def pick(k, x):
                return jax.lax.switch(k, [jnp.sin, jnp.cos, jnp.square], x)
            return {"cond": jax.vmap(piece)(xs), "switch": jax.vmap(pick)(ks, xs)}
      """,
      torch="""
        def branches(xs, ks):
            options = torch.stack([torch.sin(xs), torch.cos(xs), xs ** 2])  # every branch, then pick
            return {"cond": torch.where(xs < 0, xs ** 2, xs),
                    "switch": options[ks, torch.arange(len(xs))]}
      """,
      notes=(("Under `jit`, `if x < 0:` on a traced value raises `TracerBoolConversionError`; `lax.cond` "
              "decides on the device."),
             ("Under `vmap` with a batched predicate, `cond` becomes a `select`: both branches run for every "
              "example and the results are picked, like `where`. Only unbatched `cond` skips work."),
             "Both branches are traced and must return the same shapes and dtypes."),
      apis=("jax.lax.cond", "jax.lax.switch"),
      module="08", also=("06", "34")),
    E("round to bfloat16", "Elementwise math",
      "Keep a float32's range but only 8 bits of precision (7 stored): the rounding that happens every time a "
      "value is stored in bfloat16. `reduce_precision` simulates it without changing the dtype.",
      r"\operatorname{bf16}(x) = \operatorname{round}_\text{even}\big(x,\ 8 \text{ significant bits}\big)",
      r"$\mathrm{fl}(x)$ (the floating-point rounding of $x$)", "(n,) -> (n,) float32", "to_bf16(x)",
      lambda r: {"x": (10 * r.normal(size=8)).astype(np.float32).astype(np.float64)},
      python="""
        import struct

        def to_bf16(x):  # round float32 bits to the top 16, ties to even
            out = []
            for v in x:
                bits = struct.unpack("<I", struct.pack("<f", v))[0]
                bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
                out.append(struct.unpack("<f", struct.pack("<I", bits))[0])
            return out
      """,
      numpy="""
        def to_bf16(x):
            bits = x.astype(np.float32).view(np.uint32)
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
            return bits.astype(np.uint32).view(np.float32)
      """,
      jax="""
        def to_bf16(x):
            return jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)
            # same values as x.astype(jnp.bfloat16).astype(jnp.float32)
      """,
      torch="""
        def to_bf16(x):
            return x.to(torch.bfloat16).float()
      """,
      notes=(("bfloat16 keeps float32's 8 exponent bits (max about $3.4\\times10^{38}$) and drops mantissa "
              "bits (2–3 significant digits); float16 does the reverse (5 exponent, 10 mantissa) and overflows "
              "above 65504 — `np.float16(70000.)` is `inf`, bfloat16 rounds it to 70144."),
             ("This is why mixed precision (Module 08) stores in bfloat16 but accumulates sums in float32: "
              "adding a small number to a large bfloat16 one rounds it away."),
             "The bit trick ignores NaN handling; the library versions keep NaNs NaN."),
      apis=("jax.lax.reduce_precision",),
      tol=1e-12, module="08", also=("11", "18")),
]
