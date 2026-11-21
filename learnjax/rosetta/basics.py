"""Rosetta entries for Part 1 (Modules 01–05): arrays, linear algebra, autodiff, probability, optimizers."""
import numpy as np

from learnjax.rosetta.entry import Entry as E
from learnjax.rosetta.entry import orthogonal, probs, symmetric

ENTRIES: list[E] = [
    # ---------------------------------------------------------------- Arrays and reductions (Module 01)
    E("sum along an axis", "Arrays and reductions",
      "Collapse one axis by adding along it: summing each row of a matrix gives one number per row.",
      r"s_i = \sum_{j} x_{ij}", r"$\sum_j x_{ij}$, $x\mathbf{1}$ (times a vector of ones)",
      "(m, n) -> (m,)", "row_sums(x)", lambda r: {"x": r.normal(size=(3, 4))},
      python="""
        def row_sums(x):
            return [sum(row) for row in x]
      """,
      numpy="""
        def row_sums(x):
            return np.sum(x, axis=1)
      """,
      jax="""
        def row_sums(x):
            return jnp.sum(x, axis=1)
      """,
      torch="""
        def row_sums(x):
            return torch.sum(x, dim=1)
      """,
      einsum='"ij->i"',
      notes=("NumPy and JAX say `axis=` and `keepdims=`; PyTorch says `dim=` and `keepdim=`.",
             "Negative axes count from the end: `axis=-1` is the last axis, the usual choice for features."),
      apis=("jax.numpy.sum",),
      module="01", also=("04",)),
    E("broadcasting (outer sum)", "Arrays and reductions",
      "Combine every element of one vector with every element of another, without writing the loop: size-1 "
      "axes stretch to match.",
      r"z_{ij} = x_i + y_j", r"$x\mathbf{1}^\top + \mathbf{1}y^\top$", "(m,), (n,) -> (m, n)", "outer_sum(x, y)",
      lambda r: {"x": r.normal(size=3), "y": r.normal(size=4)},
      python="""
        def outer_sum(x, y):
            return [[xi + yj for yj in y] for xi in x]
      """,
      numpy="""
        def outer_sum(x, y):
            return x[:, None] + y[None, :]
      """,
      jax="""
        def outer_sum(x, y):
            return x[:, None] + y[None, :]
      """,
      torch="""
        def outer_sum(x, y):
            return x[:, None] + y[None, :]
      """,
      notes=(("The rules are the same in all three libraries: align shapes from the right; a size-1 axis "
             "stretches."), "`None` and `np.newaxis` / `jnp.newaxis` are the same thing: insert a size-1 axis."),
      apis=("jax.numpy.newaxis", "jax.numpy.broadcast_shapes"),
      module="01", also=("15",)),
    E("cumulative sum", "Arrays and reductions",
      "A running total: entry i is the sum of everything up to and including i.",
      r"c_i = \sum_{k \le i} x_k", "", "(n,) -> (n,)", "running_total(x)", lambda r: {"x": r.normal(size=6)},
      python="""
        def running_total(x):
            out, total = [], 0.0
            for v in x:
                total += v
                out.append(total)
            return out
      """,
      numpy="""
        def running_total(x):
            return np.cumsum(x)
      """,
      jax="""
        def running_total(x):
            return jnp.cumsum(x)
      """,
      torch="""
        def running_total(x):
            return torch.cumsum(x, dim=0)
      """,
      notes=("`torch.cumsum` needs `dim`; NumPy and JAX flatten the array if you leave `axis` out.",
             ("Any associative running operation (not just +) can be done in parallel with "
             "`jax.lax.associative_scan` (Module 36).")),
      apis=("jax.numpy.cumsum",),
      module="01", also=("36",)),
    E("scatter-add (update at indices)", "Arrays and reductions",
      "Add values into chosen positions of an array; when an index repeats, its additions accumulate.",
      r"y = x;\quad y_{i_k} \mathrel{+}= v_k \;\text{ for each } k", "", "(n,), (k,) int, (k,) -> (n,)",
      "scatter_add(x, idx, v)",
      lambda r: {"x": r.normal(size=6), "idx": np.array([0, 2, 2, 5]), "v": r.normal(size=4)},
      python="""
        def scatter_add(x, idx, v):
            y = list(x)  # copy: leave x untouched
            for i, vi in zip(idx, v):
                y[i] += vi
            return y
      """,
      numpy="""
        def scatter_add(x, idx, v):
            y = x.copy()
            np.add.at(y, idx, v)  # not y[idx] += v, which keeps only one of the repeated 2s
            return y
      """,
      jax="""
        def scatter_add(x, idx, v):
            return x.at[idx].add(v)  # returns a new array; repeated indices accumulate
      """,
      torch="""
        def scatter_add(x, idx, v):
            return x.index_add(0, idx, v)
      """,
      notes=(("JAX arrays are immutable: `x[i] = v` is an error, `x.at[i].set(v)` / `.add(v)` return a new "
             "array (under `jit`, XLA usually updates in place anyway)."),
             "NumPy's `y[idx] += v` silently drops repeated indices; `np.add.at` and JAX's `.at[].add` don't.",
             ("This is the backward pass of indexing: the gradient of `x[idx]` is a scatter-add (Module 13's "
             "embedding lookups).")),
      apis=("jax.numpy.ndarray.at",),
      module="01", also=("13", "17")),
    E("batched matrix multiply (einsum)", "Arrays and reductions",
      "Multiply a stack of matrices by another stack, pair by pair. `einsum` spells it with index letters, the "
      "same way the formula does.",
      r"C_{bij} = \sum_{k} A_{bik}\, B_{bkj}", r"$C_b = A_b B_b$ for each batch $b$",
      "(b, m, k), (b, k, n) -> (b, m, n)", "bmm(A, B)",
      lambda r: {"A": r.normal(size=(2, 3, 4)), "B": r.normal(size=(2, 4, 5))},
      python="""
        def bmm(A, B):
            C = []
            for Ab, Bb in zip(A, B):
                C.append([[sum(Ab[i][k] * Bb[k][j] for k in range(len(Bb)))
                           for j in range(len(Bb[0]))] for i in range(len(Ab))])
            return C
      """,
      numpy="""
        def bmm(A, B):
            return np.einsum("bik,bkj->bij", A, B)  # or A @ B
      """,
      jax="""
        def bmm(A, B):
            return jnp.einsum("bik,bkj->bij", A, B)  # or A @ B
      """,
      torch="""
        def bmm(A, B):
            return torch.einsum("bik,bkj->bij", A, B)  # or A @ B, torch.bmm(A, B)
      """,
      einsum='"bik,bkj->bij"',
      notes=("Read an einsum string as the formula: letters on both inputs but not the output (k) are summed.",
             "`@` treats every axis before the last two as a batch axis, in all three libraries."),
      apis=("jax.numpy.einsum", "jax.numpy.matmul"),
      module="01", also=("15", "16")),
    # ---------------------------------------------------------------- Linear algebra (Module 02)
    E("dot product", "Linear algebra",
      "Multiply matching entries and add them up: large when two vectors point the same way, zero when they "
      "are perpendicular.",
      r"x \cdot y = \sum_i x_i\, y_i", r"$\langle x, y\rangle$, $x^\top y$", "(n,), (n,) -> ()", "dot(x, y)",
      lambda r: {"x": r.normal(size=5), "y": r.normal(size=5)},
      python="""
        def dot(x, y):
            return sum(xi * yi for xi, yi in zip(x, y))
      """,
      numpy="""
        def dot(x, y):
            return np.dot(x, y)  # or x @ y
      """,
      jax="""
        def dot(x, y):
            return jnp.dot(x, y)  # or x @ y
      """,
      torch="""
        def dot(x, y):
            return torch.dot(x, y)  # 1-D only; x @ y works too
      """,
      einsum='"i,i->"',
      apis=("jax.numpy.dot",),
      module="02", also=("15",)),
    E("linear layer", "Linear algebra",
      "Multiply by a weight matrix and add a bias: the building block of every network in this course.",
      r"y = x W + b", r"$y = Wx + b$ with $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ (column vectors)",
      "(batch, d_in), (d_in, d_out), (d_out,) -> (batch, d_out)", "linear(x, W, b)",
      lambda r: {"x": r.normal(size=(5, 3)), "W": r.normal(size=(3, 4)), "b": r.normal(size=4)},
      python="""
        def linear(x, W, b):
            return [[sum(row[k] * W[k][j] for k in range(len(W))) + b[j] for j in range(len(b))]
                    for row in x]
      """,
      numpy="""
        def linear(x, W, b):
            return x @ W + b
      """,
      jax="""
        def linear(x, W, b):
            return x @ W + b  # what flax.nnx.Linear computes (kernel shape (d_in, d_out))
      """,
      torch="""
        def linear(x, W, b):
            return F.linear(x, W.T, b)  # nn.Linear stores weight as (d_out, d_in)
      """,
      einsum='"bi,io->bo"',
      notes=(("Papers usually write column vectors, $Wx$; code keeps a batch of row vectors, `x @ W`. Same math, "
             "transposed: the code's W is the paper's $W^\\top$."),
             ("Flax stores the kernel as (d_in, d_out); PyTorch's `nn.Linear.weight` is (d_out, d_in). Transpose "
             "when you port weights (Module 27 does).")),
      apis=("flax.nnx.Linear",),
      module="02", also=("09", "10", "27")),
    E("L2 norm", "Linear algebra",
      "The length of a vector: square root of the sum of squares.",
      r"\lVert x \rVert_2 = \sqrt{\textstyle\sum_i x_i^2}", r"$\lVert x\rVert$, $|x|$", "(n,) -> ()", "norm(x)",
      lambda r: {"x": r.normal(size=5)},
      python="""
        def norm(x):
            return math.sqrt(sum(v * v for v in x))
      """,
      numpy="""
        def norm(x):
            return np.linalg.norm(x)
      """,
      jax="""
        def norm(x):
            return jnp.linalg.norm(x)
      """,
      torch="""
        def norm(x):
            return torch.linalg.vector_norm(x)
      """,
      notes=(("On a matrix, `np.linalg.norm` / `jnp.linalg.norm` default to the Frobenius norm (all entries); "
             "pass `axis=-1` for one norm per row."),),
      apis=("jax.numpy.linalg.norm",),
      module="02", also=("05", "11")),
    E("cosine similarity", "Linear algebra",
      "How aligned two vectors are, ignoring length: the dot product of the two unit vectors, from -1 to 1.",
      r"\cos\theta = \frac{x \cdot y}{\lVert x\rVert\, \lVert y\rVert}", r"$\mathrm{sim}(x, y)$",
      "(n,), (n,) -> ()", "cosine(x, y)", lambda r: {"x": r.normal(size=5), "y": r.normal(size=5)},
      python="""
        def cosine(x, y):
            dot = sum(a * b for a, b in zip(x, y))
            return dot / (math.sqrt(sum(a * a for a in x)) * math.sqrt(sum(b * b for b in y)))
      """,
      numpy="""
        def cosine(x, y):
            return x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
      """,
      jax="""
        def cosine(x, y):
            return optax.losses.cosine_similarity(x, y)
      """,
      jax_by_hand="""
        def cosine(x, y):
            return x @ y / (jnp.linalg.norm(x) * jnp.linalg.norm(y))
      """,
      torch="""
        def cosine(x, y):
            return F.cosine_similarity(x, y, dim=0)
      """,
      notes=(("Library versions guard against dividing by zero with a small epsilon, so a zero vector gives 0 "
             "instead of NaN."),),
      apis=(),
      module="02", also=("30", "38")),
    E("solve a linear system", "Linear algebra",
      "Find the x that the matrix A maps to b — undo the transformation without computing the inverse.",
      r"A x = b \;\Rightarrow\; x = A^{-1} b", "", "(n, n), (n,) -> (n,)", "solve(A, b)",
      lambda r: {"A": r.normal(size=(4, 4)) + 4 * np.eye(4), "b": r.normal(size=4)},
      python="""
        def solve(A, b):  # Gaussian elimination with partial pivoting
            n = len(A)
            M = [list(row) + [bi] for row, bi in zip(A, b)]
            for c in range(n):
                p = max(range(c, n), key=lambda r: abs(M[r][c]))
                M[c], M[p] = M[p], M[c]
                for r in range(c + 1, n):
                    f = M[r][c] / M[c][c]
                    M[r] = [mr - f * mc for mr, mc in zip(M[r], M[c])]
            x = [0.0] * n
            for r in reversed(range(n)):
                x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
            return x
      """,
      numpy="""
        def solve(A, b):
            return np.linalg.solve(A, b)
      """,
      jax="""
        def solve(A, b):
            return jnp.linalg.solve(A, b)
      """,
      torch="""
        def solve(A, b):
            return torch.linalg.solve(A, b)
      """,
      notes=("Papers write $A^{-1}b$; code calls `solve`. Forming the inverse is slower and less accurate.",),
      apis=("jax.numpy.linalg.solve", "jax.numpy.linalg.inv"),
      module="02"),
    E("eigenvalues of a symmetric matrix", "Linear algebra",
      "The stretch factors along the directions a symmetric matrix only stretches, never rotates.",
      r"A = Q \Lambda Q^\top,\quad \Lambda = \mathrm{diag}(\lambda_1 \le \dots \le \lambda_n)",
      r"$A v = \lambda v$", "(n, n) -> (n,)", "eigvalsh(A)",
      lambda r: {"A": symmetric(r, 4)},
      python="""
        def eigvalsh(A, sweeps=50):  # Jacobi: rotate away off-diagonal entries until A is diagonal
            n = len(A)
            A = [list(row) for row in A]
            for _ in range(sweeps):
                if sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j) < 1e-20:
                    break
                for p in range(n):
                    for q in range(p + 1, n):
                        if A[p][q] == 0:
                            continue
                        theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
                        t = math.copysign(1, theta) / (abs(theta) + math.sqrt(theta * theta + 1))
                        c = 1 / math.sqrt(t * t + 1)
                        s = t * c
                        for k in range(n):  # columns p, q
                            akp, akq = A[k][p], A[k][q]
                            A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
                        for k in range(n):  # rows p, q
                            apk, aqk = A[p][k], A[q][k]
                            A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
            return sorted(A[i][i] for i in range(n))
      """,
      numpy="""
        def eigvalsh(A):
            return np.linalg.eigvalsh(A)  # np.linalg.eigh(A) also returns the eigenvectors
      """,
      jax="""
        def eigvalsh(A):
            return jnp.linalg.eigvalsh(A)
      """,
      torch="""
        def eigvalsh(A):
            return torch.linalg.eigvalsh(A)
      """,
      notes=("All three return eigenvalues in ascending order.",
             ("Eigenvectors are only defined up to sign, so different libraries (or GPUs) can return $v$ or "
             "$-v$; compare $Q\\Lambda Q^\\top$, not $Q$.")),
      apis=("jax.numpy.linalg.eigh",),
      module="02"),
    E("trace of a matrix", "Linear algebra",
      "Add up the diagonal. It equals the sum of the eigenvalues, so it doesn't change when you rotate the "
      "matrix into another basis; for a confusion matrix it counts the correct answers.",
      r"\mathrm{tr}(A) = \sum_{i=1}^{n} A_{ii} = \sum_{i=1}^{n} \lambda_i",
      r"$\mathrm{tr}(AB) = \mathrm{tr}(BA)$", "(n, n) -> scalar", "trace(A)",
      lambda r: {"A": r.normal(size=(4, 4))},
      python="""
        def trace(A):
            return sum(A[i][i] for i in range(len(A)))
      """,
      numpy="""
        def trace(A):
            return np.trace(A)
      """,
      jax="""
        def trace(A):
            return jnp.trace(A)
      """,
      torch="""
        def trace(A):
            return torch.trace(A)  # 2-D only; for a batch: torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)
      """,
      notes=(("Batches of matrices: `jnp.trace(A, axis1=-2, axis2=-1)` gives one trace per matrix. PyTorch's "
              "`torch.trace` takes only a single 2-D matrix."),
             ("Papers use the trace to write sums compactly: the Frobenius inner product is "
              "$\\langle A, B\\rangle = \\mathrm{tr}(A^\\top B)$, and the KL divergence between two Gaussians "
              "contains $\\mathrm{tr}(\\Sigma_2^{-1}\\Sigma_1)$.")),
      apis=("jax.numpy.trace",),
      module="09", also=("43",)),
    E("low-rank approximation (SVD)", "Linear algebra",
      "Keep only the k biggest stretches of a matrix: the best rank-k copy of it.",
      r"A_k = \sum_{i=1}^{k} \sigma_i\, u_i v_i^\top = U_k \Sigma_k V_k^\top", r"$A = U \Sigma V^\top$",
      "(m, n), k -> (m, n)", "low_rank(A, k)",
      lambda r: {"A": (orthogonal(r, 5)[:, :4] * [5.0, 3.0, 1.0, 0.5]) @ orthogonal(r, 4).T, "k": 2},
      python="""
        def low_rank(A, k, iters=100):  # power iteration finds the top singular pair; subtract it; repeat
            m, n = len(A), len(A[0])
            R = [list(row) for row in A]
            out = [[0.0] * n for _ in range(m)]
            for _ in range(k):
                v = [1.0] * n
                for _ in range(iters):
                    u = [sum(R[i][j] * v[j] for j in range(n)) for i in range(m)]
                    v = [sum(R[i][j] * u[i] for i in range(m)) for j in range(n)]
                    size = math.sqrt(sum(x * x for x in v))
                    v = [x / size for x in v]
                u = [sum(R[i][j] * v[j] for j in range(n)) for i in range(m)]  # sigma * u
                for i in range(m):
                    for j in range(n):
                        out[i][j] += u[i] * v[j]
                        R[i][j] -= u[i] * v[j]
            return out
      """,
      numpy="""
        def low_rank(A, k):
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            return (U[:, :k] * S[:k]) @ Vt[:k]
      """,
      jax="""
        def low_rank(A, k):
            U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
            return (U[:, :k] * S[:k]) @ Vt[:k]
      """,
      torch="""
        def low_rank(A, k):
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            return (U[:, :k] * S[:k]) @ Vh[:k]
      """,
      notes=(("NumPy and JAX return $V^\\top$ (called `Vt` or `vh`), not $V$; PyTorch's `torch.linalg.svd` "
             "returns `Vh` too (the older `torch.svd` returned `V`)."),
             "LoRA (Module 20) is this idea trained instead of computed: a weight update kept at rank r."),
      apis=("jax.numpy.linalg.svd", "jax.numpy.linalg.matrix_rank"),
      module="02", also=("20",)),
    # ---------------------------------------------------------------- Calculus and autodiff (Module 03)
    E("gradient", "Calculus and autodiff",
      "The direction that increases f fastest: one partial derivative per input.",
      r"\nabla f(x) = \Big(\tfrac{\partial f}{\partial x_1}, \dots, \tfrac{\partial f}{\partial x_n}\Big),"
      r"\quad f(x) = \textstyle\sum_i \tanh(x_i)^2",
      r"$\nabla_x f$, $\partial f / \partial x$, $\mathbf{g}$", "(n,) -> (n,)", "grad_f(x)",
      lambda r: {"x": r.normal(size=5)},
      python="""
        def grad_f(x, h=1e-6):  # central differences: the definition of a derivative
            def f(x):
                return sum(math.tanh(v) ** 2 for v in x)
            g = []
            for i in range(len(x)):
                up, down = list(x), list(x)
                up[i] += h
                down[i] -= h
                g.append((f(up) - f(down)) / (2 * h))
            return g
      """,
      numpy="""
        def grad_f(x):  # NumPy has no autodiff: derive it by hand
            t = np.tanh(x)
            return 2 * t * (1 - t ** 2)
      """,
      jax="""
        def f(x):
            return jnp.sum(jnp.tanh(x) ** 2)

        grad_f = jax.grad(f)
      """,
      torch="""
        def grad_f(x):
            x = x.detach().requires_grad_(True)
            torch.sum(torch.tanh(x) ** 2).backward()
            return x.grad
      """,
      notes=(("`jax.grad` turns a function into its gradient function; PyTorch records operations on tensors "
             "with `requires_grad` and fills `.grad` on `.backward()`. `torch.func.grad` is the JAX-style "
             "version."), "The function must return a scalar; for vector outputs use a Jacobian."),
      apis=("jax.grad",),
      module="03", also=("05", "09")),
    E("Jacobian", "Calculus and autodiff",
      "Every partial derivative of a vector function at once: row i is the gradient of output i.",
      r"J_{ij} = \frac{\partial f_i}{\partial x_j},\quad f(x) = \tanh(W x)",
      r"$\partial f / \partial x$, $Df(x)$, $\nabla f^\top$", "(m, n), (n,) -> (m, n)", "jac(W, x)",
      lambda r: {"W": r.normal(size=(3, 4)), "x": r.normal(size=4)},
      python="""
        def jac(W, x, h=1e-6):
            def f(x):
                return [math.tanh(sum(w * v for w, v in zip(row, x))) for row in W]
            J = [[0.0] * len(x) for _ in W]
            for j in range(len(x)):
                up, down = list(x), list(x)
                up[j] += h
                down[j] -= h
                for i, (a, b) in enumerate(zip(f(up), f(down))):
                    J[i][j] = (a - b) / (2 * h)
            return J
      """,
      numpy="""
        def jac(W, x):  # chain rule by hand: diag(1 - tanh^2) @ W
            t = np.tanh(W @ x)
            return (1 - t ** 2)[:, None] * W
      """,
      jax="""
        def jac(W, x):
            return jax.jacobian(lambda x: jnp.tanh(W @ x))(x)
      """,
      torch="""
        def jac(W, x):
            return torch.autograd.functional.jacobian(lambda x: torch.tanh(W @ x), x)
      """,
      notes=(("`jax.jacobian` is reverse mode (`jacrev`): one backward pass per output. `jax.jacfwd` is forward "
             "mode, one pass per input — cheaper when there are fewer inputs than outputs."),),
      apis=("jax.jacobian", "jax.jacfwd"),
      module="03", also=("08",)),
    # ---------------------------------------------------------------- Probability and information (Module 04)
    E("softmax", "Probability and information",
      "Turn any scores into probabilities: exponentiate (all positive) and divide by the total (sum to 1). "
      "Bigger scores get exponentially more weight.",
      r"\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}",
      r"$\sigma(z)_i$, $p_i \propto e^{z_i}$, $\mathrm{softmax}(QK^\top/\sqrt{d})$ in attention",
      "(..., n) -> (..., n), along the last axis", "softmax(x)", lambda r: {"x": 3 * r.normal(size=(3, 5))},
      python="""
        def softmax(x):
            out = []
            for row in x:
                m = max(row)  # subtract the max so exp never overflows
                e = [math.exp(v - m) for v in row]
                total = sum(e)
                out.append([v / total for v in e])
            return out
      """,
      numpy="""
        def softmax(x):
            e = np.exp(x - x.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
      """,
      jax="""
        def softmax(x):
            return jax.nn.softmax(x, axis=-1)
      """,
      jax_by_hand="""
        def softmax(x):
            e = jnp.exp(x - x.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
      """,
      torch="""
        def softmax(x):
            return torch.softmax(x, dim=-1)
      """,
      notes=(("Subtracting the max changes nothing mathematically ($e^{x-m}/\\sum e^{x-m}$ cancels) but keeps "
             "`exp` from overflowing: the course's first numerical-stability trick."),
             "Papers' $\\sigma$ can mean softmax or the logistic sigmoid; check the shape."),
      apis=("jax.nn.softmax",),
      module="04", also=("10", "15", "16")),
    E("log-sum-exp", "Probability and information",
      "A smooth maximum, and the normalizer in log space: what you subtract from scores to get log "
      "probabilities.",
      r"\mathrm{LSE}(x) = \log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m},\; m = \max_i x_i",
      r"$\log Z$ (the log partition function)", "(..., n) -> (...)", "lse(x)",
      lambda r: {"x": 40 * r.normal(size=(3, 5))},
      python="""
        def lse(x):
            out = []
            for row in x:
                m = max(row)
                out.append(m + math.log(sum(math.exp(v - m) for v in row)))
            return out
      """,
      numpy="""
        def lse(x):
            m = x.max(axis=-1)
            return m + np.log(np.exp(x - m[..., None]).sum(axis=-1))
      """,
      jax="""
        def lse(x):
            return jax.nn.logsumexp(x, axis=-1)
      """,
      torch="""
        def lse(x):
            return torch.logsumexp(x, dim=-1)
      """,
      notes=(("The test inputs are large (scale 40) on purpose: `log(sum(exp(x)))` overflows to inf in float32; "
             "every version here subtracts the max first."), "`log_softmax(x) = x - LSE(x)`."),
      apis=("jax.nn.logsumexp",),
      module="04", also=("14", "35")),
    E("cross-entropy from logits", "Probability and information",
      "How surprised the model is by the right answer, averaged: minus the log probability it gave the true "
      "class. Training a language model minimizes exactly this.",
      r"\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \log \mathrm{softmax}(z_n)_{y_n}",
      r"$H(p, q) = -\sum_i p_i \log q_i$ (with one-hot $p$); negative log-likelihood (NLL)",
      "(N, C) logits, (N,) int labels -> ()", "cross_entropy(logits, y)",
      lambda r: {"logits": 2 * r.normal(size=(5, 4)), "y": np.array([0, 3, 1, 1, 2])},
      python="""
        def cross_entropy(logits, y):
            total = 0.0
            for z, label in zip(logits, y):
                m = max(z)
                log_z = m + math.log(sum(math.exp(v - m) for v in z))
                total += log_z - z[label]  # -log softmax(z)[label]
            return total / len(y)
      """,
      numpy="""
        def cross_entropy(logits, y):
            m = logits.max(axis=-1, keepdims=True)
            logp = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
            return -logp[np.arange(len(y)), y].mean()
      """,
      jax="""
        def cross_entropy(logits, y):
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
      """,
      jax_by_hand="""
        def cross_entropy(logits, y):
            logp = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.take_along_axis(logp, y[:, None], axis=-1).mean()
      """,
      torch="""
        def cross_entropy(logits, y):
            return F.cross_entropy(logits, y)  # takes logits, averages by default
      """,
      notes=("All of these take logits, not probabilities: never softmax first.",
             ("`F.cross_entropy` averages by default (`reduction=\"mean\"`); optax returns one loss per example, "
             "so you call `.mean()`."), "In nats (natural log). Divide by $\\ln 2$ for bits per token."),
      apis=("optax.softmax_cross_entropy_with_integer_labels", "jax.numpy.take_along_axis"),
      module="04", also=("10", "14", "16")),
    E("KL divergence", "Probability and information",
      "The extra surprise from using q when the truth is p: zero only when they match, and not symmetric.",
      r"D_{\mathrm{KL}}(p \,\Vert\, q) = \sum_i p_i \log \frac{p_i}{q_i}", r"$\mathrm{KL}[p \,\Vert\, q]$",
      "(..., n), (..., n) -> (...)", "kl(p, q)", lambda r: {"p": probs(r, (3, 5)), "q": probs(r, (3, 5))},
      python="""
        def kl(p, q):
            return [sum(pi * math.log(pi / qi) for pi, qi in zip(pr, qr)) for pr, qr in zip(p, q)]
      """,
      numpy="""
        def kl(p, q):
            return np.sum(p * (np.log(p) - np.log(q)), axis=-1)
      """,
      jax="""
        def kl(p, q):
            return optax.losses.kl_divergence(jnp.log(q), p)  # (log of the approximation, target)
      """,
      jax_by_hand="""
        def kl(p, q):
            return jnp.sum(p * (jnp.log(p) - jnp.log(q)), axis=-1)
      """,
      torch="""
        def kl(p, q):
            return F.kl_div(q.log(), p, reduction="none").sum(-1)  # also (log q, p)
      """,
      notes=(("Argument-order trap: the paper writes $D(p\\Vert q)$, but `optax.losses.kl_divergence` and "
             "`F.kl_div` both take the *log* of q first and p second."),
             ("`F.kl_div`'s default `reduction=\"mean\"` averages over every element, not per example; use "
             "`\"batchmean\"` or sum yourself.")),
      apis=(),
      module="04", also=("33", "35")),
    # ---------------------------------------------------------------- Optimization (Module 05)
    E("Adam update", "Optimization",
      "Gradient descent with a running average of the gradient (momentum) divided by a running average of its "
      "square (a per-parameter step size), both corrected for starting at zero.",
      r"m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad "
      r"\theta_t = \theta_{t-1} - \eta\,\frac{m_t / (1-\beta_1^t)}{\sqrt{v_t / (1-\beta_2^t)} + \epsilon}",
      r"$\hat m_t, \hat v_t$ (bias-corrected); $\alpha$ for the learning rate",
      "params (n,), gradients (steps, n) -> params (n,)", "adam(p, grads)",
      lambda r: {"p": r.normal(size=4), "grads": r.normal(size=(3, 4))},
      python="""
        def adam(p, grads, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
            p, m, v = list(p), [0.0] * len(p), [0.0] * len(p)
            for t, g in enumerate(grads, start=1):
                for i in range(len(p)):
                    m[i] = b1 * m[i] + (1 - b1) * g[i]
                    v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
                    m_hat, v_hat = m[i] / (1 - b1 ** t), v[i] / (1 - b2 ** t)
                    p[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)
            return p
      """,
      numpy="""
        def adam(p, grads, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
            m, v = np.zeros_like(p), np.zeros_like(p)
            for t, g in enumerate(grads, start=1):
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g ** 2
                p = p - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            return p
      """,
      jax="""
        def adam(p, grads, lr=0.1):
            opt = optax.adam(lr)
            state = opt.init(p)
            for g in grads:
                updates, state = opt.update(g, state, p)
                p = optax.apply_updates(p, updates)
            return p
      """,
      jax_by_hand="""
        def adam(p, grads, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
            m, v = jnp.zeros_like(p), jnp.zeros_like(p)
            for t, g in enumerate(grads, start=1):
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g ** 2
                p = p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
            return p
      """,
      torch="""
        def adam(p, grads, lr=0.1):
            p = p.clone().requires_grad_(True)
            opt = torch.optim.Adam([p], lr=lr)
            for g in grads:
                p.grad = g.clone()
                opt.step()
            return p.detach()
      """,
      notes=(("optax separates the rule from the state: `opt.update` returns updates and new state, and you "
             "apply them; PyTorch's optimizer owns the state and changes parameters in place."),
             ("The update is the *negative* step in optax (`apply_updates` adds it); a hand-written version "
             "subtracts."), "AdamW adds `- lr * weight_decay * p` outside the fraction (Module 05 §8)."),
      apis=("optax.adam", "optax.apply_updates"),
      module="05", also=("10", "11", "16")),
    E("clip by global norm", "Optimization",
      "If all the gradients together are too long, shrink them all by the same factor so their combined length "
      "is c. Direction is kept; only the size changes.",
      r"g_\ell \leftarrow g_\ell \cdot \min\!\Big(1, \frac{c}{\lVert g \rVert}\Big),\quad "
      r"\lVert g \rVert = \sqrt{\textstyle\sum_\ell \lVert g_\ell \rVert_2^2}", "",
      "pytree of arrays, c -> same pytree", "clip_global(grads, c)",
      lambda r: {"grads": {"w": 3 * r.normal(size=(3, 4)), "b": 3 * r.normal(size=4)}, "c": 1.0},
      python="""
        def clip_global(grads, c):
            def squares(x):
                return sum(squares(v) for v in x) if isinstance(x, list) else x * x
            def scale(x, s):
                return [scale(v, s) for v in x] if isinstance(x, list) else x * s
            total = math.sqrt(sum(squares(g) for g in grads.values()))
            s = min(1.0, c / total)
            return {name: scale(g, s) for name, g in grads.items()}
      """,
      numpy="""
        def clip_global(grads, c):
            total = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
            s = min(1.0, c / total)
            return {name: g * s for name, g in grads.items()}
      """,
      jax="""
        def clip_global(grads, c):
            clip = optax.clip_by_global_norm(c)
            clipped, _ = clip.update(grads, clip.init(grads))
            return clipped
      """,
      jax_by_hand="""
        def clip_global(grads, c):
            s = jnp.minimum(1.0, c / optax.global_norm(grads))
            return jax.tree.map(lambda g: g * s, grads)
      """,
      torch="""
        def clip_global(grads, c):
            params = {name: torch.zeros_like(g, requires_grad=True) for name, g in grads.items()}
            for name, p in params.items():
                p.grad = grads[name].clone()
            torch.nn.utils.clip_grad_norm_(list(params.values()), c)  # rescales .grad in place
            return {name: p.grad for name, p in params.items()}
      """,
      notes=(("The norm is over *all* parameters together, not each one; per-tensor clipping is a different "
             "rule."), ("In optax it is one link in a chain: `optax.chain(optax.clip_by_global_norm(1.0), "
             "optax.adamw(...))`. PyTorch clips `.grad` in place between `backward()` and `step()`."),
             "PyTorch adds 1e-6 to the norm before dividing, so its result is a hair smaller."),
      apis=("optax.clip_by_global_norm", "optax.chain"),
      module="05", also=("02", "11", "16")),
]
