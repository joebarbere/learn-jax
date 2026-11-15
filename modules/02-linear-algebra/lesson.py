import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="02 · Linear algebra")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt

    import learnjax as lj

    ex = lj.load_exercises(mo.notebook_dir())
    return ex, jax, jnp, lj, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 02 · Linear algebra as geometry

    **Goal:** see what a matrix *does*. A neural network layer is "multiply by a matrix, then bend";
    an embedding is a vector whose *direction* carries meaning; LoRA (Module 20) is a low-rank matrix.
    All of that is easier with pictures than with index notation.

    **You will:** treat vectors as arrows and measure similarity with the dot product · watch a 2×2
    matrix bend the plane (sliders!) · see why composition is multiplication and the determinant is an
    area · solve linear systems · find the directions a matrix only stretches (eigenvectors) · take a
    matrix apart with the SVD · compress an image with a low-rank approximation.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from**

    The ideas here are old. Matrices were developed as a subject in the 1800s, and the singular value
    decomposition was found independently by Beltrami (1873) and Jordan (1874); Golub and Kahan made it
    practical on computers in 1965. Deep learning is, underneath, a very large amount of matrix
    multiplication. The low-rank idea came back in 2020–21: Aghajanyan, Zettlemoyer & Gupta showed fine-
    tuning lives in a small subspace, and Hu et al. turned that into LoRA (2021), which you'll build in
    Module 20.

    *Timeline: Module 00b.*
    """), kind="info")
    return


@app.cell
def _(np, plt):
    def draw_transform(ax, m, title=""):
        """Plot the unit grid, unit circle and basis vectors before (faint) and after transform m."""
        m = np.asarray(m, dtype=float)
        t = np.linspace(0, 2 * np.pi, 200)
        circle = np.stack([np.cos(t), np.sin(t)])
        for s in np.linspace(-1, 1, 5):
            line_v = np.stack([np.full(50, s), np.linspace(-1, 1, 50)])
            line_h = line_v[::-1]
            for ln in (line_v, line_h):
                ax.plot(*ln, color="0.85", lw=0.8)
                ax.plot(*(m @ ln), color="tab:blue", lw=0.8, alpha=0.6)
        ax.plot(*circle, color="0.7", lw=1, ls="--")
        ax.plot(*(m @ circle), color="tab:orange", lw=2)
        for vec, col in (((1, 0), "tab:red"), ((0, 1), "tab:green")):
            img = m @ np.array(vec, float)
            ax.annotate("", xy=img, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=2))
        ax.set_aspect("equal")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.axvline(0, color="0.5", lw=0.5)
        ax.set_title(title, fontsize=9)

    def small_fig(ncols=1, w=3.2, h=3.2):
        fig, axes = plt.subplots(1, ncols, figsize=(w * ncols, h))
        return fig, axes

    return draw_transform, small_fig


@app.cell
def _(mo, np, small_fig):
    _a, _b = np.array([2.0, 1.0]), np.array([-0.5, 1.5])
    _fig, _ax = small_fig(w=3.6, h=3.4)
    for _v, _c, _lbl in ((_a, "tab:blue", "a"), (_b, "tab:orange", "b"), (_a + _b, "tab:green", "a + b"),
                         (-0.5 * _a, "tab:purple", "-0.5 a")):
        _ax.annotate("", xy=_v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=_c, lw=2))
        _ax.text(*(_v * 1.08), _lbl, color=_c, fontsize=9)
    _ax.plot([_a[0], _a[0] + _b[0]], [_a[1], _a[1] + _b[1]], ls=":", color="0.5")
    _ax.set_xlim(-1.5, 2.5)
    _ax.set_ylim(-1, 3)
    _ax.set_aspect("equal")
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    mo.hstack([mo.md(r"""
    ## 1 · Vectors are arrows

    A vector like $a = (2, 1)$ is an arrow from the origin. Two operations do almost everything:

    * **Add**: $a + b$ — walk along $a$, then along $b$ (tip to tail).
    * **Scale**: $c\,a$ — stretch by $c$; a negative $c$ flips it.

    In ML the arrows have hundreds of dimensions instead of two — the deepseek model you trained
    represents each token as a 384-dimensional vector — but the geometry is the same, and 2-D
    pictures carry the intuition.
    """), _fig], widths=[3, 2])
    return


@app.cell
def _(jnp, mo):
    mo.md(rf"""
    ## 2 · The dot product measures alignment

    $a \cdot b = \sum_i a_i b_i = |a|\,|b|\cos\theta$, where $\theta$ is the angle between them. Divide
    out the lengths and you get **cosine similarity**, $\cos\theta \in [-1, 1]$: 1 for the same direction,
    0 for perpendicular, −1 for opposite.

    | | value |
    |---|---|
    | $(1, 0)\cdot(0, 1)$ | `{jnp.dot(jnp.array([1., 0.]), jnp.array([0., 1.]))}` — perpendicular |
    | $(1, 2)\cdot(2, 4)$ | `{jnp.dot(jnp.array([1., 2.]), jnp.array([2., 4.]))}` — same direction, big |
    | length of $(3, 4)$, [`jnp.linalg.norm`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.norm.html#jax.numpy.linalg.norm) | `{jnp.linalg.norm(jnp.array([3., 4.]))}` — length, $\sqrt{{a \cdot a}}$ |

    This is *the* similarity measure for embeddings: two words, images or documents are "close" when
    their vectors point the same way. Attention (Module 15) is a table of dot products between every
    pair of tokens.
    """)
    return


@app.cell
def _(ex, jnp, lj, mo, plt):
    _words = ["king", "queen", "apple", "banana", "truck"]
    # Hand-made 4-d "embeddings": axes are (royalty, person, fruit, vehicle).
    _emb = jnp.array([[0.9, 0.8, 0.0, 0.1], [0.85, 0.9, 0.05, 0.0], [0.0, 0.05, 0.95, 0.1],
                      [0.05, 0.0, 0.9, 0.0], [0.1, 0.1, 0.0, 0.95]])
    _cs = lj.attempt(ex.cosine_similarity, _emb[:, None, :], _emb[None, :, :])
    if _cs.ok:
        _fig, _ax = plt.subplots(figsize=(3.8, 3.2))
        _im = _ax.imshow(_cs.value, vmin=-1, vmax=1, cmap="RdBu_r")
        _ax.set_xticks(range(5), _words, rotation=45, fontsize=8)
        _ax.set_yticks(range(5), _words, fontsize=8)
        for _i in range(5):
            for _j in range(5):
                _v = float(_cs.value[_i, _j])
                _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(_v) > 0.6 else "#1f2430")
        _ax.set_title("cosine similarity (your function)", fontsize=9)
        _fig.colorbar(_im, ax=_ax, shrink=0.8)
        _fig.tight_layout()
        _view = _fig
    else:
        _view = _cs.view(mo)
    mo.vstack([mo.md("**Your turn — `cosine_similarity`:** five hand-made embeddings whose axes mean "
                     "(royalty, person, fruit, vehicle). Broadcasting `(5, 1, 4)` against `(1, 5, 4)` "
                     "gives all 25 pairs at once."), _view])
    return


@app.cell
def _(mo):
    angle = mo.ui.slider(-180, 180, value=30, step=5, label="rotate (degrees)", show_value=True)
    sx = mo.ui.slider(-2.0, 2.0, value=1.5, step=0.1, label="scale x", show_value=True)
    sy = mo.ui.slider(-2.0, 2.0, value=0.7, step=0.1, label="scale y", show_value=True)
    shear = mo.ui.slider(-1.5, 1.5, value=0.0, step=0.1, label="shear", show_value=True)
    mo.vstack([mo.md(r"""
    ## 3 · A matrix is a transformation of space

    A 2×2 matrix $M$ sends every point $p$ to $Mp$. It's completely described by where it sends the two
    basis arrows $(1,0)$ and $(0,1)$ — those images are $M$'s **columns** (red and green below). Straight
    lines stay straight and the origin stays put: that's what *linear* means.

    Build $M$ = rotate · shear · scale with the sliders. Grey is before, blue grid and orange circle after.
    """), mo.hstack([angle, sx]), mo.hstack([sy, shear])])
    return angle, shear, sx, sy


@app.cell
def _(angle, jnp, shear, sx, sy):
    _t = jnp.deg2rad(angle.value)
    R = jnp.array([[jnp.cos(_t), -jnp.sin(_t)], [jnp.sin(_t), jnp.cos(_t)]])
    H = jnp.array([[1.0, shear.value], [0.0, 1.0]])
    S = jnp.diag(jnp.array([sx.value, sy.value]))
    M = R @ H @ S
    return M, R, S


@app.cell
def _(M, draw_transform, jnp, mo, np, small_fig):
    _fig, _ax = small_fig(w=4, h=4)
    draw_transform(_ax, M, "M = rotate · shear · scale")
    _fig.tight_layout()
    _det = float(jnp.linalg.det(M))
    _m = np.asarray(M)
    _note = ("negative: the plane is **flipped** (red→green now turns clockwise)" if _det < -1e-6 else
             "zero: the plane is **squashed onto a line** — information is lost, no inverse exists"
             if abs(_det) < 1e-6 else "positive: orientation preserved")
    mo.hstack([_fig, mo.md(f"""
    $M = \\begin{{bmatrix}} {_m[0, 0]:.2f} & {_m[0, 1]:.2f} \\\\ {_m[1, 0]:.2f} & {_m[1, 1]:.2f} \\end{{bmatrix}}$

    **Determinant** $\\det M$ = `{_det:.3f}`

    The determinant is the factor by which $M$ scales **area**: the unit square (area 1) becomes a
    parallelogram of area $|\\det M|$. Its sign says whether orientation flips. Here it's {_note}.

    Rotations have $\\det = 1$ (they don't change area); a scale by $(s_x, s_y)$ has
    $\\det = s_x s_y$. Try setting *scale y* to 0.
    """)], widths=[2, 3])
    return


@app.cell
def _(R, S, draw_transform, jnp, mo, small_fig):
    _fig, _axes = small_fig(ncols=2, w=3.2, h=3.2)
    draw_transform(_axes[0], R @ S, "R @ S: scale first, then rotate")
    draw_transform(_axes[1], S @ R, "S @ R: rotate first, then scale")
    _fig.tight_layout()
    _same = bool(jnp.allclose(R @ S, S @ R, atol=1e-5))
    mo.vstack([mo.md(rf"""
    ## 4 · Composition is multiplication — and order matters

    Applying $S$ and then $R$ to a point is $R(Sp) = (RS)p$: one combined matrix. A deep network is a
    long chain of these (with a nonlinearity between each, otherwise the whole chain would collapse into
    a single matrix). Matrix products read **right to left**, and in general $RS \neq SR$ — with your
    current sliders they are {"equal (try unequal scales)" if _same else "different"}:
    """), _fig])
    return


@app.cell
def _(jax, jnp, mo):
    _A = jax.random.normal(jax.random.key(0), (512, 512))
    _x_true = jax.random.normal(jax.random.key(1), (512,))
    _b = _A @ _x_true
    _x_solve = jnp.linalg.solve(_A, _b)
    _x_inv = jnp.linalg.inv(_A) @ _b
    _sing = jnp.array([[1.0, 2.0], [2.0, 4.0]])
    mo.md(rf"""
    ## 5 · Undoing a transformation: inverses and `solve`

    If $Mx = b$, then $x = M^{{-1}}b$ — *if* $M$ is invertible ($\det M \ne 0$). For a random 512×512
    system on the GPU:

    | method | max error in $x$ |
    |---|---|
    | [`jnp.linalg.solve(A, b)`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.solve.html#jax.numpy.linalg.solve) | `{float(jnp.abs(_x_solve - _x_true).max()):.2e}` |
    | [`jnp.linalg.inv(A) @ b`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.inv.html#jax.numpy.linalg.inv) | `{float(jnp.abs(_x_inv - _x_true).max()):.2e}` |

    Both work here, but prefer `solve`: it's cheaper and usually more accurate, because forming the
    inverse explicitly amplifies rounding error. A singular matrix like $\begin{{bmatrix}}1&2\\2&4\end{{bmatrix}}$
    (second row = 2 × first, $\det$ = `{abs(float(jnp.linalg.det(_sing))):.1f}`) squashes the plane onto a
    line; many inputs map to the same output, so there's nothing to invert.
    """)
    return


@app.cell
def _(draw_transform, jnp, mo, np, small_fig):
    _Msym = jnp.array([[2.0, 0.8], [0.8, 1.0]])
    _vals, _vecs = jnp.linalg.eigh(_Msym)
    _fig, _ax = small_fig(w=4, h=4)
    draw_transform(_ax, _Msym, "eigenvectors only stretch")
    for _k in range(2):
        _v = np.asarray(_vecs[:, _k])
        _ax.annotate("", xy=_v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1.5, ls="--"))
        _ax.annotate("", xy=float(_vals[_k]) * _v, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
    _fig.tight_layout()
    mo.hstack([_fig, mo.md(rf"""
    ## 6 · Eigenvectors: directions that only stretch

    Most arrows change direction under $M$. An **eigenvector** $v$ doesn't: $Mv = \lambda v$ — it's only
    stretched by its **eigenvalue** $\lambda$.

    For the symmetric matrix $M = \begin{{bmatrix}}2 & 0.8\\0.8 & 1\end{{bmatrix}}$, [`jnp.linalg.eigh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.eigh.html#jax.numpy.linalg.eigh) gives
    $\lambda$ = `{', '.join(f'{float(_x):.3f}' for _x in _vals)}`. The dashed black arrows are the eigenvectors;
    the solid ones are where $M$ sends them — same line, scaled by $\lambda$. They line up with the long
    and short axes of the orange ellipse.

    Symmetric matrices (like covariance matrices, or $A^\top A$) always have perpendicular eigenvectors
    and real eigenvalues. Eigenvalues also decide whether repeated multiplication explodes ($|\lambda|>1$)
    or dies out ($|\lambda|<1$) — the root of exploding/vanishing gradients in deep networks.
    """)], widths=[2, 3])
    return


@app.cell
def _(M, draw_transform, jnp, mo, np, small_fig):
    _U, _s, _Vt = jnp.linalg.svd(M)
    _fig, _axes = small_fig(ncols=3, w=2.6, h=2.6)
    draw_transform(_axes[0], _Vt, "1. rotate (Vᵀ)")
    draw_transform(_axes[1], jnp.diag(_s) @ _Vt, "2. stretch (Σ)")
    draw_transform(_axes[2], _U @ jnp.diag(_s) @ _Vt, "3. rotate (U) = M")
    _fig.tight_layout()
    mo.vstack([mo.md(rf"""
    ## 7 · The SVD: every matrix is rotate · stretch · rotate

    Any matrix — not just symmetric ones — factors as $M = U\Sigma V^\top$: a rotation (or reflection)
    $V^\top$, a stretch along the axes by the **singular values** $\Sigma$, and another rotation $U$.
    Geometrically, $M$ turns the unit circle into an ellipse whose semi-axes are the singular values.

    For the matrix from your sliders in Section 3: singular values = `{', '.join(f'{float(_x):.3f}' for _x in _s)}`
    (the ellipse's half-widths), and their product equals $|\det M|$ = `{abs(float(jnp.linalg.det(M))):.3f}`.
    """), _fig])
    return


@app.cell
def _(jnp, plt):
    import matplotlib.cbook as _cbook

    _rgb = plt.imread(_cbook.get_sample_data("grace_hopper.jpg"))
    photo = jnp.asarray(_rgb[..., :3].mean(-1) / 255.0, jnp.float32)  # grayscale, 600 x 512
    photo_svd = jnp.linalg.svd(photo, full_matrices=False)
    return photo, photo_svd


@app.cell
def _(mo):
    rank = mo.ui.slider(steps=[1, 2, 5, 10, 20, 40, 80, 160, 512], value=20, show_value=True,
                        label="rank k")
    mo.vstack([mo.md(r"""
    ## 8 · Low-rank approximation: keep the biggest stretches

    The SVD sorts the stretches from largest to smallest. Keep only the top $k$ and you get the **best
    possible rank-$k$ approximation** of the matrix (Eckart–Young theorem). A rank-$k$ matrix of size
    $m \times n$ is stored as $U_k$ ($m \times k$), $k$ singular values and $V_k^\top$ ($k \times n$):
    $k(m + n + 1)$ numbers instead of $mn$.

    Here the matrix is a 600×512 grayscale photo (Grace Hopper, from matplotlib's sample data):
    """), rank])
    return (rank,)


@app.cell
def _(jnp, mo, photo, photo_svd, plt, rank):
    _U, _s, _Vt = photo_svd
    _k = rank.value
    _approx = (_U[:, :_k] * _s[:_k]) @ _Vt[:_k]
    _m, _n = photo.shape
    _err = jnp.cumsum(_s[::-1] ** 2)[::-1]  # residual energy if we keep the first k
    _rel = jnp.sqrt(jnp.concatenate([_err[1:], jnp.zeros(1)]) / _err[0])
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    _axes[0].imshow(photo, cmap="gray")
    _axes[0].set_title("original", fontsize=9)
    _axes[1].imshow(jnp.clip(_approx, 0, 1), cmap="gray")
    _axes[1].set_title(f"rank {_k}", fontsize=9)
    for _a in _axes[:2]:
        _a.axis("off")
    _axes[2].loglog(jnp.arange(1, _s.shape[0] + 1), _rel, lw=1.5)
    _axes[2].axvline(_k, color="tab:red", ls="--")
    _axes[2].set_xlabel("rank k")
    _axes[2].set_ylabel("relative error ‖M − Mₖ‖ / ‖M‖")
    _axes[2].grid(alpha=0.3, which="both")
    _fig.tight_layout()
    _store = _k * (_m + _n + 1)
    mo.vstack([_fig, mo.md(f"""
    Rank {_k}: **{_store:,} numbers vs {_m * _n:,}** ({100 * _store / (_m * _n):.1f}% of the storage),
    relative error **{float(_rel[_k - 1]):.3f}**. The singular values fall off fast, so a little rank
    captures most of the picture.

    **Why this matters for LLMs (Module 20):** fine-tuning a 7B model changes weight matrices of size
    4096×4096. LoRA bets that the *change* $\\Delta W$ is close to low-rank, and learns it directly as
    $\\Delta W = BA$ with $B$: 4096×r, $A$: r×4096 and r ≈ 8–64 — about 0.4–3% of the parameters.
    """)])
    return


@app.cell
def _(jax, jnp, mo):
    _v = jnp.array([3.0, -4.0])
    _W = jax.random.normal(jax.random.key(3), (64, 32))
    _spec = float(jnp.linalg.svd(_W, compute_uv=False)[0])
    mo.md(rf"""
    ## 9 · Norms: how big is it?

    | norm | for $v = (3, -4)$ / a 64×32 random $W$ | meaning |
    |---|---|---|
    | L2 $\sqrt{{\sum v_i^2}}$ | `{float(jnp.linalg.norm(_v)):.2f}` | ordinary length |
    | L1 $\sum_i \lvert v_i \rvert$ | `{float(jnp.linalg.norm(_v, 1)):.2f}` | "taxicab" length; pushes toward sparsity as a penalty |
    | L∞ $\max_i \lvert v_i \rvert$ | `{float(jnp.linalg.norm(_v, jnp.inf)):.2f}` | the largest entry |
    | Frobenius $\sqrt{{\sum W_{{ij}}^2}}$ | `{float(jnp.linalg.norm(_W)):.2f}` | L2 of the matrix flattened |
    | spectral (largest singular value) | `{_spec:.2f}` | most a matrix can stretch any vector |

    You'll see norms constantly: weight decay penalizes the L2 norm of weights, and the deepseek
    training script clips the **global gradient norm** to 1.0 (`optax.clip_by_global_norm(1.0)`) so one
    bad batch can't take a huge step.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 · Your turn

    Open `modules/02-linear-algebra/exercises.py`. You already met `cosine_similarity` above; the rest
    build the pictures in this lesson from scratch. `make test N=02` grades them.
    """)
    return


@app.cell
def _(mo):
    turn = mo.ui.slider(0, 360, value=60, step=15, show_value=True, label="rotate the house (degrees)")
    turn
    return (turn,)


@app.cell
def _(ex, jnp, lj, mo, np, plt, turn):
    _house = jnp.array([[-1, -1], [1, -1], [1, 0.5], [0, 1.5], [-1, 0.5], [-1, -1]], jnp.float32)
    _r = lj.attempt(ex.rotation_matrix, float(jnp.deg2rad(turn.value)))
    _items = [mo.md("**`rotation_matrix` + `transform_points`**:")]
    if _r.ok:
        _moved = lj.attempt(ex.transform_points, _r.value, _house)
        if _moved.ok:
            _fig, _ax = plt.subplots(figsize=(3.4, 3.4))
            _ax.fill(*np.asarray(_house).T, alpha=0.2, color="0.5", label="original")
            _ax.fill(*np.asarray(_moved.value).T, alpha=0.5, color="tab:blue", label="rotated")
            _ax.set_xlim(-2, 2)
            _ax.set_ylim(-2, 2)
            _ax.set_aspect("equal")
            _ax.legend(fontsize=7, loc="lower right")
            _fig.tight_layout()
            _items.append(_fig)
        else:
            _items.append(_moved.view(mo))
    else:
        _items.append(_r.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(ex, jax, jnp, lj, mo, np, photo):
    _v, _u = jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 1.0, 0.0])
    _p = lj.attempt(ex.project, _v, _u)
    _items = [mo.md("**`project((1, 2, 3)` onto `(1, 1, 0))`** — expect (1.5, 1.5, 0):"), _p.view(mo)]
    if _p.ok:
        _items.append(mo.md(f"residual · u = `{float(jnp.dot(_v - _p.value, _u)):.2e}` (should be ~0)"))
    _la = lj.attempt(ex.low_rank_approx, photo, 20)
    _items.append(mo.md("**`low_rank_approx(photo, 20)`**:"))
    if _la.ok:
        _rel = float(jnp.linalg.norm(photo - _la.value) / jnp.linalg.norm(photo))
        _items.append(mo.md(f"shape `{_la.value.shape}`, relative error `{_rel:.3f}` (Section 8 shows the reference)"))
    else:
        _items.append(_la.view(mo))
    _A = jax.random.normal(jax.random.key(4), (6, 4))
    _q = lj.attempt(ex.gram_schmidt, _A)
    _items.append(mo.md("**`gram_schmidt`** on a random 6×4 matrix — $Q^\\top Q$ should be the identity:"))
    _items.append(mo.md(f"```\n{np.round(np.asarray(_q.value.T @ _q.value), 4)}\n```") if _q.ok else _q.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(2))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * 3Blue1Brown, [Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) — the
      video series this lesson's pictures are based on; watch chapters 1–4, 6, 7 and 14
    * Gilbert Strang, [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
      lectures, and his short [Linear Algebra for Everyone](https://math.mit.edu/~gs/everyone/)
    * Jeremy Kun, [Singular Value Decomposition Part 1](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/)
    * Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) —
      skim the introduction now, read it properly in Module 20

    **Next:** Module 03 — derivatives and automatic differentiation, built from scratch.
    """)
    return


if __name__ == "__main__":
    app.run()
