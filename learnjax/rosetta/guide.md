## Reading paper notation

Papers write one example at a time, as column vectors; code works on a batch of rows. Most of the work of
turning a formula into code is putting that batch axis back and picking the axis each sum runs over.

| paper | means | code |
|---|---|---|
| $x \in \mathbb{R}^{d}$, $W \in \mathbb{R}^{m \times d}$, $Wx$ | one example, a column vector | a batch `x` of shape `(B, d)`: `x @ W.T`, or store `W` as `(d, m)` (Flax) and write `x @ W` |
| $x^{(i)}$, $y^{(i)}$ | the $i$-th training example | row `i` of the batch axis |
| $h_t$, $x_{1:T}$ | step $t$ of a sequence | the time axis; a recurrence over it is `lax.scan` |
| $W^{(l)}$, $W_\ell$ | layer $\ell$'s weights | a list of layers, or a stacked leading axis you `scan` over |
| $\sum_j A_{ij} B_{jk}$ | sum over the repeated index | `einsum("ij,jk->ik", A, B)`; every repeated index is summed |
| $a^\top b$, $\langle a, b \rangle$ | dot product | `a @ b`, `jnp.dot`, or `(a * b).sum(-1)` for batches |
| $a \odot b$ | elementwise (Hadamard) product | `a * b` |
| $a \otimes b$, $a b^\top$ | outer product (or Kronecker, by context) | `a[:, None] * b[None, :]`, `jnp.outer`, `jnp.kron` |
| $\lVert x \rVert$, $\lVert x \rVert_2$ | Euclidean length | `jnp.linalg.norm(x, axis=-1)` |
| $\mathbb{E}_{x \sim p}[f(x)]$ | average under $p$ | sample, then `.mean()` over the sample axis |
| $\mathbb{1}[c]$, $[c]$ | 1 if $c$ holds, else 0 | `c.astype(jnp.float32)`, or `jnp.where(c, 1.0, 0.0)` |
| $\operatorname{softmax}(z)_i$ | normalize over $i$ | `jax.nn.softmax(z, axis=...)`: check which axis $i$ is (keys, for attention) |
| $\log$ | natural log | `jnp.log`; papers that mean base 2 say "bits" |
| $\sigma(x)$ | the sigmoid, *or* a standard deviation | read the context: `jax.nn.sigmoid`, or `.std()` |
| $\mathcal{N}(\mu, \sigma^2)$ | normal with **variance** $\sigma^2$ | `mu + sigma * jax.random.normal(key, shape)`: scale by the std, not the variance |
| $\nabla_\theta \mathcal{L}$, $\partial \mathcal{L} / \partial \theta$ | gradient with respect to the parameters | `jax.grad(loss)(params)` |
| $\hat{y}$, $\theta$, $\tilde{x}$ | a prediction, the parameters, a perturbed or proposed copy | names, not operations |
| $j \le i$ in a sum (attention) | causal: position $i$ sees only the past | a lower-triangular mask, `jnp.tril` |
| $p(y \mid x)$ | a probability | code keeps logits or $\log p$; multiply probabilities by adding logs |

Three habits help. Write the shapes next to the symbols before writing code (each entry's **Shapes** line does
this). Write the formula as loops first: the plain-Python tab is the formula read literally, one index per
loop, so a wrong axis shows up as a wrong loop. Then let `vmap` add the batch axis, so the code can stay as
close to the one-example formula as the paper is.

## Translating NumPy to JAX

`jax.numpy` copies NumPy's names and broadcasting, so most code ports by changing `np.` to `jnp.`. The rest is
these differences, roughly in the order you'll hit them.

1. **Arrays are immutable.** `x[i] = v` becomes `x = x.at[i].set(v)`; `x[i] += v` becomes `x.at[i].add(v)`.
   Note `x.at[idx].add(v)` accumulates repeated indices, as `np.add.at` does, while NumPy's `x[idx] += v`
   applies a repeated index only once. In-place methods have functional versions: `a.sort()` becomes
   `a = jnp.sort(a)`, `rng.shuffle(a)` becomes `jax.random.permutation(key, a)`.
2. **Randomness is explicit.** `rng = np.random.default_rng(0)` becomes `key = jax.random.key(0)`; split the
   key (`k1, k2 = jax.random.split(key)`) and pass one to each call. The same key always gives the same numbers.
3. **float32 by default.** NumPy computes in float64; JAX in float32 (and int32), unless you enable x64. Expect
   agreement to about 1e-6 relative, not 1e-15, and check that a "stable" formula really is stable in float32.
4. **Out-of-range indices don't raise.** A read clamps to the last element and a write is dropped. NumPy's
   `IndexError` won't save you; test index arithmetic on small cases.
5. **Under `jit`, shapes are fixed and values are unknown.** `if x > 0:` on an array fails: use `jnp.where`
   for both-branch elementwise choices or `lax.cond` for real branches. Boolean masking `x[x > 0]` has a
   data-dependent shape: use `jnp.where(mask, x, 0)` and a masked sum, or `jnp.nonzero(..., size=k)`.
6. **Loops.** A Python loop over a fixed range is unrolled when traced, which is fine for a few steps and slow
   to compile for thousands. Use `lax.fori_loop` or `lax.scan` (which also returns the per-step outputs), and
   `lax.while_loop` when the trip count depends on values.
7. **Per-row loops become `vmap`.** `np.stack([f(row) for row in X])` becomes `jax.vmap(f)(X)`.
8. **Nested lists and dicts of arrays are pytrees.** `jax.tree.map(lambda a, b: a - lr * b, params, grads)`
   replaces the loop over parameters.
9. **Arrays live on the GPU.** `np.asarray(x)` copies one back to the host; calls return before the GPU
   finishes, so time with `.block_until_ready()`.

For PyTorch code, the same list applies plus a few renames: `dim=` is `axis=`, `x.backward()` and `.grad`
become `jax.grad` of a function, `nn.Module` state becomes Flax NNX, and `nn.Linear` stores its weight as
`(out, in)` where Flax stores `(in, out)`. Each entry's **Where they differ** box lists the traps for that
operation (default `eps`, biased versus unbiased std, GELU's approximation, AdamW's default weight decay).
