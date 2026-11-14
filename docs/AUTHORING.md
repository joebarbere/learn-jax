# Authoring a module

The reference implementation is `modules/00-environment/`. Copy its shape.

## Audience
A strong software developer who is new to ML and rusty on linear algebra and calculus. Explain the
intuition before the math, show the math in LaTeX only when it earns its place, and make every idea
something you can *see* (a plot or a table) or *change* (a slider). No hand-waving: if a lesson says
"X is faster", it measures X. Tie back to what the learner already did (the deepseek-v3-jax replication:
36.2M-param MoE model, bf16, 35k tok/s on the RX 7600 XT) when it helps.

## Files in `modules/NN-slug/`
- `lesson.py` — a marimo notebook. Sections numbered `## 1 · …`. Order: goal & "You will" list →
  intuition → build it from scratch → library version → GPU experiment → **Your turn** cells that call
  exercises → `## Further reading` (3–5 high-quality links) → "**Next:** Module NN+1 — …".
- `exercises.py` — module docstring explaining how to work; 4–7 functions with precise docstrings
  (shapes, dtypes, edge cases) whose body is `raise NotImplementedError`. Progressive difficulty.
- `solutions.py` — reference implementations, same signatures.
- `test_exercises.py` — pytest functions taking the `impl` fixture (see `conftest.py`). Test behaviour,
  not implementation; include an edge case per function; use `pytest.approx` / `numpy.testing`.
  Must pass on solutions and fail on stubs. Keep each module's tests under ~20 s on the GPU.

## marimo rules (the notebook is a dataflow graph, not a script)
- First cell imports `marimo as mo`, `jax`, `jax.numpy as jnp`, `learnjax as lj`, and loads
  `ex = lj.load_exercises(mo.notebook_dir())`; returns them.
- A global name may be defined in **one** cell only. Use `_underscore` names for cell-local variables.
- Every cell is `@app.cell def _(deps...): ...; return (exported,)` — parameters are the globals it reads.
- A cell's last expression is its output. UI elements (`mo.ui.slider`, `mo.ui.radio`, …) must be
  displayed in one cell and their `.value` read in another.
- Call exercises only through `lj.attempt(ex.fn, ...)` and render with `.view(mo)`; guard downstream
  use with `if _a.ok:` so the notebook runs cleanly when exercises are still stubs.
- Keep each cell fast (< ~10 s on the GPU); the whole lesson should run headless in < ~2 min.
- Plots: matplotlib, return the `Figure` (or `plt.gca()`), `plt.close` not needed. Keep them small.
- Quizzes: a `mo.ui.radio` cell plus a feedback cell with `mo.callout(..., kind="success"|"warn")`.

## Verify before calling a module done (all inside the container)
```bash
R=containers/run.sh; M=modules/NN-slug
$R marimo check $M/lesson.py                                         # structure lint: no output = ok
$R python -m pytest $M                                               # must FAIL (stubs)
LEARNJAX_SOLUTIONS=1 $R python -m pytest $M                          # must PASS
$R python $M/lesson.py                                               # runs with stubs, exit 0
LEARNJAX_SOLUTIONS=1 $R python tools/lesson_outputs.py $M/lesson.py  # read every rendered output; 0 errors
$R ruff check $M/exercises.py $M/solutions.py $M/test_exercises.py
```
Read the `lesson_outputs.py` text yourself: numbers sensible, prose matches what the cells show.

## Known marimo pitfalls
- A multi-line value (an array repr, joined rows) interpolated into an indented `mo.md(f"""...""")`
  breaks the dedent and renders as a code block. Re-indent it: `text.replace("\n", "\n    ")`, or
  build the markdown without indentation.
- A literal `|` inside LaTeX in a markdown table cell (`$|v|$`) splits the column. Use `\lvert v \rvert`.
- float32 results from `jnp.linalg.*` print float noise via `.tolist()`; format numbers explicitly.
- `tools/lesson_outputs.py` shows top-level images and text; callouts and plots nested in
  `mo.vstack`/`hstack` may not appear in its text — check those cells in the marimo UI.

## Data, training time, and outputs
- Datasets: `from learnjax.data import fashion_mnist, tiny_shakespeare, FASHION_CLASSES` (NumPy arrays,
  cached in the `learn-jax-cache` volume). Don't add downloads elsewhere without updating learnjax/data.py.
- Training lessons may train for real, but keep each training cell < ~60 s and the lesson < ~4 min.
  Prefer a few epochs + a note on what more training would give.
- Files a lesson writes (checkpoints, logs) go under `/course/runs/<module>/` (gitignored).

## PyTorch side-by-side notebooks (`torch_compare.py`)
- Runs in the PyTorch image (`IMAGE=localhost/learn-jax-torch containers/run.sh ...`), which has **no JAX**.
  Import only `learnjax.data` (NumPy-only) from the course package — never `learnjax.course`/`timing`/`gpu`.
- Mirror the JAX lesson's model and data so numbers are comparable; time with `torch.cuda.synchronize()`.
  `torch.cuda` means "the GPU" on ROCm builds too — say so once.
- Verify: `IMAGE=localhost/learn-jax-torch containers/run.sh python $M/torch_compare.py` (exit 0) and
  `IMAGE=localhost/learn-jax-torch containers/run.sh python tools/lesson_outputs.py $M/torch_compare.py`.
- XLA "excess precision": under `jit`, a float32→bfloat16→float32 round trip may be optimized away
  (reports zero error). To *demonstrate* bf16 rounding use `jax.lax.reduce_precision(x, 8, 7)`
  (exponent, mantissa bits) or compute outside jit.
- `jax.config.update("jax_debug_nans", True)` may not take effect inside marimo; use the scoped
  `with jax.debug_nans(True):` form.
- Flax 0.12: read parameters as `module.kernel[...]`, not `.value` (deprecated, prints a warning).
