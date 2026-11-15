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
LEARNJAX_SOLUTIONS=1 $R python tools/lesson_outputs.py --full $M/lesson.py  # read every output; 0 errors
$R ruff check $M/exercises.py $M/solutions.py $M/test_exercises.py
```
Read the `lesson_outputs.py` text yourself: numbers sensible, prose matches what the cells show.

## Known marimo pitfalls
- A multi-line value (an array repr, joined rows) interpolated into `mo.md(f"""...""")` whose text is indented
  *deeper than the cell body* (e.g. inside a helper `def` in the cell) breaks the dedent and renders as code
  blocks. marimo dedents each cell body first, so markdown at the usual 4-space indent is fine, even inside an
  `if`. `python3 tools/check_md_indent.py modules/*/lesson.py` finds the risky cases (0 in the course, Nov 2026).
  Note the exported notebook stores already-dedented markdown, so rendered outputs can't reveal this.
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
  The image is optional (~26 GB): if it's missing, `make build-torch` first. `make check` skips torch notebooks
  without it, so build it before verifying one.
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

## Dependencies
- Add a package to `containers/requirements.txt` (or `-torch.txt` / `-book.txt`) with an exact pin, run
  `make lock`, then rebuild (`make build`). Commit both the `.txt` and the regenerated `.lock`.
- `containers/lock.sh` resolves with uv *inside the base image*, holding every preinstalled package at its
  version (only packages the `.txt` names may move, and the lock says so). Images install the lock with
  `uv pip install --no-deps` and fail the build on `uv pip check` errors.
- For a reproducible re-lock, pass a date: `UV_EXCLUDE_NEWER=2026-11-15T00:00:00Z make lock`.

## Phase 3+ additions
- `learnjax.data.detective_corpus()` → the deepseek-v3-jax v1.0 tokens (train ~21.7M / val ~0.30M,
  uint16) + its 8,192-vocab BPE `tokenizer_path`. GPT-2's tokenizer: `tokenizers.Tokenizer.from_pretrained("gpt2")`.
- Longer training runs (minutes, not seconds) go in a standalone `modules/NN-slug/train.py` the learner
  runs with `containers/run.sh python modules/NN-slug/train.py`; it writes `/course/runs/NN-slug/`
  (metrics.jsonl, checkpoint). The lesson trains a short version live and, if the long run's outputs
  exist, plots/loads them instead (and says how to produce them if not).
- PyTorch SDPA on this card: the flash / memory-efficient backends of `F.scaled_dot_product_attention`
  are off by default ("No available kernel") and need `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, set
  before torch first runs attention (module 15: ~40× faster, 27× less memory at T=4096). It's
  experimental, so notebooks opt in explicitly and say so; the image does not set it.
- JAX has no fused attention on ROCm: `jax.nn.dot_product_attention(implementation="cudnn")` fails;
  the default falls back to plain XLA.
- marimo markdown swallows `<|endoftext|>`-style tokens as HTML even in backticks: `html.escape` them.
- Pass large arrays (token datasets) into jitted functions as arguments, don't close over them —
  closures bake multi-MB constants into the compiled program. Small closed-over arrays bite too: if a big
  intermediate depends *only* on constants (e.g. an (L, D, N) kernel from closed-over parameters and
  `arange(L)`), XLA constant-folds it on the CPU at compile time (module 36: 3+ min).
- On ROCm, `x ** arange(L)` (broadcast power) inside a reduction compiles very slowly even with arguments;
  write `exp(k * log(x))` for positive x (module 36: 193 s → 1 s). If a cell logs XLA's "Very slow
  compile?" alarm, find the op and fix it — lessons don't use a persistent compilation cache.
- ROCm codegen bugs found in module 38 (both reproducible on an idle GPU, both with workarounds):
  gathering rows of a big bf16 table and projecting them in one fused kernel under jit + scan + grad
  page-faults and hangs the GPU → project the whole table first, then gather; and fusing scoring with
  `top_k` in one jitted program can fail with `hipError_t(98)` "Failed to set shared memory size"
  (shape-dependent; `optimization_barrier` doesn't help) → split into two jitted functions.
- Run GPU jobs one at a time (`flock <lockfile> containers/run.sh ...`): concurrent jobs slowed a 3-minute
  run to 81 minutes and coincided with a NaN and a segfault that didn't reproduce alone.

## Seminal papers (every module)
Add the module's 2–5 seminal papers to `learnjax/papers.py` (title, authors, year, link, one line on why it
matters *for this module*). Prefer arXiv abs links or DOIs. The lesson renders them with a cell just before
`## Further reading`: `python3 tools/add_papers_cell.py modules/NN-slug/lesson.py` inserts it. Then
`make papers` checks every link (arXiv titles against the arXiv API) and regenerates docs/PAPERS.md.

## Part 5: real pretrained models (`learnjax.llama`)
- `params, cfg, tok = llama.load(llama.SMOLLM_360M)` (or `SMOLLM_135M`, a base model with the same
  tokenizer — a natural draft model). Pure-JAX Llama: stacked layers + `lax.scan`, GQA, rotate-half RoPE,
  preallocated KV cache (`init_cache`, `forward(params, tokens, cfg, cache, pos)`), `generate`,
  `chat_prompt` (ChatML; stop on `<|im_end|>`). Verified against HF transformers by
  `tools/check_llama.py` (logits 3e-4, greedy tokens identical). Baseline (idle GPU, 360M bf16):
  ~119–135 tok/s decode with `generate`'s Python loop; ~172–175 tok/s with an on-device `fori_loop`
  (module 18). Time with the *same* cache shape as warmup — a new `n_new` changes the cache shape and
  recompiles (an early "68 tok/s" figure was that artifact).
- Weights download once into the HF cache volume. Don't modify learnjax/llama.py from a module — copy
  and adapt into the module if a lesson needs a variant, and report it.
- PyTorch image has transformers 5.23 / peft 0.21 / accelerate 1.15 for side-by-side notebooks.
- To measure what XLA *would* allocate for a huge shape without allocating it, compile only with autotuning
  off: `jax.jit(f).lower(x).compile(compiler_options={"xla_gpu_autotune_level": 0})` (autotuning
  materializes buffers and can OOM).

## Links: attribution and JAX docs (every module)
- **Attribute every external link** to who made it: authors for papers ("Kingma & Ba, [Adam](…)"), the person or
  organization for posts, talks, videos, docs, repos, models and datasets ("JAX docs: […](…)", "Horace He, […](…)",
  "ggml-org, [llama.cpp server README](…)", "[SmolLM2-360M-Instruct](…) by Hugging Face"). When a lesson builds on
  or quotes someone's work (code, an exercise idea, a bug), name them and link it at that point, not only in
  Further reading. New major sources also go in the README's Credits.
- **Link the first mention of each JAX API** in a lesson's prose to its reference page:
  `containers/run.sh python tools/link_jax_docs.py modules/NN-slug/lesson.py` (targets come from the docs' own
  Sphinx inventory, so they exist; it skips code, f-string values and already-linked mentions; idempotent).
- `make links` checks every external link resolves (`tools/check_links.py`; `--list` prints each with context
  for an attribution review). `make papers` does the same for papers and companion videos.

## History (every module) and Part 10 milestone modules
- Every lesson opens (right after its intro cell) with a **Where this came from** callout (`# history box`):
  3–6 plain-language sentences — who, when, what problem they were stuck on, what came before and after —
  grounded in the module's registered papers; ends with "Timeline: Module 00b". No invented quotes; use the
  year or decade when an exact date is uncertain.
- Module numbers are registry keys; the **display label** is the slug prefix (`00b-history` → label "00b",
  number 28). Use `learnjax.syllabus.number(slug)` / `label(slug)`; never parse numbers from slugs.
- **Part 10 "Milestones, rebuilt in JAX"** (modules 29–37) retells the field's history by rebuilding each
  milestone at small scale on the RX 7600 XT. Each one: states which earlier module to take first; tells the
  story of the original (people, year, the problem, what it beat, what replaced it); rebuilds it in JAX the
  way it was done *then* (period-faithful first), then shows what we'd do *now*; measures something real on
  the GPU; and links forward/back to the main modules. JAX ecosystem libraries available in the image:
  `mctx` (DeepMind MCTS), `pgx` (GPU game environments), `gymnax` (classic RL environments).
