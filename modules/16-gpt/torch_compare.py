import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="16 · GPT in PyTorch")


@app.cell
def _():
    import os

    # Must be set before PyTorch runs its first attention call: on this card (gfx1102) the fused
    # flash / memory-efficient SDPA kernels (AOTriton) are experimental and off by default.
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

    import marimo as mo
    import math
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from learnjax.data import tiny_shakespeare  # NumPy-only part of the course package (no JAX here)

    return F, SDPBackend, math, mo, nn, np, os, plt, sdpa_kernel, time, tiny_shakespeare, torch


@app.cell
def _(mo):
    mo.md(r"""
    # 16 · The same GPT in PyTorch (side-by-side)

    This notebook runs in the **PyTorch** image (`make torch-lesson N=16`), which has no JAX. It builds the
    lesson's character GPT — vocabulary 65, context 128, width 128, 4 layers, 4 heads, pre-LayerNorm, GELU
    MLP, tied embeddings, GPT-2 init — and trains it with the same recipe: batches of 64 random windows,
    AdamW (β = 0.9, 0.95, weight decay 0.1), warmup 100 → peak 2e-3 → cosine to 2e-4 over 2,000 steps,
    clipping at global norm 1, bfloat16 compute.

    Unlike Module 10's tiny MLP, this model does real arithmetic per step, so the comparison is about
    **kernels** — especially attention — more than Python overhead.
    """)
    return


@app.cell
def _(mo, os, torch):
    _props = torch.cuda.get_device_properties(0)
    mo.md(f"""
    ## 1 · The device

    | | |
    |---|---|
    | PyTorch | `{torch.__version__}` (HIP `{torch.version.hip}`) |
    | device | `{torch.cuda.get_device_name(0)}` — `{_props.gcnArchName}`, {_props.total_memory / 2**30:.1f} GiB |
    | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `{os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")}` |

    As before, `torch.cuda` / `device="cuda"` means "the GPU" on ROCm builds too, and this image needs no
    gfx override. The environment variable opts in to AMD's **AOTriton** fused-attention kernels, which
    are *experimental* on RDNA3 consumer cards like this one; without it,
    `F.scaled_dot_product_attention` silently falls back to the unfused "math" implementation. Section 4
    measures both.
    """)
    return


@app.cell
def _(F, math, nn, torch):
    class Block(nn.Module):
        def __init__(self, d, n_heads, n_layers):
            super().__init__()
            self.n_heads = n_heads
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.fc, self.proj = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
            for lin in (self.qkv, self.fc):
                nn.init.normal_(lin.weight, std=0.02)
            for lin in (self.o, self.proj):  # writes into the residual stream: scaled init
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * n_layers))
            for lin in (self.qkv, self.o, self.fc, self.proj):
                nn.init.zeros_(lin.bias)

        def forward(self, x):
            B, T, D = x.shape
            q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
            q, k, v = (a.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) for a in (q, k, v))
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, hd)
            x = x + self.o(att.transpose(1, 2).reshape(B, T, D))
            return x + self.proj(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))

    class GPT(nn.Module):
        def __init__(self, vocab=65, seq_len=128, d=128, n_layers=4, n_heads=4):
            super().__init__()
            self.wte, self.wpe = nn.Embedding(vocab, d), nn.Embedding(seq_len, d)
            nn.init.normal_(self.wte.weight, std=0.02)
            nn.init.normal_(self.wpe.weight, std=0.02)
            self.blocks = nn.ModuleList(Block(d, n_heads, n_layers) for _ in range(n_layers))
            self.ln_f = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab, bias=False)
            self.head.weight = self.wte.weight  # weight tying: one tensor, two uses

        def forward(self, tokens):
            T = tokens.shape[1]
            x = self.wte(tokens) + self.wpe(torch.arange(T, device=tokens.device))
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.ln_f(x)).float()

    return (GPT,)


@app.cell
def _(GPT, mo):
    _n = sum(p.numel() for p in GPT().parameters())
    _formula = 65 * 128 + 128 * 128 + 4 * (12 * 128**2 + 13 * 128) + 2 * 128  # the lesson's count_params
    mo.md(f"""
    ## 2 · The model

    `GPT()` has **{_n:,} parameters** — the lesson's closed-form `count_params(cfg_char)` gives
    {_formula:,}{" ✅" if _n == _formula else " ❌"}.
    `model.parameters()` de-duplicates the tied `wte`/`head` tensor, just as the JAX pytree simply never
    has a second copy.
    """)
    return


@app.cell
def _(np, tiny_shakespeare, torch):
    _text = tiny_shakespeare()
    _chars = sorted(set(_text))
    _ids = np.array([{c: i for i, c in enumerate(_chars)}[c] for c in _text], dtype=np.int64)
    _cut = int(0.9 * len(_ids))
    train_ids = torch.from_numpy(_ids[:_cut]).cuda()
    val_ids = torch.from_numpy(_ids[_cut:]).cuda()
    itos = _chars

    def get_batch(data, batch, seq_len, generator=None):
        """Random windows of seq_len + 1 tokens, sampled on the GPU (the lesson's vmap+dynamic_slice)."""
        starts = torch.randint(0, data.numel() - seq_len, (batch,), device=data.device, generator=generator)
        return data[starts[:, None] + torch.arange(seq_len + 1, device=data.device)]

    return get_batch, itos, train_ids, val_ids


@app.cell
def _(F, GPT, get_batch, math, torch, train_ids):
    def make_training(compile_model=False, steps=2000, seed=0):
        torch.manual_seed(seed)
        model = GPT().cuda()
        decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=2e-3, betas=(0.9, 0.95))

        def lr_at(step):  # = optax.warmup_cosine_decay_schedule(0, 2e-3, 100, 2000, 2e-4)
            if step < 100:
                return 2e-3 * step / 100
            frac = (step - 100) / (steps - 100)
            return 2e-4 + (2e-3 - 2e-4) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))

        if compile_model == "reduce-overhead":  # also capture the step as a HIP graph: fewer launches
            fwd = torch.compile(model, mode="reduce-overhead")
        else:
            fwd = torch.compile(model) if compile_model else model

        def step_fn(step):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            batch = get_batch(train_ids, 64, 128)
            with torch.autocast("cuda", dtype=torch.bfloat16):  # bf16 compute, f32 master weights
                logits = fwd(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            return loss

        return model, step_fn

    return (make_training,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Speed: which attention, eager or compiled

    Each variant runs 20 warm-up steps (compilation, autotuning) and then times 100 steps with
    `torch.cuda.synchronize()` around the clock. Tokens per step: 64 × 128 = 8,192.
    """)
    return


@app.cell
def _(SDPBackend, make_training, mo, sdpa_kernel, time, torch):
    def _tok_s(backends, compile_model):
        _model, step_fn = make_training(compile_model)
        with sdpa_kernel(backends):
            for s in range(20):
                step_fn(s)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for s in range(20, 120):
                step_fn(s)
            torch.cuda.synchronize()
        return 100 * 64 * 128 / (time.perf_counter() - t0)

    _fused = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    speeds = {}
    for _name, _backends, _compile in (("eager, math attention (no opt-in)", [SDPBackend.MATH], False),
                                       ("eager, fused attention (AOTriton)", _fused, False),
                                       ("torch.compile, fused attention", _fused, True),
                                       ("torch.compile reduce-overhead, fused", _fused, "reduce-overhead")):
        try:
            speeds[_name] = _tok_s(_backends, _compile)
        except RuntimeError as _e:  # a backend this build can't run on this card
            speeds[_name] = None
            print(f"{_name}: {str(_e).splitlines()[0]}")
    _rows = "\n".join(f"| {k} | {v:,.0f} |" if v else f"| {k} | not available |" for k, v in speeds.items())
    mo.md("| PyTorch variant | tokens/s |\n|---|---|\n" + _rows)
    return (speeds,)


@app.cell
def _(make_training, math, mo, np, torch, val_ids, get_batch, time, SDPBackend, sdpa_kernel):
    _fused = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    model, _step = make_training(compile_model=False)
    _g = torch.Generator(device="cuda").manual_seed(100)
    _val_batches = [get_batch(val_ids, 64, 128, _g) for _ in range(8)]
    torch_log = []
    _t0 = time.perf_counter()
    with sdpa_kernel(_fused):
        for _s in range(1, 2001):
            _loss = _step(_s - 1)
            if _s % 100 == 0:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.eval()
                    _v = np.mean([torch.nn.functional.cross_entropy(
                        model(b[:, :-1]).reshape(-1, 65), b[:, 1:].reshape(-1)).item() for b in _val_batches])
                    model.train()
                torch_log.append((_s, _loss.item(), _v))
    torch.cuda.synchronize()
    mo.md(f"""
    ## 4 · Training to the end

    2,000 steps (eager, fused attention) took **{time.perf_counter() - _t0:.1f} s** including evaluation;
    final validation loss **{torch_log[-1][2]:.3f}** nats/char (perplexity {math.exp(torch_log[-1][2]):.1f}).
    """)
    return model, torch_log


@app.cell
def _(plt, torch_log):
    _s, _tr, _va = zip(*torch_log)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_s, _tr, label="train (last batch)")
    _ax.plot(_s, _va, "o-", ms=3, label="validation")
    _ax.set(xlabel="step", ylabel="cross-entropy (nats/char)", title="PyTorch character GPT")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(itos, mo, model, torch):
    @torch.no_grad()
    def _sample(prompt, n, temperature=0.8):
        model.eval()
        toks = torch.tensor([[itos.index(c) for c in prompt]], device="cuda")
        g = torch.Generator(device="cuda").manual_seed(1)
        for _ in range(n):
            logits = model(toks[:, -128:])[0, -1] / temperature
            toks = torch.cat([toks, torch.multinomial(torch.softmax(logits, -1), 1, generator=g)[None]], dim=1)
        return "".join(itos[i] for i in toks[0].tolist())

    mo.md(f"**Sample, temperature 0.8:**\n\n```\n{_sample('ROMEO:' + chr(10), 300)}\n```")
    return


@app.cell
def _(mo, speeds, torch_log):
    # Measured in the JAX lesson (modules/16-gpt/lesson.py) on the same RX 7600 XT when this was written —
    # this image has no JAX, so they can't be re-measured here.
    JAX_TOK_S, JAX_VAL = 732_712, 1.54
    _best = max(v for v in speeds.values() if v)
    _rows = "\n".join(f"| PyTorch: {k} | {v:,.0f} | {v / JAX_TOK_S:.2f}× |" if v else f"| PyTorch: {k} | — | — |"
                      for k, v in speeds.items())
    mo.md(f"""
    ## 5 · JAX vs PyTorch on this model

    | variant | tokens/s | vs JAX |
    |---|---|---|
    | **JAX: jit + lax.scan, attention written as einsums (XLA-fused)** | {JAX_TOK_S:,} | 1.00× |
    {_rows.replace(chr(10), chr(10) + "    ")}

    Final validation loss after 2,000 steps: JAX {JAX_VAL:.3f}, PyTorch {torch_log[-1][2]:.3f} — the same
    model and recipe land in the same place (different random streams, so not bit-identical).

    What's going on (measured on an otherwise idle GPU — rerun this while other jobs use the card and
    every number drops):
    * **Fused attention is PyTorch's biggest lever here.** Compare the first two PyTorch rows: with the
      AOTriton opt-in, attention is one kernel that never writes the T×T score matrix to memory; without
      it, PyTorch runs matmul → mask → softmax → matmul as separate kernels and is well behind JAX.
    * **With fused attention and `torch.compile`, PyTorch beats JAX on this model.** JAX on ROCm has *no*
      fused attention kernel (the `cudnn` implementation of `jax.nn.dot_product_attention` doesn't exist on
      AMD), so XLA builds attention from the einsum → mask → softmax → einsum chain, fusing what it can but
      still materializing the scores. That's the likely gap — Module 22 profiles both to check, and Module 23
      looks at writing a fused attention kernel for JAX with Pallas. The gap grows with context length.
    * **Overhead is no longer the story.** Eager PyTorch already matches JAX's 100-steps-per-dispatch
      `lax.scan`, and `mode="reduce-overhead"` (replaying the step as a captured GPU graph) adds nothing —
      unlike Module 10's tiny MLP, each step here does ~{6 * 818_048 * 8192 / 1e9:.0f} GFLOP of real work.
      `torch.compile`'s gain comes from fusing the elementwise work (LayerNorm, GELU, residual adds), the job
      XLA already does for JAX.
    * The lesson's MFU table shows where this goes: at train.py's width both frameworks spend most of their
      time in the same hipBLASLt matrix multiplies. Best PyTorch here: {_best:,.0f} tokens/s.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · The same model, two idioms

    | | JAX (lesson) | PyTorch (here) |
    |---|---|---|
    | parameters | a pytree of arrays from `init_gpt(key, cfg)` | `nn.Module` attributes, `model.parameters()` |
    | weight tying | `x @ params["wte"].T` — just use the array twice | `head.weight = wte.weight` — share one `Parameter` |
    | causal attention | einsum → `jnp.where(mask, …)` → softmax → einsum | `F.scaled_dot_product_attention(…, is_causal=True)` |
    | mixed precision | cast the param pytree to bf16 inside the loss | `torch.autocast("cuda", dtype=torch.bfloat16)` |
    | random batches | `get_batch(key, …)` with `vmap(dynamic_slice)`, explicit key | `torch.randint` with the global (or a `Generator`) RNG |
    | training loop | `lax.scan` over steps inside one `jit` | Python loop; `torch.compile` for fusion |
    | init | written out per array in `init_gpt` | `nn.init.*` on each layer after construction |
    | sampling | jitted fixed-shape window, pad right | slice the last 128 tokens, variable shape (eager) |
    """)
    return


if __name__ == "__main__":
    app.run()
