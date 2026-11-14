import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="20 · LoRA in PyTorch + PEFT")


@app.cell
def _():
    import os

    # Opt in to the experimental fused attention kernels on RDNA3 (see Module 15) before torch loads.
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

    import marimo as mo
    import importlib.util
    import math
    import sys
    import time

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    _spec = importlib.util.spec_from_file_location("m20_task", mo.notebook_dir() / "task.py")
    task = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = task
    _spec.loader.exec_module(task)
    REPO = "HuggingFaceTB/SmolLM2-360M-Instruct"
    return (AutoModelForCausalLM, LoraConfig, REPO, Tokenizer, get_peft_model, math, mo, np, snapshot_download,
            task, time, torch)


@app.cell
def _(mo, torch):
    mo.md(f"""
    # 20 · The same fine-tune in PyTorch + PEFT

    Same model (SmolLM2-360M-Instruct), same task and data (`task.py`), same LoRA setup (rank 8,
    α = 16, `q_proj` + `v_proj`, no dropout), same optimizer (AdamW 1e-3, 20 warmup steps, cosine to
    10%, clip 1.0), batch 16, 120 steps. Here the LoRA comes from Hugging Face **PEFT** instead of
    your own `lora_apply`.

    PyTorch {torch.__version__} on **{torch.cuda.get_device_name(0)}** — on ROCm builds `torch.cuda`
    means "the GPU". Fused attention is opted in via `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
    (experimental on this card; Module 15).
    """)
    return


@app.cell
def _(AutoModelForCausalLM, REPO, Tokenizer, snapshot_download, task, time, torch):
    _t = time.perf_counter()
    _path = snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = Tokenizer.from_file(f"{_path}/tokenizer.json")       # identical token ids to the JAX lesson
    base = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).to("cuda").eval()
    splits = task.make_splits()
    enc = {_k: task.encode_batch(tok, _v) for _k, _v in splits.items()}
    load_s = time.perf_counter() - _t
    return base, enc, load_s, splits, tok


@app.cell
def _(np, task, torch):
    @torch.no_grad()
    def exact_match(model, batch, chunk=16):
        preds = []
        for i in range(0, len(batch["tokens"]), chunk):
            x = torch.from_numpy(batch["tokens"][i:i + chunk]).long().cuda()
            preds.append(model(input_ids=x).logits.argmax(-1).cpu().numpy())
        return task.teacher_forced_exact_match(np.concatenate(preds), batch)

    return (exact_match,)


@app.cell
def _(base, enc, exact_match, load_s, mo):
    base_em = {_s: float(exact_match(base, enc[_s]).mean()) for _s in ("test", "heldout")}
    mo.md(f"""
    ## 1 · Before

    Loaded in {load_s:.1f} s. Base model exact match (teacher-forced, same grading as the JAX lesson):
    **test {base_em['test']:.0%}**, **heldout {base_em['heldout']:.0%}**.
    """)
    return (base_em,)


@app.cell
def _(LoraConfig, base, get_peft_model, mo):
    cfg_lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0,
                          bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg_lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in model.parameters())
    _dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
    mo.md(f"""
    ## 2 · Wrap the model with PEFT

    ```python
    LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
    model = get_peft_model(base, cfg_lora)
    ```

    Trainable: **{trainable:,}** of {_total:,} ({trainable / _total:.2%}) — the JAX lesson's
    `lora_param_count(cfg, 8, ["wq", "wv"])` gives the same 819,200. Adapter dtype: {', '.join(_dtypes)}
    (PEFT keeps adapters in float32 on a bf16 model, like our f32 A and B). One difference: PEFT
    initializes A with Kaiming-uniform; the JAX lesson uses a normal with std 1/√d_in. B is zero
    in both.
    """)
    return model, trainable


@app.cell
def _(enc, math, model, np, time, torch):
    STEPS, BATCH, LR, WARMUP = 120, 16, 1e-3, 20
    _opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)

    def _lr_mult(step):          # same shape as optax.warmup_cosine_decay_schedule(0, LR, 20, 120, 0.1*LR)
        if step < WARMUP:
            return step / WARMUP
        frac = (step - WARMUP) / (STEPS - WARMUP)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac))

    _sched = torch.optim.lr_scheduler.LambdaLR(_opt, _lr_mult)
    _rng = np.random.default_rng(1)
    _n = len(enc["train"]["tokens"])
    losses, _step_times = [], []
    model.train()
    _perm = _rng.permutation(_n)
    _t0 = time.perf_counter()
    for _s in range(STEPS):
        _i = (_s * BATCH) % (_n - BATCH)
        _idx = _perm[_i:_i + BATCH]
        _x = torch.from_numpy(enc["train"]["tokens"][_idx]).long().cuda()
        _m = torch.from_numpy(enc["train"]["loss_mask"][_idx]).cuda()
        torch.cuda.synchronize()
        _ts = time.perf_counter()
        _logits = model(input_ids=_x).logits.float()
        _nll = torch.nn.functional.cross_entropy(_logits[:, :-1].reshape(-1, _logits.shape[-1]),
                                                 _x[:, 1:].reshape(-1), reduction="none").view(BATCH, -1)
        _loss = (_nll * _m[:, :-1]).sum() / _m.sum()
        _opt.zero_grad(set_to_none=True)
        _loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        _opt.step()
        _sched.step()
        torch.cuda.synchronize()
        _step_times.append(time.perf_counter() - _ts)
        losses.append(_loss.item())
    model.eval()
    torch_timing = {"total": time.perf_counter() - _t0, "per_step": float(np.median(_step_times[1:]))}
    return losses, torch_timing


@app.cell
def _(base_em, enc, exact_match, losses, mo, model, torch_timing, trainable):
    after_em = {_s: float(exact_match(model, enc[_s]).mean()) for _s in ("test", "heldout")}
    mo.md(f"""
    ## 3 · After

    | | base | PEFT LoRA (PyTorch) | your LoRA (JAX, measured when written) |
    |---|---|---|---|
    | test exact match | {base_em['test']:.0%} | **{after_em['test']:.0%}** | 100% |
    | heldout exact match | {base_em['heldout']:.0%} | **{after_em['heldout']:.0%}** | 100% |
    | trainable params | — | {trainable:,} | 819,200 |
    | time per step | — | **{torch_timing["per_step"] * 1e3:.0f} ms** | ~180 ms |
    | final training loss | — | {losses[-1]:.2e} | ~2e-4 |

    (The JAX numbers are hardcoded from the JAX lesson on this card — this image has no JAX.) Both
    frameworks learn the task completely; the step times are in the same ballpark because both spend
    them in the same place: a forward and backward pass through 362M frozen bf16 weights. LoRA makes the
    *optimizer* cheap, not the model.
    """)
    return (after_em,)


@app.cell
def _(mo, model, splits, task, tok, torch):
    _end = tok.token_to_id("<|im_end|>")
    _rows = []
    with torch.no_grad():
        for _e in splits["heldout"][:3]:
            _ids = torch.tensor([tok.encode(task.chatml_prompt(_e["text"])).ids]).cuda()
            _out = model.generate(input_ids=_ids, max_new_tokens=40, do_sample=False, eos_token_id=_end)
            _text = tok.decode([t for t in _out[0, _ids.shape[1]:].tolist() if t != _end])
            _rows.append(f"* {_e['text']}\n  → `{_text}` {'✅' if _text == _e['answer'] else '❌'}")
    mo.md("Generated with `model.generate` (greedy, unseen templates):\n\n" + "\n".join(_rows))
    return


@app.cell
def _(mo, model):
    import pathlib

    _dir = pathlib.Path("/course/runs/20-lora/peft_adapter")
    model.save_pretrained(_dir)
    _size = sum(f.stat().st_size for f in _dir.iterdir())
    mo.md(f"""
    ## 4 · Save and merge

    `model.save_pretrained(...)` writes only the adapter: **{_size / 2**20:.1f} MiB** in
    `runs/20-lora/peft_adapter/` (safetensors + its config). `model.merge_and_unload()` is PEFT's
    `merge_lora`: it folds $\\frac{{\\alpha}}{{r}}BA$ into each weight and returns a plain transformers model.

    | JAX lesson (yours) | PEFT |
    |---|---|
    | `init_adapters` + `lora_init` | `get_peft_model(base, LoraConfig(...))` |
    | `lora_apply` inside the layer | PEFT swaps `nn.Linear` for a `lora.Linear` wrapper |
    | grad w.r.t. adapters only | `requires_grad=False` on base weights |
    | `merge_lora` | `merge_and_unload()` |
    | `np.savez` adapter | `save_pretrained` (safetensors + adapter_config.json) |

    Same math, different idiom: JAX keeps the adapter as a separate pytree you pass around; PEFT
    mutates the model object in place.
    """)
    return


if __name__ == "__main__":
    app.run()
