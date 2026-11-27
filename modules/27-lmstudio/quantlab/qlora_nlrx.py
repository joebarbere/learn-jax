"""QLoRA fine-tune of Qwen3.5-9B on NL-RX (English → regex), and exact-match evaluation. §16's model.

Runs in the PyTorch image (it includes bitsandbytes); ~50 min to train one epoch on the
RX 7600 XT (500 steps × 16 examples, 8.8 GiB peak), ~4 min to evaluate 500 test examples.

    python qlora_nlrx.py train --out DIR        # writes a peft adapter to DIR
    python qlora_nlrx.py eval [--adapter DIR] [--shots 5] [--n 500] --out FILE.json

The base model is loaded in 4-bit NF4 (bitsandbytes) and frozen; LoRA r=16 trains on every linear layer in bf16
(Dettmers et al. 2023, QLoRA). Loss is on the answer tokens only. Thinking is switched off in the prompt
(`enable_thinking=False`, an empty <think></think> block): the model learns to answer directly, which is also the
format it must be served with afterwards (§16).
"""
import argparse
import json
import os
import random
import sys
import time

# PyTorch's "expandable segments" allocator corrupts gradients on this stack (ROCm 7.16, torch 2.14, gfx1102):
# with it, ~15% of updates came out NaN (with garbage like 4.7e32 beside them) and one hour-long run hung the GPU
# with a page fault. Without it: 0 bad updates in 500 steps, same speed and memory. Refuse to run with it on.
for _v in ("PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
    if "expandable_segments:true" in os.environ.get(_v, "").lower().replace(" ", ""):
        sys.exit(f"{_v} sets expandable_segments: it corrupts gradients on ROCm here (see module 27 §16). Unset it.")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from learnjax.data import nlrx

MODEL = "Qwen/Qwen3.5-9B"
p = argparse.ArgumentParser()
p.add_argument("mode", choices=["train", "eval"])
p.add_argument("--out", required=True)
p.add_argument("--adapter", help="eval: a trained adapter directory (omit for the base model)")
p.add_argument("--shots", type=int, default=0, help="eval: in-context examples from the training set")
p.add_argument("--n", type=int, default=500, help="eval: first n test examples")
p.add_argument("--epochs", type=float, default=1)
p.add_argument("--bs", type=int, default=16, help="examples per optimizer step")
p.add_argument("--accum", type=int, default=4, help="micro-batches per step (bs/accum examples each)")
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--r", type=int, default=16)
p.add_argument("--max-steps", type=int, default=0)
p.add_argument("--log-every", type=int, default=25)
a = p.parse_args()

data = nlrx("turk")
train, test = data["train"], data["test"]
tok = AutoTokenizer.from_pretrained(MODEL)


def prompt(msgs, shots=()):
    m = [msgs[0]]
    for s in shots:
        m += [s[1], s[2]]
    m.append(msgs[1])
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)


bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0}, dtype=torch.bfloat16)
print(f"loaded in {time.time() - t0:.0f}s, VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

if a.mode == "eval":
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    tok.padding_side = "left"
    shots = random.Random(0).sample(train, a.shots) if a.shots else ()
    rows, hits, out, t0 = test[: a.n], 0, [], time.time()
    for i in range(0, len(rows), 16):
        chunk = rows[i: i + 16]
        enc = tok([prompt(m, shots) for m in chunk], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=tok.pad_token_id)
        for m, ids in zip(chunk, g[:, enc.input_ids.shape[1]:]):
            gen = tok.decode(ids, skip_special_tokens=True).strip()
            hits += gen == m[2]["content"]
            out.append({"nl": m[1]["content"], "gt": m[2]["content"], "gen": gen})
    res = {"exact_match": hits / len(rows), "n": len(rows), "shots": a.shots, "adapter": bool(a.adapter),
           "seconds": round(time.time() - t0), "outputs": out}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"RESULT exact match {hits}/{len(rows)} = {hits / len(rows):.1%}  ({res['seconds']}s)")
    sys.exit()

from peft import LoraConfig, get_peft_model

# Not peft's prepare_model_for_kbit_training: it upcasts the ~1B-parameter embedding (248K vocabulary) to fp32,
# about 4 GB, and runs out of memory at 16 GB.
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules="all-linear"))
model.print_trainable_parameters()
ex = []
for m in train:
    pr = tok(prompt(m), add_special_tokens=False).input_ids
    ans = tok(m[2]["content"] + tok.eos_token, add_special_tokens=False).input_ids
    ex.append((pr + ans, [-100] * len(pr) + ans))      # loss only on the answer
steps = int(len(ex) * a.epochs) // a.bs
if a.max_steps:
    steps = min(steps, a.max_steps)
params = [q for q in model.parameters() if q.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 20) * max(0.0, 1 - s / steps))
rng, order, skipped, log = random.Random(0), [], 0, []
model.train()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for s in range(steps):
    if len(order) < a.bs:
        order += rng.sample(range(len(ex)), len(ex))
    batch, order = order[: a.bs], order[a.bs:]
    mb = a.bs // a.accum
    for k in range(a.accum):
        idx = batch[k * mb: (k + 1) * mb]
        L = max(len(ex[i][0]) for i in idx)
        ids = torch.tensor([ex[i][0] + [tok.pad_token_id] * (L - len(ex[i][0])) for i in idx], device="cuda")
        lab = torch.tensor([ex[i][1] + [-100] * (L - len(ex[i][1])) for i in idx], device="cuda")
        att = torch.tensor([[1] * len(ex[i][0]) + [0] * (L - len(ex[i][0])) for i in idx], device="cuda")
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        (loss / a.accum).backward()
    gn = torch.nn.utils.clip_grad_norm_([q for q in params if q.grad is not None], 1.0)
    if not torch.isfinite(gn):   # cheap insurance: one bad batch must not turn every weight into NaN
        skipped += 1
        opt.zero_grad()
        sched.step()
        print(f"step {s + 1}: non-finite gradient norm, update skipped (total {skipped})", flush=True)
        continue
    opt.step()
    sched.step()
    opt.zero_grad()
    log.append({"step": s + 1, "loss": round(loss.item(), 4), "grad_norm": round(gn.item(), 3)})
    if s % a.log_every == 0 or s == steps - 1:
        el = time.time() - t0
        print(f"step {s + 1}/{steps} loss {loss.item():.4f} gnorm {gn.item():.3f}  {el / (s + 1):.2f}s/step  "
              f"eta {el / (s + 1) * (steps - s - 1) / 60:.0f}min  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
              flush=True)
model.save_pretrained(a.out)
summary = {"steps": steps, "skipped": skipped, "minutes": round((time.time() - t0) / 60, 1),
           "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1), "log": log}
json.dump(summary, open(os.path.join(a.out, "train_log.json"), "w"))
print(f"TRAIN DONE: {steps} steps ({skipped} skipped) in {summary['minutes']} min, peak VRAM {summary['peak_gib']} GiB")
