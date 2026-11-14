"""Check learnjax.llama against Hugging Face transformers (float32, same prompt).

    IMAGE=localhost/learn-jax-torch containers/run.sh python tools/check_llama.py reference   # torch image
    containers/run.sh python tools/check_llama.py compare                                      # JAX image
"""
import sys
from pathlib import Path

import numpy as np

REPO = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "The detective looked at the muddy footprints and said"
OUT = Path("/course/runs/check_llama.npz")
N_GEN = 20


def reference():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0].numpy()
        gen = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)[0, ids.shape[1]:].numpy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, ids=ids[0].numpy(), logits=logits, gen=gen)
    print(f"saved HF reference: {ids.shape[1]} prompt tokens, greedy: {tok.decode(gen)!r}")


def compare():
    import jax
    import jax.numpy as jnp

    from learnjax import llama
    ref = np.load(OUT)
    params, cfg, tok = llama.load(REPO, dtype="float32")
    ids = jnp.asarray(ref["ids"])[None]
    with jax.default_matmul_precision("highest"):
        logits, _ = llama.forward_jit(params, ids, cfg)
        full_err = float(jnp.max(jnp.abs(logits[0] - ref["logits"])))
        # cached path: prefill all but the last 3 tokens, then feed them one at a time
        T = ids.shape[1]
        cache = llama.init_cache(cfg, 1, T + 4)
        _, cache = llama.forward_jit(params, ids[:, : T - 3], cfg, cache, 0)
        step_logits = []
        for t in range(T - 3, T):
            lg, cache = llama.forward_jit(params, ids[:, t : t + 1], cfg, cache, t)
            step_logits.append(lg[0, 0])
        cache_err = float(jnp.max(jnp.abs(jnp.stack(step_logits) - ref["logits"][T - 3:])))
        gen = llama.generate(params, cfg, ref["ids"].tolist(), N_GEN, jax.random.key(0), temperature=0.0)
    scale = float(np.abs(ref["logits"]).max())
    same = gen == ref["gen"].tolist()
    print(f"full forward  max |Δlogit| = {full_err:.2e}  (logit scale {scale:.1f})")
    print(f"KV-cached     max |Δlogit| = {cache_err:.2e}")
    print(f"greedy {N_GEN} tokens identical to HF: {same}   {tok.decode(gen)!r}")
    ok = full_err < 1e-2 * scale and cache_err < 1e-2 * scale and same
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(reference() if sys.argv[1] == "reference" else compare())
