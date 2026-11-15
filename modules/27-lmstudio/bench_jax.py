"""JAX column of Module 27's "same model, four ways" table: SmolLM2-360M-Instruct greedy decode, bf16,
Module 18's on-device decode loop (prefill + 128 tokens in one compiled program).

    containers/run.sh python modules/27-lmstudio/bench_jax.py      # prints one JSON line
"""
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from learnjax import llama

spec = importlib.util.spec_from_file_location("m27_infer18", ROOT / "modules" / "18-inference" / "infer.py")
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)

PROMPT = "Write a short story about a lighthouse keeper, about 150 words."
N_NEW = 128

params, cfg, tok = llama.load(llama.SMOLLM_360M)
prompt = jnp.asarray(llama.chat_prompt(tok, PROMPT), jnp.int32)[None]
key = jax.random.key(0)
jax.block_until_ready(infer.decode_on_device(params, cfg, prompt, N_NEW, key))       # compile
times = []
for _ in range(3):
    t = time.perf_counter()
    jax.block_until_ready(infer.decode_on_device(params, cfg, prompt, N_NEW, key))
    times.append(time.perf_counter() - t)
# prefill is inside the timed program; it is a few ms for a ~40-token prompt
print(json.dumps({"engine": "JAX (ROCm, bf16, on-device loop)", "model": "SmolLM2-360M-Instruct bf16",
                  "gen_tok_s": N_NEW / statistics.median(times), "prompt_tokens": int(prompt.shape[1])}))
