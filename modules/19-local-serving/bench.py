"""Benchmark llama.cpp on this GPU: every backend × model × prompt length, using the server's own timings.

Run on the HOST (it restarts the server container between configurations):

    python3 modules/19-local-serving/bench.py            # full matrix, ~5–10 min
    python3 modules/19-local-serving/bench.py --quick    # one model, both backends

Writes runs/19-local-serving/bench.json, which the lesson plots. Standard library only.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "containers" / "llamacpp.sh"
OUT = ROOT / "runs" / "19-local-serving" / "bench.json"
URL = "http://127.0.0.1:8080"

MODELS = {
    "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf": {"params_b": 1.54, "quant": "Q4_K_M"},
    "qwen2.5-coder-1.5b-instruct-q8_0.gguf": {"params_b": 1.54, "quant": "Q8_0"},
    "qwen2.5-coder-14b-instruct-q4_k_m.gguf": {"params_b": 14.77, "quant": "Q4_K_M"},
}
PROMPT_TOKENS = (512, 4096)
N_PREDICT = 128


def post(path: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def make_prompt(n_tokens: int) -> list[int]:
    """A prompt of exactly n tokens: tokenize some code, repeat it, truncate."""
    code = (ROOT / "learnjax" / "llama.py").read_text()
    ids = post("/tokenize", {"content": code})["tokens"]
    return (ids * (n_tokens // len(ids) + 1))[:n_tokens]


def measure(n_prompt: int) -> dict:
    prompt = make_prompt(n_prompt)
    post("/completion", {"prompt": prompt[:16], "n_predict": 4, "cache_prompt": False})   # warm up
    r = post("/completion", {"prompt": prompt, "n_predict": N_PREDICT, "cache_prompt": False,
                             "temperature": 0, "ignore_eos": True})
    t = r["timings"]
    return {"prompt_tokens": t["prompt_n"], "prompt_tok_s": t["prompt_per_second"],
            "gen_tokens": t["predicted_n"], "gen_tok_s": t["predicted_per_second"]}


def vram_used_gb() -> float | None:
    try:
        card = next(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"))
        return int(card.read_text()) / 2**30
    except (StopIteration, OSError):
        return None


def main() -> None:
    quick = "--quick" in sys.argv
    models = ["qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"] if quick else list(MODELS)
    results = []
    for backend in ("vulkan", "rocm"):
        for model in models:
            subprocess.run([str(SCRIPT), "start", backend, model], check=True, stdout=subprocess.DEVNULL)
            time.sleep(2)
            vram = vram_used_gb()
            for n in PROMPT_TOKENS:
                m = measure(n) | {"backend": backend, "model": model, "vram_gb": vram} | MODELS[model]
                results.append(m)
                print(f"{backend:6} {model:42} prompt {m['prompt_tokens']:5}: "
                      f"pp {m['prompt_tok_s']:8.1f} tok/s   tg {m['gen_tok_s']:6.1f} tok/s   "
                      f"vram {vram or 0:.1f} GB", flush=True)
    subprocess.run([str(SCRIPT), "stop"], check=False, stdout=subprocess.DEVNULL)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"measured": time.strftime("%Y-%m-%d"), "results": results}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
