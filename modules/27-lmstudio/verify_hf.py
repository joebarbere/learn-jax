"""Check every export loads in Hugging Face transformers and computes the same logits as JAX.

    IMAGE=localhost/learn-jax-torch containers/run.sh python modules/27-lmstudio/verify_hf.py

Runs in the PyTorch image (no JAX there): reads runs/27-lmstudio/refs/<name>.npz written by export.py,
runs the same probe tokens through transformers in float32 on the CPU, and writes verify.json with the
max |Δlogit|, whether the greedy next token agrees at every position, and whether the HF tokenizer
reproduces the JAX tokenizer's ids. NumPy + torch + transformers only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).resolve().parents[2] / "runs" / "27-lmstudio"


def main() -> int:
    results = {}
    heldout = (OUT / "heldout.txt").read_text()
    for ref in sorted((OUT / "refs").glob("*.npz")):
        name = ref.stem
        d = OUT / "hf" / name
        z = np.load(ref)
        model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
        tok = AutoTokenizer.from_pretrained(d)
        with torch.no_grad():
            logits = model(torch.tensor(z["ids"])[None]).logits[0].numpy()
        diff = float(np.abs(logits - z["logits"]).max())
        agree = float(np.mean(logits.argmax(-1) == z["logits"].argmax(-1)))
        # the probe ids were made by the JAX-side tokenizer: does the HF tokenizer agree on the same text?
        text = heldout[:2000] if not name.startswith("gpt16") else tok.decode(z["ids"].tolist())
        hf_ids = tok(text, add_special_tokens=False).input_ids[: len(z["ids"])]
        results[name] = {"max_abs_logit_diff": diff, "logit_scale": float(np.abs(z["logits"]).max()),
                         "greedy_agreement": agree, "tokenizer_ids_match": hf_ids == z["ids"].tolist()}
        print(f"{name:24} max|Δlogit| {diff:.2e} (scale {results[name]['logit_scale']:.1f})  "
              f"greedy agree {agree:.3f}  tokenizer match {results[name]['tokenizer_ids_match']}")
    (OUT / "verify.json").write_text(json.dumps(results, indent=2))
    return 0 if all(r["greedy_agreement"] == 1.0 for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
