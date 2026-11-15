"""Export the course's models from JAX into Hugging Face format (the doorway to GGUF and LM Studio).

    containers/run.sh python modules/27-lmstudio/export.py            # all models that exist
    containers/run.sh python modules/27-lmstudio/export.py --only gpt16-detective

Writes runs/27-lmstudio/:
  hf/<name>/              config.json + model.safetensors (float32) + tokenizer files
  refs/<name>.npz         probe token ids + JAX float32 logits (checked against transformers by verify_hf.py)
  heldout.txt             held-out detective prose for perplexity / bits-per-byte (decoded Module 16 val tokens)
  jax_reference.json      what the JAX models score on their tasks (LM Studio is compared against this)
  export.json             round-trip and bookkeeping results

Models: smollm2-360m-instruct (round-trip check only — converted straight from the HF download),
lora20-orders (Module 20's LoRA merged in float32), grpo26-arith (Module 26's GRPO adapter merged in
float32), gpt16-detective (Module 16's GPT in GPT-2 layout). Missing runs are skipped with a note.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
MODULES = HERE.parent
ROOT = MODULES.parent
sys.path.insert(0, str(ROOT))
from learnjax import llama
from learnjax.course import load_exercises
from learnjax.data import detective_corpus

RUNS = ROOT / "runs"
OUT = RUNS / "27-lmstudio"
N_PROBE = 64


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ex = load_exercises(HERE)              # the learner's export functions (or the reference with LEARNJAX_SOLUTIONS=1)


def _save_safetensors(path: Path, tensors: dict) -> None:
    from safetensors.numpy import save_file
    save_file({k: np.ascontiguousarray(np.asarray(v, np.float32)) for k, v in tensors.items()}, str(path),
              metadata={"format": "pt"})


def _copy_tokenizer_files(src: Path, dst: Path) -> None:
    for f in src.iterdir():
        if f.suffix in (".json", ".txt") and f.name != "config.json" and "results" not in f.name \
                and f.name != "trainer_state.json":
            shutil.copy(f, dst / f.name)


def write_llama(name: str, params: dict, cfg, src_dir: Path) -> Path:
    d = OUT / "hf" / name
    d.mkdir(parents=True, exist_ok=True)
    _save_safetensors(d / "model.safetensors", ex.export_llama_hf(params, cfg))
    hf_cfg = json.loads((src_dir / "config.json").read_text())
    hf_cfg["torch_dtype"] = "float32"
    (d / "config.json").write_text(json.dumps(hf_cfg, indent=2))
    _copy_tokenizer_files(src_dir, d)
    return d


def jax_logits_llama(params, cfg, ids: np.ndarray) -> np.ndarray:
    with jax.default_matmul_precision("highest"):
        logits, _ = llama.forward_jit(params, jnp.asarray(ids)[None], cfg)
    return np.asarray(logits[0])


def merge_f32(params, adapters: dict, alpha: float, rank: int) -> dict:
    """W + (alpha/r)·A·B with everything in float32. Merging into bf16 weights rounds most of a small
    LoRA update away (Module 20 measured logit drift up to 1.25); float32 keeps it."""
    layers = dict(params["layers"])
    for name, ab in adapters.items():
        A, B = jnp.asarray(ab["A"], jnp.float32), jnp.asarray(ab["B"], jnp.float32)
        layers[name] = layers[name].astype(jnp.float32) + (alpha / rank) * jnp.einsum("lir,lro->lio", A, B)
    return {**params, "layers": layers}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "refs").mkdir(exist_ok=True)
    report = json.loads((OUT / "export.json").read_text()) if (OUT / "export.json").exists() else {}
    jax_ref = json.loads((OUT / "jax_reference.json").read_text()) if (OUT / "jax_reference.json").exists() else {}
    want = (lambda n: args.only in (None, n))

    # held-out text (detective val, decoded) shared by every perplexity measurement
    det = detective_corpus()
    from tokenizers import Tokenizer
    det_tok = Tokenizer.from_file(str(det["tokenizer_path"]))
    heldout = det_tok.decode(det["val"][:12000].tolist())
    (OUT / "heldout.txt").write_text(heldout)
    n_bytes = len(heldout.encode())
    report["heldout"] = {"chars": len(heldout), "bytes": n_bytes,
                         "detective_tokens": len(det_tok.encode(heldout).ids)}

    # ---- SmolLM2: the inverse loader must round-trip a real checkpoint bit for bit
    src360 = llama.download(llama.SMOLLM_360M)
    base, cfg, tok = llama.load(llama.SMOLLM_360M, dtype="float32")
    report["heldout"]["smollm_tokens"] = len(tok.encode(heldout).ids)
    if want("smollm2-360m-instruct"):
        from safetensors.numpy import load_file
        raw = load_file(str(next(src360.glob("*.safetensors"))))
        out = ex.export_llama_hf(base, cfg)
        mism = [k for k in raw if k not in out or not np.array_equal(out[k], raw[k].astype(np.float32))]
        report["smollm2-360m-instruct"] = {"tensors": len(raw), "exact_round_trip": not mism and set(out) == set(raw),
                                           "mismatches": mism[:5], "hf_dir": str(src360)}
        print(f"[smollm2] round trip: {len(raw)} tensors, exact={report['smollm2-360m-instruct']['exact_round_trip']}")

    probe = np.asarray(tok.encode(heldout[:2000]).ids[:N_PROBE], np.int32)

    # ---- Module 20: LoRA (JSON order extraction), merged in float32
    a20 = RUNS / "20-lora" / "adapter_qv_r8.npz"
    if want("lora20-orders"):
        if not a20.exists():
            report["lora20-orders"] = {"skipped": "run Module 20 first (runs/20-lora/adapter_qv_r8.npz)"}
        else:
            z = np.load(a20)
            adapters = {k.split("/")[0]: {} for k in z.files}
            for k in z.files:
                adapters[k.split("/")[0]][k.split("/")[1]] = z[k]
            merged = merge_f32(base, adapters, alpha=16.0, rank=8)
            d = write_llama("lora20-orders", merged, cfg, src360)
            np.savez(OUT / "refs" / "lora20-orders.npz", ids=probe, logits=jax_logits_llama(merged, cfg, probe))
            task = _load(MODULES / "20-lora" / "task.py", "m27_task20")
            held = task.make_splits(seed=0)["heldout"][:40]
            end = tok.token_to_id("<|im_end|>")
            ok = []
            for e in held:
                out_ids = llama.generate(merged, cfg, tok.encode(task.chatml_prompt(e["text"])).ids, 40,
                                         jax.random.key(0), temperature=0.0, stop_ids=(end,))
                ok.append(tok.decode([t for t in out_ids if t != end]).strip() == e["answer"])
            jax_ref["lora20-orders"] = {"task": "Module 20 heldout (unseen templates), first 40", "n": len(ok),
                                        "exact_match": float(np.mean(ok))}
            report["lora20-orders"] = {"hf_dir": str(d), "alpha": 16.0, "rank": 8, "targets": sorted(adapters)}
            print(f"[lora20] exported; JAX merged-f32 exact match on 40 heldout = {np.mean(ok):.3f}")

    # ---- Module 26: GRPO adapter, merged in float32
    a26 = RUNS / "26-reasoning" / "adapters.npz"
    if want("grpo26-arith"):
        if not a26.exists():
            report["grpo26-arith"] = {"skipped": "run Module 26's train.py first (runs/26-reasoning/adapters.npz)"}
        else:
            R = _load(MODULES / "26-reasoning" / "reasoning.py", "m27_reasoning26")
            sol26 = load_exercises(MODULES / "26-reasoning", solutions=True)
            z = np.load(a26)
            alpha, rank = float(z["alpha"]), int(z["rank"])
            adapters = R.load_adapters(a26)
            merged = merge_f32(base, adapters, alpha=alpha, rank=rank)
            d = write_llama("grpo26-arith", merged, cfg, src360)
            np.savez(OUT / "refs" / "grpo26-arith.npz", ids=probe, logits=jax_logits_llama(merged, cfg, probe))
            test = R.make_split(1, 64)
            pb, _, _ = llama.load(llama.SMOLLM_360M)                        # bf16, as Module 26 trained
            acc_base, _ = R.greedy_accuracy(pb, {}, cfg_bf16(cfg), tok, test, sol26.verify_answer)
            acc_grpo, _ = R.greedy_accuracy(pb, adapters, cfg_bf16(cfg), tok, test, sol26.verify_answer)
            jax_ref["grpo26-arith"] = {"task": "Module 26 test split (seed 1), first 64, greedy", "n": len(test),
                                       "accuracy_base": acc_base, "accuracy_grpo": acc_grpo}
            report["grpo26-arith"] = {"hf_dir": str(d), "alpha": alpha, "rank": rank, "targets": sorted(adapters)}
            print(f"[grpo26] exported; JAX greedy accuracy base {acc_base:.3f} -> GRPO {acc_grpo:.3f}")

    # ---- Module 16: the detective GPT → GPT-2 layout
    if want("gpt16-detective"):
        run16 = RUNS / "16-gpt"
        if not (run16 / "params.npz").exists():
            report["gpt16-detective"] = {"skipped": "run Module 16's train.py first (runs/16-gpt/params.npz)"}
        else:
            t16 = _load(MODULES / "16-gpt" / "train.py", "m27_train16")
            sol16 = load_exercises(MODULES / "16-gpt", solutions=True)
            p16, c16, _ = t16.load_checkpoint(sol16, run16)
            d = OUT / "hf" / "gpt16-detective"
            d.mkdir(parents=True, exist_ok=True)
            _save_safetensors(d / "model.safetensors", ex.gpt_to_gpt2_state_dict(p16, c16.n_layers))
            hf_cfg = {"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2", "vocab_size": c16.vocab,
                      "n_positions": c16.seq_len, "n_ctx": c16.seq_len, "n_embd": c16.d_model,
                      "n_layer": c16.n_layers, "n_head": c16.n_heads, "n_inner": None,
                      "activation_function": "gelu_new",          # jax.nn.gelu's default is the tanh approximation
                      "layer_norm_epsilon": 1e-5, "tie_word_embeddings": True, "scale_attn_weights": True,
                      "resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0,
                      "bos_token_id": 0, "eos_token_id": 0, "torch_dtype": "float32"}
            (d / "config.json").write_text(json.dumps(hf_cfg, indent=2))
            shutil.copy(det["tokenizer_path"], d / "tokenizer.json")
            tok_cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<|doc|>", "eos_token": "<|doc|>",
                       "model_max_length": c16.seq_len, "clean_up_tokenization_spaces": False}
            (d / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2))
            (d / "special_tokens_map.json").write_text(json.dumps({"bos_token": "<|doc|>", "eos_token": "<|doc|>"}))
            gprobe = np.asarray(det["val"][:N_PROBE], np.int32)
            with jax.default_matmul_precision("highest"):
                glog = np.asarray(sol16.gpt_apply(p16, jnp.asarray(gprobe)[None], c16)[0])
            np.savez(OUT / "refs" / "gpt16-detective.npz", ids=gprobe, logits=glog)
            report["gpt16-detective"] = {"hf_dir": str(d), "n_params": int(sum(x.size for x in jax.tree.leaves(p16)))}
            print(f"[gpt16] exported GPT-2 layout ({report['gpt16-detective']['n_params']:,} params)")

    (OUT / "export.json").write_text(json.dumps(report, indent=2))
    (OUT / "jax_reference.json").write_text(json.dumps(jax_ref, indent=2))
    return 0


def cfg_bf16(cfg):
    import dataclasses
    return dataclasses.replace(cfg, dtype="bfloat16")


if __name__ == "__main__":
    sys.exit(main())
