"""Merge a peft LoRA adapter into Qwen3.5-9B's original safetensors in float32 (§16). Keeps the checkpoint layout
(Qwen3_5ForConditionalGeneration) that convert_hf_to_gguf.py accepts. Merged tensors are written as float32 so the
update isn't rounded (the lesson of §3); tensors without an adapter keep their original bf16 bytes. Prints the
relative size of the update, which §16 shows.

    python merge_f32.py SNAPSHOT_DIR ADAPTER_DIR OUT_DIR"""
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

snap, adir, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((adir / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / cfg["r"]
ad = safe_open(adir / "adapter_model.safetensors", "pt")
deltas = {}
for k in ad.keys():  # noqa: SIM118 (safe_open handles have keys(), not __iter__)
    if ".lora_A." not in k:
        continue
    mod = k.split(".lora_A.")[0].removeprefix("base_model.model.model.")
    A = ad.get_tensor(k).float()
    B = ad.get_tensor(k.replace(".lora_A.", ".lora_B.")).float()
    deltas[f"model.language_model.{mod}.weight"] = (B @ A) * scale
print(f"{len(deltas)} LoRA deltas, scale {scale}")

for f in snap.iterdir():   # configs, tokenizer, templates: copy (resolving HF cache symlinks)
    if not f.name.endswith(".safetensors"):
        shutil.copyfile(f.resolve(), out / f.name)
index = json.loads((snap / "model.safetensors.index.json").read_text())
merged, rel = 0, []
for shard in sorted(set(index["weight_map"].values())):
    t = {}
    with safe_open(snap / shard, "pt") as s:
        for k in s.keys():  # noqa: SIM118 (safe_open handles have keys(), not __iter__)
            w = s.get_tensor(k)
            if k in deltas:
                d = deltas.pop(k)
                assert d.shape == w.shape, (k, d.shape, w.shape)
                wf = w.float()
                rel.append((d.norm() / wf.norm()).item())
                w = wf + d
                merged += 1
            t[k] = w.contiguous()
    save_file(t, out / shard, metadata={"format": "pt"})
    print("wrote", shard, flush=True)
assert not deltas, f"unmatched LoRA modules: {list(deltas)[:5]}"
print(f"merged {merged} tensors; relative update |dW|/|W|: median {sorted(rel)[len(rel)//2]:.2e}, max {max(rel):.2e}")
