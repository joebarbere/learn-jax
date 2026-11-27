#!/usr/bin/env bash
# §16 · The quantizer's workflow II: a real 9B. QLoRA-fine-tune Qwen3.5-9B on NL-RX, merge in float32, convert to
# an f32 GGUF, build importance matrices, quantize Q8_0 → IQ2_XXS, score every quant, and import the pick into
# LM Studio. Optional and heavy: ~2.5 hours on the RX 7600 XT, ~90 GB of free disk at the peak (~50 GB kept).
# Needs the PyTorch image (make build-torch) and `make export-models` (for the held-out prose).
#
#   make quant-lab-9b      # → runs/27-lmstudio/quantlab9b/summary.json (the lesson reads it)
#
# Every step skips work whose output already exists, so an interrupted run resumes. KEEP_F32=1 keeps the 31 GB
# merged checkpoint and the 37 GB f32 GGUF (deleted by default once the quants exist; this script rebuilds them).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$here/../../.." && pwd)"
G="$repo/containers/gguf.sh"; LOCK=/tmp/learn-jax-gpu.lock
out="$repo/runs/27-lmstudio/quantlab9b"; R=/course/runs/27-lmstudio/quantlab9b; Q=/course/modules/27-lmstudio/quantlab
mkdir -p "$out"
podman image exists localhost/learn-jax-torch || { echo "needs the PyTorch image: make build-torch"; exit 1; }
[ -s "$repo/runs/27-lmstudio/heldout.txt" ] || { echo "run 'make export-models' first"; exit 1; }
avail=$(df --output=avail -BG "$out" | tail -1 | tr -dc 0-9)
[ -s "$out/m-Q8_0-none.gguf" ] || [ "$avail" -ge 90 ] || { echo "needs ~90 GB free here, have ${avail} GB"; exit 1; }

# QLoRA needs bitsandbytes, which the PyTorch image doesn't ship: install the pinned wheel into each throwaway
# container. (0.50.2 has no ROCm 7.16 build and loads its 7.15 one with a warning; that works.)
torch() { flock "$LOCK" env IMAGE=localhost/learn-jax-torch "$repo/containers/run.sh" \
            sh -c "uv pip install --system -q bitsandbytes==0.50.2 && $*"; }

IMAGE=localhost/learn-jax-torch "$repo/containers/run.sh" python $Q/prep.py nlrx $R

# 1 · Fine-tune (~50 min) and score against the base model
[ -s "$out/adapter/adapter_model.safetensors" ] || torch python $Q/qlora_nlrx.py train --out $R/adapter
[ -s "$out/eval-base-5shot.json" ] || torch python $Q/qlora_nlrx.py eval --shots 5 --out $R/eval-base-5shot.json
[ -s "$out/eval-ft.json" ] || torch python $Q/qlora_nlrx.py eval --adapter $R/adapter --out $R/eval-ft.json

# 2 · Merge in float32 and convert to an f32 GGUF: the reference every quant is made from
if [ ! -s "$out/m-Q8_0-none.gguf" ] || [ ! -s "$out/m-IQ2_XXS-generic.gguf" ]; then
  if [ ! -s "$out/f32.gguf" ]; then
    [ -s "$out/merged/model.safetensors.index.json" ] || IMAGE=localhost/learn-jax-torch "$repo/containers/run.sh" sh -c \
      "python $Q/merge_f32.py \$(ls -d /cache/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/* | head -1) $R/adapter $R/merged" \
      | tee "$out/merge.log" | tail -1
    "$G" convert $R/merged $R/f32.gguf --outtype f32 2>&1 | tail -1
    [ "${KEEP_F32:-0}" = 1 ] || rm -rf "$out/merged"
  fi
fi

# 3 · Q8_0, then importance matrices from it on the GPU (the f32 file is 37 GB, too big for 16 GB of VRAM)
[ -s "$out/m-Q8_0-none.gguf" ] || "$G" quantize $R/f32.gguf $R/m-Q8_0-none.gguf Q8_0 > "$out/m-Q8_0-none.quantize.log" 2>&1
for c in task generic; do
  [ -s "$out/imat-$c.gguf" ] || flock "$LOCK" "$G" imatrix -m $R/m-Q8_0-none.gguf -f $R/calib_$c.txt \
      -o $R/imat-$c.gguf -c 512 -ngl 99 > "$out/imat-$c.log" 2>&1
  echo "imatrix $c: $(grep -oE 'computing over [0-9]+ chunks' "$out/imat-$c.log" | tail -1)"
done

# 4 · Lower-bit quants. Qwen3.5's multi-token-prediction head (blk.32, used only for speculative decoding) never runs
# while the imatrix is collected, so it has no importance data and the very-low-bit types refuse to quantize it
# blind ("Missing importance matrix ... The result will be garbage"): keep it at Q8_0 (+0.26 GB).
for q in Q4_K_M IQ4_XS IQ3_XXS IQ2_XXS; do
  for c in none task generic; do
    case $q in IQ*) [ $c = none ] && continue;; esac
    f="m-$q-$c"; [ -s "$out/$f.gguf" ] && continue
    extra=(); [ $c != none ] && extra=(--imatrix $R/imat-$c.gguf)
    case $q in IQ3*|IQ2*) extra+=(--tensor-type 'blk\.32\.=q8_0');; esac
    "$G" quantize "${extra[@]}" $R/f32.gguf $R/$f.gguf $q > "$out/$f.quantize.log" 2>&1 \
      || { echo "quantize $q/$c failed (see $out/$f.quantize.log)"; continue; }
    echo "$q/$c: $(grep -oE '[0-9.]+ BPW' "$out/$f.quantize.log" | tail -1)"
  done
done
[ "${KEEP_F32:-0}" = 1 ] || rm -f "$out/f32.gguf"

# 5 · Score every quant (~3 min each) and summarize
flock "$LOCK" "$G" python $Q/score.py $R
"$G" python $Q/summarize.py $R

# 6 · The pick, for LM Studio: IQ4_XS with the task imatrix, with thinking forced off in its chat template.
# The fine-tune only ever saw the empty-<think> format; LM Studio passes enable_thinking=true from its own setting,
# so a template default isn't enough.
if [ ! -s "$out/qwen3.5-9b-nlrx-iq4_xs.gguf" ]; then
  "$G" python - <<EOF
import sys; sys.path.insert(0, "/app/gguf-py")
from gguf import GGUFReader
f = GGUFReader("$R/m-IQ4_XS-task.gguf").fields["tokenizer.chat_template"]
t = bytes(f.parts[f.data[0]]).decode()
open("$R/chat_template_nothink.jinja", "w").write("{%- set enable_thinking = false %}\n" + t)
EOF
  "$G" python /app/gguf-py/gguf/scripts/gguf_new_metadata.py $R/m-IQ4_XS-task.gguf $R/qwen3.5-9b-nlrx-iq4_xs.gguf \
      --chat-template "$(cat "$out/chat_template_nothink.jinja")" --force 2>&1 | tail -1
fi
if podman container exists learn-jax-lmstudio; then
  podman exec learn-jax-lmstudio test -e /root/.lmstudio/models/learnjax/qwen3.5-9b-nlrx-iq4_xs/qwen3.5-9b-nlrx-iq4_xs.gguf \
    || "$repo/containers/lmstudio.sh" import $R/qwen3.5-9b-nlrx-iq4_xs.gguf learnjax/qwen3.5-9b-nlrx-iq4_xs 2>&1 | tail -1
else
  echo "LM Studio isn't running: 'make lms-up', then re-run this script to import learnjax/qwen3.5-9b-nlrx-iq4_xs"
fi
