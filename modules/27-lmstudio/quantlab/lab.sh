#!/usr/bin/env bash
# §15 · The quantizer's workflow I: your Module 20 LoRA (SmolLM2-360M, merged in float32 by export.py).
# Importance matrices from two calibration sets × quant types, each scored on the orders task and on perplexity.
# Runs on the host; every step runs in a course container. ~10 minutes. Needs `make export-models` first.
#
#   modules/27-lmstudio/quantlab/lab.sh        # → runs/27-lmstudio/quantlab/results.json
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$here/../../.." && pwd)"
G="$repo/containers/gguf.sh"; LOCK=/tmp/learn-jax-gpu.lock
out="$repo/runs/27-lmstudio/quantlab"; R=/course/runs/27-lmstudio/quantlab
F16=/course/runs/27-lmstudio/gguf/lora20-orders-f16.gguf
[ -s "$repo/runs/27-lmstudio/gguf/lora20-orders-f16.gguf" ] || { echo "run 'make export-models' first"; exit 1; }
mkdir -p "$out"

"$repo/containers/run.sh" python modules/27-lmstudio/quantlab/prep.py orders "$R"

for c in task generic; do       # an importance matrix: which weights the activations lean on, per calibration set
  [ -s "$out/imat-$c.gguf" ] || flock "$LOCK" "$G" imatrix -m "$F16" -f "$R/calib_$c.txt" -o "$R/imat-$c.gguf" \
      -c 256 -ngl 99 > "$out/imat-$c.log" 2>&1
  echo "imatrix $c: $(grep -oE 'computing over [0-9]+ chunks' "$out/imat-$c.log" | tail -1)"
done

for q in Q8_0 Q4_K_M Q3_K_S Q2_K IQ3_XXS IQ2_XS; do
  for c in none task generic; do
    case $q in IQ*) [ $c = none ] && continue;; esac     # i-quants need an imatrix
    case $q in Q8_0) [ $c != none ] && continue;; esac   # Q8_0 ignores it
    f="m-$q-$c"; [ -s "$out/$f.gguf" ] && continue
    extra=(); [ $c != none ] && extra=(--imatrix "$R/imat-$c.gguf")
    "$G" quantize "${extra[@]}" "$F16" "$R/$f.gguf" "$q" > "$out/$f.quantize.log" 2>&1 \
      || { echo "quantize $q/$c failed (see $out/$f.quantize.log)"; continue; }
    echo "$q/$c: $(grep -oE '[0-9.]+ BPW' "$out/$f.quantize.log" | tail -1)"
  done
done

flock "$LOCK" "$G" python /course/modules/27-lmstudio/quantlab/score.py "$R" "$F16"
