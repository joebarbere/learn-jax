#!/usr/bin/env bash
# Convert every export to GGUF, quantize, and measure perplexity on held-out text (runs on the host; each
# step runs in ggml-org's pinned llama.cpp image via containers/gguf.sh). Run after export.py (+ verify_hf.py).
#
#   modules/27-lmstudio/build_gguf.sh            # writes runs/27-lmstudio/gguf/*.gguf and gguf_quality.json
#
# Base precision per model:
#   smollm2-360m-instruct  bf16 (the published weights are bf16)
#   lora20 / grpo26        f16 AND bf16 from the float32 merge — f16 keeps 3 more mantissa bits, which is
#                          what a small LoRA update needs to survive (Module 20); the lesson compares both
#   gpt16-detective        f32 → f16 (tiny model; its 384-wide rows aren't multiples of 256, so k-quants
#                          fall back to Q5_0/Q8_0 for some tensors — llama-quantize says so)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$here/../.." && pwd)"
G="$repo/containers/gguf.sh"
R=/course/runs/27-lmstudio
out="$repo/runs/27-lmstudio"
mkdir -p "$out/gguf" "$out/ppl"
snap360=$(podman run --rm -v learn-jax-cache:/cache docker.io/library/ubuntu:24.04 \
  bash -c 'ls -d /cache/hf/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/* | head -1')

convert() {  # name src_dir outtype
  local f="$out/gguf/$1-$3.gguf"
  [ -s "$f" ] && return 0
  if [ "$1" = gpt16-detective ]; then
    "$G" python /course/modules/27-lmstudio/convert_custom_bpe.py "$2" --outfile "$R/gguf/$1-$3.gguf" --outtype "$3" 2>&1 | tail -1
  else
    "$G" convert "$2" "$R/gguf/$1-$3.gguf" --outtype "$3" 2>&1 | tail -1
  fi
}
quant() {    # name base_type QTYPE
  local f="$out/gguf/$1-${3,,}.gguf"
  [ -s "$f" ] && return 0
  "$G" quantize "$R/gguf/$1-$2.gguf" "$R/gguf/$1-${3,,}.gguf" "$3" > "$out/gguf/$1-${3,,}.quantize.log" 2>&1 || echo "quantize failed: $1 $3 (see $out/gguf/$1-${3,,}.quantize.log)"
  grep -m3 -E "fallback|quant size" "$out/gguf/$1-${3,,}.quantize.log" || true
}
ppl() {      # name type ctx
  local log="$out/ppl/$1-$2.log"
  [ -s "$log" ] && grep -q "Final estimate" "$log" && return 0
  "$G" perplexity -m "$R/gguf/$1-$2.gguf" -f "$R/heldout.txt" -c "$3" -ngl 99 > "$log" 2>&1 || true
  grep -o "Final estimate: PPL = [0-9.]*" "$log" || echo "perplexity failed for $1-$2 (see $log)"
}

snap135=$(podman run --rm -v learn-jax-cache:/cache docker.io/library/ubuntu:24.04 \
  bash -c 'ls -d /cache/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/* | head -1')
convert smollm2-135m "$snap135" bf16       # same tokenizer as the 360M: a speculative-decoding draft model
convert smollm2-360m-instruct "$snap360" bf16
for q in Q8_0 Q4_K_M; do quant smollm2-360m-instruct bf16 "$q"; done
for t in bf16 q8_0 q4_k_m; do ppl smollm2-360m-instruct "$t" 512; done

for m in lora20-orders grpo26-arith; do
  [ -d "$out/hf/$m" ] || { echo "skip $m (run export.py)"; continue; }
  convert "$m" "$R/hf/$m" f16
  convert "$m" "$R/hf/$m" bf16
  for q in Q8_0 Q4_K_M; do quant "$m" f16 "$q"; done
  for t in f16 bf16 q8_0 q4_k_m; do ppl "$m" "$t" 512; done
done

if [ -d "$out/hf/gpt16-detective" ]; then
  convert gpt16-detective "$R/hf/gpt16-detective" f32
  convert gpt16-detective "$R/hf/gpt16-detective" f16
  for q in Q8_0 Q4_K_M; do quant gpt16-detective f16 "$q"; done
  for t in f32 f16 q8_0 q4_k_m; do ppl gpt16-detective "$t" 256; done
fi

python3 - "$out" <<'EOF'
import json, re, sys
from pathlib import Path
out = Path(sys.argv[1]); res = {}
for log in sorted((out / "ppl").glob("*.log")):
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", log.read_text())
    name, typ = log.stem.rsplit("-", 1)
    gg = out / "gguf" / f"{log.stem}.gguf"
    res.setdefault(name, {})[typ] = {"ppl": float(m.group(1)) if m else None,
                                     "size_mb": round(gg.stat().st_size / 1e6, 1) if gg.exists() else None}
(out / "gguf_quality.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
EOF
