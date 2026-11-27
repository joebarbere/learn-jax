#!/usr/bin/env bash
# llama.cpp's model tools (convert_hf_to_gguf.py, llama-quantize, llama-perplexity, llama-imatrix) for Module 27, from
# ggml-org's official `full-vulkan` image at the same build Module 19 serves with (b12375), pinned by digest.
# The repo is mounted at /course and the course cache volume at /cache (HF downloads, GGUF files).
#
#   containers/gguf.sh convert <hf_model_dir> <out.gguf> [--outtype bf16|f16|q8_0]
#   containers/gguf.sh quantize <in.gguf> <out.gguf> <Q8_0|Q4_K_M|...>
#   containers/gguf.sh perplexity -m <model.gguf> -f <text file> [-c 512 ...]
#   containers/gguf.sh imatrix -m <model.gguf> -f <calibration text> -o <imatrix.gguf> [-c 512 -ngl 99 ...]
#   containers/gguf.sh python <script.py> [args...]   (e.g. modules/27-lmstudio/convert_custom_bpe.py)
#   containers/gguf.sh shell
set -euo pipefail
IMAGE="ghcr.io/ggml-org/llama.cpp:full-vulkan-b12375@sha256:b3e2c930b4d67647dc42431d0fea19304d04b92564fd5d5befad0a3554d2eaf5"
repo="$(cd "$(dirname "$0")/.." && pwd)"
run=(podman run --rm -i --device /dev/dri --security-opt label=disable --group-add keep-groups
     -v "$repo":/course -v learn-jax-cache:/cache -w /course -e LD_LIBRARY_PATH=/app --entrypoint "")
cmd="${1:-help}"; shift || true
case "$cmd" in
  convert)    in="$1"; out="$2"; shift 2
              exec "${run[@]}" "$IMAGE" python3 /app/convert_hf_to_gguf.py "$in" --outfile "$out" "$@" ;;
  quantize)   exec "${run[@]}" "$IMAGE" /app/llama-quantize "$@" ;;
  perplexity) exec "${run[@]}" "$IMAGE" /app/llama-perplexity "$@" ;;
  imatrix)    exec "${run[@]}" "$IMAGE" /app/llama-imatrix "$@" ;;
  python)     exec "${run[@]}" "$IMAGE" python3 "$@" ;;
  shell)      exec "${run[@]}" -it "$IMAGE" bash ;;
  *) sed -n 2,12p "$0" ;;
esac
