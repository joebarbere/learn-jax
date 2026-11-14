#!/usr/bin/env bash
# llama.cpp's OpenAI-compatible server on the RX 7600 XT, for Module 19 (local serving).
#
#   containers/llamacpp.sh download                 # the course's GGUF models → learn-jax-cache:/cache/gguf (~12 GB)
#   containers/llamacpp.sh start [vulkan|rocm] [model.gguf] [extra llama-server args...]
#   containers/llamacpp.sh stop | status | logs | models
#
# The server listens on http://127.0.0.1:8080 (host only — for editors/CLIs on this machine) and joins the
# podman network `learn-jax-net` as `llama`, so course containers started with NETWORK=learn-jax-net reach
# it at http://llama:8080. (Rootless pasta maps host.containers.internal to the host's LAN address, not its
# loopback, so a 127.0.0.1-only port isn't reachable that way — by design.)
# Images are ggml-org's official builds, pinned by digest:
#   vulkan: needs only /dev/dri (Mesa RADV) — no ROCm at all, 0.9 GB image
#   rocm:   HIP build (ROCm 7.2.1, gfx1102 is a native target) — needs /dev/kfd too, 23 GB image
set -euo pipefail

TAG=b12375
VULKAN_IMAGE="ghcr.io/ggml-org/llama.cpp:server-vulkan-${TAG}@sha256:b6016ab1fe392f163a0dc601161820685321c70c414c496104e370a3396ee9f6"
ROCM_IMAGE="ghcr.io/ggml-org/llama.cpp:server-rocm-${TAG}@sha256:eeb8278276991fe964e9b602f088d5c5ac3d75ba5833bd9bf6fe50907ad33f7d"
NAME=learn-jax-llama
NET=learn-jax-net
PORT="${PORT:-8080}"
DEFAULT_MODEL=qwen2.5-coder-14b-instruct-q4_k_m.gguf
# repo  file  (all Apache-2.0, from Qwen's official GGUF repos)
MODELS=(
  "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
  "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF qwen2.5-coder-1.5b-instruct-q8_0.gguf"
  "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF qwen2.5-coder-14b-instruct-q4_k_m.gguf"
)
here="$(cd "$(dirname "$0")" && pwd)"

cmd="${1:-help}"; shift || true
case "$cmd" in
  download)
    for m in "${MODELS[@]}"; do
      read -r repo file <<<"$m"
      "$here/run.sh" python -c "from huggingface_hub import hf_hub_download as d; print(d('$repo', '$file', local_dir='/cache/gguf'))"
    done ;;
  start)
    backend="${1:-vulkan}"; model="${2:-$DEFAULT_MODEL}"; shift $(( $# > 2 ? 2 : $# )) || true
    case "$backend" in
      vulkan) image="$VULKAN_IMAGE"; devices=(--device /dev/dri) ;;
      rocm)   image="$ROCM_IMAGE";   devices=(--device /dev/kfd --device /dev/dri) ;;
      *) echo "backend must be vulkan or rocm" >&2; exit 2 ;;
    esac
    podman rm -f "$NAME" >/dev/null 2>&1 || true
    podman network exists "$NET" || podman network create "$NET" >/dev/null
    podman run -d --name "$NAME" --network "$NET" --network-alias llama "${devices[@]}" \
      --security-opt label=disable --group-add keep-groups \
      -p "127.0.0.1:${PORT}:8080" -v learn-jax-cache:/cache:ro \
      -e LLAMA_ARG_HOST=0.0.0.0 -e LLAMA_ARG_PORT=8080 \
      "$image" -m "/cache/gguf/$model" --alias "${model%.gguf}" \
      -ngl 999 -c "${CTX:-8192}" --jinja --metrics "$@" >/dev/null
    echo "starting $backend server with $model on http://127.0.0.1:${PORT} ..."
    for _ in $(seq 1 120); do
      if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then echo "ready"; exit 0; fi
      if ! podman container exists "$NAME" || [[ "$(podman inspect -f '{{.State.Running}}' "$NAME")" != true ]]; then
        echo "server exited:"; podman logs --tail 30 "$NAME"; exit 1
      fi
      sleep 1
    done
    echo "not ready after 120 s — see: containers/llamacpp.sh logs"; exit 1 ;;
  stop)   podman rm -f "$NAME" >/dev/null 2>&1 && echo "stopped" || echo "not running" ;;
  status) curl -s "http://127.0.0.1:${PORT}/health" && echo || echo "not running" ;;
  logs)   podman logs --tail "${1:-40}" "$NAME" ;;
  models) "$here/run.sh" bash -c 'ls -lh /cache/gguf/*.gguf 2>/dev/null || echo "none yet: containers/llamacpp.sh download"' ;;
  *) sed -n '2,12p' "$0" ;;
esac
