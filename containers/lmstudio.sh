#!/usr/bin/env bash
# LM Studio headless (llmster + lms CLI) on the RX 7600 XT, for Module 27.
#
#   containers/lmstudio.sh build                     # build the image (downloads LM Studio's binaries)
#   containers/lmstudio.sh start [vulkan|rocm]       # server on http://127.0.0.1:1234 (host only)
#   containers/lmstudio.sh stop | status | logs
#   containers/lmstudio.sh import <path-in-container> <publisher/repo>   # symlink a GGUF into LM Studio
#   containers/lmstudio.sh lms <args...>             # any lms command, e.g. `lms ls`, `lms load <id>`
#
# The server joins podman network `learn-jax-net` as `lmstudio`, so lessons started with
# NETWORK=learn-jax-net (the Makefile does this automatically) reach it at http://lmstudio:1234.
# Inside the container the repo is mounted read-only at /course and the course cache at /cache, so GGUFs
# under runs/27-lmstudio/gguf (→ /course/runs/...) and /cache/gguf (Module 19) can be imported as symlinks.
# LM Studio's own model index lives in the `learn-jax-lmstudio-models` volume.
# Engines (llama.cpp 2.73.0, Vulkan and ROCm) are pinned in the image; see Containerfile.lmstudio.
set -euo pipefail
IMAGE=localhost/learn-jax-lmstudio
NAME=learn-jax-lmstudio
NET=learn-jax-net
PORT="${PORT:-1234}"
repo="$(cd "$(dirname "$0")/.." && pwd)"

cmd="${1:-help}"; shift || true
case "$cmd" in
  build)
    podman build -t "$IMAGE" -f "$repo/containers/Containerfile.lmstudio" "$repo/containers" ;;
  start)
    engine="${1:-vulkan}"
    podman rm -f "$NAME" >/dev/null 2>&1 || true
    podman network exists "$NET" || podman network create "$NET" >/dev/null
    podman run -d --name "$NAME" --network "$NET" --network-alias lmstudio \
      -p "127.0.0.1:${PORT}:1234" \
      --device /dev/dri --device /dev/kfd --security-opt label=disable --group-add keep-groups \
      -v learn-jax-lmstudio-models:/root/.lmstudio/models \
      -v learn-jax-cache:/cache:ro -v "$repo":/course:ro \
      -e LMS_ENGINE="$engine" "$IMAGE" >/dev/null
    for _ in $(seq 60); do
      curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "LM Studio ($engine) at http://127.0.0.1:${PORT}"; exit 0; }
      sleep 1
    done
    echo "server did not come up; see: containers/lmstudio.sh logs" >&2; exit 1 ;;
  stop)   podman rm -f "$NAME" >/dev/null 2>&1 && echo stopped || echo "not running" ;;
  status) podman exec "$NAME" lms ps 2>/dev/null || echo "not running" ;;
  logs)   podman logs --tail 80 "$NAME" ;;
  import)
    path="$1"; repo_id="$2"
    podman exec "$NAME" lms import -y -l --user-repo "$repo_id" "$path" ;;
  lms)    podman exec "$NAME" lms "$@" ;;
  *) sed -n 2,16p "$0" ;;
esac
