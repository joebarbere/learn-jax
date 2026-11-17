#!/usr/bin/env bash
# Run a command in the learn-jax container: GPU passed through, repo at /course, downloads cached in a
# named volume. Usage:
#   containers/run.sh python modules/00-environment/lesson.py
#   PORT=2718 containers/run.sh marimo edit ...              (publishes 127.0.0.1:$PORT)
#   DETACH=1 NAME=learn-jax PORT=2718 containers/run.sh ...  (background)
#   IMAGE=localhost/learn-jax-torch containers/run.sh ...    (PyTorch side-by-side image)
#   NETWORK=learn-jax-net containers/run.sh ...               (reach the llama.cpp server, module 19)
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
image="${IMAGE:-localhost/learn-jax}"
# A course image that isn't built (or was removed to save disk space): say how to get it back.
if ! podman image exists "$image"; then
  case "$image" in
    localhost/learn-jax) hint="make build   (~19 GB; downloads the rocm/jax base if it's gone)" ;;
    localhost/learn-jax-torch) hint="make build-torch   (~26 GB, mostly the rocm/pytorch base; only the PyTorch side-by-side notebooks need it)" ;;
    localhost/learn-jax-book) hint="make book-image   (~1.4 GB)" ;;
    *) hint="podman pull $image" ;;
  esac
  echo "containers/run.sh: image $image is not here. Restore it with: $hint" >&2
  exit 2
fi
args=(--rm)
[[ -t 0 && -t 1 ]] && args+=(-it)
[[ "${DETACH:-}" == 1 ]] && args=(-d --name "${NAME:-learn-jax}")
[[ -n "${PORT:-}" ]] && args+=(-p "127.0.0.1:${PORT}:${PORT}")
# Join the llama.cpp server's network (module 19) so lessons can reach it at http://llama:8080.
[[ -n "${NETWORK:-}" ]] && args+=(--network "$NETWORK")
for v in LEARNJAX_SOLUTIONS LEARNJAX_BOOK JAX_PLATFORMS XLA_FLAGS HSA_OVERRIDE_GFX_VERSION MATPLOTLIBRC; do
  [[ -n "${!v:-}" ]] && args+=(-e "$v=${!v}")
done
exec podman run "${args[@]}" \
  --device /dev/kfd --device /dev/dri \
  --security-opt label=disable --group-add keep-groups --ipc=host \
  -v "$repo":/course -v learn-jax-cache:/cache \
  -w /course "$image" "$@"
