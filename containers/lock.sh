#!/usr/bin/env bash
# Regenerate the lockfiles the course images install from (make lock). For each image:
#
#   containers/requirements[-torch|-book].txt   what you ask for (top-level pins, edited by hand)
#   containers/requirements[-torch|-book].lock  everything the image adds, pinned (generated: don't edit)
#
# Why not `uv pip compile`? The base images ship packages PyPI doesn't have (the ROCm JAX plugin,
# torch 2.14.0+rocm10.1.0, the rocm-sdk wheels), so a resolver working from PyPI alone would pick the CUDA
# torch and its nvidia-* dependencies. Instead, this resolves *inside the real base image*: it installs the
# requirements with uv while every preinstalled package is held at its exact version, and records the
# packages that were added (or moved because requirements.txt pins them by name). The Containerfile then
# installs that list with --no-deps, so a build can neither pull in anything unlisted nor move a package the
# base image provides. Set UV_EXCLUDE_NEWER=<date> to ignore releases after a date (reproducible re-locks).
#
#   containers/lock.sh                 # all three
#   containers/lock.sh torch           # one: jax | torch | book
set -euo pipefail
cd "$(dirname "$0")"
UV_VERSION=0.13.6   # keep in step with the COPY --from=ghcr.io/astral-sh/uv:… lines in the Containerfiles

lock_one() {  # lock_one <name> <Containerfile> <requirements.txt> <requirements.lock>
  local name=$1 file=$2 req=$3 out=$4 base
  base=$(sed -n 's/^FROM //p' "$file" | head -1)
  echo "[$name] resolving $req against $base"
  podman run --rm --security-opt label=disable -v "$PWD":/c -e REQ="$req" -e OUT="$out" -e BASE="$base" \
    -e UV_VERSION="$UV_VERSION" -e UV_EXCLUDE_NEWER="${UV_EXCLUDE_NEWER:-}" --entrypoint sh "$base" -euc '
    [ -n "$UV_EXCLUDE_NEWER" ] || unset UV_EXCLUDE_NEWER
    python3 -m pip install -q --disable-pip-version-check --root-user-action=ignore --target /tmp/uv "uv==$UV_VERSION"
    python3 /c/lockdiff.py freeze > /tmp/before.txt
    python3 /c/lockdiff.py hold "/c/$REQ" /tmp/before.txt > /tmp/hold.txt
    /tmp/uv/bin/uv pip install --system --quiet -r "/c/$REQ" -c /tmp/hold.txt
    python3 /c/lockdiff.py freeze > /tmp/after.txt
    python3 /c/lockdiff.py write "/c/$REQ" "/c/$OUT" /tmp/before.txt /tmp/after.txt'
  echo "[$name] wrote containers/$out"
}

case "${1:-all}" in
  jax|all) lock_one jax Containerfile requirements.txt requirements.lock ;;&
  torch|all) lock_one torch Containerfile.torch requirements-torch.txt requirements-torch.lock ;;&
  book|all) lock_one book Containerfile.book requirements-book.txt requirements-book.lock ;;
  jax|torch) ;;
  *) echo "usage: $0 [jax|torch|book|all]" >&2; exit 2 ;;
esac
