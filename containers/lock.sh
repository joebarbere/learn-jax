#!/usr/bin/env bash
# Regenerate the lockfiles the course images install from (make lock). For each image:
#
#   containers/requirements[-torch|-book].txt   what you ask for (top-level pins, edited by hand)
#   containers/requirements[-torch|-book].lock  everything the image adds, pinned (generated: don't edit)
#
# The JAX image (lock_jax) has no base image to resolve against: JAX, its ROCm plugin and ROCm itself are pip
# packages, so it is a plain `uv pip compile` with hashes. ROCm 10's wheels live only on AMD's index; PyPI has
# empty placeholder packages with the same names (rocm, rocm-sdk-device-*), so uv must look at AMD's index first
# (unsafe-best-match) and the lock pins every file by hash. overrides.txt lifts stale upper bounds.
#
# The torch and book images: why not `uv pip compile`? Their base images ship packages PyPI doesn't have (the ROCm JAX plugin,
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

lock_jax() {  # runs uv on the host: it only resolves versions, nothing is installed
  echo "[jax] resolving requirements.txt (PyPI + AMD's ROCm index)"
  uv pip compile requirements.txt -o requirements.lock --quiet --override overrides.txt \
    --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ --python-version 3.12 \
    --python-platform x86_64-manylinux_2_28 --index-strategy unsafe-best-match --generate-hashes --emit-index-url \
    ${UV_EXCLUDE_NEWER:+--exclude-newer "$UV_EXCLUDE_NEWER"}
  echo "[jax] wrote containers/requirements.lock ($(grep -c '^[a-z]' requirements.lock) packages)"
}

case "${1:-all}" in
  jax|all) lock_jax ;;&
  torch|all) lock_one torch Containerfile.torch requirements-torch.txt requirements-torch.lock ;;&
  book|all) lock_one book Containerfile.book requirements-book.txt requirements-book.lock ;;
  jax|torch) ;;
  *) echo "usage: $0 [jax|torch|book|all]" >&2; exit 2 ;;
esac
