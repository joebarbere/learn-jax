#!/usr/bin/env bash
# Lock the experimental image's packages (requirements-next.txt -> requirements-next.lock), with hashes.
#
# Unlike lock.sh, nothing comes from a base image: JAX, its ROCm plugin and ROCm itself (TheRock wheels) are all
# pip packages. ROCm 10's wheels live only on AMD's index; PyPI has empty placeholder packages with the same
# names (rocm, rocm-sdk-device-gfx1102), so uv must look at AMD's index first (uv's default for
# --extra-index-url) and the lock pins every file by hash.
set -euo pipefail
cd "$(dirname "$0")"
AMD_INDEX=https://stable.repo.amd.com/rocm/whl-next/
uv pip compile requirements-next.txt -o requirements-next.lock --quiet --override overrides-next.txt \
  --extra-index-url "$AMD_INDEX" --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --index-strategy unsafe-best-match --generate-hashes --emit-index-url ${UV_EXCLUDE_NEWER:+--exclude-newer "$UV_EXCLUDE_NEWER"}
echo "wrote containers/requirements-next.lock ($(grep -c '^[a-z]' requirements-next.lock) packages)"
