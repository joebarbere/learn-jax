#!/usr/bin/env bash
# Container entrypoint for LM Studio headless (llmster): select the engine, start the daemon and the
# OpenAI/Anthropic/REST-compatible server on 0.0.0.0:1234 *inside* the container (the host only publishes
# it on 127.0.0.1), then follow the server log so the container stays up.
set -euo pipefail
engine="${LMS_ENGINE:-vulkan}"
case "$engine" in
  vulkan) pack="llama.cpp-linux-x86_64-vulkan-avx2@${LMS_RUNTIME_VERSION}" ;;
  rocm)   pack="llama.cpp-linux-x86_64-amd-rocm-avx2@${LMS_RUNTIME_VERSION}" ;;
  *) echo "LMS_ENGINE must be vulkan or rocm" >&2; exit 2 ;;
esac
lms daemon up
# the daemon indexes its engines asynchronously after `daemon up` returns: retry the selection briefly
for _ in $(seq 30); do lms runtime select "$pack" >/dev/null 2>&1 && break; sleep 1; done
lms runtime select "$pack"
lms server start --port 1234 --bind 0.0.0.0
echo "LM Studio server up: engine=$pack"
exec lms log stream
