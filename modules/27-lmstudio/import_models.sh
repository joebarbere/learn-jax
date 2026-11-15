#!/usr/bin/env bash
# Import the course's GGUF files into the running LM Studio container (symlinks — no copies), one
# publisher/repo per file so every quant gets its own model key: learnjax/<file stem>.
#   modules/27-lmstudio/import_models.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$here/../.." && pwd)"
L="$repo/containers/lmstudio.sh"
for f in "$repo"/runs/27-lmstudio/gguf/*.gguf; do
  stem="$(basename "$f" .gguf)"
  podman exec learn-jax-lmstudio test -e "/root/.lmstudio/models/learnjax/$stem/$stem.gguf" && continue
  "$L" import "/course/runs/27-lmstudio/gguf/$stem.gguf" "learnjax/$stem" 2>&1 | tail -1
done
# Qwen2.5-1.5B-Instruct (Apache-2.0, ~1 GB): the lesson's tool-calling model (Qwen2.5-Coder's GGUF template
# doesn't produce calls LM Studio parses — see §9). Downloaded once into the course cache.
"$repo/containers/run.sh" python -c "from huggingface_hub import hf_hub_download as d; \
d('Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='/cache/gguf')" >/dev/null
podman exec learn-jax-lmstudio test -e /root/.lmstudio/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  || "$L" import /cache/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf Qwen/Qwen2.5-1.5B-Instruct-GGUF 2>&1 | tail -1
# Module 19's coding models, if downloaded
for f in qwen2.5-coder-1.5b-instruct-q4_k_m; do
  podman exec learn-jax-lmstudio test -e "/cache/gguf/$f.gguf" || continue
  podman exec learn-jax-lmstudio test -e "/root/.lmstudio/models/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/$f.gguf" && continue
  "$L" import "/cache/gguf/$f.gguf" "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF" 2>&1 | tail -1
done
"$L" lms ls
