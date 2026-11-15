#!/usr/bin/env bash
# "The same model, four ways": SmolLM2-360M-Instruct generation speed in JAX, raw llama.cpp, and LM Studio
# on its Vulkan and ROCm engines — plus LM Studio speculative decoding (135M draft). Runs on the host; run
# on an otherwise idle GPU. Restarts the LM Studio container (Vulkan, then ROCm, then Vulkan again).
#
#   modules/27-lmstudio/bench.sh        # writes runs/27-lmstudio/bench.json
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$here/../.." && pwd)"
out="$repo/runs/27-lmstudio"; L="$repo/containers/lmstudio.sh"; G="$repo/containers/gguf.sh"
rows="$out/bench.rows.jsonl"; : > "$rows"
PROMPT="Write a short story about a lighthouse keeper, about 150 words."

echo "[1/4] JAX"
"$repo/containers/run.sh" python modules/27-lmstudio/bench_jax.py 2>/dev/null | tail -1 >> "$rows"

echo "[2/4] raw llama.cpp (llama-bench, Vulkan)"
for q in bf16 q8_0 q4_k_m; do
  "$G" python - <<EOF >> "$rows"
import json, subprocess
r = subprocess.run(["/app/llama-bench", "-m", "/course/runs/27-lmstudio/gguf/smollm2-360m-instruct-$q.gguf",
                    "-ngl", "99", "-p", "512", "-n", "128", "-o", "json"], capture_output=True, text=True)
for t in json.loads(r.stdout):
    if t["n_gen"]:
        print(json.dumps({"engine": "llama.cpp b12375 (Vulkan, llama-bench)", "model": "SmolLM2-360M-Instruct $q",
                          "gen_tok_s": t["avg_ts"]}))
EOF
done

lms_bench() {  # engine model_key [label_suffix]
  python3 - "$1" "$2" "${3:-}" <<'EOF'
import json, statistics, sys, urllib.request
engine, key, suffix = sys.argv[1:4]
B = "http://127.0.0.1:1234"
body = {"model": key, "input": "Write a short story about a lighthouse keeper, about 150 words.",
        "temperature": 0, "max_output_tokens": 128}
tps = []
for _ in range(4):
    r = urllib.request.Request(B + "/api/v1/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    tps.append(json.loads(urllib.request.urlopen(r, timeout=300).read())["stats"]["tokens_per_second"])
print(json.dumps({"engine": f"LM Studio 0.4.28 / llmster 0.0.27 ({engine}, llama.cpp 2.73.0){suffix}",
                  "model": "SmolLM2-360M-Instruct " + key.split("@")[-1], "gen_tok_s": statistics.median(tps[1:])}))
EOF
}

for engine in vulkan rocm; do
  echo "[3/4] LM Studio ($engine)"
  "$L" start "$engine" >/dev/null
  "$here/import_models.sh" >/dev/null 2>&1 || true
  for q in bf16 q8_0 q4_k_m; do
    "$L" lms unload --all >/dev/null 2>&1 || true
    "$L" lms load "smollm2-360m-instruct@$q" -y --gpu max >/dev/null 2>&1
    lms_bench "$engine" "smollm2-360m-instruct@$q" >> "$rows"
  done
done

echo "[4/4] LM Studio speculative decoding (Vulkan): 360M target, 135M draft"
"$L" start vulkan >/dev/null
"$L" lms unload --all >/dev/null 2>&1 || true
"$L" lms load "smollm2-360m-instruct@bf16" -y --gpu max >/dev/null 2>&1
lms_bench vulkan "smollm2-360m-instruct@bf16" ", no draft" >> "$rows"
"$L" lms unload --all >/dev/null 2>&1 || true
"$L" lms load "smollm2-360m-instruct@bf16" -y --gpu max --identifier spec360 \
  --speculative-draft-simple --speculative-draft-model smollm2-135m >/dev/null 2>&1
python3 - <<'EOF' >> "$rows"
import json, statistics, urllib.request
B = "http://127.0.0.1:1234"
body = {"model": "spec360", "input": "Write a short story about a lighthouse keeper, about 150 words.",
        "temperature": 0, "max_output_tokens": 128}
tps, stats = [], None
for _ in range(4):
    r = urllib.request.Request(B + "/api/v1/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    stats = json.loads(urllib.request.urlopen(r, timeout=300).read())["stats"]
    tps.append(stats["tokens_per_second"])
print(json.dumps({"engine": "LM Studio 0.4.28 / llmster 0.0.27 (vulkan, llama.cpp 2.73.0), 135M draft",
                  "model": "SmolLM2-360M-Instruct bf16", "gen_tok_s": statistics.median(tps[1:]), "stats": stats}))
EOF
"$L" lms unload --all >/dev/null 2>&1 || true

python3 - "$rows" "$out/bench.json" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
json.dump({"measured": __import__("time").strftime("%Y-%m-%d"), "rows": rows}, open(sys.argv[2], "w"), indent=2)
for r in rows: print(f"{r['gen_tok_s']:8.1f} tok/s  {r['model']:34} {r['engine']}")
EOF
