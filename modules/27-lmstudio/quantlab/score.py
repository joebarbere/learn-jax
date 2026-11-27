"""Score GGUF models on a task (exact match, greedy) through llama-server, plus perplexity. Runs in llama.cpp's image.

    python3 score.py DIR [REFERENCE.gguf ...]

Scores DIR/m-<QUANT>-<CALIB>.gguf (CALIB: none, task or generic imatrix) and any extra reference models on
DIR/tests.jsonl, measures perplexity on DIR/ppl_text.txt, and writes DIR/results.json. Already-scored models are
skipped, so an interrupted run resumes. Each entry also records how often the model gives exactly the same answer
as the highest-precision model in the run: a steadier signal than exact match, which is noisy at a few hundred
examples.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DIR = sys.argv[1]
URL = "http://127.0.0.1:8099"
rows = [json.loads(line) for line in open(f"{DIR}/tests.jsonl")]
path = f"{DIR}/results.json"
res = json.load(open(path)) if os.path.exists(path) else {}


def post(body):
    r = urllib.request.Request(URL + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))


def bpw(log_path):
    """Real bits per weight from llama-quantize's log ("quant size = 5129.21 MiB (4.68 BPW)")."""
    if os.path.exists(log_path):
        m = re.search(r"quant size\s*=\s*[\d.]+ MiB \(([\d.]+) BPW\)", open(log_path).read())
        return float(m.group(1)) if m else None
    return None


models = sys.argv[2:] + sorted(glob.glob(f"{DIR}/m-*.gguf"))
for m in models:
    base = os.path.basename(m).removesuffix(".gguf")
    name = base[2:] if base.startswith("m-") else base.rsplit("-", 1)[-1].upper() + "-none"
    if name in res:
        continue
    srv = subprocess.Popen(["/app/llama-server", "-m", m, "-ngl", "99", "-c", "4096", "-np", "4", "--port", "8099"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(600):
        try:
            urllib.request.urlopen(URL + "/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    t0 = time.time()
    with ThreadPoolExecutor(4) as pool:     # 4 server slots
        outs = list(pool.map(lambda r: post({"prompt": r["prompt"], "n_predict": 96, "temperature": 0,
                                             "stop": ["<|im_end|>"], "cache_prompt": False})["content"].strip(), rows))
    srv.terminate()
    srv.wait()
    seconds = time.time() - t0
    ppl = subprocess.run(["/app/llama-perplexity", "-m", m, "-f", f"{DIR}/ppl_text.txt", "-c", "512", "-ngl", "99"],
                         capture_output=True, text=True)
    pm = re.search(r"Final estimate: PPL = ([\d.]+)", ppl.stdout + ppl.stderr)
    ok = sum(o == r["answer"] for o, r in zip(outs, rows))
    res[name] = {"exact_match": ok / len(rows), "n": len(rows), "ppl": float(pm.group(1)) if pm else None,
                 "size_gb": round(os.path.getsize(m) / 1e9, 3), "bpw": bpw(m.removesuffix(".gguf") + ".quantize.log"),
                 "seconds": round(seconds), "outputs": outs}
    print(f"{name:18s} {res[name]['size_gb']:6.2f} GB  exact {ok}/{len(rows)} = {ok / len(rows):5.1%}  "
          f"ppl {res[name]['ppl']}  ({seconds:.0f}s)", flush=True)
    json.dump(res, open(path, "w"), indent=1)

ref = max(res, key=lambda k: res[k]["size_gb"])        # the highest-precision model scored
for k, v in res.items():
    v["agree_with_ref"] = sum(a == b for a, b in zip(v["outputs"], res[ref]["outputs"])) / len(rows)
    v["ref"] = ref
json.dump(res, open(path, "w"), indent=1)
