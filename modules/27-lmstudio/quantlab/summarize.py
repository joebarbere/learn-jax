"""Condense a quantization lab's outputs into summary.json (no per-example outputs): what §16 reads.

    python3 summarize.py DIR [OUT.json]

Reads DIR/results.json (score.py), and if present DIR/adapter/train_log.json, DIR/eval-*.json and DIR/merge.log.
modules/27-lmstudio/quantlab/reference_9b.json is this file from the course author's run, so the lesson can show
§16's numbers without the 2.5-hour run.
"""
import json
import os
import re
import sys
from pathlib import Path

d = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "summary.json")
res = json.loads(Path(d, "results.json").read_text())
s = {"quants": {k: {f: v[f] for f in ("exact_match", "n", "ppl", "size_gb", "bpw", "agree_with_ref", "ref")}
                for k, v in res.items()}}
tl = os.path.join(d, "adapter", "train_log.json")
if os.path.exists(tl):
    t = json.loads(Path(tl).read_text())
    s["train"] = {k: t[k] for k in ("steps", "skipped", "minutes", "peak_gib")}
    s["train"]["loss"] = [[r["step"], r["loss"]] for r in t["log"]][::5]       # every 5th step is plenty to plot
for name in ("base-5shot", "ft"):
    p = os.path.join(d, f"eval-{name}.json")
    if os.path.exists(p):
        e = json.loads(Path(p).read_text())
        s[f"eval_{name.replace('-', '_')}"] = {k: e[k] for k in ("exact_match", "n", "shots", "seconds")}
        if name == "ft":
            s["examples"] = [o for o in e["outputs"] if o["gen"] != o["gt"]][:6] + \
                            [o for o in e["outputs"] if o["gen"] == o["gt"]][:3]
ml = os.path.join(d, "merge.log")
if os.path.exists(ml):
    m = re.search(r"median ([\d.e+-]+), max ([\d.e+-]+)", Path(ml).read_text())
    if m:
        s["update_rel"] = {"median": float(m.group(1)), "max": float(m.group(2))}
Path(out).write_text(json.dumps(s, indent=1))
print(f"wrote {out}: {len(s['quants'])} quants" + (f", train {s['train']['steps']} steps" if "train" in s else ""))
