"""Module 39: the full measurements behind the lesson's tables (~7–10 min on an RX 7600 XT).

    containers/run.sh python modules/39-agents/bench.py                       # uses YOUR exercises
    LEARNJAX_SOLUTIONS=1 containers/run.sh python modules/39-agents/bench.py  # the reference

Runs a 40-task suite (arithmetic, dates, paper lookups, two-step questions) through: prompt-only vs constrained
tool calls; single agent vs manager vs handoff; and Anthropic's five workflow patterns; plus the memory recall
test over three long conversations and the KV-reuse timing. Writes /course/runs/39-agents/bench.json, which the
lesson loads (it runs a small live version otherwise).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

import learnjax as lj
from learnjax import llama
from learnjax.data import radio_abstracts

HERE = Path(__file__).resolve().parent
OUT = Path("/course/runs/39-agents")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def setup(ex38=None):
    """Model, tools (with Module 38's index over 4,000 papers), and the task suite. Shared with the lesson."""
    A = _load("m39_agents", HERE / "agents.py")
    rag = _load("m38_rag", HERE.parent / "38-rag" / "rag.py")
    ex38 = ex38 or lj.load_exercises(HERE.parent / "38-rag", solutions=True)
    papers = A.paper_subset(radio_abstracts())
    index = rag.build_index([p["title"] + ". " + p["abstract"] for p in papers], ex38)
    params, cfg, tok = llama.load(llama.SMOLLM_360M)
    lm = A.LM(params, cfg, tok)
    tools = {**A.basic_tools(), "search_papers": A.paper_tool(index, papers)}
    return A, rag, ex38, papers, lm, tools


def score(A, tasks, outs):
    oks = [A.check(t, o["answer"]) for t, o in zip(tasks, outs, strict=True)]
    by = {}
    for t, ok in zip(tasks, oks, strict=True):
        by.setdefault(t["kind"], []).append(ok)
    return {"success": float(np.mean(oks)), "tokens": float(np.mean([o["tokens"] for o in outs])),
            "by_kind": {k: float(np.mean(v)) for k, v in by.items()}, "n": len(tasks)}


def main():
    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    ex = lj.load_exercises(HERE)
    A, rag, ex38, papers, lm, tools = setup()
    box = A.Toolbox(lm, tools)
    tasks = A.make_tasks(papers, n_each=10)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"n_tasks": len(tasks), "model": "SmolLM2-360M-Instruct"}

    def save():
        (OUT / "bench.json").write_text(json.dumps(res, indent=1, default=float))

    t0 = time.perf_counter()
    examples = A.EXAMPLES + A.DATE_EXAMPLES + A.LIBRARY_EXAMPLES
    pre = box.prefix(("agent", tuple(tools), tuple(examples)), tools, examples)   # the same cached prefix box.agent uses
    for name, mode, cal in [("prompt", "prompt", False), ("constrained_raw", "constrained", False),
                            ("constrained", "constrained", True)]:
        s = A.evaluate(lm, ex, tasks, tools, mode, prefix=pre, calibrate=cal)
        s.pop("rows")
        res[f"agent_{name}"] = s
        print(name, {k: v for k, v in s.items() if k != "by_kind"}, s["by_kind"], flush=True)
    save()

    for name, fn in [("single", A.ma_single), ("manager", A.ma_manager), ("handoff", A.ma_handoff)]:
        outs = [fn(box, ex, t) for t in tasks]
        res[f"multi_{name}"] = score(A, tasks, outs)
        if name == "handoff":
            want = {"arithmetic": "math", "dates": "dates", "lookup": "library"}
            res["multi_handoff"]["routing_accuracy"] = float(np.mean(
                [o["handoff"] == want[t["kind"]] for t, o in zip(tasks, outs, strict=True) if t["kind"] in want]))
        print(name, res[f"multi_{name}"], flush=True)
        save()

    two = [t for t in tasks if t["kind"] == "two-step"]
    small = [t for k in ("arithmetic", "dates", "lookup", "two-step") for t in tasks if t["kind"] == k][::5]
    for name, fn, subset in [("routing", A.wf_routing, tasks), ("chaining", A.wf_chaining, two),
                             ("orchestrator", A.wf_orchestrator, two), ("evaluator", A.wf_evaluator, tasks),
                             ("voting", A.wf_voting, small)]:
        outs = [fn(box, ex, t) for t in subset]
        res[f"workflow_{name}"] = score(A, subset, outs)
        base = [A.check(t, A.ma_single(box, ex, t)["answer"]) for t in subset]
        res[f"workflow_{name}"]["single_agent_same_tasks"] = float(np.mean(base))
        if name == "routing":
            want = {"arithmetic": "math", "dates": "dates", "lookup": "library"}
            res["workflow_routing"]["routing_accuracy"] = float(np.mean(
                [o["route"] == want[t["kind"]] for t, o in zip(subset, outs, strict=True) if t["kind"] in want]))
        print(name, res[f"workflow_{name}"], flush=True)
        save()

    mem = []
    for seed in range(3):
        msgs, qs = A.make_memory_test(seed=seed)
        m = A.memory_methods(lm, ex, rag, ex38, msgs, qs)
        mem.append({k: {kk: vv for kk, vv in v.items() if kk != "replies"} for k, v in m.items()})
    res["memory"] = {k: {s: float(np.nanmean([m[k][s] for m in mem])) for s in ("recall", "tokens", "updated")}
                     for k in mem[0]}
    print("memory", res["memory"], flush=True)
    res["kv_timing"] = A.kv_reuse_timing(lm, n_turns=24)
    res["seconds"] = time.perf_counter() - t0
    save()
    print(f"wrote {OUT / 'bench.json'} in {res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
