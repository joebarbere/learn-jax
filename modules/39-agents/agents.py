"""Module 39 helpers: a small agent stack on SmolLM2-360M-Instruct, in JAX, on one GPU.

* `LM` / `State`: chat with a KV cache you keep between turns (prefill only what's new; fork a state for free,
  because JAX arrays are immutable).
* Tools, a verifiable task suite, and `run_agent`: the loop (act → observe → repeat until an exit condition),
  with free generation ("prompt-only") or constrained decoding of the tool-call JSON.
* Memory experiments, workflow patterns and multi-agent patterns are built from these pieces (see bench.py).

The learner's exercise functions are passed in as `ex`, so every number comes from their code.
"""
from __future__ import annotations

import ast
import collections
import datetime
import operator
import random
import re
import time
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from learnjax import llama

BUCKETS = (1, 16, 32, 64, 128, 256, 512, 1024)
CACHE_LEN = 2048


# ------------------------------------------------------------------------------------------ the model, with a cache

@partial(jax.jit, static_argnames=("cfg",))
def _forward_last(params, tokens, cfg, cache, pos, last):
    """learnjax.llama.forward, but the LM head runs only on row `last` (prefill doesn't need every logit)."""
    T = tokens.shape[1]
    positions = jnp.broadcast_to(pos + jnp.arange(T), (1, T))
    x = params["embed"][tokens]

    def body(x, layer):
        lp, ckv = layer
        return llama._layer(cfg, x, lp, positions, ckv, pos)

    x, (k, v) = jax.lax.scan(body, x, (params["layers"], (cache["k"], cache["v"])))
    h = llama.rms_norm(x[0, last], params["final_norm"], cfg.norm_eps)
    head = params["embed"].T if cfg.tied else params["head"]
    return (h @ head).astype(jnp.float32), {"k": k, "v": v}


@partial(jax.jit, static_argnames=("cfg",))
def _forward_all(params, tokens, cfg, cache, pos):
    """Log-probabilities at every position of a short chunk (scoring candidates); the cache is not kept."""
    logits, _ = llama.forward(params, tokens, cfg, cache, pos)
    return jax.nn.log_softmax(logits[0], -1)


@dataclass(frozen=True)
class State:
    """Where a conversation is: the KV cache, how many positions are filled, and the next-token logits."""

    cache: dict
    pos: int
    logits: jax.Array | None = None


class LM:
    def __init__(self, params, cfg, tok, cache_len: int = CACHE_LEN):
        self.params, self.cfg, self.tok, self.cache_len = params, cfg, tok, cache_len
        self.end = tok.token_to_id("<|im_end|>")
        self.strings = [tok.decode([i]) for i in range(cfg.vocab)]
        self.stats = collections.Counter()
        self._masks: dict = {}

    # -- token counting / encoding
    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def new(self) -> State:
        return State(llama.init_cache(self.cfg, 1, self.cache_len), 0)

    # -- prefill and decode
    def feed(self, state: State, text_or_ids) -> State:
        """Prefill new tokens after what the cache already holds. Chunks are padded to a few bucket sizes so
        XLA compiles each size once; the padding lands past the last real position and is overwritten next."""
        ids = self.encode(text_or_ids) if isinstance(text_or_ids, str) else list(text_or_ids)
        cache, pos, logits = state.cache, state.pos, state.logits
        if pos + len(ids) > self.cache_len:
            raise OverflowError(f"conversation needs {pos + len(ids)} positions; the cache holds {self.cache_len}")
        while ids:
            n = min(len(ids), BUCKETS[-1])
            b = next(x for x in BUCKETS if x >= n)
            chunk = np.full((1, b), self.end, np.int32)
            chunk[0, :n] = ids[:n]
            logits, cache = _forward_last(self.params, jnp.asarray(chunk), self.cfg, cache, pos, n - 1)
            pos, ids = pos + n, ids[n:]
            self.stats["prefill"] += n
        return State(cache, pos, logits)

    def logprob(self, state: State, text: str) -> np.ndarray:
        """Per-token log-probabilities of `text` as the continuation of `state` (state itself unchanged)."""
        ids = self.encode(text)
        first = float(jax.nn.log_softmax(state.logits)[ids[0]])
        if len(ids) == 1:
            return np.array([first])
        b = next(x for x in BUCKETS if x >= len(ids))
        chunk = np.full((1, b), self.end, np.int32)
        chunk[0, :len(ids)] = ids
        lp = np.asarray(_forward_all(self.params, jnp.asarray(chunk), self.cfg, state.cache, state.pos))
        self.stats["scored"] += len(ids)
        return np.concatenate([[first], lp[np.arange(len(ids) - 1), ids[1:]]])

    def generate(self, state: State, max_new: int, stop=(), mask=None, terminators=None, temperature=0.0,
                 key=None) -> tuple[str, State, str]:
        """Decode until a stop string, <|im_end|>, a terminator token (constrained mode) or max_new tokens.
        Returns (text, state, why). A terminator token is *not* fed: the caller writes the canonical closing."""
        out: list[int] = []
        why = "max_new"
        for i in range(max_new):
            lg = np.asarray(state.logits, np.float32)
            if mask is not None:
                allowed = mask if (i > 0 or terminators is None) else mask & ~terminators
                lg = np.where(allowed, lg, -np.inf)
            if temperature > 0:
                key, sub = jax.random.split(key)
                t = int(jax.random.categorical(sub, jnp.asarray(lg) / temperature))
            else:
                t = int(lg.argmax())
            if terminators is not None and terminators[t] and i > 0:
                why = "terminator"
                break
            if t == self.end:
                why = "end"
                break
            out.append(t)
            state = self.feed(state, [t])
            self.stats["decode"] += 1
            text = self.tok.decode(out)
            if any(s in text for s in stop):
                why = "stop"
                break
        return self.tok.decode(out), state, why

    def mask(self, ex, kind: str):
        """(allowed, terminators) boolean vocab masks for one argument type, built with the learner's token_mask."""
        if kind not in self._masks:
            chars, terms = ARG_CHARS[kind]
            allowed = np.asarray(ex.token_mask(self.strings, chars), bool)
            term = np.asarray(ex.token_mask(self.strings, "", terms), bool)
            self._masks[kind] = (allowed | term, term)
        return self._masks[kind]


ARG_CHARS = {   # characters a value may contain, and what ends it (tokens that *start* with one of these)
    "expression": ("0123456789+-*/(). ", ('"',)),
    "date": ("0123456789-", ('"',)),
    "integer": ("0123456789-", (",", "}", " ")),
    "text": ("".join(chr(c) for c in range(32, 127) if chr(c) not in '"\\<>{}') + "’–—éüö", ('"',)),
    "answer": ("".join(chr(c) for c in range(32, 127) if chr(c) not in "<>\n") + "’–—éüö", ("<",)),
}


def turn(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


ASSISTANT = "<|im_start|>assistant\n"


# ------------------------------------------------------------------------------------------------------- tools

@dataclass
class Tool:
    name: str
    description: str
    params: list[tuple[str, str]]          # (argument name, ARG_CHARS kind)
    fn: callable
    irreversible: bool = False

    def signature(self) -> str:
        args = ", ".join(f'"{a}": <{k}>' for a, k in self.params)
        return f'- {self.name}: {self.description} Arguments: {{{args}}}'


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def calculator(expression: str):
    """Arithmetic only (no names, no calls): parse the expression and walk the syntax tree."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError("only numbers and + - * / // % ( ) are allowed")
    v = ev(ast.parse(expression.strip(), mode="eval").body)
    return int(v) if float(v).is_integer() else round(v, 6)


def _date(s: str) -> datetime.date:
    return datetime.date.fromisoformat(s.strip())


def days_between(start: str, end: str) -> int:
    return abs((_date(end) - _date(start)).days)


def weekday(date: str) -> str:
    return _date(date).strftime("%A")


def basic_tools() -> dict[str, Tool]:
    return {t.name: t for t in [
        Tool("calculator", "Evaluate an arithmetic expression exactly.", [("expression", "expression")], calculator),
        Tool("days_between", "Number of days between two dates (YYYY-MM-DD).", [("start", "date"), ("end", "date")],
             days_between),
        Tool("weekday", "Day of the week of a date (YYYY-MM-DD).", [("date", "date")], weekday),
    ]}


def paper_tool(index, papers) -> Tool:
    def search_papers(query: str) -> str:
        hits = index.search(query, k=3)
        return " | ".join(f'"{papers[d]["title"]}" published {papers[d]["published"]}' for d, _ in hits)
    return Tool("search_papers", "Search arXiv radio-astronomy papers by title words; returns titles with dates.",
                [("query", "text")], search_papers)


def paper_subset(papers: list[dict], n: int = 4000, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(papers, n)


# --------------------------------------------------------------------------------------------- the task suite

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def make_tasks(papers: list[dict], n_each: int = 10, seed: int = 0) -> list[dict]:
    """Questions a 360M model can't answer from memory but can with the right tool, each with an exact check."""
    rng = random.Random(seed)
    tasks = []

    def rand_date():
        return datetime.date(1990, 1, 1) + datetime.timedelta(days=rng.randrange(365 * 40))

    for _ in range(n_each):
        a, b, c = rng.randint(12, 99), rng.randint(12, 99), rng.randint(100, 999)
        op = rng.choice(["+", "-"])
        expr = f"{a} * {b} {op} {c}"
        tasks.append({"kind": "arithmetic", "q": f"What is {expr}?", "answer": str(calculator(expr))})
    for i in range(n_each):
        d1, d2 = sorted([rand_date(), rand_date()])
        if i % 2 == 0:
            tasks.append({"kind": "dates", "q": f"How many days are there between {d1} and {d2}?",
                          "answer": str((d2 - d1).days)})
        else:
            tasks.append({"kind": "dates", "q": f"What day of the week was {d1}?", "answer": d1.strftime("%A")})
    picks = rng.sample(range(len(papers)), 3 * n_each)
    for i in range(n_each):
        p = papers[picks[i]]
        tasks.append({"kind": "lookup", "q": f'In which year was the arXiv paper "{p["title"]}" published?',
                      "answer": p["published"][:4]})
    for i in range(n_each):
        p, q = papers[picks[n_each + 2 * i]], papers[picks[n_each + 2 * i + 1]]
        days = abs((_date(p["published"]) - _date(q["published"])).days)
        tasks.append({"kind": "two-step", "q": f'How many days apart were the arXiv papers "{p["title"]}" and '
                      f'"{q["title"]}" published?', "answer": str(days)})
    return tasks


def check(task: dict, answer: str | None) -> bool:
    if answer is None:
        return False
    a = answer.strip().rstrip(".").replace(",", "")
    if task["kind"] == "dates" and task["answer"] in WEEKDAYS:
        return task["answer"].lower() in a.lower()
    nums = re.findall(r"-?\d+", a)
    return bool(nums) and nums[0] == task["answer"]


# ------------------------------------------------------------------------------------------------- the loop

SYSTEM = """You are an assistant that solves problems with tools. You cannot do arithmetic, date or paper lookups
in your head. Available tools:
{tools}
To use a tool, reply with exactly one line:
<tool_call>{{"name": "<tool name>", "arguments": {{...}}}}</tool_call>
The result comes back as <tool_result>...</tool_result>. When you know the answer, reply with exactly:
<answer>the answer</answer>"""

EXAMPLES = [   # one worked example as chat turns (Module 20: examples in the shape the model was trained on)
    ("user", "What is 25 * 4 + 7?"),
    ("assistant", '<tool_call>{"name": "calculator", "arguments": {"expression": "25 * 4 + 7"}}</tool_call>'),
    ("user", "<tool_result>107</tool_result>"),
    ("assistant", "<answer>107</answer>"),
]


def system_prompt(tools: dict[str, Tool], examples=EXAMPLES, system: str = SYSTEM) -> str:
    text = turn("system", system.format(tools="\n".join(t.signature() for t in tools.values())))
    return text + "".join(turn(r, c) for r, c in examples)


@dataclass
class Run:
    answer: str | None = None
    exit: str = "max_turns"
    calls: int = 0
    valid_calls: int = 0
    invalid: int = 0
    transcript: list = field(default_factory=list)
    tokens: int = 0
    seconds: float = 0.0


ANSWER = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)


def scores(lm: LM, state: State, candidates: list[str], base: State | None = None) -> list[np.ndarray]:
    """Per-token log-probabilities of each candidate; with `base` (the same prompt with a content-free question),
    subtract the candidate's log-probability there too: what the *question* adds over the model's prior
    (contextual calibration, Zhao et al. 2021). Spread the difference over the candidate's tokens."""
    out = []
    for c in candidates:
        lp = lm.logprob(state, c)
        if base is not None:
            lp = lp - lm.logprob(base, c)
        out.append(lp)
    return out


def constrained_turn(lm: LM, ex, state: State, tools: dict[str, Tool], temperature=0.0, key=None,
                     base: State | None = None) -> tuple[str, State]:
    """One assistant turn under constrained decoding: choose action and tool by log-probability, generate each
    argument under its token mask, and write every bit of JSON syntax ourselves. Always well-formed."""
    choice = ex.select_by_logprob([lm.logprob(state, s) for s in ("<tool_call>", "<answer>")])
    if choice == 1:
        state = lm.feed(state, "<answer>")
        allowed, term = lm.mask(ex, "answer")
        val, state, _ = lm.generate(state, 24, mask=allowed, terminators=term, temperature=temperature, key=key)
        text = f"<answer>{val.strip()}</answer>"
        return text, lm.feed(state, "</answer>")
    names = list(tools)
    heads = [f'<tool_call>{{"name": "{n}", "arguments": {{' for n in names]
    k = ex.select_by_logprob(scores(lm, state, heads, base))
    tool, text = tools[names[k]], heads[k]
    state = lm.feed(state, heads[k])
    for j, (arg, kind) in enumerate(tool.params):
        quote = '"' if kind != "integer" else ""
        piece = (", " if j else "") + f'"{arg}": ' + quote
        state = lm.feed(state, piece)
        allowed, term = lm.mask(ex, kind)
        val, state, _ = lm.generate(state, 40, mask=allowed, terminators=term, temperature=temperature,
                                    key=None if key is None else jax.random.fold_in(key, j))
        text += piece + val + quote
        if quote:
            state = lm.feed(state, '"')
    state = lm.feed(state, "}}</tool_call>")
    return text + "}}</tool_call>", state


def run_agent(lm: LM, ex, task: str, tools: dict[str, Tool], mode: str = "prompt", max_turns: int = 4,
              prefix: State | None = None, allowed: set | None = None, approve=None, examples=EXAMPLES,
              system: str = SYSTEM, temperature: float = 0.0, key=None, calibrate: bool = True) -> Run:
    """The agent loop. Exit conditions: an <answer>, max_turns, or a guardrail stop."""
    t0, before = time.perf_counter(), sum(lm.stats.values())
    run = Run()
    state = prefix if prefix is not None else lm.feed(lm.new(), system_prompt(tools, examples, system))
    base = lm.feed(state, turn("user", "N/A") + ASSISTANT) if (calibrate and mode == "constrained") else None
    state = lm.feed(state, turn("user", task) + ASSISTANT)
    for i in range(max_turns):
        k = None if key is None else jax.random.fold_in(key, i)
        if mode == "constrained":
            text, state = constrained_turn(lm, ex, state, tools, temperature, k, base)
        else:
            text, state, _ = lm.generate(state, 80, stop=("</tool_call>", "</answer>"), temperature=temperature, key=k)
        state = lm.feed(state, "<|im_end|>\n")
        run.transcript.append(("assistant", text))
        call = ex.parse_tool_call(text)
        if call is not None:
            run.calls += 1
            tool = tools.get(call["name"])
            ok = tool is not None and set(call["arguments"]) == {a for a, _ in tool.params}
            run.valid_calls += ok
            if tool is not None and tool.irreversible and not (approve and approve(call)):
                result = "error: this action needs a human's approval, which was not given"
                run.exit = "guardrail"
            else:
                result = ex.dispatch_tool({n: t.fn for n, t in tools.items()}, call, allowed)
            run.transcript.append(("tool", result))
            if run.exit == "guardrail":
                break
            state = lm.feed(state, turn("user", f"<tool_result>{result}</tool_result>") + ASSISTANT)
            continue
        m = ANSWER.search(text)
        if m:
            run.answer, run.exit = m.group(1).strip(), "answer"
            break
        if "<tool_call>" in text:
            run.calls += 1
        run.invalid += 1
        state = lm.feed(state, turn("user", "Reply with one <tool_call> or an <answer>.") + ASSISTANT)
    run.tokens = sum(lm.stats.values()) - before
    run.seconds = time.perf_counter() - t0
    return run


def evaluate(lm, ex, tasks, tools, mode, prefix=None, **kw) -> dict:
    rows = []
    for t in tasks:
        r = run_agent(lm, ex, t["q"], tools, mode, prefix=prefix, **kw)
        rows.append({"kind": t["kind"], "ok": check(t, r.answer), "calls": r.calls, "valid": r.valid_calls,
                     "invalid": r.invalid, "tokens": r.tokens, "seconds": r.seconds, "exit": r.exit,
                     "answer": r.answer})
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    calls = sum(r["calls"] for r in rows)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r["ok"])
    return {"success": float(np.mean([r["ok"] for r in rows])),
            "valid_rate": (sum(r["valid"] for r in rows) / calls) if calls else float("nan"),
            "tokens": float(np.mean([r["tokens"] for r in rows])),
            "seconds": float(np.mean([r["seconds"] for r in rows])),
            "by_kind": {k: float(np.mean(v)) for k, v in by.items()}, "n": len(rows), "rows": rows}


# -------------------------------------------------------------------------------------- short-term memory: KV reuse

FILLER = [
    ("Can you suggest a name for a small sailing boat?", "How about Wavelet? It's short and easy to shout."),
    ("What's a good way to keep basil alive indoors?", "Give it a sunny window and water when the top soil is dry."),
    ("Tell me a fun fact about owls.", "Owls can't move their eyes, so they turn their heads instead."),
    ("Any tips for a long train journey?", "Bring snacks, a charged phone, and something to read offline."),
    ("How long should I boil an egg for a runny yolk?", "About six minutes from boiling, then into cold water."),
    ("What's the difference between a lake and a pond?", "Mostly size and depth; ponds are shallow enough for light."),
    ("Recommend a board game for four people.", "Try Ticket to Ride: easy rules and quick turns."),
    ("Why is the sky blue?", "Air scatters short blue wavelengths of sunlight more than red ones."),
    ("How do I remove a coffee stain from a shirt?", "Rinse with cold water, then dab on a little dish soap."),
    ("Give me a word that rhymes with orange.", "Nothing rhymes perfectly; 'door hinge' is the usual joke."),
]


def filler_turns(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        u, a = rng.choice(FILLER)
        out += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return out


def kv_reuse_timing(lm: LM, n_turns: int = 14, seed: int = 0) -> dict:
    """ms to get the next-token logits after each new turn: (a) prefill only the new turn into the kept cache,
    (b) re-prefill the whole conversation from an empty cache. Second pass timed (first pass compiles)."""
    system = turn("system", "You are a helpful assistant.")
    turns = ["".join(turn(m["role"], m["content"]) for m in filler_turns(1, seed + i)) for i in range(n_turns)]
    result = {}
    for _ in range(2):
        reuse, scratch, lengths = [], [], []
        state, history = lm.feed(lm.new(), system), system
        for t in turns:
            history += t
            t0 = time.perf_counter()
            state = lm.feed(state, t)
            state.logits.block_until_ready()
            reuse.append(1e3 * (time.perf_counter() - t0))
            t0 = time.perf_counter()
            s2 = lm.feed(lm.new(), history)
            s2.logits.block_until_ready()
            scratch.append(1e3 * (time.perf_counter() - t0))
            lengths.append(state.pos)
        result = {"reuse_ms": reuse, "scratch_ms": scratch, "context_tokens": lengths}
    return result


# ---------------------------------------------------------------------------------- long-term memory: a recall test

FACT_KINDS = [
    ("locker code", lambda r: str(r.randint(1000, 9999))),
    ("sister's name", lambda r: r.choice(["Imogen", "Priya", "Tamsin", "Beatriz", "Yuki", "Odette"])),
    ("flight number", lambda r: f"{r.choice(['UA', 'DL', 'BA', 'AF'])}{r.randint(100, 999)}"),
    ("hotel room number", lambda r: str(r.randint(100, 999))),
    ("cat's name", lambda r: r.choice(["Pickle", "Juniper", "Mochi", "Biscuit", "Nimbus"])),
    ("dentist appointment day", lambda r: r.choice(WEEKDAYS)),
    ("project codename", lambda r: r.choice(["Bluefin", "Larkspur", "Granite", "Halcyon", "Quillon"])),
    ("bike lock combination", lambda r: str(r.randint(100, 999))),
    ("favourite tea", lambda r: r.choice(["Assam", "oolong", "sencha", "rooibos", "Darjeeling"])),
    ("gym membership number", lambda r: str(r.randint(10000, 99999))),
]


def make_memory_test(n_filler: int = 36, n_updates: int = 2, seed: int = 0):
    """A long conversation: 10 personal facts scattered among small talk, two of them corrected later.
    Returns (messages, questions) with questions = [(question, answer, kind)]."""
    rng = random.Random(seed)
    facts = {k: f(rng) for k, f in FACT_KINDS}
    msgs = filler_turns(n_filler, seed)
    slots = sorted(rng.sample(range(len(msgs) // 2), len(facts)))
    for inserted, ((k, v), s) in enumerate(zip(facts.items(), slots, strict=True)):
        i = 2 * s + 2 * inserted
        msgs[i:i] = [{"role": "user", "content": f"By the way, my {k} is {v}."},
                     {"role": "assistant", "content": "Got it, I'll remember that."}]
    updated = rng.sample(list(facts)[:5], n_updates)
    for k in updated:
        new = dict(FACT_KINDS)[k](rng)
        while new == facts[k]:
            new = dict(FACT_KINDS)[k](rng)
        facts[k] = new
        msgs += [{"role": "user", "content": f"Correction: my {k} is now {new}."},
                 {"role": "assistant", "content": "Thanks, updated."}] + filler_turns(2, seed + 99)
    qs = [(f"What is my {k}?", v, "updated" if k in updated else "stated once") for k, v in facts.items()]
    return msgs, qs


ASK = "Answer the question using the conversation above. Reply with just the answer, a few words at most."


def answer_from(lm: LM, state: State, question: str) -> str:
    s = lm.feed(state, turn("user", f"{ASK}\n\n{question}") + ASSISTANT)
    text, _, _ = lm.generate(s, 16)
    return text.strip()


def _render(msgs) -> str:
    return "".join(turn(m["role"], m["content"]) for m in msgs)


def memory_methods(lm: LM, ex, rag, ex38, msgs, questions, window_budget: int = 300, summarize_every: int = 12):
    """Recall accuracy and prompt tokens per question for five memory strategies."""
    system = turn("system", "You are a helpful assistant with a good memory.")
    out = {}

    def score(name, contexts):
        ok, toks = [], []
        for (q, a, kind), ctx in zip(questions, contexts, strict=True):
            st = lm.feed(lm.new(), system + ctx)
            reply = answer_from(lm, st, q)
            ok.append((a.lower() in reply.lower(), kind, reply))
            toks.append(st.pos)
        out[name] = {"recall": float(np.mean([o[0] for o in ok])), "tokens": float(np.mean(toks)),
                     "updated": float(np.mean([o[0] for o in ok if o[1] == "updated"] or [np.nan])),
                     "replies": [o[2] for o in ok]}

    full = _render(msgs)
    score("full conversation in context", [full] * len(questions))
    win = ex.trim_window(msgs, window_budget, lm.count)
    score(f"sliding window ({window_budget} tokens)", [_render(win)] * len(questions))

    # running summary: every `summarize_every` messages, the model rewrites its notes
    notes = ""
    for i in range(0, len(msgs), summarize_every):
        chunk = _render(msgs[i:i + summarize_every])
        p = (system + turn("user", "Notes so far:\n" + (notes or "(none)") + "\n\nNew conversation:\n" + chunk
                           + "\nRewrite the notes as a short list of every personal fact the user stated "
                             "(keep old facts; replace any the user corrected).") + ASSISTANT)
        notes, _, _ = lm.generate(lm.feed(lm.new(), p), 120)
        notes = notes.strip()
    tail = _render(ex.trim_window(msgs, 120, lm.count))
    score("running summary + recent turns", [turn("system", "Notes: " + notes) + tail] * len(questions))

    # episodic log searched with BM25 (Module 38), ranked by recency + relevance
    user_turns = [(i, m["content"]) for i, m in enumerate(msgs) if m["role"] == "user"]
    idx = rag.build_index([c for _, c in user_turns], ex38)
    ctxs = []
    for q, _, _ in questions:
        hits = idx.search(q, k=min(8, len(user_turns)))
        ids = [d for d, _ in hits]
        ages = [len(msgs) - user_turns[d][0] for d in ids]
        rel = [s for _, s in hits]
        order = [j for j in np.argsort(-ex.recency_relevance_score(ages, rel))[:3] if rel[j] >= 0.5 * max(rel)]
        picked = [ids[j] for j in order][::-1]      # weak matches dropped; best LAST, next to the question
        ctxs.append("".join(turn("user", user_turns[d][1]) for d in picked))
    score("episodic log + retrieval (recency × relevance)", ctxs)

    # a structured ledger in SQLite, written by a rule as facts arrive
    import sqlite3
    db = sqlite3.connect(":memory:")
    db.execute("create table facts (turn integer, key text, value text)")
    pat = re.compile(r"my (.+?) is (?:now )?(.+?)\.$")
    for i, m in enumerate(msgs):
        hit = pat.search(m["content"]) if m["role"] == "user" else None
        if hit:
            db.execute("insert into facts values (?, ?, ?)", (i, hit.group(1), hit.group(2)))
    ctxs = []
    for q, _, _ in questions:
        key = re.search(r"my (.+)\?", q).group(1)
        row = db.execute("select value from facts where key = ? order by turn desc limit 1", (key,)).fetchone()
        ctxs.append(turn("system", f"Known fact: my {key} is {row[0] if row else 'unknown'}."))
    score("structured ledger (SQLite)", ctxs)
    return out


# ------------------------------------------------------------------------------- workflows (Anthropic's five patterns)

DATE_EXAMPLES = [
    ("user", "How many days are there between 2003-02-10 and 2003-03-01?"),
    ("assistant", ('<tool_call>{"name": "days_between", "arguments": {"start": "2003-02-10", "end": "2003-03-01"}}'
                   "</tool_call>")),
    ("user", "<tool_result>19</tool_result>"),
    ("assistant", "<answer>19</answer>"),
    ("user", "What day of the week was 2001-05-17?"),
    ("assistant", '<tool_call>{"name": "weekday", "arguments": {"date": "2001-05-17"}}</tool_call>'),
    ("user", "<tool_result>Thursday</tool_result>"),
    ("assistant", "<answer>Thursday</answer>"),
]
LIBRARY_EXAMPLES = [
    ("user", 'In which year was the arXiv paper "Pulsar timing with MeerKAT" published?'),
    ("assistant", ('<tool_call>{"name": "search_papers", "arguments": {"query": "Pulsar timing with MeerKAT"}}'
                   "</tool_call>")),
    ("user", ('<tool_result>"Pulsar timing with MeerKAT" published 2020-03-02 | "Timing of millisecond pulsars" '
              "published 2011-07-19</tool_result>")),
    ("assistant", "<answer>2020</answer>"),
]
SPECIALISTS = {   # name -> (tools, examples, what it's for)
    "math": (["calculator"], EXAMPLES, "arithmetic"),
    "dates": (["days_between", "weekday"], DATE_EXAMPLES, "dates: days between two dates, or a weekday"),
    "library": (["search_papers"], LIBRARY_EXAMPLES, "finding arXiv papers and their publication dates"),
}


class Toolbox:
    """Tools plus cached prompt prefixes: each agent's system prompt is prefilled once and forked per task."""

    def __init__(self, lm: LM, tools: dict[str, Tool]):
        self.lm, self.tools, self._prefix = lm, tools, {}

    def subset(self, names) -> dict[str, Tool]:
        return {n: self.tools[n] for n in names}

    def prefix(self, key, tools, examples, system=SYSTEM) -> State:
        if key not in self._prefix:
            self._prefix[key] = self.lm.feed(self.lm.new(), system_prompt(tools, examples, system))
        return self._prefix[key]

    def agent(self, ex, question, names=None, examples=None, mode="constrained", **kw) -> Run:
        names = list(names or self.tools)
        if examples is None:
            examples = EXAMPLES + DATE_EXAMPLES + LIBRARY_EXAMPLES if len(names) > 1 else EXAMPLES
        tools = self.subset(names)
        return run_agent(self.lm, ex, question, tools, mode,
                         prefix=self.prefix(("agent", tuple(names), tuple(examples)), tools, examples), **kw)

    def specialist(self, ex, name, question, **kw) -> Run:
        names, examples, _ = SPECIALISTS[name]
        return self.agent(ex, question, names, examples, **kw)


def _tokens(lm):
    return sum(lm.stats.values())


def classify(lm: LM, ex, question: str, labels: dict[str, str], calibrate: bool = True) -> str:
    """Routing: pick a label by log-probability of its name after a classification prompt (calibrated)."""
    menu = "\n".join(f"- {k}: {v}" for k, v in labels.items())
    head = lm.feed(lm.new(), turn("system", "Classify the question. Categories:\n" + menu))
    s = lm.feed(head, turn("user", question) + ASSISTANT + "Category:")
    base = lm.feed(head, turn("user", "N/A") + ASSISTANT + "Category:") if calibrate else None
    names = list(labels)
    return names[ex.select_by_logprob(scores(lm, s, [" " + n for n in names], base))]


def wf_routing(box: Toolbox, ex, task, calibrate: bool = True) -> dict:
    t0 = _tokens(box.lm)
    route = classify(box.lm, ex, task["q"], {k: v[2] for k, v in SPECIALISTS.items()}, calibrate)
    r = box.specialist(ex, route, task["q"])
    return {"answer": r.answer, "route": route, "tokens": _tokens(box.lm) - t0}


def wf_voting(box: Toolbox, ex, task, n: int = 5, temperature: float = 0.7, seed: int = 0) -> dict:
    t0 = _tokens(box.lm)
    answers = [box.agent(ex, task["q"], temperature=temperature, key=jax.random.key(seed + i)).answer
               for i in range(n)]
    return {"answer": ex.majority_vote([a.strip() if a else None for a in answers]), "votes": answers,
            "tokens": _tokens(box.lm) - t0}


TITLES = re.compile(r'"([^"]+)"')
ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


def wf_chaining(box: Toolbox, ex, task) -> dict:
    """Prompt chaining for the two-step questions: a fixed sequence of small agent calls with programmatic gates."""
    t0 = _tokens(box.lm)
    dates = []
    for title in TITLES.findall(task["q"])[:2]:
        r = box.specialist(ex, "library", f'On what date (YYYY-MM-DD) was the arXiv paper "{title}" published?')
        found = ISO.findall((r.answer or "") + " ".join(t for k, t in r.transcript if k == "tool"))
        if not found:                                   # gate: no date, stop the chain
            return {"answer": None, "gate": "no date", "tokens": _tokens(box.lm) - t0}
        dates.append(found[0])
    r = box.specialist(ex, "dates", f"How many days are there between {dates[0]} and {dates[1]}?")
    return {"answer": r.answer, "dates": dates, "tokens": _tokens(box.lm) - t0}


def wf_orchestrator(box: Toolbox, ex, task, max_workers: int = 3) -> dict:
    """Orchestrator-workers: the model writes the sub-questions, workers (agents) answer each, then a final agent
    combines the results. Unlike chaining, the plan is decided at run time by the model."""
    lm, t0 = box.lm, _tokens(box.lm)
    s = lm.feed(lm.new(), turn("system", "Break the question into at most three simple sub-questions, one per line, "
                                         "each answerable with one tool call. Write only the sub-questions.")
                + turn("user", task["q"]) + ASSISTANT)
    plan, _, _ = lm.generate(s, 120)
    subs = [ln.strip("-*0123456789. ").strip() for ln in plan.splitlines() if ln.strip()][:max_workers]
    results = [(q, box.agent(ex, q).answer) for q in subs]
    notes = "\n".join(f"- {q} -> {a}" for q, a in results)
    r = box.agent(ex, f"{task['q']}\n\nFindings so far:\n{notes}")
    return {"answer": r.answer, "plan": subs, "results": results, "tokens": _tokens(lm) - t0}


def wf_evaluator(box: Toolbox, ex, task, rounds: int = 2) -> dict:
    """Evaluator-optimizer: an evaluator call judges the answer (yes/no by log-probability); on "no" the agent
    retries with that feedback (Reflexion-style)."""
    lm, t0 = box.lm, _tokens(box.lm)
    q, history = task["q"], []
    for _ in range(rounds + 1):
        r = box.agent(ex, q)
        evidence = " ".join(t for k, t in r.transcript if k == "tool")
        s = lm.feed(lm.new(), turn("system", "You check answers. Reply yes or no.")
                    + turn("user", f"Question: {task['q']}\nTool results: {evidence or '(none)'}\n"
                               f"Proposed answer: {r.answer}\nIs the proposed answer correct and supported by "
                               "the tool results?") + ASSISTANT)
        verdict = ["yes", "no"][ex.select_by_logprob([lm.logprob(s, w) for w in ("yes", "no")])]
        history.append((r.answer, verdict))
        if verdict == "yes":
            break
        q = f"{task['q']}\n(A reviewer judged the answer {r.answer!r} wrong. Check your tool calls and try again.)"
    return {"answer": history[-1][0], "history": history, "tokens": _tokens(lm) - t0}


# ------------------------------------------------------------------------- single agent vs multi-agent (OpenAI guide)

def ma_single(box: Toolbox, ex, task) -> dict:
    t0 = _tokens(box.lm)
    r = box.agent(ex, task["q"])
    return {"answer": r.answer, "tokens": _tokens(box.lm) - t0}


MANAGER_EXAMPLES = [
    ("user", "What is 25 * 4 + 7?"),
    ("assistant", '<tool_call>{"name": "math", "arguments": {"request": "What is 25 * 4 + 7?"}}</tool_call>'),
    ("user", "<tool_result>107</tool_result>"),
    ("assistant", "<answer>107</answer>"),
]


def ma_manager(box: Toolbox, ex, task, max_turns: int = 5) -> dict:
    """Manager pattern: specialists are the manager's tools; it delegates, reads their answers, and replies."""
    t0 = _tokens(box.lm)
    staff = {name: Tool(name, f"An expert agent for {desc}. Send it one self-contained request.",
                        [("request", "text")],
                        lambda request, _n=name: box.specialist(ex, _n, request).answer or "no answer")
             for name, (_, _, desc) in SPECIALISTS.items()}
    pre = box.prefix(("manager",), staff, MANAGER_EXAMPLES)
    r = run_agent(box.lm, ex, task["q"], staff, "constrained", max_turns=max_turns, prefix=pre)
    return {"answer": r.answer, "calls": [c for c in r.transcript], "tokens": _tokens(box.lm) - t0}


def ma_handoff(box: Toolbox, ex, task, calibrate: bool = True) -> dict:
    """Decentralized pattern: a triage agent hands the conversation to one specialist (a one-way transfer, which in
    OpenAI's Agents SDK is itself a tool call), and that specialist owns the reply."""
    t0 = _tokens(box.lm)
    transfers = {f"transfer_to_{n}": Tool(f"transfer_to_{n}", f"Hand the user to the {desc} agent.", [], lambda: "ok")
                 for n, (_, _, desc) in SPECIALISTS.items()}
    ex_turns = [("user", "What is 25 * 4 + 7?"),
                ("assistant", '<tool_call>{"name": "transfer_to_math", "arguments": {}}</tool_call>')]
    pre = box.prefix(("triage",), transfers, ex_turns)
    s = box.lm.feed(pre, turn("user", task["q"]) + ASSISTANT)
    base = box.lm.feed(pre, turn("user", "N/A") + ASSISTANT) if calibrate else None
    text, _ = constrained_turn(box.lm, ex, s, transfers, base=base)
    call = ex.parse_tool_call(text)
    target = call["name"].removeprefix("transfer_to_") if call else None
    r = box.specialist(ex, target, task["q"]) if target in SPECIALISTS else None
    return {"answer": r.answer if r else None, "handoff": target, "tokens": _tokens(box.lm) - t0}


# ------------------------------------------------------------------------------------------------------- MCP client

def with_mcp(server_path, fn, timeout: float = 180):
    """Launch an MCP server over stdio, then run `fn(call, tools)` in a worker thread: `call(name, args) -> str`
    is a synchronous bridge to the async MCP session, `tools` is the server's tool list."""
    import asyncio
    import os
    import sys
    import threading

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def main():
        env = {**os.environ, "JAX_PLATFORMS": "cpu"}
        params = StdioServerParameters(command=sys.executable, args=[str(server_path)], env=env, cwd="/course")
        async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
            await session.initialize()
            listed = await session.list_tools()
            loop = asyncio.get_running_loop()

            def call(name, args):
                res = asyncio.run_coroutine_threadsafe(session.call_tool(name, args), loop).result(timeout)
                return "".join(getattr(c, "text", "") for c in res.content)

            return await asyncio.to_thread(fn, call, listed.tools)

    out = {}

    def runner():
        try:
            out["value"] = asyncio.run(main())
        except BaseException as e:  # noqa: BLE001 - surfaced to the caller
            out["error"] = e

    t = threading.Thread(target=runner)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
    return out["value"]
