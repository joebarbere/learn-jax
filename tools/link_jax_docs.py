"""Link the first mention of each JAX API in a lesson's prose to its page in the JAX API docs.

    containers/run.sh python tools/link_jax_docs.py modules/*/lesson.py        # rewrite in place
    containers/run.sh python tools/link_jax_docs.py --check modules/*/lesson.py # report unlinked first mentions

Targets come from the JAX docs' own Sphinx inventory (https://docs.jax.dev/en/latest/objects.inv), so every
link points at a page that exists. Only markdown text inside `mo.md(...)` cells is touched (never code), and
only inline code spans containing a JAX name: `jax.jit`, `jnp.einsum`, `k1, k2 = jax.random.split(key)`,
`x.at[i].set(v)` (as `jax.numpy.ndarray.at`). The first mention per lesson gets the link (the whole code span
becomes the link text); later mentions stay plain.
"""
from __future__ import annotations

import re
import sys
import urllib.request
import zlib
from pathlib import Path

DOCS = "https://docs.jax.dev/en/latest/"
CACHE = Path("/cache/learnjax/jax_objects.inv")
ALIASES = {"jnp.": "jax.numpy.", "lax.": "jax.lax.", "jr.": "jax.random.", "jsp.": "jax.scipy."}
SPECIAL = {".at[": "jax.numpy.ndarray.at", "jax.Array": "jax.Array"}
# bare core transforms, linked only when the whole code span is just that word (e.g. `jit`, `vmap()`)
BARE = {w: f"jax.{w}" for w in ["jit", "grad", "vmap", "pmap", "jvp", "vjp", "value_and_grad", "jacobian",
                                "hessian", "make_jaxpr", "device_put", "block_until_ready", "shard_map",
                                "checkpoint", "eval_shape", "named_scope"]}


def inventory() -> dict[str, str]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(DOCS + "objects.inv", headers={"User-Agent": "learn-jax-course"})
        CACHE.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    raw = CACHE.read_bytes()
    body = zlib.decompress(raw.split(b"\n", 4)[4]).decode()
    out = {}
    for ln in body.splitlines():
        m = re.match(r"(\S+)\s+py:\S+\s+-?\d+\s+(\S+)\s+", ln)
        if m:
            name, uri = m.groups()
            out.setdefault(name, DOCS + (uri[:-1] + name if uri.endswith("$") else uri))
    return out


INV = None
# an inline code span not already inside a link, and JAX names inside it
SPAN = re.compile(r"(?<!\[)`([^`\n]+)`(?!\]\()")
NAME = re.compile(r"(?<![\w.])((?:jax|jnp|lax|jr|jsp)\.[A-Za-z_][\w.]*|\.at\[)")


def resolve(name: str) -> str | None:
    if ".at[" in name:
        return INV.get(SPECIAL[".at["])
    for short, full in ALIASES.items():
        if name.startswith(short):
            name = full + name[len(short):]
    parts = name.split(".")
    while len(parts) >= 2:                      # jax.random.key(0).foo → try longest documented prefix
        cand = ".".join(parts)
        if cand in INV:
            return INV[cand]
        parts.pop()
    return None


def md_spans(src: str):
    """(start, end) of the string literal(s) passed to mo.md(...) calls."""
    for m in re.finditer(r"mo\.md\(\s*(?:[rf]{0,2})(\"\"\"|'''|\"|')", src):
        q = m.group(1)
        start = m.end()
        end = src.find(q, start)
        if end > 0:
            yield start, end


def process(path: Path, check: bool) -> tuple[int, list[str]]:
    src = path.read_text()
    seen: set[str] = set()
    for m in re.finditer(r"\]\((https://docs\.jax\.dev/[^)]+)\)", src):   # already linked
        seen.add(m.group(1))
    edits, missing = [], []
    for s, e in md_spans(src):
        for t in SPAN.finditer(src, s, e):
            if "{" in t.group(1):                      # f-string interpolation shows a value, not prose
                continue
            word = t.group(1).strip().removesuffix("()")
            if word in BARE:
                url = INV.get(BARE[word])
                if url and url not in seen:
                    seen.add(url)
                    edits.append((t.start(), t.end(), f"[{t.group(0)}]({url})"))
                continue
            for n in NAME.finditer(t.group(1)):          # first JAX name in this span not yet linked
                url = resolve(n.group(1))
                if url is None:
                    missing.append(n.group(1))
                    continue
                if url not in seen:
                    seen.add(url)
                    edits.append((t.start(), t.end(), f"[{t.group(0)}]({url})"))
                    break
    if not check and edits:
        for a, b, rep in sorted(edits, reverse=True):
            src = src[:a] + rep + src[b:]
        path.write_text(src)
    return len(edits), sorted(set(missing))


def main(argv: list[str]) -> int:
    global INV
    check = "--check" in argv
    INV = inventory()
    todo = 0
    for p in [Path(a) for a in argv if not a.startswith("--")]:
        n, missing = process(p, check)
        todo += n
        verb = "would link" if check else "linked"
        print(f"{p}: {verb} {n} first mentions" + (f"; no docs page for {missing}" if missing else ""))
    return 1 if (check and todo) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
