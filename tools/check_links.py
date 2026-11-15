"""Check every external link in the lessons (and README/docs) resolves, and list them for attribution review.

    containers/run.sh python tools/check_links.py                 # check all links, print problems
    containers/run.sh python tools/check_links.py --list          # also print every link with its context

Attribution rule (docs/AUTHORING.md): every external link names who made it — authors for papers, the
person or organization for posts, talks, videos, docs and repos (e.g. "JAX docs: …", "Horace He, …",
"Karpathy's nanoGPT (GitHub)"). Generated sections (papers, companion lectures, JAX API links) satisfy it
by construction; `--list` is for reviewing the hand-written ones.
"""
from __future__ import annotations

import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (learn-jax course link checker)"}
LINK = re.compile(r"\[([^\]\n]+)\]\((https?://(?:[^()\s]|\([^()\s]*\))+)\)")  # allows one level of (...) in URLs
BOT_BLOCKED = {"doi.org", "platform.openai.com", "openai.com", "www.reddit.com", "x.com", "twitter.com",
               "www.linkedin.com", "dl.acm.org"}


def files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    return sorted(root.glob("modules/*/lesson.py")) + sorted(root.glob("modules/*/torch_compare.py")) + \
        [root / "README.md"] + sorted(root.glob("docs/*.md"))


def status(url: str) -> int | str:
    try:
        req = urllib.request.Request(url, headers=UA, method="GET")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except Exception as ex:  # noqa: BLE001
        return getattr(ex, "code", None) or type(ex).__name__


def main(argv: list[str]) -> int:
    listing = "--list" in argv
    found: dict[str, list[tuple[str, str, str]]] = {}
    for f in files():
        text = f.read_text()
        for m in LINK.finditer(text):
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            context = text[line_start:line_end].strip()
            found.setdefault(m.group(2).split("#")[0], []).append((f.parent.name if f.name != "README.md" else "README",
                                                                   m.group(1), context))
    with ThreadPoolExecutor(16) as ex:
        codes = dict(zip(found, ex.map(status, found)))
    bad = 0
    for url, uses in sorted(found.items()):
        code = codes[url]
        host = url.split("/")[2]
        ok = (isinstance(code, int) and 200 <= code < 300) or (code in (403, 429) and host in BOT_BLOCKED) or (code == 429)
        bad += not ok
        if not ok or listing:
            mark = "ok " if ok else "BAD"
            for where, label, ctx in uses[: (len(uses) if listing else 1)]:
                print(f"{mark} {code} [{where}] {label[:50]!r} {url}" + (f"\n        {ctx[:160]}" if listing else ""))
    print(f"\n{len(found)} unique external links, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
