"""Verify every link in learnjax/papers.py: arXiv IDs against the arXiv API (title match), others resolve.

    containers/run.sh python tools/check_papers.py
"""
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

from learnjax.papers import COMPANIONS, PAPERS

UA = {"User-Agent": "learn-jax-course link checker"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " "))


def main() -> int:
    papers = {p.url: p for ps in PAPERS.values() for p in ps}
    ids = {u.rsplit("/", 1)[1]: p for u, p in papers.items() if "arxiv.org/abs/" in u}
    bad = 0
    q = "https://export.arxiv.org/api/query?max_results=200&id_list=" + ",".join(ids)
    feed = ET.fromstring(urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=60).read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in feed.findall("a:entry", ns):
        aid = re.sub(r"v\d+$", "", e.find("a:id", ns).text.rsplit("/", 1)[1])
        title = " ".join(e.find("a:title", ns).text.split())
        p = ids.pop(aid, None)
        if p is None:
            continue
        score = SequenceMatcher(None, norm(p.title), norm(title)).ratio()
        ok = score > 0.8 or norm(p.title) in norm(title)
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} arXiv {aid:12} {score:.2f}  {p.title[:60]}" + ("" if ok else f"\n      arXiv says: {title}"))
    for aid, p in ids.items():
        bad += 1
        print(f"BAD arXiv {aid:12} not returned  {p.title[:60]}")
    for url, p in papers.items():
        if "arxiv.org/abs/" in url:
            continue
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60)
            print(f"ok  {r.status} {url[:70]}  ({p.title[:40]})")
        except Exception as ex:  # noqa: BLE001
            code = getattr(ex, "code", None)
            ok = code in (403, 418)  # publisher bot-blocking: the DOI resolved to a real page
            bad += not ok
            print(f"{'ok?' if ok else 'BAD'} {code or type(ex).__name__} {url[:70]}  ({p.title[:40]})")
    videos = {v.url: v for series in COMPANIONS.values() for lectures in series.values() for v in lectures}
    for url, v in videos.items():
        try:
            o = "https://www.youtube.com/oembed?format=json&url=" + url
            title = json.loads(urllib.request.urlopen(urllib.request.Request(o, headers=UA), timeout=30).read())["title"]
            print(f"ok  video {url[-11:]}  {title[:70]}")
        except Exception as ex:  # noqa: BLE001
            bad += 1
            print(f"BAD video {url[-11:]}  {type(ex).__name__}  ({v.title[:50]})")
    n = len(papers) + len(videos)
    print(f"\n{len(papers)} paper links + {len(videos)} videos = {n}, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
