"""Verify every link in learnjax/papers.py: arXiv IDs against the arXiv API (title match), others resolve.

    containers/run.sh python tools/check_papers.py
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from html import unescape

from learnjax.papers import PAPERS, all_companions

UA = {"User-Agent": "learn-jax-course link checker"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " "))


def api_titles(keys, ns, tries=3):
    """(id, title) pairs from the arXiv API, or None if it keeps answering 429."""
    out = []
    for i in range(0, len(keys), 100):                  # the API returns at most max_results per query
        q = "https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(keys[i:i + 100])
        for t in range(tries):
            time.sleep(3 if i or t else 0)              # arXiv asks for one request every 3 s
            try:
                feed = ET.fromstring(urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=60).read())
                break
            except urllib.error.HTTPError as ex:
                if ex.code != 429:
                    raise
                time.sleep(10 * (t + 1))
        else:
            return None
        for e in feed.findall("a:entry", ns):
            out.append((re.sub(r"v\d+$", "", e.find("a:id", ns).text.rsplit("/", 1)[1]),
                        " ".join(e.find("a:title", ns).text.split())))
    return out


def page_titles(keys):
    """(id, title) pairs from each abstract page's citation_title meta tag."""
    out = []
    for i, aid in enumerate(keys):
        if i:
            time.sleep(3)
        try:
            html = urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/abs/{aid}", headers=UA),
                                          timeout=60).read().decode("utf-8", "replace")
        except urllib.error.HTTPError:
            continue                                     # reported below as "not returned"
        m = re.search(r'<meta name="citation_title" content="([^"]*)"', html)
        if m:
            out.append((aid, " ".join(unescape(m.group(1)).split())))
    return out


def main() -> int:
    papers = {p.url: p for ps in PAPERS.values() for p in ps}
    ids = {u.rsplit("/", 1)[1]: p for u, p in papers.items() if "arxiv.org/abs/" in u}
    bad = 0
    ns = {"a": "http://www.w3.org/2005/Atom"}
    titles = api_titles(list(ids), ns)
    if titles is None:                                   # API rate-limited: read each abstract page instead
        print("arXiv API refused (HTTP 429); reading titles from abstract pages, one every 3 s")
        titles = page_titles(list(ids))
    for aid, title in titles:
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
    videos = {v.url: v for n in range(40) for lectures in all_companions(n).values() for v in lectures}
    for url, v in videos.items():
        if "youtube.com" not in url:                         # written chapters / repos: must resolve
            try:
                code = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).status
                print(f"ok  page  {code} {url[:70]}")
            except Exception as ex:  # noqa: BLE001
                bad += 1
                print(f"BAD page  {getattr(ex, 'code', type(ex).__name__)} {url[:70]}")
            continue
        try:
            o = "https://www.youtube.com/oembed?format=json&url=" + url
            title = json.loads(urllib.request.urlopen(urllib.request.Request(o, headers=UA), timeout=30).read())["title"]
            print(f"ok  video {url[-11:]}  {title[:70]}")
        except Exception as ex:  # noqa: BLE001
            bad += 1
            print(f"BAD video {url[-11:]}  {type(ex).__name__}  ({v.title[:50]})")
    n = len(papers) + len(videos)
    print(f"\n{len(papers)} paper links + {len(videos)} companion lectures/readings = {n}, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
