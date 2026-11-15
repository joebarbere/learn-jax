"""Fetch Module 38's corpus from the arXiv API: radio-related astro-ph abstracts, 1993 to Oct 2026.

    containers/run.sh python tools/fetch_radio.py        # what `make rag-data` runs if aws-ai's copy is absent

Ported from the owner's aws-ai repo (python/scripts/radio_corpus.py, `abstracts`, MIT), with the same query,
so the corpus matches the one aws-ai's local-kb benchmark used. arXiv's API terms: export.arxiv.org, one
request every 3 seconds, a single connection. Resumable: years already fetched are skipped. ~10–20 minutes.
Writes $LEARNJAX_CACHE/radio/radio-abstracts.jsonl (abstracts are read locally, never redistributed).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from learnjax.data import CACHE, RADIO_FILE

API = "https://export.arxiv.org/api/query"
USER_AGENT = "learn-jax-course/0.1 (one request per 3 s)"
PAUSE = 3.0
RADIO = ('abs:"radio telescope" OR abs:"radio astronomy" OR abs:"radio interferometer" OR '
         'abs:"radio interferometry" OR abs:VLBI OR abs:"radio frequency interference" OR '
         'abs:"radio array" OR abs:"21 cm"')
QUERY = ("(cat:astro-ph OR cat:astro-ph.IM OR cat:astro-ph.GA OR cat:astro-ph.HE OR "
         f"cat:astro-ph.CO OR cat:astro-ph.SR) AND ({RADIO} OR abs:radio OR abs:pulsar) AND ")
YEARS = range(1993, 2027)    # one query per year keeps each under the API's 30k-result cap
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except OSError as exc:
            if attempt == 4:
                raise
            print(f"  retrying after {exc}", file=sys.stderr)
            time.sleep(PAUSE * (attempt + 2))
    raise AssertionError("unreachable")


def search(query: str, limit: int = 30000, page: int = 2000):
    start = 0
    while start < limit:
        params = urllib.parse.urlencode({"search_query": query, "start": start, "max_results": min(page, limit - start),
                                         "sortBy": "submittedDate", "sortOrder": "ascending"})
        feed = ET.fromstring(get(f"{API}?{params}"))
        entries = feed.findall("a:entry", NS)
        total = int(feed.findtext("os:totalResults", "0", NS))
        for e in entries:
            versioned = e.findtext("a:id", "", NS).rsplit("/abs/", 1)[1]
            yield {"id": versioned.rsplit("v", 1)[0], "version": versioned,
                   "published": e.findtext("a:published", "", NS)[:10],
                   "category": e.find("arxiv:primary_category", NS).get("term"),
                   "title": " ".join(e.findtext("a:title", "", NS).split()),
                   "abstract": " ".join(e.findtext("a:summary", "", NS).split())}
        start += len(entries)
        if not entries or start >= total:
            return
        time.sleep(PAUSE)


def main():
    out_path = CACHE / "radio" / RADIO_FILE
    done_path = out_path.with_suffix(".years")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = {json.loads(x)["id"] for x in out_path.read_text().splitlines()} if out_path.exists() else set()
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    for year in YEARS:
        if str(year) in done:
            continue
        end = "10312359" if year == 2026 else "12312359"
        added = 0
        with out_path.open("a") as out:
            for entry in search(QUERY + f"submittedDate:[{year}01010000 TO {year}{end}]"):
                if entry["id"] not in seen:
                    seen.add(entry["id"])
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    added += 1
        with done_path.open("a") as f:
            f.write(f"{year}\n")
        print(f"{year}: {added} abstracts", flush=True)
        time.sleep(PAUSE)
    print(f"{out_path}: {len(seen)} abstracts")


if __name__ == "__main__":
    main()
