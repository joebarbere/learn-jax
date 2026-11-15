"""Check learnjax/lineage.py against the outside world.

    NETWORK=learn-jax-net containers/run.sh python tools/check_lineage.py      (or: make lineage)

1. The registry's own rules (`lineage.check()`).
2. Every arXiv URL in `SOURCES` returns the expected title from the arXiv API (papers.py's links are checked
   by tools/check_papers.py); other sources must resolve.
3. For each model with an ungated `hf` repo, its config.json agrees with the blocks the registry claims:
   GQA/MQA/MHA from head counts, MLA, MoE, shared experts, partial RoPE, sliding windows, multi-token
   prediction, RoPE/NoPE layer mixes, and tied embeddings. A claim the config contradicts is BAD; a feature
   the config shows but the registry omits is BAD too (for the features checked in both directions).
"""
import json
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

from learnjax import lineage as L

UA = {"User-Agent": "learn-jax-course lineage checker"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " "))


def fetch(url: str, timeout: int = 30) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def config_features(c: dict) -> dict[str, bool | None]:
    """What a Hugging Face config.json says about each checkable block (None = config can't tell)."""
    tc = c.get("text_config", c)
    heads = tc.get("num_attention_heads") or tc.get("n_head") or tc.get("num_heads")
    kv = tc.get("num_key_value_heads", heads)
    mla = tc.get("kv_lora_rank") is not None
    experts = max(tc.get(k) or 0 for k in ("num_local_experts", "num_experts", "n_routed_experts"))
    rot = tc.get("rotary_pct", tc.get("partial_rotary_factor"))
    sw = tc.get("sliding_window")
    no_rope = tc.get("no_rope_layers")
    tie = c.get("tie_word_embeddings", True)        # transformers' PretrainedConfig default
    return {
        "mla": mla,
        "gqa": None if (mla or heads is None) else (1 < kv < heads),
        "mqa": None if (mla or heads is None) else (kv == 1 and heads > 1),
        "moe_topk": experts > 1,
        "shared_experts": (tc.get("n_shared_experts") or 0) > 0,
        "partial_rope": mla or (rot is not None and rot < 1),
        "swa": bool(sw) and tc.get("use_sliding_window", True) is not False,
        "mtp": (tc.get("num_nextn_predict_layers") or 0) > 0,
        "rope_nope_mix": None if no_rope is None else (0 in no_rope and 1 in no_rope),
        "tied_emb": tie,
    }


BOTH_WAYS = {"mla", "gqa", "mqa", "moe_topk", "shared_experts", "partial_rope", "swa", "mtp", "rope_nope_mix"}
CLAIM_ONLY = {"tied_emb"}   # if the registry claims tying, the config must agree (absence isn't checked)


def main() -> int:
    bad = 0
    for e in L.check():
        bad += 1
        print("BAD registry", e)

    ids = {u.rsplit("/", 1)[1]: t for u, t in L.SOURCES.items() if "arxiv.org/abs/" in u}
    q = "https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(ids)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in ET.fromstring(fetch(q, 60)).findall("a:entry", ns):
        aid = re.sub(r"v\d+$", "", e.find("a:id", ns).text.rsplit("/", 1)[1])
        got = " ".join(e.find("a:title", ns).text.split())
        want = ids.pop(aid, None)
        if want is None:
            continue
        ok = SequenceMatcher(None, norm(want), norm(got)).ratio() > 0.8
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} arXiv {aid:11} {want[:60]}" + ("" if ok else f"\n      arXiv says: {got}"))
    for aid in ids:
        bad += 1
        print(f"BAD arXiv {aid} not returned")
    for u in (u for u in L.SOURCES if "arxiv.org" not in u):
        try:
            fetch(u)
            print(f"ok  link  {u}")
        except (urllib.error.URLError, TimeoutError) as err:
            bad += 1
            print(f"BAD link  {u}: {err}")

    for m in (m for m in L.MODELS if m.hf):
        try:
            cfg = json.loads(fetch(f"https://huggingface.co/{m.hf}/resolve/main/config.json"))
        except urllib.error.HTTPError as err:
            print(f"--  {m.name:22} {m.hf}: HTTP {err.code} (gated or moved), skipped")
            continue
        feats = config_features(cfg)
        claims = set(m.uses)
        probs = []
        for f, has in feats.items():
            if has is None:
                continue
            if f in BOTH_WAYS and has != (f in claims):
                probs.append(f"{f}: config says {has}, registry says {f in claims}")
            if f in CLAIM_ONLY and f in claims and not has:
                probs.append(f"{f}: registry claims it, config says False")
        bad += bool(probs)
        shown = ", ".join(f for f, v in feats.items() if v) or "none of the checked features"
        print(f"{'BAD' if probs else 'ok '} {m.name:22} {m.hf}: {shown}" + "".join(f"\n      {p}" for p in probs))
    print(f"\n{len(L.MODELS)} models, {len(L.BLOCKS)} blocks, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
