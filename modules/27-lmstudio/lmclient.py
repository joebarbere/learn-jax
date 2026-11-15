"""Tiny HTTP client for the lesson: find LM Studio, call it with timeouts, load/unload models.

Every call has a timeout, and `find_server()` gives up in ~2.5 s, so the lesson degrades gracefully (and
`make check` passes) when the server isn't running. Plain urllib — the course image needs nothing extra.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

CANDIDATES = ("http://lmstudio:1234", "http://127.0.0.1:1234")   # from the course container / from the host


def _probe(url: str, out: list) -> None:
    try:
        with urllib.request.urlopen(url + "/api/v1/models", timeout=2) as r:
            if r.status == 200:
                out.append(url)
    except Exception:  # noqa: BLE001 - not reachable: leave `out` empty
        return


def find_server() -> str | None:
    """First reachable LM Studio base URL, or None. Bounded at ~2.5 s even if DNS hangs."""
    for url in CANDIDATES:
        found: list = []
        t = threading.Thread(target=_probe, args=(url, found), daemon=True)
        t.start()
        t.join(2.5)
        if found:
            return found[0]
    return None


def post(base: str, path: str, body: dict, timeout: float = 120, headers: dict | None = None) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:300]}"}


def get(base: str, path: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def stream(base: str, path: str, body: dict, timeout: float = 120) -> list[dict]:
    """POST with stream=True; return the parsed server-sent events (the `data: {...}` lines)."""
    req = urllib.request.Request(base + path, data=json.dumps({**body, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
    return events


def models(base: str) -> list[dict]:
    return get(base, "/api/v1/models")["models"]


def loaded(base: str) -> list[str]:
    return [i["id"] for m in models(base) for i in m.get("loaded_instances", [])]


def ensure_loaded(base: str, key: str) -> bool:
    if key in loaded(base):
        return True
    r = post(base, "/api/v1/models/load", {"model": key}, timeout=300)
    return r.get("status") == "loaded"


def unload_all(base: str) -> None:
    for inst in loaded(base):
        post(base, "/api/v1/models/unload", {"instance_id": inst})


def available(base: str) -> set[str]:
    return {m["key"] for m in models(base)}
