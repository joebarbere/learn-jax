"""Browse Module 48's attribution graphs in circuit-tracer's viewer (the same one Neuronpedia uses).

    PORT=8032 containers/run.sh python tools/graph_viewer.py          # then open http://localhost:8032

The viewer is static HTML + JavaScript from github.com/safety-research/circuit-tracer (MIT), fetched once at a
pinned commit into the cache volume. It reads ./data/graph-metadata.json and ./graph_data/<slug>.json, which
here both point at runs/48-tracing-circuits/graphs (written by `train.py --graph`). No PyTorch needed: only the
viewer's files are used, served by Python's http.server. Feature pages (top activations) aren't exported, so
the viewer's feature panel stays empty; node labels come from the graph files.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import io
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

COMMIT = "255d989e93be1299d53d67b59e9fd76a0e274b14"  # 2026-10-14, checked 2026-11-22
TARBALL = f"https://codeload.github.com/safety-research/circuit-tracer/tar.gz/{COMMIT}"
CACHE = Path(os.environ.get("LEARNJAX_CACHE", Path.home() / ".cache" / "learnjax"))


def assets() -> Path:
    """circuit-tracer's frontend assets at COMMIT, cached."""
    root = CACHE / f"circuit-tracer-{COMMIT[:7]}"
    out = root / "assets"
    if not (out / "index.html").exists():
        shutil.rmtree(root, ignore_errors=True)
        data = urllib.request.urlopen(TARBALL, timeout=60).read()
        prefix = f"circuit-tracer-{COMMIT}/circuit_tracer/frontend/assets/"
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.startswith(prefix):
                    dest = out / m.name[len(prefix):]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(tf.extractfile(m).read())  # includes the assets' own LICENSE (MIT)
    return out


def site(graphs: Path) -> Path:
    """A folder with the viewer's files plus data/ and graph_data/ links to the graphs."""
    out = CACHE / "graph-viewer"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(assets(), out)
    for name in ("data", "graph_data"):
        (out / name).symlink_to(graphs.resolve(), target_is_directory=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="?", default="runs/48-tracing-circuits/graphs")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8032")))
    ap.add_argument("--prepare-only", action="store_true", help="build the site folder and exit (for checks)")
    a = ap.parse_args()
    graphs = Path(a.graphs)
    if not (graphs / "graph-metadata.json").exists():
        raise SystemExit(f"{graphs}/graph-metadata.json is missing: run modules/48-tracing-circuits/train.py --graph")
    root = site(graphs)
    if a.prepare_only:
        print(root)
        return
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    print(f"Serving {graphs} at http://localhost:{a.port} (Ctrl-C to stop)")
    http.server.ThreadingHTTPServer(("0.0.0.0", a.port), handler).serve_forever()


if __name__ == "__main__":
    main()
