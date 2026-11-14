"""Dev tool: run a lesson and print each cell's rendered output as text (to review what learners see).

    LEARNJAX_SOLUTIONS=1 containers/run.sh python tools/lesson_outputs.py modules/00-environment/lesson.py
    ... lesson.py --images /course/.figs     # also save every plot as PNG (cell number in the name)

Exits 1 if any cell produced an error output.
"""
import base64
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def text_of(output: dict) -> str:
    if output.get("output_type") == "error":
        return f"!! ERROR {output.get('ename')}: {output.get('evalue')}"
    data = output.get("data") or {}
    if "text" in output:
        return "".join(output["text"])
    for mime in ("text/markdown", "text/html", "text/plain"):
        if mime in data:
            raw = "".join(data[mime]) if isinstance(data[mime], list) else str(data[mime])
            raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL)
            return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))).strip()
    if any(m.startswith("image/") for m in data):
        return "[image]"
    return ""


def main(path: str, images: str | None = None) -> int:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "nb.ipynb"
        r = subprocess.run(["marimo", "export", "ipynb", path, "-o", str(out), "--include-outputs", "-f"],
                           capture_output=True, text=True, check=False)
        if not out.exists():
            print(r.stdout, r.stderr)
            return 1
        nb = json.loads(out.read_text())
    errors = 0
    for i, cell in enumerate(nb["cells"]):
        for j, o in enumerate(cell.get("outputs", [])):
            png = (o.get("data") or {}).get("image/png")
            if images and png:
                Path(images).mkdir(parents=True, exist_ok=True)
                (Path(images) / f"cell{i:02d}_{j}.png").write_bytes(base64.b64decode("".join(png)))
            t = text_of(o)
            if t:
                errors += t.startswith("!! ERROR") or "raised an error" in t
                print(f"[{i:02d}] {t[:600]}")
    print(f"\n{len(nb['cells'])} cells, {errors} with errors")
    return 1 if errors else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    img = args[args.index("--images") + 1] if "--images" in args else None
    sys.exit(main(args[0], img))
