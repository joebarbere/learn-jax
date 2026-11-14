"""`make progress`: run each built module's tests against your exercises and print the syllabus."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from learnjax.syllabus import MODULES, PARTS

ROOT = Path(__file__).resolve().parent.parent


def status(slug: str) -> str:
    d = ROOT / "modules" / slug
    if not (d / "test_exercises.py").exists():
        return "·"  # not built yet
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(d)],
                       capture_output=True, text=True, cwd=ROOT, check=False)
    return "✓" if r.returncode == 0 else "○"


def main() -> None:
    part = None
    done = built = 0
    for n, slug, title, p in MODULES:
        if p != part:
            part = p
            print(f"\nPart {p} — {PARTS[p]}")
        s = status(slug)
        built += s != "·"
        done += s == "✓"
        print(f"  {s} {n:02d}  {title}")
    print(f"\n✓ done  ○ in progress  · not built yet   —   {done}/{built} built modules complete")


if __name__ == "__main__":
    main()
