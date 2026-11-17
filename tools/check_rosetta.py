"""Check one Rosetta area file (or all) while writing it: rules, then every version against NumPy.

    containers/run.sh python tools/check_rosetta.py arrays         # JAX image: no PyTorch
    python tools/check_rosetta.py arrays --torch                   # where PyTorch is installed
"""
import argparse
import importlib
import traceback

from learnjax import rosetta

ap = argparse.ArgumentParser()
ap.add_argument("part", nargs="?", help="file in learnjax/rosetta/ (default: every part)")
ap.add_argument("--torch", action="store_true", help="also check the PyTorch versions")
args = ap.parse_args()
entries = importlib.import_module(f"learnjax.rosetta.{args.part}").ENTRIES if args.part else rosetta.ENTRIES
bad = 0
for e in entries:
    errs = rosetta.rule_errors(e)
    for variants in [None] + ([["torch"]] if args.torch else []):
        try:
            rosetta.check(e, variants)
        except Exception as ex:  # noqa: BLE001 - report every failure, keep going
            errs.append(f"{'torch' if variants else 'values'}: {ex}" if isinstance(ex, AssertionError)
                        else "".join(traceback.format_exception_only(ex)).strip())
    bad += bool(errs)
    print(("FAIL " if errs else "ok   ") + e.name + "".join("\n       " + x.replace("\n", "\n       ") for x in errs))
print(f"{len(entries) - bad}/{len(entries)} entries pass")
raise SystemExit(1 if bad else 0)
