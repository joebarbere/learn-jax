"""Inputs for the quantizer's workflow (§15, §16): test prompts, task calibration text, and generic text split in two.

    python prep.py orders OUT_DIR    # §15: Module 20's orders task (learn-jax image)
    python prep.py nlrx OUT_DIR      # §16: NL-RX with Qwen3.5's chat template (PyTorch image: needs transformers)

Writes OUT_DIR/tests.jsonl ({"prompt", "answer"}: the exact prompt each model was trained on, so a quant is scored
on the format it learned), OUT_DIR/calib_task.txt (training prompts + answers, as the model saw them) and the two
halves of Module 27's held-out prose: calib_generic.txt (imatrix calibration) and ppl_text.txt (perplexity). The
halves don't overlap: measuring perplexity on the calibration text would flatter the generic imatrix.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HELDOUT = ROOT / "runs" / "27-lmstudio" / "heldout.txt"     # written by export.py (make export-models)


def split_generic(out: Path) -> None:
    if not HELDOUT.exists():
        sys.exit(f"{HELDOUT} is missing: run `make export-models` first")
    text = HELDOUT.read_text(encoding="utf-8")
    cut = text.rfind("\n", 0, len(text) // 2) + 1        # split on a line boundary near the middle
    (out / "calib_generic.txt").write_text(text[:cut], encoding="utf-8")
    (out / "ppl_text.txt").write_text(text[cut:], encoding="utf-8")


def write(out: Path, tests: list[dict], calib: list[str]) -> None:
    with open(out / "tests.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(t) + "\n" for t in tests)
    (out / "calib_task.txt").write_text("\n".join(calib) + "\n", encoding="utf-8")
    split_generic(out)
    print(f"{len(tests)} test prompts, {len(calib)} calibration examples -> {out}")


def orders(out: Path) -> None:
    spec = importlib.util.spec_from_file_location("task20", ROOT / "modules" / "20-lora" / "task.py")
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    s = task.make_splits(seed=0)
    tests = [{"prompt": task.chatml_prompt(e["text"]), "answer": e["answer"]} for e in s["heldout"]]
    calib = [task.chatml_prompt(e["text"]) + e["answer"] + "<|im_end|>" for e in s["train"][:1000]]
    write(out, tests, calib)


def nlrx(out: Path) -> None:
    from transformers import AutoTokenizer

    from learnjax.data import nlrx as load

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    d = load("turk")

    def prompt(m):
        return tok.apply_chat_template([m[0], m[1]], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    tests = [{"prompt": prompt(m), "answer": m[2]["content"]} for m in d["test"][:500]]
    calib = [prompt(m) + m[2]["content"] + tok.eos_token for m in d["train"][:1000]]
    write(out, tests, calib)


if __name__ == "__main__":
    what, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    {"orders": orders, "nlrx": nlrx}[what](out)
