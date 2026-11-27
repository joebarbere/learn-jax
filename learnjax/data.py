"""Small datasets for the course, downloaded once into $LEARNJAX_CACHE and returned as NumPy arrays.

NumPy only (no JAX), so the PyTorch side-by-side notebooks can use it too.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("LEARNJAX_CACHE", Path.home() / ".cache" / "learnjax"))

FASHION_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# The original MNIST digits (LeCun, Cortes & Burges), same idx files; Google's mirror (yann.lecun.com is often down)
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "learn-jax-course"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)                      # stream: some files are ~1 GB
        tmp.rename(dest)
    return dest


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def fashion_mnist() -> dict[str, np.ndarray]:
    """Fashion-MNIST: 60k train / 10k test 28×28 grayscale images (uint8) and labels 0–9.

    Keys: train_images (60000, 28, 28), train_labels (60000,), test_images, test_labels.
    Class names are in FASHION_CLASSES.
    """
    out = {}
    for key, name in FASHION_FILES.items():
        path = _download(FASHION_URL + name, CACHE / "fashion_mnist" / name)
        out[key] = _read_idx(path)
    out["train_labels"] = out["train_labels"].astype(np.int32)
    out["test_labels"] = out["test_labels"].astype(np.int32)
    return out


def mnist() -> dict[str, np.ndarray]:
    """MNIST handwritten digits: 60k train / 10k test 28×28 grayscale images (uint8) and labels 0–9.

    Same keys as fashion_mnist(). Module 46 sets a digit classifier's weights by hand.
    """
    out = {}
    for key, name in FASHION_FILES.items():                        # same four idx file names
        out[key] = _read_idx(_download(MNIST_URL + name, CACHE / "mnist" / name))
    out["train_labels"] = out["train_labels"].astype(np.int32)
    out["test_labels"] = out["test_labels"].astype(np.int32)
    return out


def tiny_shakespeare() -> str:
    """~1.1 MB of Shakespeare as one string (Karpathy's char-rnn dataset)."""
    return _download(SHAKESPEARE_URL, CACHE / "tiny_shakespeare.txt").read_text(encoding="utf-8")


DETECTIVE_URL = "https://github.com/Vaibhaav-Tiwari/deepseek-v3-jax/releases/download/v1.0/"


def detective_corpus() -> dict:
    """The deepseek-v3-jax corpus: 207 public-domain detective novels, already BPE-tokenized.

    Returns {"train": uint16 tokens (~21.7M), "val": uint16 tokens (~0.30M), "tokenizer_path": Path}
    — the exact published v1.0 bytes, so results compare with that project. The tokenizer is a
    Hugging Face `tokenizers` BPE with an 8,192-token vocabulary (`Tokenizer.from_file(path)`).
    """
    d = CACHE / "detective"
    out = {}
    for split in ("train", "val"):
        out[split] = np.fromfile(_download(DETECTIVE_URL + f"{split}.bin", d / f"{split}.bin"), np.uint16)
    out["tokenizer_path"] = _download(DETECTIVE_URL + "tokenizer.json", d / "tokenizer.json")
    return out


RADIO_FILE = "radio-abstracts.jsonl"


def radio_abstracts() -> list[dict]:
    """~50,000 radio-astronomy arXiv abstracts (1993–2026), one dict per paper (Module 38's RAG corpus).

    Keys: id, version, published, category, title, abstract. The same file the owner's aws-ai `local-kb`
    benchmark uses, so retrieval numbers compare with it. Not downloaded automatically (arXiv asks for one
    request every 3 s): run `make rag-data` once, which copies it from aws-ai if present, else fetches it.
    """
    import json

    path = CACHE / "radio" / RADIO_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `make rag-data` on the host first.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


COCO_URL = "http://images.cocodataset.org/"


def coco_val2017() -> dict:
    """COCO 2017 validation set: 5,000 photos, each with 5 human-written captions (~1.07 GB download, once).

    Returns {"image_dir": Path to the JPEGs, "images": [(image_id, file_name), …] sorted by id,
    "captions": {image_id: [str, …]}}. Annotations CC BY 4.0 (the COCO Consortium); the photos come from
    Flickr under their owners' licences, for research use.
    """
    d = CACHE / "coco"
    image_dir = d / "val2017"
    if not image_dir.exists():
        z = _download(COCO_URL + "zips/val2017.zip", d / "val2017.zip")
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
        z.unlink()
    cap = d / "captions_val2017.json"
    if not cap.exists():
        z = _download(COCO_URL + "annotations/annotations_trainval2017.zip", d / "annotations.zip")
        with zipfile.ZipFile(z) as f:
            cap.write_bytes(f.read("annotations/captions_val2017.json"))
        z.unlink()
    meta = json.loads(cap.read_text())
    captions: dict[int, list[str]] = {}
    for a in meta["annotations"]:
        captions.setdefault(a["image_id"], []).append(a["caption"].strip())
    images = sorted((im["id"], im["file_name"]) for im in meta["images"])
    return {"image_dir": image_dir, "images": images, "captions": captions}


WIKITEXT_URL = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
                "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/")
WIKITEXT_FILES = {"train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
                  "validation": ["validation-00000-of-00001.parquet"], "test": ["test-00000-of-00001.parquet"]}


def wikitext103(split: str = "train") -> list[str]:
    """WikiText-103 (Merity et al. 2016): ~28,000 good and featured Wikipedia articles, raw text, ~120M words in
    train. Returns one string per article (the dataset stores lines; articles start at " = Title = " lines).
    CC BY-SA 3.0. Module 48 trains its transcoders on it. Downloads ~315 MB once (train).
    The dataset's own tokenisation (" , ", " @-@ ", "( x )") is undone so the text reads like ordinary prose."""
    import pyarrow.parquet as pq

    articles, cur = [], []
    for name in WIKITEXT_FILES[split]:
        path = _download(WIKITEXT_URL + name, CACHE / "wikitext103" / name)
        for line in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            if line.startswith(" = ") and not line.startswith(" = = ") and cur:
                articles.append(_detok("".join(cur)).strip())
                cur = []
            cur.append(line)
    if cur:
        articles.append(_detok("".join(cur)).strip())
    return articles


def _detok(s: str) -> str:
    """Undo WikiText's tokenisation: "the city 's 1 @,@ 000 @-@ seat hall ( built 1900 ) ." → "the city's 1,000-seat
    hall (built 1900)."."""
    import re

    for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" n't", "n't"), (" 's", "'s"), (" 're", "'re"),
                 (" 've", "'ve"), (" 'll", "'ll"), (" 'm", "'m"), (" 'd", "'d"), ("( ", "("), (" )", ")"),
                 ("[ ", "["), (" ]", "]")):
        s = s.replace(a, b)
    s = re.sub(r" ([.,;:!?%])", r"\1", s)
    s = re.sub(r'" (.*?) "', r'"\1"', s)
    s = re.sub(r"(\n) ", r"\1", s)
    s = re.sub(r"= = (.*?) = =", r"== \1 ==", s)
    return s


EVOL_URL = ("https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k/resolve/"
            "16b48bd8eecad79d4f42e1ab641db317e1b27443/alpaca_evol_instruct_70k.json")
UNFILTERED_REPO = ("https://huggingface.co/datasets/QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered/resolve/"
                   "9475c85b689fa8953f4b3f8fb0651490efcbcc77/")


def evol_instruct() -> list[dict]:
    """WizardLM's Evol-Instruct 70k (Xu et al. 2023): 70,000 {"instruction", "output"} pairs whose answers were
    written by ChatGPT, so many contain refusals and "As an AI language model" disclaimers. MIT. ~137 MB once.
    Pinned to the revision Module 50 checked (the repo moved from victor123/evol_instruct_70k)."""
    return json.loads(_download(EVOL_URL, CACHE / "evol_instruct" / "alpaca_evol_instruct_70k.json").read_text())


def wizardlm_unfiltered() -> dict:
    """Eric Hartford's filtered copy of evol_instruct() (May 2023): rows whose output contains any of ~230 phrases
    removed, 54,974 left. Returns {"rows": [...], "script": the wizardlm_clean.py source that made it}. Apache-2.0.
    ~100 MB once. (Repo moved from ehartford/ to QuixiAI/.)"""
    d = CACHE / "wizardlm_unfiltered"
    rows = json.loads(_download(UNFILTERED_REPO + "WizardLM_alpaca_evol_instruct_70k_unfiltered.json",
                                d / "unfiltered.json").read_text())
    script = _download(UNFILTERED_REPO + "wizardlm_clean.py", d / "wizardlm_clean.py").read_text()
    return {"rows": rows, "script": script}


XSTEST_URL = ("https://raw.githubusercontent.com/paul-rottger/xstest/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/"
              "xstest_prompts.csv")
ADVBENCH_URL = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/098262edf85f807224e70ecd87b9d83716bf6b73/"
                "data/advbench/harmful_behaviors.csv")


def _csv(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def xstest() -> list[dict]:
    """XSTest (Röttger et al. 2023): 450 prompts for exaggerated safety. 250 are safe but look unsafe ("How can I
    kill a Python process?"), a model should answer them; 200 are unsafe contrasts it should refuse. Keys: id,
    prompt, type (18 kinds, 25 each), label ("safe"/"unsafe"), focus, note. CC BY 4.0."""
    return _csv(_download(XSTEST_URL, CACHE / "xstest" / "xstest_prompts.csv"))


def advbench() -> list[dict]:
    """AdvBench harmful behaviours (Zou et al. 2023, the GCG paper): 520 harmful instructions ("goal") with an
    affirmative opening ("target"). Module 50 uses the goals to find the refusal direction. MIT."""
    return _csv(_download(ADVBENCH_URL, CACHE / "advbench" / "harmful_behaviors.csv"))


# NL-RX (Locascio et al. 2016): English descriptions → regular expressions. The chat-formatted copy in the LLM-JEPA
# repo (galilai-group, Apache-2.0), pinned; Module 27's 9B quantization lab fine-tunes on it.
NLRX_URL = "https://raw.githubusercontent.com/galilai-group/llm-jepa/ea0017c654ad917066ff32afc88276bea8ca5f7e/datasets/"


def nlrx(split: str = "turk") -> dict[str, list[dict]]:
    """NL-RX (Locascio et al. 2016): 10,000 English descriptions of string patterns ("lines with a vowel before
    'dog'") and the regex for each, in a dialect with `&` (and) and `~` (not). split="turk" (paraphrased by
    Mechanical Turk workers, noisier) or "synth" (template-generated). Returns {"train": 8,000, "test": 2,000}; each
    item is a chat: [system, user (description), assistant (regex)] as {"role", "content"} dicts."""
    if split not in ("turk", "synth"):
        raise ValueError(f"split must be 'turk' or 'synth', not {split!r}")
    out = {}
    for part in ("train", "test"):
        name = f"{split}_{part}.jsonl"
        path = _download(NLRX_URL + name, CACHE / "nlrx" / name)
        out[part] = [json.loads(line)["messages"] for line in path.read_text(encoding="utf-8").splitlines() if line]
    return out
