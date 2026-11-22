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
