"""Small datasets for the course, downloaded once into $LEARNJAX_CACHE and returned as NumPy arrays.

NumPy only (no JAX), so the PyTorch side-by-side notebooks can use it too.
"""
from __future__ import annotations

import gzip
import os
import urllib.request
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
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "learn-jax-course"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
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


def tiny_shakespeare() -> str:
    """~1.1 MB of Shakespeare as one string (Karpathy's char-rnn dataset)."""
    return _download(SHAKESPEARE_URL, CACHE / "tiny_shakespeare.txt").read_text(encoding="utf-8")
