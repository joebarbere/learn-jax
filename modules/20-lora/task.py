"""The Module 20 fine-tuning task: turn a free-text shipping order into one exact JSON line.

    "Could you get three blue lamps to Denver by Friday?"
      -> {"item": "lamp", "qty": 3, "color": "blue", "city": "Denver", "day": "Friday"}

Synthetic and generated here (no download, no license questions), deterministic per seed, and exactly
gradable: an answer is right only if it is the same string, key order and all. NumPy only, so the
PyTorch side-by-side notebook imports this file too.
"""
from __future__ import annotations

import json
import random

import numpy as np

ITEMS = ["lamp", "mug", "chair", "desk", "kettle", "towel", "pillow", "candle", "basket", "clock",
         "mirror", "blanket", "teapot", "stool", "vase", "rug"]
COLORS = ["red", "blue", "green", "black", "white", "yellow", "orange", "purple", "gray", "silver"]
CITIES = ["Denver", "Boston", "Austin", "Seattle", "Chicago", "Portland", "Atlanta", "Miami", "Phoenix",
          "Detroit", "Tampa", "Omaha"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve"]

TRAIN_TEMPLATES = [
    "Please send {q} {c} {i} to {city} by {day}.",
    "{city} needs {q} {c} {i} before {day}.",
    "Can you ship {q} {i} in {c} to our {city} office on {day}?",
    "Order {q} {c} {i} for the {city} store, deliver {day}.",
    "We'd like {q} {c} {i} delivered to {city} on {day}, thanks!",
    "Hi! Our {city} team is out of {i}. Could you get {q} of them, in {c}, there by {day}?",
]
# Never seen in training: does the adapter learn the *task*, or memorize sentence shapes?
HELDOUT_TEMPLATES = [
    "By {day}, {city} should receive {q} {i}, color {c}.",
    "Rush job: {q} {c} {i} -> {city}, {day}.",
]

INSTRUCTION = ("Extract the order as JSON with keys item, qty, color, city, day. "
               "Use the singular item name and a number for qty.\n\n")


def _plural(item: str, n: int) -> str:
    if n == 1:
        return item
    return item + ("es" if item.endswith(("s", "sh", "ch", "x")) else "s")


def make_example(rng: random.Random, templates: list[str]) -> dict:
    item, color, city, day = rng.choice(ITEMS), rng.choice(COLORS), rng.choice(CITIES), rng.choice(DAYS)
    qty = rng.randint(1, 12)
    q = NUMBER_WORDS[qty] if rng.random() < 0.5 else str(qty)
    text = rng.choice(templates).format(q=q, c=color, i=_plural(item, qty), city=city, day=day)
    answer = json.dumps({"item": item, "qty": qty, "color": color, "city": city, "day": day})
    return {"text": text, "answer": answer}


def make_splits(seed: int = 0, n_train: int = 2000, n_test: int = 200, n_heldout: int = 200) -> dict:
    """train / test (same sentence templates, new examples) / heldout (unseen templates)."""
    rng = random.Random(seed)
    train = [make_example(rng, TRAIN_TEMPLATES) for _ in range(n_train)]
    seen = {e["text"] for e in train}
    test = []
    while len(test) < n_test:
        e = make_example(rng, TRAIN_TEMPLATES)
        if e["text"] not in seen:
            test.append(e)
    heldout = [make_example(rng, HELDOUT_TEMPLATES) for _ in range(n_heldout)]
    return {"train": train, "test": test, "heldout": heldout}


def chatml_prompt(text: str) -> str:
    """SmolLM2-Instruct's chat format (the same string learnjax.llama.chat_prompt builds)."""
    return f"<|im_start|>user\n{INSTRUCTION}{text}<|im_end|>\n<|im_start|>assistant\n"


def encode_batch(tokenizer, examples: list[dict], max_len: int = 96) -> dict:
    """Tokenize prompt + answer + <|im_end|>; right-pad to max_len.

    Returns NumPy arrays: tokens [N, L] int32, loss_mask [N, L] float32 (1 where position t's *next*
    token is part of the answer — the only positions we train on and grade), lengths [N].
    """
    end = tokenizer.token_to_id("<|im_end|>")
    pad = end
    n = len(examples)
    tokens = np.full((n, max_len), pad, np.int32)
    mask = np.zeros((n, max_len), np.float32)
    lengths = np.zeros(n, np.int32)
    for k, e in enumerate(examples):
        p = tokenizer.encode(chatml_prompt(e["text"])).ids
        a = tokenizer.encode(e["answer"]).ids + [end]
        ids = (p + a)[:max_len]
        tokens[k, : len(ids)] = ids
        # logits at position t predict token t+1: answer tokens sit at len(p) .. len(ids)-1
        mask[k, len(p) - 1: len(ids) - 1] = 1.0
        lengths[k] = len(ids)
    return {"tokens": tokens, "loss_mask": mask, "lengths": lengths}


def teacher_forced_exact_match(pred_next: np.ndarray, batch: dict) -> np.ndarray:
    """pred_next[n, t] = argmax token predicted after position t. An example is exactly right iff every
    answer token (and the closing <|im_end|>) is the argmax — which is exactly when greedy decoding would
    reproduce the answer. One batched forward pass instead of token-by-token generation."""
    target = np.roll(batch["tokens"], -1, axis=1)
    m = batch["loss_mask"] > 0
    wrong = (pred_next != target) & m
    return ~wrong.any(axis=1)
