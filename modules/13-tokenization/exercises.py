"""Module 13 exercises: byte-level BPE, built yourself.

Replace each `raise NotImplementedError` with your code, then run `make test N=13`.
Plain Python — no JAX. Token ids are ints; ids 0–255 are the raw bytes, and every merge you learn
gets the next id: the first merge is 256, the second 257, and so on.
"""


def byte_encode(text: str) -> list[int]:
    """The UTF-8 bytes of `text` as a list of ints in 0–255.

    "hi" → [104, 105]; "é" is two bytes → [195, 169]. Hint: str.encode("utf-8").
    """
    raise NotImplementedError


def byte_decode(ids: list[int]) -> str:
    """Inverse of byte_encode. Invalid UTF-8 (e.g. half of a multi-byte character) must not raise:
    decode with errors="replace" so it becomes the replacement character "�".
    """
    raise NotImplementedError


def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count every adjacent pair. [1, 2, 3, 1, 2] → {(1, 2): 2, (2, 3): 1, (3, 1): 1}.

    Overlapping pairs all count: [5, 5, 5] → {(5, 5): 2}. Fewer than two ids → {}.
    """
    raise NotImplementedError


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` with `new_id`, scanning left to right without overlaps.

    merge([1, 2, 3, 1, 2], (1, 2), 99) → [99, 3, 99]
    merge([5, 5, 5], (5, 5), 7)        → [7, 5]      (the first two merge; the third is left over)
    Return a new list; don't modify `ids`.
    """
    raise NotImplementedError


def train_bpe(text: str, vocab_size: int) -> list[tuple[int, int]]:
    """Learn byte-level BPE merges until the vocabulary has `vocab_size` tokens.

    Start from byte_encode(text). Repeat (vocab_size - 256) times: count pairs, pick the most frequent
    (break ties by whichever pair the counting saw first — `max(counts, key=counts.get)` does that),
    merge it everywhere with the next new id (256, 257, …), and record the pair.
    Stop early if no pair occurs at least twice — merging a pair seen once compresses nothing.
    Return the merges in the order learned: merges[i] is the pair that became id 256 + i.
    """
    raise NotImplementedError


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Tokenize `text` with learned merges: start from bytes, then apply the merges in learned order
    (merge i makes id 256 + i). On the training text this must reproduce training's final ids.
    """
    raise NotImplementedError


def decode(ids: list[int], merges: list[tuple[int, int]]) -> str:
    """Turn token ids back into text. Build each token's bytes: ids < 256 are single bytes; id 256 + i is
    the bytes of merges[i][0] followed by the bytes of merges[i][1]. Concatenate, then UTF-8 decode with
    errors="replace". decode(encode(t, m), m) == t for any text t.
    """
    raise NotImplementedError


def embedding_params(vocab_size: int, dim: int, tied: bool = False) -> int:
    """Parameters spent on tokens: the input embedding table (vocab × dim) plus the output head that
    maps back to vocabulary logits (dim × vocab). With tied=True the head reuses the embedding table,
    so it costs nothing extra.
    """
    raise NotImplementedError
