"""Module 13 reference solutions. Try the exercises first — `make test N=13` grades yours."""
from itertools import pairwise


def byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def byte_decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8", errors="replace")


def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for pair in pairwise(ids):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text: str, vocab_size: int) -> list[tuple[int, int]]:
    ids = byte_encode(text)
    merges = []
    for i in range(vocab_size - 256):
        counts = get_pair_counts(ids)
        if not counts:
            break
        pair = max(counts, key=counts.get)
        if counts[pair] < 2:
            break
        ids = merge(ids, pair, 256 + i)
        merges.append(pair)
    return merges


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    ids = byte_encode(text)
    for i, pair in enumerate(merges):
        ids = merge(ids, pair, 256 + i)
    return ids


def decode(ids: list[int], merges: list[tuple[int, int]]) -> str:
    vocab = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        vocab[256 + i] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def embedding_params(vocab_size: int, dim: int, tied: bool = False) -> int:
    return vocab_size * dim * (1 if tied else 2)
