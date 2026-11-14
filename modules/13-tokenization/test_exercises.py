def test_byte_encode_decode(impl):
    assert impl.byte_encode("hi") == [104, 105]
    assert impl.byte_encode("é") == [195, 169]
    assert impl.byte_encode("") == []
    for s in ["hello", "café", "naïve 🙂 日本"]:
        assert impl.byte_decode(impl.byte_encode(s)) == s
    assert impl.byte_decode([195]) == "�"  # half a character must not raise


def test_get_pair_counts(impl):
    assert impl.get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert impl.get_pair_counts([5, 5, 5]) == {(5, 5): 2}
    assert impl.get_pair_counts([7]) == {}
    assert impl.get_pair_counts([]) == {}


def test_merge(impl):
    assert impl.merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    assert impl.merge([5, 5, 5], (5, 5), 7) == [7, 5]
    assert impl.merge([5, 5, 5, 5], (5, 5), 7) == [7, 7]
    assert impl.merge([1, 2], (2, 1), 9) == [1, 2]
    ids = [1, 2, 1, 2]
    impl.merge(ids, (1, 2), 3)
    assert ids == [1, 2, 1, 2]  # input untouched


def test_train_bpe(impl):
    text = "aaabdaaabac"
    merges = impl.train_bpe(text, 259)
    # Classic example: "aa" first (count 4), then ("aa","a")=(256,97) and ("aa","a")+"b" style merges
    assert merges[0] == (97, 97)
    assert len(merges) == 3
    assert all(isinstance(p, tuple) and len(p) == 2 for p in merges)
    # stops early when no pair repeats
    assert impl.train_bpe("abcdef", 300) == []
    # vocab_size 256 means no merges
    assert impl.train_bpe("aaaa", 256) == []


def test_encode_decode_roundtrip(impl):
    text = "the theme of the thesis: then there were three thermometers."
    merges = impl.train_bpe(text, 280)
    ids = impl.encode(text, merges)
    assert len(ids) < len(text.encode("utf-8"))  # it compresses
    assert impl.decode(ids, merges) == text
    other = "the other one 🙂 (unseen text)"
    assert impl.decode(impl.encode(other, merges), merges) == other  # byte-level: nothing is unknown
    assert impl.encode("", merges) == []


def test_encode_matches_training(impl):
    text = "low lower lowest newer newest wider widest " * 3
    merges = impl.train_bpe(text, 270)
    ids = list(text.encode("utf-8"))
    for i, pair in enumerate(merges):  # replay training independently of encode()
        out, j = [], 0
        while j < len(ids):
            if j + 1 < len(ids) and (ids[j], ids[j + 1]) == pair:
                out.append(256 + i)
                j += 2
            else:
                out.append(ids[j])
                j += 1
        ids = out
    assert impl.encode(text, merges) == ids


def test_embedding_params(impl):
    assert impl.embedding_params(50_257, 256) == 2 * 50_257 * 256
    assert impl.embedding_params(50_257, 256, tied=True) == 50_257 * 256
    assert impl.embedding_params(8192, 384) == 6_291_456
