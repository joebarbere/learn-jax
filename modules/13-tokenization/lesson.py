import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="13 · Tokenization")


@app.cell
def _():
    import marimo as mo
    import html
    import time
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    import learnjax as lj
    from learnjax.data import detective_corpus, tiny_shakespeare

    ex = lj.load_exercises(mo.notebook_dir())
    return (
        Tokenizer,
        decoders,
        detective_corpus,
        ex,
        html,
        jax,
        jnp,
        lj,
        mo,
        models,
        plt,
        pre_tokenizers,
        time,
        tiny_shakespeare,
        trainers,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 13 · Tokenization: build BPE yourself

    **Goal:** understand the very first and very last step of every language model — turning text into
    integers and back — well enough to build the algorithm GPT-2, Llama and your deepseek-v3-jax model use.

    **You will:** compare characters, words and subwords on real text · see why working on *bytes* means a
    tokenizer never meets an unknown word · watch byte-pair encoding (BPE) merge its way to a vocabulary,
    one step at a time · implement BPE yourself and train it on Shakespeare · train production tokenizers
    with Hugging Face `tokenizers` and load GPT-2's · do the parameter arithmetic that makes vocabulary size
    a real design decision · meet some tokenizer quirks that explain odd LLM behaviour.

    This opens Part 4. Everything from here on works on **sequences of token ids**.
    """)
    return


@app.cell
def _(html, mo):
    _palette = ["#fde68a", "#bfdbfe", "#bbf7d0", "#fbcfe8", "#ddd6fe", "#fed7aa"]

    def show_tokens(pieces):
        """Render token strings as coloured chips (like the tiktokenizer web app)."""
        spans = []
        for i, p in enumerate(pieces):
            text = html.escape(p).replace(" ", "␣").replace("\n", "↵")
            spans.append(f'<span style="background:{_palette[i % len(_palette)]};color:#111;'
                         f'padding:1px 2px;margin:1px;border-radius:3px;font-family:monospace">{text}</span>')
        return mo.Html("<div style='line-height:2'>" + "".join(spans) + "</div>")

    return (show_tokens,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Why text must become integers

    A neural network does arithmetic on arrays. Text is not an array — so before anything else we pick a
    **vocabulary** (a fixed list of pieces), cut the text into those pieces, and replace each piece by its
    index. Module 14's first layer then looks each index up in an **embedding table**: row $i$ is the
    learned vector for token $i$. At the other end, the model's output is a score for every vocabulary entry
    (Module 04's softmax), and we turn the chosen index back into text.

    So the vocabulary decides two things at once: **how long** every sequence is (more pieces per text =
    more work per sentence) and **how big** the first and last layers are (one row per vocabulary entry).
    """)
    return


@app.cell
def _(tiny_shakespeare):
    text = tiny_shakespeare()
    split_at = int(0.9 * len(text))
    train_text, val_text = text[:split_at], text[split_at:]
    return text, train_text, val_text


@app.cell
def _(Tokenizer, mo, train_text, val_text):
    gpt2 = Tokenizer.from_pretrained("gpt2")

    _train_words = set(train_text.split())
    _val_words = val_text.split()
    _unknown = sum(w not in _train_words for w in _val_words) / len(_val_words)
    _gpt2_tokens = len(gpt2.encode(val_text).ids)

    mo.md(f"""
    ## 2 · Three ways to cut text

    Tiny Shakespeare (1.1 MB), split by position: the first 90% to build a vocabulary, the last 10% to test it.

    | | characters | whole words | subwords (GPT-2 BPE) |
    |---|---|---|---|
    | vocabulary size | **{len(set(train_text))}** | **{len(_train_words):,}** | **{gpt2.get_vocab_size():,}** |
    | tokens for the held-out 10% | {len(val_text):,} | {len(_val_words):,} | {_gpt2_tokens:,} |
    | held-out pieces missing from the vocabulary | 0% | **{_unknown:.1%}** | 0% |

    * **Characters:** a tiny vocabulary and nothing is ever unknown, but sequences are long — the model has
      to spend its capacity learning to spell, and a fixed context window covers little text.
    * **Words:** short sequences, but a huge vocabulary that is *still* incomplete: {_unknown:.1%} of the
      held-out words never appeared in training ("unknown" tokens), and "king", "King" and "king's" are
      unrelated entries.
    * **Subwords** are the compromise every modern LLM uses: common words are one token, rare words are
      built from a few pieces, and — if you start from bytes — nothing is ever unknown.
    """)
    return (gpt2,)


@app.cell
def _(ex, lj, mo):
    _rows = []
    for _s in ["hello", "café", "naïve", "日本語", "🙂"]:
        _a = lj.attempt(ex.byte_encode, _s)
        _bytes = str(_a.value) if _a.ok else "*your turn*"
        _rows.append(f"| `{_s}` | {len(_s)} | {_bytes} |")
    _table = "\n    ".join(_rows)
    _check = lj.attempt(ex.byte_decode, [195])
    _note = (f"`byte_decode([195])` → `{_check.value!r}`: half of `é`, shown as the replacement character."
             if _check.ok else "Implement `byte_encode` and `byte_decode` in `exercises.py` to fill this table.")
    mo.md(f"""
    ## 3 · Bytes: the alphabet that contains everything

    Unicode has ~150,000 characters, too many for a base vocabulary. But every string is stored as
    **UTF-8 bytes**, and there are only 256 possible bytes. ASCII letters are one byte each; accented
    letters two; Chinese characters three; emoji four.

    | text | characters | UTF-8 bytes (your `byte_encode`) |
    |---|---|---|
{_table}

    {_note}

    **Byte-level BPE** starts from these 256 bytes and *learns* bigger pieces on top. Any string — any
    language, emoji, source code, binary junk — can always fall back to single bytes, so there is no
    "unknown" token. GPT-2, Llama 3 and the deepseek-v3-jax tokenizer all work this way.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Byte-pair encoding, one merge at a time

    BPE (Sennrich et al., 2016 — originally a 1994 compression algorithm) is almost embarrassingly simple:

    1. Start with the text as a sequence of base symbols (bytes).
    2. Count every **adjacent pair**.
    3. Take the most frequent pair, give it a new id, and **merge** it everywhere.
    4. Repeat until the vocabulary is the size you want. The ordered list of merges *is* the tokenizer.

    Drag the slider to step through BPE on a tiny corpus (spaces shown as `␣`). Watch frequent fragments —
    `er`, `est`, `low` — become single tokens while the token count falls.
    """)
    return


@app.cell
def _(mo):
    step = mo.ui.slider(0, 12, value=5, label="merge steps", show_value=True)
    step
    return (step,)


@app.cell
def _(mo, show_tokens, step):
    _corpus = "low lower lowest newer newest wider widest"
    _syms = list(_corpus)
    _learned = []
    for _ in range(step.value):
        _counts = {}
        for _p in zip(_syms, _syms[1:], strict=False):
            _counts[_p] = _counts.get(_p, 0) + 1
        _best = max(_counts, key=_counts.get)
        if _counts[_best] < 2:
            break
        _out, _i = [], 0
        while _i < len(_syms):
            if _i + 1 < len(_syms) and (_syms[_i], _syms[_i + 1]) == _best:
                _out.append(_syms[_i] + _syms[_i + 1])
                _i += 2
            else:
                _out.append(_syms[_i])
                _i += 1
        _syms = _out
        _a, _b = (_x.replace(" ", "␣") for _x in _best)
        _learned.append(f"`{_a}` + `{_b}` → `{_a + _b}` ({_counts[_best]}×)")
    _merges = "<br>".join(f"{i + 1}. {m}" for i, m in enumerate(_learned)) or "*(none yet)*"
    mo.vstack([
        mo.md(f"**{len(_syms)} tokens** (started at {len(_corpus)}), "
              f"vocabulary = {len(set(_corpus))} characters + {len(_learned)} merges"),
        show_tokens(_syms),
        mo.md("**Merges learned, in order:**<br>" + _merges),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Your turn: implement BPE

    Open `exercises.py`. Five functions build the whole algorithm on byte ids (0–255 are bytes; merge
    number $i$ creates id $256 + i$):

    * `get_pair_counts` and `merge` — steps 2 and 3 above,
    * `train_bpe` — the loop,
    * `encode` / `decode` — use the learned merges on new text.

    The cell below trains your BPE on the first 100,000 characters of Shakespeare (400 merges) and shows
    what it learned.
    """)
    return


@app.cell
def _(ex, lj, mo, time, train_text):
    _snippet = train_text[:100_000]
    _t0 = time.perf_counter()
    bpe_try = lj.attempt(ex.train_bpe, _snippet, 256 + 400)
    _secs = time.perf_counter() - _t0
    my_merges = bpe_try.value if bpe_try.ok else None
    if bpe_try.ok:
        _vocab = {i: bytes([i]) for i in range(256)}
        for _i, (_a, _b) in enumerate(my_merges):
            _vocab[256 + _i] = _vocab[_a] + _vocab[_b]
        _shown = ", ".join(f"`{_vocab[256 + i].decode('utf-8', 'replace')!r}`" for i in range(min(30, len(my_merges))))
        _out = mo.md(f"""
        Learned **{len(my_merges)} merges** in {_secs:.1f} s (pure Python — fine for learning, far too slow
        for gigabytes; that's what the Rust library in Section 6 is for).

        First 30 tokens learned, most frequent first: {_shown}

        Word endings glued to a space (`'e '`, `'t '`, `'s '`) and common pairs like `'th'` come first; whole
        words such as `'the '` and `'you'` follow within a few dozen merges.
        """)
    else:
        _out = bpe_try.view(mo)
    _out
    return (my_merges,)


@app.cell
def _(ex, lj, mo, my_merges, show_tokens, val_text):
    _items = []
    if my_merges is not None:
        _sample = val_text[200:420]
        _enc = lj.attempt(ex.encode, _sample, my_merges)
        if _enc.ok:
            _dec = lj.attempt(ex.decode, _enc.value, my_merges)
            _pieces = []
            _vocab = {i: bytes([i]) for i in range(256)}
            for _i, (_a, _b) in enumerate(my_merges):
                _vocab[256 + _i] = _vocab[_a] + _vocab[_b]
            _pieces = [_vocab[t].decode("utf-8", "replace") for t in _enc.value]
            _ratio = len(_sample.encode("utf-8")) / len(_enc.value)
            _round = _dec.ok and _dec.value == _sample
            _items += [
                mo.md(f"**Held-out text, tokenized by your BPE:** {len(_enc.value)} tokens for "
                      f"{len(_sample.encode('utf-8'))} bytes — **{_ratio:.2f} bytes/token**. "
                      f"Round trip `decode(encode(x)) == x`: **{_round}**"),
                show_tokens(_pieces),
            ]
        else:
            _items.append(_enc.view(mo))
    else:
        _items.append(mo.md("*Train your BPE above first.*"))
    mo.vstack(_items)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · The library way: Hugging Face `tokenizers`

    Same algorithm, written in Rust, with two practical additions:

    * **Pre-tokenization:** text is first split into word-ish chunks (on spaces and punctuation), and merges
      never cross chunk boundaries. That's why GPT-2 tokens look like `Ġthe` — `Ġ` is how its byte-level
      alphabet *displays* a leading space byte — and why " the" and "the" are different tokens.
    * **Speed:** it trains on the full 1.1 MB of Shakespeare in about a second.

    Below we train byte-level BPE tokenizers of several vocabulary sizes and measure compression on the
    held-out text.
    """)
    return


@app.cell
def _(Tokenizer, decoders, models, pre_tokenizers, time, train_text, trainers, val_text):
    def train_hf_bpe(vocab_size, lines):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|doc|>"],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      show_progress=False)
        tok.train_from_iterator(lines, trainer)
        return tok

    _lines = train_text.splitlines(keepends=True)
    _val_bytes = len(val_text.encode("utf-8"))
    hf_sizes = [300, 500, 1000, 2000, 4000, 8192]
    hf_results = []
    for _v in hf_sizes:
        _t0 = time.perf_counter()
        _tok = train_hf_bpe(_v, _lines)
        _secs = time.perf_counter() - _t0
        _n = len(_tok.encode(val_text).ids)
        hf_results.append({"vocab": _v, "tokens": _n, "bytes_per_token": _val_bytes / _n, "secs": _secs})
    return hf_results, hf_sizes


@app.cell
def _(gpt2, hf_results, mo, plt, val_text):
    _val_bytes = len(val_text.encode("utf-8"))
    _gpt2_bpt = _val_bytes / len(gpt2.encode(val_text).ids)
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _ax.plot([r["vocab"] for r in hf_results], [r["bytes_per_token"] for r in hf_results], "o-",
             label="byte-level BPE trained on Shakespeare")
    _ax.axhline(_gpt2_bpt, color="C1", ls="--", label=f"GPT-2 (50,257 vocab, trained on web text): {_gpt2_bpt:.2f}")
    _ax.axhline(1.0, color="gray", lw=0.8, label="raw bytes")
    _ax.set(xscale="log", xlabel="vocabulary size", ylabel="bytes per token (held-out)",
            title="Bigger vocabulary → fewer, longer tokens")
    _ax.set_xticks([r["vocab"] for r in hf_results], [f"{r['vocab']:,}" for r in hf_results])
    _ax.minorticks_off()
    _ax.legend(fontsize=8, loc="lower right")
    _rows = "\n".join(f"| {r['vocab']:,} | {r['tokens']:,} | {r['bytes_per_token']:.2f} | {r['secs']:.1f} s |"
                      for r in hf_results)
    mo.vstack([_fig, mo.md(f"""
| vocabulary | held-out tokens | bytes/token | training time |
|---|---|---|---|
{_rows}

Compression improves with vocabulary size, with diminishing returns. At {hf_results[-1]['vocab']:,} tokens, a
tokenizer trained on *this* text gets {hf_results[-1]['bytes_per_token']:.2f} bytes/token — better than GPT-2's
{_gpt2_bpt:.2f} with a vocabulary six times smaller, because GPT-2 spent its vocabulary on web text.
Tokenizers are specialised to their training data, which is why domain-specific models often train their own.
""")])
    return


@app.cell
def _(Tokenizer, detective_corpus, gpt2, mo, show_tokens):
    detective_tok = Tokenizer.from_file(str(detective_corpus()["tokenizer_path"]))
    _s = "The detective lit his pipe. \"It was the butler,\" said Holmes, in 1891."
    _g = gpt2.encode(_s)
    _d = detective_tok.encode(_s)
    _clean = lambda toks: [t.replace("Ġ", " ").replace("Ċ", "\n") for t in toks]  # noqa: E731
    mo.vstack([
        mo.md(f"""
    ## 7 · Real tokenizers side by side

    The same sentence through GPT-2's tokenizer ({gpt2.get_vocab_size():,} tokens) and the deepseek-v3-jax
    tokenizer you trained with ({detective_tok.get_vocab_size():,} tokens, learned on the detective novels):
    """),
        mo.md(f"**GPT-2 — {len(_g.ids)} tokens**"), show_tokens(_clean(_g.tokens)),
        mo.md(f"**deepseek-v3-jax detective BPE — {len(_d.ids)} tokens**"), show_tokens(_clean(_d.tokens)),
        mo.md("`␣` marks a leading space (GPT-2's byte alphabet stores it as `Ġ`). Both learned that ` the`, "
              "` detective` and ` said` deserve a token. GPT-2 needs two pieces for ` butler`; the detective "
              "tokenizer, trained on novels full of butlers, has it as one — but spends three on `1891`. "
              "Each tokenizer is shaped by the text it was trained on. Across a whole corpus, the smaller "
              "vocabulary splits rare words more often; that is the price of a smaller embedding table."),
    ])
    return


@app.cell
def _(ex, lj, mo):
    _gpt2_emb = lj.attempt(ex.embedding_params, 50_257, 256)
    _ds_emb = lj.attempt(ex.embedding_params, 8192, 384)
    _ll3_emb = lj.attempt(ex.embedding_params, 128_256, 4096)
    if _gpt2_emb.ok and _ds_emb.ok and _ll3_emb.ok:
        # Llama-style layer for the blog config: dim 256, 8 heads, 4 KV heads (head_dim 32), FFN hidden 768
        _attn = 256 * 256 + 2 * 256 * (4 * 32) + 256 * 256
        _ffn = 3 * 256 * 768
        _layers = 6 * (_attn + _ffn + 2 * 256)
        _blog_total = _gpt2_emb.value + _layers + 256
        _out = mo.md(f"""
    ## 8 · Vocabulary size costs parameters

    Every vocabulary entry needs a row in the input embedding table and a column in the output head
    (unless the two are *tied* — shared). Your `embedding_params` makes the arithmetic concrete:

    | model | vocab × dim | embedding + head params | share of the model |
    |---|---|---|---|
    | "LLaMA 3 in pure JAX" blog model | 50,257 × 256 | **{_gpt2_emb.value / 1e6:.1f}M** | ~{_gpt2_emb.value / _blog_total:.0%} of ~{_blog_total / 1e6:.0f}M |
    | your deepseek-v3-jax model | 8,192 × 384 | **{_ds_emb.value / 1e6:.1f}M** | {_ds_emb.value / 36.2e6:.0%} of 36.2M |
    | Llama 3 8B | 128,256 × 4,096 | **{_ll3_emb.value / 1e9:.2f}B** | {_ll3_emb.value / 8.03e9:.0%} of 8.0B |

    **A cautionary tale.** A popular blog post builds "LLaMA 3 in pure JAX" with the GPT-2 tokenizer
    (50,257 tokens), `dim=256`, 6 layers, 8 query heads and 4 key/value heads, and calls it "~2 million
    parameters". The 6 transformer layers really are small — about {_layers / 1e6:.1f}M (attention
    {_attn:,} + feed-forward {_ffn:,} per layer, with Llama's 768-wide feed-forward). But the vocabulary adds
    {_gpt2_emb.value / 1e6:.1f}M, so the model is **~{_blog_total / 1e6:.0f}M parameters**, and over 80% of
    them only look tokens up. At small scale, *the vocabulary is the model*.

    That is why your deepseek-v3-jax run trained its own 8,192-token tokenizer: a 36M model can't afford
    a 50K vocabulary. Llama 3 went the other way — 128K tokens, because at 8B parameters the table is a
    modest share and better compression means more text per context window and per FLOP. (Llama 3 also uses
    its own tokenizer — not GPT-2's — and a RoPE base of 500,000 rather than 10,000; Module 17.)
    """)
    else:
        _out = mo.vstack([mo.md("## 8 · Vocabulary size costs parameters"),
                          _gpt2_emb.view(mo)])
    _out
    return


@app.cell
def _(gpt2, html, mo):
    _rows = []
    for _s in [" the", "the", "The", " THE", "12345", " 2026", " SolidGoldMagikarp", "SolidGoldMagikarp",
               "<|endoftext|>"]:
        _e = gpt2.encode(_s)
        _esc = lambda x: html.escape(x).replace("|", "&#124;")  # noqa: E731  (a raw | would split the table cell)
        _toks = " · ".join(f"<code>{_esc(t)}</code>" for t in _e.tokens)
        _rows.append(f"| <code>{_esc(repr(_s))}</code> | {len(_e.ids)} | {_toks} | {_e.ids} |")
    _table = "\n    ".join(_rows)
    mo.md(f"""
    ## 9 · Quirks that explain odd model behaviour

    | input | tokens | pieces | ids |
    |---|---|---|---|
{_table}

    * **Leading spaces and case matter.** `" the"`, `"the"` and `"The"` are three unrelated ids; the model
      has to learn they mean the same word.
    * **Numbers are chopped arbitrarily** (`123` + `45`), one reason LLMs are shaky at arithmetic. Newer
      tokenizers split digits one at a time on purpose.
    * **Glitch tokens.** `" SolidGoldMagikarp"` (a Reddit username) got its own GPT-2 token because it was
      frequent in the *tokenizer's* training data — but almost absent from the *model's*. Its embedding was
      barely trained, and early GPT models produced bizarre output when asked to repeat it.
    * **Special tokens** like <code>&lt;|endoftext|&gt;</code> (GPT-2, id 50256) or <code>&lt;|doc|&gt;</code>
      (id 0 in the deepseek-v3-jax tokenizer) are added by hand to mark document boundaries; the model learns that what follows is
      unrelated to what came before.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "It has only 256 base symbols and can always fall back to single bytes": "a",
            "Its vocabulary includes every word in every language": "b",
            "It replaces unknown words with a special unknown-word token": "c",
        },
        label="**Check yourself:** why can a byte-level BPE tokenizer encode *any* string?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _feedback = {
        "a": mo.callout(mo.md("Right. Every string is UTF-8 bytes; merges are shortcuts on top. A word the "
                              "tokenizer never saw is just spelled with smaller pieces."), kind="success"),
        "b": mo.callout(mo.md("No — the vocabulary is a few thousand to ~128K entries. Rare words are built "
                              "from several pieces instead."), kind="warn"),
        "c": mo.callout(mo.md("That's what *word*-level tokenizers do (and the problem byte-level BPE "
                              "removes): there is never an unknown token."), kind="warn"),
    }
    _feedback.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(13))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Andrej Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (video)
      and [minbpe](https://github.com/karpathy/minbpe) — the same exercise, with the GPT-4 regex splitter.
    * Sennrich, Haddow & Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
      (2016) — BPE for neural networks.
    * Hugging Face, [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)
      and the [`tokenizers` docs](https://huggingface.co/docs/tokenizers).
    * Rumbelow & Watkins, [SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)
      — the glitch-token story.

    **Next:** Module 14 — language modeling basics: predicting the next token, from bigram counts to a
    small neural network.
    """)
    return


if __name__ == "__main__":
    app.run()
