import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="32 · BERT vs GPT")


@app.cell
def _():
    import marimo as mo
    import importlib.util
    import sys
    import time

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from tokenizers import Tokenizer

    import learnjax as lj
    from learnjax.data import detective_corpus

    jax.config.update("jax_compilation_cache_dir", "/cache/jaxcache")
    ex = lj.load_exercises(mo.notebook_dir())
    ref = lj.load_exercises(mo.notebook_dir(), solutions=True)
    _spec = importlib.util.spec_from_file_location("m32_bertgpt", mo.notebook_dir() / "bertgpt.py")
    B = importlib.util.module_from_spec(_spec)
    sys.modules["m32_bertgpt"] = B
    _spec.loader.exec_module(B)
    corpus = detective_corpus()
    tok = Tokenizer.from_file(str(corpus["tokenizer_path"]))
    return B, corpus, ex, importlib, jax, jnp, lj, mo, np, plt, ref, sys, time, tok


@app.cell
def _(mo):
    mo.md(r"""
    # 32 · BERT vs GPT: encoders and decoders (2018)

    **Part 10, Milestones rebuilt in JAX.** Take Modules 15 and 16 first (and Module 30 for word vectors).

    **Goal:** see why one year, 2018, produced two very different answers to "how should a Transformer
    read text?", by training both on the same detective novels on your RX 7600 XT and putting them through
    the same tests.

    **You will:** build BERT's bidirectional mask next to GPT's causal one · implement BERT's 80/10/10
    masking recipe and its masked-LM loss · pretrain a small BERT and a same-size GPT on the same tokens ·
    race them at fill-in-the-blank (BERT reads both sides) and at writing (GPT's home ground) · fine-tune
    both to tell four detective novels apart · watch one word, *safe*, get different vectors in different
    sentences, which word2vec (Module 30) can't do.
    """)
    return


@app.cell
def _(mo):
    # history box
    mo.callout(mo.md(r"""
    **Where this came from.** By 2017 every NLP system started from word2vec- or GloVe-style vectors: one
    vector per word, whatever its sentence. In early 2018 Matthew Peters and colleagues at the Allen
    Institute showed with ELMo that vectors taken from a language model *in context* did much better. In
    June 2018 Alec Radford and colleagues at OpenAI pretrained a Transformer decoder left to right (GPT)
    and fine-tuned it for each task. In October 2018 Jacob Devlin and colleagues at Google published BERT,
    a Transformer *encoder* trained to fill in masked words using both sides, and it topped nearly every
    benchmark at once. For the next few years "BERTology" dominated NLP; then scaled-up decoders (GPT-2,
    GPT-3) turned out to be the ones that could write. Timeline: Module 00b.
    """), kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · One vector per word, then one vector per word *in context*

    Module 30's word2vec gives *safe* one vector, whether the sentence is about a strongbox or about being
    out of danger. The 2018 idea was to read the whole sentence with a deep network first and take the
    word's vector *after* that. Three designs appeared within months:

    | Model (2018) | Who | Reads | Trained to |
    |---|---|---|---|
    | ELMo | Peters et al., Allen Institute | two LSTMs (Module 31), one each way | predict the next and previous word |
    | GPT | Radford et al., OpenAI | Transformer **decoder**: left to right | predict the next token |
    | BERT | Devlin et al., Google | Transformer **encoder**: both directions at once | fill in masked tokens |

    BERT's trick is what made "both directions at once" possible. If every position can see every other
    position, predicting the next token is trivial: the answer is visible. So BERT hides about 15% of the
    tokens and asks the model to recover them (a *masked language model*, MLM). It also trained on
    "next-sentence prediction"; RoBERTa (Liu et al., 2019) showed that part wasn't needed and that BERT had
    simply been undertrained. Google's T5 (2019) combined an encoder and a decoder, the original 2017
    Transformer layout.

    Our two models below share **everything**: Module 16's block, the same width and depth, the same
    tokenizer and the same novels. Only the attention mask and the training objective differ.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Two masks, one stack

    In Module 15 you masked attention so position $t$ only sees positions $\le t$. BERT simply removes the
    mask. Implement both in `exercises.py`: `causal_mask(T)` and `bidirectional_mask(T)`.
    """)
    return


@app.cell
def _(ex, lj, mo, np, plt):
    _c = lj.attempt(ex.causal_mask, 8)
    _b = lj.attempt(ex.bidirectional_mask, 8)
    if _c.ok and _b.ok:
        _fig, _ax = plt.subplots(1, 2, figsize=(7, 3.2))
        for _a, _m, _t in [(_ax[0], _c.value, "GPT: causal (decoder)"), (_ax[1], _b.value, "BERT: bidirectional (encoder)")]:
            _a.imshow(np.asarray(_m), cmap="Blues", vmin=0, vmax=1.4)
            _a.set_title(_t, fontsize=10)
            _a.set_xlabel("key position (what it may look at)")
            _a.set_ylabel("query position")
        _fig.tight_layout()
        _out = _fig
    else:
        _out = mo.vstack([_c.view(mo), _b.view(mo)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · BERT's masking recipe

    Pick each token with probability 15%. Those are the prediction targets. Of the picked tokens, replace
    80% with a special `[MASK]` token, 10% with a random token, and leave 10% unchanged. Why not always
    `[MASK]`? Because `[MASK]` never appears when the model is later fine-tuned or used, so the model must
    also learn to produce good vectors for ordinary tokens it can't trust. Implement `mask_tokens` and
    `mlm_loss` (cross-entropy averaged over the target positions only).

    Our vocabulary is the detective corpus's 8,192 BPE tokens plus two new ones: `[MASK]` = 8192 and
    `[CLS]` = 8193.
    """)
    return


@app.cell
def _(B, ex, jax, jnp, lj, mo, np, tok):
    _ids = jnp.asarray(tok.encode("The inspector found a revolver hidden under the floor of the study, "
                                  "and the butler swore he had never seen it before.").ids, jnp.int32)
    _m = lj.attempt(ex.mask_tokens, jax.random.key(4), _ids[None], B.MASK_ID, B.TEXT_VOCAB, 0.3)
    _l = lj.attempt(ex.mlm_loss, jnp.log(jnp.array([[[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]])),
                    jnp.array([[0, 2]]), jnp.array([[True, False]]))

    def _show(ids, targets):
        _parts = []
        for _i, _t in zip(np.asarray(ids)[0], np.asarray(targets)[0]):
            _s = " [MASK]" if _i == B.MASK_ID else tok.decode([int(_i)])
            _lead = " " if _s.startswith(" ") else ""
            _parts.append(f"{_lead}**{_s.strip()}**" if _t else _s)
        return "".join(_parts)

    _items = [mo.md("**Your `mask_tokens` at a 30% rate** (bold = a prediction target):")]
    if _m.ok:
        _inp, _tgt = _m.value
        _items.append(mo.md(_show(_inp, _tgt)))
        _ok = np.asarray((_inp == _ids[None]) | _tgt).all()
        _items.append(mo.md(f"Untargeted tokens unchanged: **{bool(_ok)}**."))
    else:
        _items.append(_m.view(mo))
    _items.append(mo.md("**Your `mlm_loss`** on a toy example (should be −ln 0.7 ≈ 0.357, only position 0 counts):"))
    _items.append(mo.md(f"`{float(_l.value):.4f}`") if _l.ok else _l.view(mo))
    mo.vstack(_items)
    return


@app.cell
def _(B, ex, jax, jnp, lj, ref):
    # Use your masking + loss for pretraining once they agree with the reference; otherwise the reference.
    _ids = jnp.arange(4096, dtype=jnp.int32).reshape(8, 512) % 8000
    _a = lj.attempt(ex.mask_tokens, jax.random.key(0), _ids, B.MASK_ID, B.TEXT_VOCAB, 0.15)
    _b = lj.attempt(ex.mlm_loss, jnp.zeros((2, 3, 5)), jnp.zeros((2, 3), jnp.int32), jnp.ones((2, 3), bool))
    _good = (_a.ok and _b.ok and 0.12 < float(_a.value[1].mean()) < 0.18
             and abs(float(_b.value) - float(jnp.log(5.0))) < 1e-4)
    mask_fn, loss_fn = (ex.mask_tokens, ex.mlm_loss) if _good else (ref.mask_tokens, ref.mlm_loss)
    using_yours = _good
    return loss_fn, mask_fn, using_yours


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Pretrain both, on the same novels

    Two models with the same shape: 4 layers, width 256, 4 heads, 128-token windows, about 5.3M
    parameters (most of them the 8,194 × 256 embedding table). Each trains for the same number of steps on
    the same random windows of the detective corpus, with Module 16's recipe: AdamW with warmup and cosine
    decay, gradient clipping, bf16 compute on f32 weights, and each 100-step chunk run as one [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan).
    """)
    return


@app.cell
def _(B, corpus, importlib, jax, loss_fn, mask_fn, mo, np, sys, using_yours):
    _spec = importlib.util.spec_from_file_location("m32_train", mo.notebook_dir() / "train.py")
    _train = importlib.util.module_from_spec(_spec)
    sys.modules["m32_train"] = _train
    _spec.loader.exec_module(_train)
    _saved = _train.load_all()
    cfg = B.Cfg()
    if _saved is not None:
        bert, gpt, _m = _saved
        STEPS = _m["steps"]
        bert_losses = np.repeat(np.asarray(_m["bert"]["losses"]), 10)
        gpt_losses = np.repeat(np.asarray(_m["gpt"]["losses"]), 10)
        bert_s, gpt_s = _m["bert"]["run_seconds"], _m["gpt"]["run_seconds"]
        _final = (_m["bert"]["final_loss"], _m["gpt"]["final_loss"])
        _src = (f"Loaded the **{STEPS:,}-step** models from `runs/32-bert-vs-gpt/` (made by `train.py`, which uses "
                "the reference masking and loss).")
    else:
        STEPS = 1500
        bert, bert_losses, bert_s = B.pretrain("bert", cfg, corpus["train"], jax.random.key(0), STEPS, mask_fn, loss_fn)
        gpt, gpt_losses, gpt_s = B.pretrain("gpt", cfg, corpus["train"], jax.random.key(0), STEPS, mask_fn, loss_fn)
        _final = (np.mean(bert_losses[-100:]), np.mean(gpt_losses[-100:]))
        _src = (f"Trained live for **{STEPS:,} steps** each with **{'your' if using_yours else 'the reference'}** "
                "`mask_tokens` / `mlm_loss`. For better-trained models (BERT especially needs more steps, since "
                "it learns from only 15% of positions), run `containers/run.sh python modules/32-bert-vs-gpt/train.py` "
                "once; this lesson will then load them.")
    _tok_s = STEPS * 64 * 128
    mo.md(f"""
    {_src}

    | | BERT (masked LM) | GPT (next token) |
    |---|---|---|
    | final loss (mean of last 100 steps) | {_final[0]:.3f} | {_final[1]:.3f} |
    | training time (after compile) | {bert_s:.0f} s | {gpt_s:.0f} s |
    | tokens seen | {_tok_s / 1e6:.1f}M | {_tok_s / 1e6:.1f}M |
    | tokens that produced a loss | about 15% ({_tok_s * 0.15 / 1e6:.1f}M) | all ({_tok_s / 1e6:.1f}M) |
    """)
    return bert, bert_losses, cfg, gpt, gpt_losses


@app.cell
def _(bert_losses, gpt_losses, np, plt):
    def _smooth(x, k=25):
        return np.convolve(x, np.ones(k) / k, mode="valid")

    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_smooth(bert_losses), label="BERT: masked-token loss")
    _ax.plot(_smooth(gpt_losses), label="GPT: next-token loss")
    _ax.set(xlabel="step", ylabel="cross-entropy (nats)", title="Same model size, same data, different jobs")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The two curves aren't directly comparable: BERT's job is easier per token (it sees the words on *both*
    sides of each gap) but it only learns from 15% of positions. One honest comparison is a test both can
    take: hide one word in the middle of a passage and ask each model to guess it.

    Look at the start of BERT's curve. It drops fast, then sits nearly flat around 6 nats for hundreds of
    steps before falling again. Guessing every hidden token from word frequencies alone scores about
    6.7 nats on this corpus, so on the plateau BERT is barely using context: it predicts "something common
    goes here". The second drop comes when attention learns to look at the neighbouring words. GPT has no
    such plateau, because "the next token" is strongly predicted by the token right before it, a signal
    available from step one. Sudden drops like this are a recurring theme; Module 37 catches one in the act
    (induction heads).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Fill in the blank

    For 512 held-out passages, hide the token at position 64. **BERT** gets the passage with `[MASK]` there
    and both sides visible. **GPT** can only use the 63 tokens to the left, which is all a decoder ever
    sees. Same parameters, same training tokens.
    """)
    return


@app.cell
def _(B, bert, corpus, gpt, jnp, mo, np, tok):
    fill_bert, fill_gpt = B.fill_accuracy(bert, gpt, corpus["val"])
    _s = tok.encode("The detective opened the safe and found the stolen jewels inside the velvet case.").ids
    _rows = []
    for _word in [" safe", " jewels", " velvet"]:
        _p = _s.index(tok.encode(_word).ids[0])
        _x = jnp.asarray([_s + [0] * (128 - len(_s))], jnp.int32)
        _tops = {}
        for _n, _f, _prm in [("BERT", B.bert_fill, bert), ("GPT", B.gpt_fill, gpt)]:
            _lg = np.asarray(_f(_prm, _x, jnp.asarray([_p]))[0, :B.TEXT_VOCAB])
            _tops[_n] = ", ".join(repr(tok.decode([int(t)]).strip()) for t in np.argsort(-_lg)[:5])
        _rows.append(f"| `{_word.strip()}` | {_tops['BERT']} | {_tops['GPT']} |")
    _table = "\n    ".join(_rows)
    if fill_bert > fill_gpt + 0.01:
        _verdict = ("Seeing the right-hand side (\"…and found the stolen jewels inside\") is exactly what lets an "
                    "encoder pin down a word. That made BERT the tool of choice for *understanding* tasks: "
                    "classification, search, question answering by pointing at a span.")
    else:
        _verdict = ("BERT *should* win this test, since it sees both sides of the gap, but here it doesn't "
                    "(yet). Masked LM learns from only about 15% of positions per step, so at equal steps it "
                    "is far less trained than GPT, which learns from every position. This is the cost RoBERTa "
                    "and later work attacked with longer training (and ELECTRA with a different objective). "
                    "Run `train.py` for longer-trained models and see whether the order changes.")
    mo.md(f"""
    **Top-1 accuracy on 512 hidden tokens:** BERT **{fill_bert:.1%}**, GPT **{fill_gpt:.1%}**.

    One sentence, three blanks: *"The detective opened the ___ and found the stolen ___ inside the ___ case."*
    Each model's top 5 guesses:

    | hidden word | BERT (both sides) | GPT (left only) |
    |---|---|---|
    {_table}

    {_verdict}
    """)
    return fill_bert, fill_gpt


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Now ask both to write

    GPT was trained for exactly this: predict the next token, append it, repeat. BERT was never trained
    with nothing to its right, so the naive trick (append a `[MASK]`, fill it, repeat) asks it something
    it has never seen.
    """)
    return


@app.cell
def _(B, bert, gpt, jax, mo, tok):
    _prompt = tok.encode("The inspector looked at the body and said").ids
    _g = tok.decode(B.gpt_generate(gpt, _prompt, 40, jax.random.key(3)))
    _b = tok.decode(B.bert_generate(bert, _prompt, 40))
    mo.md(f"""
    Prompt: *"The inspector looked at the body and said"*

    **GPT** (temperature 0.8, top-40): {_g!r}

    **BERT** (append `[MASK]`, fill greedily, repeat): {_b!r}

    Neither is Dashiell Hammett after about a minute of training, but GPT's continuation is shaped like
    prose and BERT's usually isn't. That difference, scaled up, is the story of 2019–2020: GPT-2 and GPT-3
    showed that a big enough decoder could do many tasks just by continuing a prompt, with no fine-tuning
    at all, and generation became the interface.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Fine-tuning: which novel is this passage from?

    BERT's 2018 recipe was *pretrain once, fine-tune for each task*: add a small head, train everything
    briefly on labelled data. Our task: given a 63-token passage, say which of four novels it came from.
    Training passages come from the first 80% of each book and test passages from the last 20%.

    How does each model turn a passage into one vector? BERT puts a `[CLS]` token in front and uses its
    output (`cls_pool`); a GPT uses the **last** token, the only one that has seen the whole passage
    (`last_token_pool`). Implement both.
    """)
    return


@app.cell
def _(B, corpus, ex, jnp, lj, mo, np):
    Xtr, ytr, Xte, yte = B.book_passages(corpus["train"])
    _h = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)
    _c = lj.attempt(ex.cls_pool, _h)
    _l = lj.attempt(ex.last_token_pool, _h, jnp.array([2, 4]))
    _books = "\n    ".join(f"| {i} | {name} | {(ytr == i).sum()} | {(yte == i).sum()} |" for i, name in enumerate(B.BOOKS.values()))
    mo.vstack([
        mo.md(f"""
    | label | novel | train passages | test passages |
    |---|---|---|---|
    {_books}
    """),
        mo.md("**Your `cls_pool`** (should be rows 0 and 12 of the toy tensor):"),
        _c.view(mo) if not _c.ok else mo.md(f"`{np.asarray(_c.value).tolist()}`"),
        mo.md("**Your `last_token_pool` with lengths [2, 4]** (rows 3 and 21):"),
        _l.view(mo) if not _l.ok else mo.md(f"`{np.asarray(_l.value).tolist()}`"),
    ])
    return Xte, Xtr, yte, ytr


@app.cell
def _(B, Xte, Xtr, bert, ex, gpt, jax, lj, mo, ref, yte, ytr):
    _yours = lj.attempt(ex.cls_pool, jax.numpy.zeros((1, 2, 2))).ok and lj.attempt(ex.last_token_pool, jax.numpy.zeros((1, 2, 2))).ok
    _cls, _last = (ex.cls_pool, ex.last_token_pool) if _yours else (ref.cls_pool, ref.last_token_pool)
    acc_bert = B.finetune_classifier("bert", bert, Xtr, ytr, Xte, yte, _cls, jax.random.key(1))
    acc_gpt = B.finetune_classifier("gpt", gpt, Xtr, ytr, Xte, yte, _last, jax.random.key(1))
    _winner = ("BERT" if acc_bert > acc_gpt + 0.02 else "GPT" if acc_gpt > acc_bert + 0.02 else "neither, clearly")
    mo.md(f"""
    After 200 fine-tuning steps (whole model plus a linear head; chance is 25%):

    | | BERT + `[CLS]` | GPT + last token |
    |---|---|---|
    | test accuracy | **{acc_bert:.1%}** | **{acc_gpt:.1%}** |

    Winner at this tiny scale: **{_winner}**. Both models learned a lot about the novels' style in a
    minute of pretraining. In 2018–2020 encoders won tasks like this on the benchmarks (GLUE, SQuAD);
    today large decoders usually match them, but small encoders are still the cheap, fast choice for
    classification and search.
    """)
    return acc_bert, acc_gpt


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · One word, many vectors

    Take the word *safe* in six sentences: three where it's a strongbox, three where it means out of
    danger. Word2vec (Module 30) gives all six the **same** vector. Our models give each occurrence its own
    vector, computed from the sentence around it. Implement `contextual_similarity` (cosine similarity
    between rows) and compare.
    """)
    return


@app.cell
def _(B, bert, ex, gpt, lj, mo, np, plt, tok):
    _sents = ["The jewels were locked in the safe behind the portrait.",
              "He forced the door of the safe with a chisel.",
              "The combination of the safe was written on a card.",
              "You will be safe here until morning.",
              "It is not safe to cross the moor at night.",
              "She was safe at last, far from the house."]
    _wid = tok.encode(" safe").ids[0]
    _vecs = {k: B.word_vectors(p, tok, _sents, _wid, k) for k, p in [("bert", bert), ("gpt", gpt)]}
    _a = lj.attempt(ex.contextual_similarity, _vecs["bert"])
    if _a.ok:
        _fig, _axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
        _stats = []
        for _ax, (_k, _v) in zip(_axes, _vecs.items()):
            _s = np.asarray(ex.contextual_similarity(_v))
            _within = (np.r_[_s[:3, :3][~np.eye(3, dtype=bool)], _s[3:, 3:][~np.eye(3, dtype=bool)]]).mean()
            _cross = _s[:3, 3:].mean()
            _stats.append((_k.upper(), _within, _cross))
            _ax.imshow(_s, vmin=-0.2, vmax=1, cmap="viridis")
            _ax.set_title(f"{_k.upper()}: 'safe' in 6 sentences", fontsize=10)
            _ax.set_xticks(range(6), ["box", "box", "box", "ok", "ok", "ok"])
            _ax.set_yticks(range(6), ["box", "box", "box", "ok", "ok", "ok"])
        _fig.tight_layout()
        _rows = "\n    ".join(f"| {k} | {w:.3f} | {c:.3f} |" for k, w, c in _stats)
        _out = mo.vstack([_fig, mo.md(f"""
    | model | same meaning (mean cosine) | different meaning (mean cosine) |
    |---|---|---|
    {_rows}
    | word2vec | 1.000 | 1.000 |

    A gap between the two columns means the model's vector for *safe* carries the sentence's meaning.
    GPT's vector at *safe* only reflects the words before it ("the safe" vs "be safe"), while BERT's also
    sees what follows ("…behind the portrait", "…until morning").
    """)])
    else:
        _out = _a.view(mo)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Why decoders won, and where encoders live on

    * **Generation is the universal interface.** A decoder can do classification ("Answer: A or B"),
      extraction and translation by writing the answer. GPT-3 (2020) showed this works without fine-tuning
      once the model is big enough, and every chat model since is a decoder.
    * **Scaling favoured the decoder objective.** Every token is a training signal (vs about 15% for MLM),
      and the same model is used for training and generating.
    * **Encoders didn't disappear.** Sentence-embedding models used for search and retrieval-augmented
      generation (RAG) are mostly encoders. ModernBERT (Warner et al., 2024) brought the architecture up to
      date. Module 27's LM Studio embeddings endpoint serves exactly this kind of model.
    * **Encoder-decoders** (the original 2017 Transformer, T5) survive in translation and speech models.

    You've now built both halves. Module 33 turns to the other big generative story of the same decade:
    images.
    """)
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options={
            "Its attention lets every token see every other token, so 'predict the next token' would be trivial": "a",
            "Masking makes training faster on GPUs": "b",
            "Next-token prediction doesn't work with Transformers": "c",
        },
        label="**Check yourself:** why does BERT need a *masked* objective instead of next-token prediction?",
    )
    quiz
    return (quiz,)


@app.cell
def _(mo, quiz):
    _fb = {
        "a": mo.callout(mo.md("Right. With a bidirectional mask the answer to 'what comes next?' is "
                              "visible in the input, so BERT hides tokens instead."), kind="success"),
        "b": mo.callout(mo.md("No: MLM actually learns from fewer positions per step (about 15%). The "
                              "reason is the bidirectional attention."), kind="warn"),
        "c": mo.callout(mo.md("It works fine: that's GPT, trained in section 4. The issue is specific to "
                              "a model that can see both sides."), kind="warn"),
    }
    _fb.get(quiz.value, mo.md(""))
    return


@app.cell
def _(mo):
    import learnjax.papers as _papers

    mo.md(_papers.papers_md(32))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further reading
    * Jay Alammar, [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) — the
      2018 transfer-learning story in pictures.
    * Rogers, Kovaleva & Rumshisky, [A Primer in BERTology](https://arxiv.org/abs/2002.12327) (2020) — what
      was learned about what BERT learns.
    * Raffel et al., [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
      (2019) — T5, the encoder-decoder answer.
    * Warner et al., [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663)
      (2024) — ModernBERT.

    **Next:** Module 33 — GANs, VAEs, diffusion and CLIP.
    """)
    return


if __name__ == "__main__":
    app.run()
