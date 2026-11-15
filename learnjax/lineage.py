"""The family tree: foundation models, the building blocks each one uses, and who introduced each block.

Two registries, one dataset:

* `BLOCKS` — the parts a model is assembled from, grouped by *slot* (position, normalization, attention, …).
  Each cites the paper that introduced it (a URL registered in `learnjax/papers.py`, or in `SOURCES` below)
  and the module where this course builds or discusses it.
* `MODELS` — about fifty models: year, organization, family, the models they build on (`parents`), and the
  blocks they use. "What did a model add?" is *derived*: the blocks it uses that none of its ancestors do
  (module 00b's `new_blocks` exercise).

Rules (checked by `check()`, which the tests and `tools/check_lineage.py` run):
parents are older than (or as old as) their children; every block is used by at least one model; a block's
`first_model`, when set, uses it, and no model in the registry uses it earlier; every paper/source URL is
registered. `tools/check_lineage.py` also cross-checks open models' Hugging Face `config.json` files
(GQA, MLA, MoE, shared experts, partial RoPE, sliding windows, multi-token prediction, NoPE layers).

`tools/gen_lineage.py` renders docs/LINEAGE.md. Module 00b draws the graph and the taxonomy.
"""
from __future__ import annotations

from typing import NamedTuple

from learnjax.papers import PAPERS


def _ax(i: str) -> str:
    return f"https://arxiv.org/abs/{i}"


class Block(NamedTuple):
    id: str
    name: str
    slot: str
    what: str              # one plain-language line
    paper: str             # URL of the introducing paper (registered in papers.py or SOURCES)
    first_model: str | None  # the registry model whose own paper introduced it, if any
    module: str | None     # display label of the module that builds or discusses it


class Model(NamedTuple):
    id: str
    name: str
    year: int
    org: str
    family: str
    parents: tuple[str, ...]
    uses: tuple[str, ...]
    note: str              # one line: why it matters in the lineage
    source: str            # paper or official post (registered in papers.py or SOURCES)
    module: str | None     # display label of the module that uses or rebuilds it
    hf: str | None = None  # an ungated Hugging Face repo whose config.json tools/check_lineage.py checks


SLOTS = ["input", "position", "normalization", "sequence mixing", "feed-forward", "connectivity",
         "objective", "post-training"]

FAMILIES = ["RNN & embeddings", "CNN", "RL & games", "encoder", "encoder-decoder", "decoder", "MoE decoder",
            "SSM & hybrid", "vision-language & diffusion"]

# Sources not in papers.py (model reports, official posts). check_lineage verifies arXiv titles.
SOURCES: dict[str, str] = {
    _ax("2204.02311"): "PaLM: Scaling Language Modeling with Pathways",
    _ax("2304.01373"): "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling",
    _ax("2211.05100"): "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model",
    _ax("2307.09288"): "Llama 2: Open Foundation and Fine-Tuned Chat Models",
    _ax("2310.06825"): "Mistral 7B",
    _ax("2401.04088"): "Mixtral of Experts",
    _ax("2412.15115"): "Qwen2.5 Technical Report",
    _ax("2505.09388"): "Qwen3 Technical Report",
    _ax("2502.02737"): "SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model",
    "https://huggingface.co/blog/smollm3": "SmolLM3: smol, multilingual, long-context reasoner",
    "https://ai.meta.com/blog/llama-4-multimodal-intelligence/": "The Llama 4 herd",
}

ATT, T5P, VIT = _ax("1706.03762"), _ax("1910.10683"), _ax("2010.11929")

BLOCKS: list[Block] = [
    # ---------------------------------------------------------------- input
    Block("bpe", "BPE subwords", "input", "Merge frequent character pairs into a subword vocabulary.",
          _ax("1508.07909"), None, "13"),
    Block("byte_bpe", "byte-level BPE", "input", "BPE over raw bytes: any text encodes, no unknown tokens.",
          "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
          "gpt2", "13"),
    Block("skipgram", "skip-gram embeddings", "input", "Learn word vectors by predicting a word's neighbours.",
          _ax("1301.3781"), "word2vec", "30"),
    Block("tied_emb", "tied input/output embeddings", "input",
          "Use the token-embedding matrix as the output layer too: fewer parameters, better perplexity.",
          _ax("1608.05859"), None, "16"),
    Block("patch_emb", "patch embeddings", "input", "Cut an image into 16×16 patches and treat each as a token.",
          VIT, "vit", None),
    # ---------------------------------------------------------------- position
    Block("sinusoidal", "sinusoidal positions", "position", "Fixed sin/cos vectors added to token embeddings.",
          ATT, "transformer", "15"),
    Block("learned_pos", "learned absolute positions", "position",
          "A trained vector per position, added to the token embedding; caps the context length.",
          _ax("1705.03122"), None, "16"),
    Block("rel_bias", "relative position bias", "position",
          "A learned number per head and distance bucket, added to attention scores.", T5P, "t5", "17"),
    Block("alibi", "ALiBi", "position", "A fixed penalty proportional to distance, added to attention scores.",
          _ax("2108.12409"), None, "17"),
    Block("rope", "RoPE", "position", "Rotate queries and keys by position; scores see only the offset.",
          _ax("2104.09864"), None, "17"),
    Block("partial_rope", "partial RoPE", "position", "Rotate only a slice of each head's features.",
          _ax("2204.06745"), "gpt_neox", "17"),
    Block("nope", "NoPE", "position", "No position signal at all: the causal mask is the only clue.",
          _ax("2203.16634"), None, "17"),
    Block("rope_nope_mix", "RoPE/NoPE hybrid layers", "position", "RoPE in most layers, none in every few.",
          "https://huggingface.co/blog/smollm3", None, "17"),
    # ---------------------------------------------------------------- normalization
    Block("batchnorm", "BatchNorm", "normalization", "Normalize each feature over the batch.",
          _ax("1502.03167"), None, "11"),
    Block("layernorm", "LayerNorm", "normalization", "Normalize each token's features to zero mean, unit variance.",
          _ax("1607.06450"), None, "11"),
    Block("rmsnorm", "RMSNorm", "normalization", "LayerNorm without the mean: rescale by root-mean-square.",
          _ax("1910.07467"), None, "11"),
    Block("post_norm", "post-norm", "normalization", "Normalize after adding the residual (the 2017 layout).",
          ATT, "transformer", None),
    Block("pre_norm", "pre-norm", "normalization",
          "Normalize each sub-layer's input; the residual path stays clean, so deep stacks train stably.",
          "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
          "gpt2", "11"),
    Block("qk_norm", "QK-Norm", "normalization", "Normalize queries and keys before the dot product.",
          _ax("2010.04245"), None, None),
    # ---------------------------------------------------------------- sequence mixing
    Block("conv", "convolution", "sequence mixing", "Slide small learned filters over the input.",
          "https://doi.org/10.1109/5.726791", "lenet", "12"),
    Block("lstm", "LSTM", "sequence mixing", "A recurrent cell with gates that keep gradients alive.",
          "https://doi.org/10.1162/neco.1997.9.8.1735", "lstm", "31"),
    Block("additive_attn", "additive attention", "sequence mixing",
          "A decoder learns where to look in the encoder's states.", _ax("1409.0473"), "rnnsearch", "31"),
    Block("mha", "multi-head self-attention", "sequence mixing",
          "Every token attends to every other, in several heads at once.", ATT, "transformer", "15"),
    Block("mqa", "multi-query attention", "sequence mixing", "All query heads share one key/value head.",
          _ax("1911.02150"), None, "17"),
    Block("gqa", "grouped-query attention", "sequence mixing", "Groups of query heads share key/value heads.",
          _ax("2305.13245"), None, "17"),
    Block("mla", "multi-head latent attention", "sequence mixing",
          "Cache a small compressed latent instead of per-head keys and values.", _ax("2405.04434"),
          "deepseek_v2", "17"),
    Block("swa", "sliding-window attention", "sequence mixing", "Each token attends only to the last W tokens.",
          _ax("2004.05150"), None, None),
    Block("s4", "structured state space (S4)", "sequence mixing",
          "A linear recurrence with a structured state matrix, trainable as a long convolution.",
          _ax("2111.00396"), "s4", "36"),
    Block("selective_ssm", "selective SSM", "sequence mixing",
          "State-space recurrence whose parameters depend on the input (Mamba's selection).", _ax("2312.00752"),
          "mamba", "36"),
    Block("ssd", "state space duality (SSD)", "sequence mixing",
          "A restricted selective SSM that runs as matmuls, like masked attention.", _ax("2405.21060"),
          "mamba2", "36"),
    Block("attn_ssm_mix", "attention/SSM hybrid layers", "sequence mixing",
          "Mostly SSM layers with an attention layer every few.", _ax("2403.19887"), "jamba", "36"),
    # ---------------------------------------------------------------- feed-forward
    Block("relu_ffn", "ReLU feed-forward", "feed-forward", "Two linear layers with a ReLU between.", ATT,
          "transformer", "10"),
    Block("gelu", "GELU", "feed-forward", "A smooth ReLU, weighting inputs by how likely they are positive.",
          _ax("1606.08415"), None, "10"),
    Block("swiglu", "SwiGLU", "feed-forward", "A gated feed-forward: silu(xW1) ⊙ xW3, then W2.",
          _ax("2002.05202"), None, "17"),
    Block("moe_topk", "top-k mixture of experts", "feed-forward",
          "A router sends each token to k of many feed-forward experts.", _ax("1701.06538"), None, "17"),
    Block("switch_top1", "top-1 routing", "feed-forward", "Route each token to a single expert.",
          _ax("2101.03961"), "switch", "17"),
    Block("shared_experts", "fine-grained + shared experts", "feed-forward",
          "Many small routed experts plus always-on shared ones.", _ax("2401.06066"), "deepseekmoe", "17"),
    Block("aux_free", "aux-loss-free balancing", "feed-forward",
          "Balance experts with a bias on the router instead of an extra loss.", _ax("2408.15664"),
          None, "17"),
    # ---------------------------------------------------------------- connectivity
    Block("residual", "residual connections", "connectivity", "Add each layer's input to its output.",
          _ax("1512.03385"), "resnet", "11"),
    Block("dropout", "dropout", "connectivity", "Randomly zero activations during training.",
          "https://jmlr.org/papers/v15/srivastava14a.html", None, "11"),
    # ---------------------------------------------------------------- objective
    Block("next_token", "next-token prediction", "objective", "Predict the next token from the ones before.",
          "https://jmlr.org/papers/v3/bengio03a.html", "nplm", "14"),
    Block("mlm", "masked language modeling", "objective", "Hide some tokens and predict them from both sides.",
          _ax("1810.04805"), "bert", "32"),
    Block("span_corruption", "span corruption", "objective", "Replace spans with sentinels; decode the spans.",
          T5P, "t5", None),
    Block("contrastive", "contrastive image-text", "objective",
          "Pull matching image/caption pairs together, push the rest apart.", _ax("2103.00020"), "clip", "33"),
    Block("diffusion", "denoising diffusion", "objective", "Learn to remove noise, step by step.",
          _ax("2006.11239"), "ddpm", "33"),
    Block("latent_diffusion", "latent diffusion", "objective", "Diffuse in an autoencoder's latent space.",
          _ax("2112.10752"), "stable_diffusion", "33"),
    Block("mtp", "multi-token prediction", "objective", "Also predict tokens two or more steps ahead.",
          _ax("2404.19737"), None, "17"),
    Block("q_learning", "deep Q-learning", "objective", "Learn action values from rewards with a network.",
          _ax("1312.5602"), "dqn", "34"),
    Block("selfplay_mcts", "self-play + MCTS", "objective",
          "Train policy and value from games against itself, with tree search as the teacher.",
          "https://doi.org/10.1038/nature24270", "alphago_zero", "34"),
    Block("learned_model", "planning with a learned model", "objective",
          "Search inside a model of the game the network learned itself.", _ax("1911.08265"), "muzero", "34"),
    # ---------------------------------------------------------------- post-training
    Block("rlhf", "RLHF", "post-training", "Fit a reward model to human preferences, then optimize against it.",
          _ax("1706.03741"), None, "35"),
    Block("dpo", "DPO", "post-training", "Optimize preferences directly, no reward model or RL loop.",
          _ax("2305.18290"), None, "35"),
    Block("grpo", "RLVR with GRPO", "post-training",
          "RL against verifiable rewards, with group-relative advantages instead of a critic.",
          _ax("2402.03300"), None, "26"),
]

G2 = "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
LLAMA_CORE = ("pre_norm", "rmsnorm", "rope", "swiglu", "residual", "next_token")

MODELS: list[Model] = [
    # ---------------------------------------------------------------- RNNs and embeddings
    Model("lstm", "LSTM", 1997, "Hochreiter & Schmidhuber", "RNN & embeddings", (), ("lstm",),
          "Gated memory that keeps gradients alive over long sequences.",
          "https://doi.org/10.1162/neco.1997.9.8.1735", "31"),
    Model("nplm", "Neural probabilistic LM", 2003, "Bengio et al.", "RNN & embeddings", (), ("next_token",),
          "The first neural language model: learned word vectors predict the next word.",
          "https://jmlr.org/papers/v3/bengio03a.html", "14"),
    Model("word2vec", "word2vec", 2013, "Google", "RNN & embeddings", ("nplm",), ("skipgram",),
          "Cheap, good word vectors at scale.", _ax("1301.3781"), "30"),
    Model("rnn_encdec", "RNN encoder-decoder", 2014, "Cho et al.", "RNN & embeddings", ("lstm",), ("next_token",),
          "Encode a sentence to a vector, decode a translation (and the GRU).", _ax("1406.1078"), "31"),
    Model("seq2seq", "seq2seq", 2014, "Google", "RNN & embeddings", ("lstm",), ("lstm", "next_token"),
          "Deep LSTMs translating end to end.", _ax("1409.3215"), "31"),
    Model("rnnsearch", "RNNsearch (Bahdanau attention)", 2014, "Bahdanau, Cho, Bengio", "RNN & embeddings",
          ("rnn_encdec",), ("additive_attn", "next_token"),
          "Attention: the decoder looks back at every encoder state.", _ax("1409.0473"), "31"),
    Model("elmo", "ELMo", 2018, "AllenNLP", "RNN & embeddings", ("lstm", "word2vec"), ("lstm", "next_token"),
          "Contextual word vectors from a bidirectional LSTM language model.", _ax("1802.05365"), "32"),
    # ---------------------------------------------------------------- CNNs
    Model("lenet", "LeNet-5", 1998, "LeCun et al.", "CNN", (), ("conv",),
          "Convolutional networks reading handwritten digits.", "https://doi.org/10.1109/5.726791", "12"),
    Model("alexnet", "AlexNet", 2012, "Krizhevsky, Sutskever, Hinton", "CNN", ("lenet",), ("conv", "dropout"),
          "A deep CNN on two GPUs wins ImageNet: the deep-learning revival.",
          "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks", "12"),
    Model("resnet", "ResNet", 2015, "Microsoft Research", "CNN", ("alexnet",), ("conv", "residual", "batchnorm"),
          "Residual connections make 100+ layer networks trainable.", _ax("1512.03385"), "11"),
    # ---------------------------------------------------------------- RL and games
    Model("dqn", "DQN", 2013, "DeepMind", "RL & games", ("alexnet",), ("conv", "q_learning"),
          "One network learns many Atari games from pixels and score.", _ax("1312.5602"), "34"),
    Model("alphago", "AlphaGo", 2016, "DeepMind", "RL & games", ("dqn",), ("conv",),
          "Policy and value networks plus tree search beat a Go champion.", "https://doi.org/10.1038/nature16961",
          "34"),
    Model("alphago_zero", "AlphaGo Zero", 2017, "DeepMind", "RL & games", ("alphago", "resnet"),
          ("conv", "residual", "batchnorm", "selfplay_mcts"),
          "No human games: self-play with search as the teacher.", "https://doi.org/10.1038/nature24270", "34"),
    Model("alphazero", "AlphaZero", 2017, "DeepMind", "RL & games", ("alphago_zero",),
          ("conv", "residual", "batchnorm", "selfplay_mcts"),
          "The same algorithm masters chess, shogi and Go.", _ax("1712.01815"), "34"),
    Model("muzero", "MuZero", 2019, "DeepMind", "RL & games", ("alphazero",),
          ("conv", "residual", "batchnorm", "selfplay_mcts", "learned_model"),
          "Plans without being told the rules.", _ax("1911.08265"), "34"),
    # ---------------------------------------------------------------- the Transformer and encoders
    Model("transformer", "Transformer", 2017, "Google", "encoder-decoder", ("rnnsearch",),
          ("mha", "sinusoidal", "post_norm", "layernorm", "relu_ffn", "residual", "dropout", "bpe", "next_token"),
          "Attention alone, no recurrence: the architecture everything after builds on.", ATT, "15"),
    Model("bert", "BERT", 2018, "Google", "encoder", ("transformer", "elmo", "gpt1"),
          ("mha", "learned_pos", "post_norm", "layernorm", "gelu", "residual", "dropout", "mlm"),
          "Bidirectional pretraining with masked tokens; fine-tune for any task.", _ax("1810.04805"), "32",
          "google-bert/bert-base-uncased"),
    Model("roberta", "RoBERTa", 2019, "Facebook AI", "encoder", ("bert",),
          ("mha", "learned_pos", "post_norm", "layernorm", "gelu", "residual", "dropout", "mlm", "byte_bpe"),
          "BERT trained longer, on more data, without next-sentence prediction.", _ax("1907.11692"), "32",
          "FacebookAI/roberta-base"),
    Model("t5", "T5", 2019, "Google", "encoder-decoder", ("transformer", "bert"),
          ("mha", "rel_bias", "pre_norm", "relu_ffn", "residual", "dropout", "span_corruption", "tied_emb"),
          "Every task as text-to-text, with relative position biases.", T5P, "32", "google-t5/t5-base"),
    Model("vit", "Vision Transformer (ViT)", 2020, "Google", "encoder", ("transformer", "resnet"),
          ("patch_emb", "mha", "learned_pos", "pre_norm", "layernorm", "gelu", "residual"),
          "A plain Transformer on image patches matches CNNs at scale.", VIT, None,
          "google/vit-base-patch16-224"),
    # ---------------------------------------------------------------- decoders
    Model("gpt1", "GPT", 2018, "OpenAI", "decoder", ("transformer",),
          ("mha", "learned_pos", "post_norm", "layernorm", "gelu", "residual", "dropout", "bpe", "next_token"),
          "Generative pretraining of a decoder-only Transformer, then fine-tuning.",
          "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf", "16"),
    Model("gpt2", "GPT-2", 2019, "OpenAI", "decoder", ("gpt1",),
          ("mha", "learned_pos", "pre_norm", "layernorm", "gelu", "residual", "dropout", "byte_bpe", "tied_emb",
           "next_token"),
          "Scale plus byte-level BPE and pre-norm; zero-shot tasks emerge.", G2, "16", "openai-community/gpt2"),
    Model("gpt3", "GPT-3", 2020, "OpenAI", "decoder", ("gpt2",),
          ("mha", "learned_pos", "pre_norm", "layernorm", "gelu", "residual", "byte_bpe", "next_token"),
          "175B parameters; tasks from a few examples in the prompt.", _ax("2005.14165"), "20"),
    Model("instructgpt", "InstructGPT", 2022, "OpenAI", "decoder", ("gpt3",),
          ("mha", "learned_pos", "pre_norm", "layernorm", "gelu", "residual", "byte_bpe", "next_token", "rlhf"),
          "Supervised instructions plus RLHF: the recipe behind ChatGPT.", _ax("2203.02155"), "35"),
    Model("gpt_neox", "GPT-NeoX-20B", 2022, "EleutherAI", "decoder", ("gpt3",),
          ("mha", "partial_rope", "pre_norm", "layernorm", "gelu", "residual", "byte_bpe", "next_token"),
          "Open 20B model; rotary positions on a quarter of each head.", _ax("2204.06745"), None,
          "EleutherAI/gpt-neox-20b"),
    Model("pythia", "Pythia", 2023, "EleutherAI", "decoder", ("gpt_neox",),
          ("mha", "partial_rope", "pre_norm", "layernorm", "gelu", "residual", "byte_bpe", "next_token"),
          "A suite of models with every checkpoint public, for studying training.", _ax("2304.01373"), None,
          "EleutherAI/pythia-160m"),
    Model("palm", "PaLM", 2022, "Google", "decoder", ("gpt3", "t5"),
          ("mqa", "rope", "swiglu", "pre_norm", "layernorm", "residual", "tied_emb", "next_token"),
          "540B parameters with SwiGLU, RoPE and multi-query attention.", _ax("2204.02311"), None),
    Model("bloom", "BLOOM", 2022, "BigScience", "decoder", ("gpt3",),
          ("mha", "alibi", "pre_norm", "layernorm", "gelu", "residual", "byte_bpe", "tied_emb", "next_token"),
          "Open, multilingual 176B model; ALiBi instead of position embeddings.", _ax("2211.05100"), None,
          "bigscience/bloom-560m"),
    Model("llama", "LLaMA", 2023, "Meta", "decoder", ("gpt3", "palm", "gpt_neox"), ("mha", "bpe") + LLAMA_CORE,
          "Open weights combining pre-norm RMSNorm, SwiGLU and RoPE: the recipe most open models copy.",
          _ax("2302.13971"), "17"),
    Model("llama2", "Llama 2", 2023, "Meta", "decoder", ("llama",), ("mha", "gqa", "rlhf", "bpe") + LLAMA_CORE,
          "Longer context, GQA in the 34B/70B models, RLHF-tuned chat models.", _ax("2307.09288"), "17"),
    Model("mistral", "Mistral 7B", 2023, "Mistral AI", "decoder", ("llama2",), ("gqa", "swa", "bpe") + LLAMA_CORE,
          "A 7B model beating Llama 2 13B, with GQA and sliding-window attention.", _ax("2310.06825"), None),
    Model("llama3", "Llama 3", 2024, "Meta", "decoder", ("llama2",), ("gqa", "dpo", "byte_bpe") + LLAMA_CORE,
          "GQA at every size, a 128K vocabulary, RoPE base 500,000, DPO in post-training.", _ax("2407.21783"),
          "17"),
    Model("qwen25", "Qwen2.5", 2024, "Alibaba", "decoder", ("llama",), ("gqa", "dpo", "grpo", "byte_bpe") + LLAMA_CORE,
          "A strong open family (0.5B–72B): GQA, bias on QKV, DPO then GRPO.", _ax("2412.15115"), "19",
          "Qwen/Qwen2.5-7B"),
    Model("smollm2", "SmolLM2", 2025, "Hugging Face", "decoder", ("llama",), ("gqa", "tied_emb", "dpo", "byte_bpe") + LLAMA_CORE,
          "Small models (135M–1.7B) from carefully staged data; the models you run in Modules 18–21.",
          _ax("2502.02737"), "18", "HuggingFaceTB/SmolLM2-360M"),
    Model("smollm3", "SmolLM3", 2025, "Hugging Face", "decoder", ("smollm2", "llama3"),
          ("gqa", "tied_emb", "rope_nope_mix", "byte_bpe") + LLAMA_CORE,
          "3B, long context; drops RoPE from every 4th layer.", "https://huggingface.co/blog/smollm3", "17",
          "HuggingFaceTB/SmolLM3-3B"),
    Model("deepseek_r1", "DeepSeek-R1", 2025, "DeepSeek", "MoE decoder", ("deepseek_v3",),
          ("mla", "partial_rope", "shared_experts", "moe_topk", "aux_free", "mtp", "grpo", "byte_bpe") + LLAMA_CORE,
          "Reasoning learned largely through RL on verifiable rewards.", _ax("2501.12948"), "26",
          "deepseek-ai/DeepSeek-R1"),
    # ---------------------------------------------------------------- MoE decoders
    Model("gshard", "GShard", 2020, "Google", "encoder-decoder", ("transformer",),
          ("mha", "moe_topk", "residual"),
          "Top-2 mixture-of-experts Transformers sharded across 2,048 TPU cores.", _ax("2006.16668"), "24"),
    Model("switch", "Switch Transformer", 2021, "Google", "encoder-decoder", ("t5", "gshard"),
          ("mha", "rel_bias", "pre_norm", "moe_topk", "switch_top1", "residual", "span_corruption"),
          "Route each token to a single expert: trillion-parameter sparse models.", _ax("2101.03961"), "17"),
    Model("mixtral", "Mixtral 8x7B", 2024, "Mistral AI", "MoE decoder", ("mistral", "gshard"),
          ("gqa", "moe_topk", "bpe") + LLAMA_CORE,
          "Open sparse MoE: 8 experts, 2 per token, 13B of 47B parameters active.", _ax("2401.04088"), None),
    Model("deepseekmoe", "DeepSeekMoE 16B", 2024, "DeepSeek", "MoE decoder", ("llama", "gshard"),
          ("mha", "moe_topk", "shared_experts", "byte_bpe") + LLAMA_CORE,
          "Many small experts plus always-on shared ones.", _ax("2401.06066"), "17"),
    Model("deepseek_v2", "DeepSeek-V2", 2024, "DeepSeek", "MoE decoder", ("deepseekmoe",),
          ("mla", "partial_rope", "moe_topk", "shared_experts", "byte_bpe") + LLAMA_CORE,
          "Multi-head latent attention shrinks the KV cache.", _ax("2405.04434"), "17", "deepseek-ai/DeepSeek-V2"),
    Model("deepseek_v3", "DeepSeek-V3", 2024, "DeepSeek", "MoE decoder", ("deepseek_v2",),
          ("mla", "partial_rope", "moe_topk", "shared_experts", "aux_free", "mtp", "grpo", "byte_bpe") + LLAMA_CORE,
          "671B MoE with aux-loss-free balancing and multi-token prediction; you replicated it in miniature.",
          _ax("2412.19437"), "17", "deepseek-ai/DeepSeek-V3"),
    Model("qwen3", "Qwen3", 2025, "Alibaba", "MoE decoder", ("qwen25",),
          ("gqa", "qk_norm", "moe_topk", "grpo", "byte_bpe") + LLAMA_CORE,
          "Dense and MoE models with QK-Norm and switchable thinking.", _ax("2505.09388"), None,
          "Qwen/Qwen3-30B-A3B"),
    Model("llama4", "Llama 4", 2025, "Meta", "MoE decoder", ("llama3",),
          ("gqa", "moe_topk", "shared_experts", "rope_nope_mix", "dpo") + LLAMA_CORE,
          "Meta's first MoE: routed plus shared experts, no RoPE in every 4th layer, natively multimodal.",
          "https://ai.meta.com/blog/llama-4-multimodal-intelligence/", None),
    # ---------------------------------------------------------------- SSMs and hybrids
    Model("s4", "S4", 2021, "Stanford", "SSM & hybrid", (), ("s4", "residual"),
          "Structured state spaces handle sequences of thousands of steps.", _ax("2111.00396"), "36"),
    Model("mamba", "Mamba", 2023, "CMU & Princeton", "SSM & hybrid", ("s4",),
          ("selective_ssm", "rmsnorm", "pre_norm", "residual", "next_token"),
          "Input-dependent state spaces: attention-quality language modeling in linear time.", _ax("2312.00752"),
          "36", "state-spaces/mamba-130m-hf"),
    Model("mamba2", "Mamba-2", 2024, "CMU & Princeton", "SSM & hybrid", ("mamba",),
          ("ssd", "rmsnorm", "pre_norm", "residual", "next_token"),
          "State space duality: SSMs as structured attention, running on tensor cores.", _ax("2405.21060"), "36"),
    Model("jamba", "Jamba", 2024, "AI21 Labs", "SSM & hybrid", ("mamba", "mixtral"),
          ("selective_ssm", "attn_ssm_mix", "gqa", "nope", "moe_topk", "rmsnorm", "pre_norm", "residual",
           "next_token"),
          "Mamba layers with attention every 8th layer and MoE, and no positional encoding.", _ax("2403.19887"),
          "36", "ai21labs/Jamba-v0.1"),
    # ---------------------------------------------------------------- vision-language and diffusion
    Model("ddpm", "DDPM", 2020, "UC Berkeley", "vision-language & diffusion", (), ("diffusion", "residual"),
          "Denoising diffusion produces high-quality images.", _ax("2006.11239"), "33"),
    Model("clip", "CLIP", 2021, "OpenAI", "vision-language & diffusion", ("vit", "gpt2"),
          ("contrastive", "patch_emb", "mha", "pre_norm", "layernorm", "residual", "byte_bpe"),
          "Images and captions in one embedding space, trained on 400M pairs.", _ax("2103.00020"), "33",
          "openai/clip-vit-base-patch32"),
    Model("stable_diffusion", "Latent diffusion / Stable Diffusion", 2022, "LMU Munich & Runway",
          "vision-language & diffusion", ("ddpm", "clip"), ("latent_diffusion", "diffusion", "residual", "mha"),
          "Diffusion in a compressed latent space, conditioned on text.", _ax("2112.10752"), "33"),
]


def _registered() -> set[str]:
    return {p.url for ps in PAPERS.values() for p in ps} | set(SOURCES)


def by_id() -> dict[str, Model]:
    return {m.id: m for m in MODELS}


def blocks_by_id() -> dict[str, Block]:
    return {b.id: b for b in BLOCKS}


def check() -> list[str]:
    """Consistency problems in the registry (empty list = fine). Run by the tests and tools/check_lineage.py."""
    from learnjax.syllabus import MODULES, label

    labels = {label(s) for _n, s, _t, _p in MODULES}
    reg, models, blocks = _registered(), by_id(), blocks_by_id()
    errs = []
    if len(models) != len(MODELS) or len(blocks) != len(BLOCKS):
        errs.append("duplicate ids")
    for m in MODELS:
        if m.family not in FAMILIES:
            errs.append(f"{m.id}: unknown family {m.family!r}")
        if m.source not in reg:
            errs.append(f"{m.id}: source not registered: {m.source}")
        if m.module is not None and m.module not in labels:
            errs.append(f"{m.id}: unknown module {m.module}")
        for p in m.parents:
            if p not in models:
                errs.append(f"{m.id}: unknown parent {p}")
            elif models[p].year > m.year:
                errs.append(f"{m.id} ({m.year}) is older than its parent {p} ({models[p].year})")
        for u in m.uses:
            if u not in blocks:
                errs.append(f"{m.id}: unknown block {u}")
        if len(set(m.uses)) != len(m.uses):
            errs.append(f"{m.id}: repeated block")
    used = {u for m in MODELS for u in m.uses}
    for b in BLOCKS:
        if b.slot not in SLOTS:
            errs.append(f"{b.id}: unknown slot {b.slot!r}")
        if b.paper not in reg:
            errs.append(f"{b.id}: paper not registered: {b.paper}")
        if b.module is not None and b.module not in labels:
            errs.append(f"{b.id}: unknown module {b.module}")
        if b.id not in used:
            errs.append(f"{b.id}: used by no model")
        if b.first_model is not None:
            fm = models.get(b.first_model)
            if fm is None or b.id not in fm.uses:
                errs.append(f"{b.id}: first_model {b.first_model} doesn't use it")
            elif fm.source != b.paper:
                errs.append(f"{b.id}: first_model {b.first_model}'s source isn't the block's paper")
            else:
                early = [m.id for m in MODELS if b.id in m.uses and m.year < fm.year]
                if early:
                    errs.append(f"{b.id}: used by {early} before its first model {fm.id} ({fm.year})")
    return errs


def compare_md(columns: list[tuple[str, tuple[str, ...]]], slots: list[str] | None = None) -> str:
    """A markdown table: one row per block that any column uses (in BLOCKS order, limited to `slots`),
    one ✓ column per (header, uses) pair, plus the module that builds the block. Lessons use it for
    "you are here" comparisons between the model you built and the registry's models."""
    blocks = blocks_by_id()
    for _h, uses in columns:
        unknown = [u for u in uses if u not in blocks]
        if unknown:
            raise KeyError(f"unknown blocks {unknown}")
    rows = [b for b in BLOCKS if (slots is None or b.slot in slots) and any(b.id in u for _h, u in columns)]
    head = "| slot | block | " + " | ".join(h for h, _u in columns) + " | module |"
    sep = "|---|---|" + "---|" * len(columns) + "---|"
    lines = [head, sep]
    for b in rows:
        cells = " | ".join("✓" if b.id in u else "" for _h, u in columns)
        lines.append(f"| {b.slot} | {b.name} | {cells} | {b.module or '—'} |")
    return "\n".join(lines)
