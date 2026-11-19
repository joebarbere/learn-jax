# The family tree: models and their building blocks

Generated from `learnjax/lineage.py` by `tools/gen_lineage.py`; module 00b (Section 12) draws it as an interactive graph. `make lineage` checks the registry's rules, the sources' arXiv titles, and every open model's Hugging Face `config.json` against the blocks listed here.

62 models, 62 building blocks in 8 slots.

## The taxonomy: building blocks by slot

*Introduced in* is the paper the block comes from. *First model* is set when that paper's own model is in the registry. *Module* is where this course builds or discusses it.

### input

| block | what it does | introduced in | first model | used by | module |
|---|---|---|---|---|---|
| **BPE subwords** | Merge frequent character pairs into a subword vocabulary. | [Sennrich, Haddow, Birch, 2015](https://arxiv.org/abs/1508.07909) | — | Transformer, GPT, LLaMA, Llama 2, Mistral 7B, Mixtral 8x7B | 13 |
| **byte-level BPE** | BPE over raw bytes: any text encodes, no unknown tokens. | [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | GPT-2 | GPT-2, RoBERTa, GPT-3, CLIP, BLOOM, GPT-NeoX-20B, InstructGPT, Pythia, DeepSeek-V2, DeepSeek-V3, DeepSeekMoE 16B, Llama 3, Qwen2.5, DeepSeek-R1, Qwen3, Qwen3-Next, SmolLM2, SmolLM3, Engram-27B | 13 |
| **skip-gram embeddings** | Learn word vectors by predicting a word's neighbours. | [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781) | word2vec | word2vec | 30 |
| **tied input/output embeddings** | Use the token-embedding matrix as the output layer too: fewer parameters, better perplexity. | [Press, Wolf, 2016](https://arxiv.org/abs/1608.05859) | — | GPT-2, T5, BLOOM, PaLM, SmolLM2, SmolLM3 | 16 |
| **patch embeddings** | Cut an image into 16×16 patches and treat each as a token. | [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929) | Vision Transformer (ViT) | Vision Transformer (ViT), CLIP | — |

### position

| block | what it does | introduced in | first model | used by | module |
|---|---|---|---|---|---|
| **sinusoidal positions** | Fixed sin/cos vectors added to token embeddings. | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) | Transformer | Transformer | 15 |
| **learned absolute positions** | A trained vector per position, added to the token embedding; caps the context length. | [Gehring et al., 2017](https://arxiv.org/abs/1705.03122) | — | BERT, GPT, GPT-2, RoBERTa, GPT-3, Vision Transformer (ViT), InstructGPT | 16 |
| **relative position bias** | A learned number per head and distance bucket, added to attention scores. | [Raffel et al., 2019](https://arxiv.org/abs/1910.10683) | T5 | T5, Switch Transformer | 17 |
| **ALiBi** | A fixed penalty proportional to distance, added to attention scores. | [Press, Smith, Lewis, 2021](https://arxiv.org/abs/2108.12409) | — | BLOOM | 17 |
| **RoPE** | Rotate queries and keys by position; scores see only the offset. | [Su et al., 2021](https://arxiv.org/abs/2104.09864) | — | PaLM, LLaMA, Llama 2, Mistral 7B, DeepSeek-V2, DeepSeek-V3, DeepSeekMoE 16B, Gemma 2, Llama 3, Mixtral 8x7B, Qwen2.5, DeepSeek-R1, Llama 4, Qwen3, SmolLM2, SmolLM3, Engram-27B | 17 |
| **partial RoPE** | Rotate only a slice of each head's features. | [Black et al., 2022](https://arxiv.org/abs/2204.06745) | GPT-NeoX-20B | GPT-NeoX-20B, Pythia, DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, Qwen3-Next | 17 |
| **NoPE** | No position signal at all: the causal mask is the only clue. | [Haviv et al., 2022](https://arxiv.org/abs/2203.16634) | — | Jamba, Nemotron 3 Super | 17 |
| **RoPE/NoPE hybrid layers** | RoPE in most layers, none in every few. | [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3) | — | Llama 4, SmolLM3 | 17 |

### normalization

| block | what it does | introduced in | first model | used by | module |
|---|---|---|---|---|---|
| **BatchNorm** | Normalize each feature over the batch. | [Ioffe, Szegedy, 2015](https://arxiv.org/abs/1502.03167) | — | ResNet, AlphaGo Zero, AlphaZero, MuZero | 11 |
| **LayerNorm** | Normalize each token's features to zero mean, unit variance. | [Ba, Kiros, Hinton, 2016](https://arxiv.org/abs/1607.06450) | — | Transformer, BERT, GPT, GPT-2, RoBERTa, GPT-3, Vision Transformer (ViT), CLIP, BLOOM, GPT-NeoX-20B, InstructGPT, PaLM, Pythia | 11 |
| **RMSNorm** | LayerNorm without the mean: rescale by root-mean-square. | [Zhang, Sennrich, 2019](https://arxiv.org/abs/1910.07467) | — | LLaMA, Llama 2, Mamba, Mistral 7B, DeepSeek-V2, DeepSeek-V3, DeepSeekMoE 16B, DeltaNet, Gated DeltaNet, Gemma 2, Jamba, Llama 3, Mamba-2, Mixtral 8x7B, Qwen2.5, DeepSeek-R1, Llama 4, Nemotron-H, Qwen3, Qwen3-Next, SmolLM2, SmolLM3, Nemotron 3 Super | 11 |
| **post-norm** | Normalize after adding the residual (the 2017 layout). | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) | Transformer | Transformer, BERT, GPT, RoBERTa | — |
| **pre-norm** | Normalize each sub-layer's input; the residual path stays clean, so deep stacks train stably. | [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | GPT-2 | GPT-2, T5, GPT-3, Vision Transformer (ViT), CLIP, Switch Transformer, BLOOM, GPT-NeoX-20B, InstructGPT, PaLM, LLaMA, Llama 2, Mamba, Mistral 7B, Pythia, DeepSeek-V2, DeepSeek-V3, DeepSeekMoE 16B, Jamba, Llama 3, Mamba-2, Mixtral 8x7B, Qwen2.5, DeepSeek-R1, Llama 4, Qwen3, SmolLM2, SmolLM3 | 11 |
| **QK-Norm** | Normalize queries and keys before the dot product. | [Henry et al., 2020](https://arxiv.org/abs/2010.04245) | — | Qwen3 | — |

### sequence mixing

| block | what it does | introduced in | first model | used by | module |
|---|---|---|---|---|---|
| **convolution** | Slide small learned filters over the input. | [LeCun et al., 1998](https://doi.org/10.1109/5.726791) | LeNet-5 | LeNet-5, AlexNet, DQN, ResNet, AlphaGo, AlphaGo Zero, AlphaZero, MuZero | 12 |
| **LSTM** | A recurrent cell with gates that keep gradients alive. | [Hochreiter, Schmidhuber, 1997](https://doi.org/10.1162/neco.1997.9.8.1735) | LSTM | LSTM, seq2seq, ELMo | 31 |
| **additive attention** | A decoder learns where to look in the encoder's states. | [Bahdanau, Cho, Bengio, 2014](https://arxiv.org/abs/1409.0473) | RNNsearch (Bahdanau attention) | RNNsearch (Bahdanau attention) | 31 |
| **multi-head self-attention** | Every token attends to every other, in several heads at once. | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) | Transformer | Transformer, BERT, GPT, GPT-2, RoBERTa, T5, GPT-3, GShard, Vision Transformer (ViT), CLIP, Switch Transformer, BLOOM, GPT-NeoX-20B, InstructGPT, Latent diffusion / Stable Diffusion, LLaMA, Llama 2, Pythia, DeepSeekMoE 16B | 15 |
| **multi-query attention** | All query heads share one key/value head. | [Shazeer, 2019](https://arxiv.org/abs/1911.02150) | — | PaLM | 17 |
| **grouped-query attention** | Groups of query heads share key/value heads. | [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245) | — | Llama 2, Mistral 7B, Gemma 2, Jamba, Llama 3, Mixtral 8x7B, Qwen2.5, Llama 4, Qwen3, Qwen3-Next, SmolLM2, SmolLM3, Nemotron 3 Super | 17 |
| **multi-head latent attention** | Cache a small compressed latent instead of per-head keys and values. | [DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434) | DeepSeek-V2 | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, Kimi Linear, Engram-27B | 17 |
| **sliding-window attention** | Each token attends only to the last W tokens. | [Beltagy, Peters, Cohan, 2020](https://arxiv.org/abs/2004.05150) | — | Mistral 7B, Gemma 2 | — |
| **structured state space (S4)** | A linear recurrence with a structured state matrix, trainable as a long convolution. | [Gu, Goel, Ré, 2021](https://arxiv.org/abs/2111.00396) | S4 | S4 | 36 |
| **selective SSM** | State-space recurrence whose parameters depend on the input (Mamba's selection). | [Gu, Dao, 2023](https://arxiv.org/abs/2312.00752) | Mamba | Mamba, Jamba | 36 |
| **state space duality (SSD)** | A restricted selective SSM that runs as matmuls, like masked attention. | [Dao, Gu, 2024](https://arxiv.org/abs/2405.21060) | Mamba-2 | Mamba-2, Nemotron-H, Nemotron 3 Super | 36 |
| **attention/SSM hybrid layers** | Mostly SSM or linear-attention layers with an attention layer every few. | [Lieber et al., 2024](https://arxiv.org/abs/2403.19887) | Jamba | Jamba, Kimi Linear, MiniMax-01, Nemotron-H, Qwen3-Next, Nemotron 3 Super | 36 |
| **linear attention** | Drop the softmax: attention becomes an RNN whose state is a d × d matrix of summed key-value products. | [Katharopoulos et al., 2020](https://arxiv.org/abs/2006.16236) | Linear Transformer | Linear Transformer, MiniMax-01 | 41 |
| **retention (fixed decay)** | Linear attention whose state fades by a fixed factor per head every step. | [Sun et al., 2023](https://arxiv.org/abs/2307.08621) | RetNet | RetNet | 41 |
| **delta rule** | Write only the error: the state overwrites a key's old value instead of adding to it. | [Schlag, Irie, Schmidhuber, 2021](https://arxiv.org/abs/2102.11174) | — | DeltaNet | 41 |
| **gated delta rule** | The delta rule plus Mamba-2's per-step decay: overwrite and forget. | [Yang, Kautz, Hatamizadeh, 2024](https://arxiv.org/abs/2412.06464) | Gated DeltaNet | Gated DeltaNet, Qwen3-Next | 41 |
| **Kimi Delta Attention** | The gated delta rule with a decay per channel instead of per head. | [Kimi Team, 2025](https://arxiv.org/abs/2510.26692) | Kimi Linear | Kimi Linear | 41 |
| **gated attention** | Multiply attention's output by a sigmoid gate computed from the layer input. | [Qiu et al., 2025](https://arxiv.org/abs/2505.06708) | — | Qwen3-Next | 41 |

### feed-forward

| block | what it does | introduced in | first model | used by | module |
|---|---|---|---|---|---|
| **ReLU feed-forward** | Two linear layers with a ReLU between. | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) | Transformer | Transformer, T5 | 10 |
| **GELU** | A smooth ReLU, weighting inputs by how likely they are positive. | [Hendrycks, Gimpel, 2016](https://arxiv.org/abs/1606.08415) | — | BERT, GPT, GPT-2, RoBERTa, GPT-3, Vision Transformer (ViT), BLOOM, GPT-NeoX-20B, InstructGPT, Pythia | 10 |
| **SwiGLU** | A gated feed-forward: silu(xW1) ⊙ xW3, then W2. | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) | — | PaLM, LLaMA, Llama 2, Mistral 7B, DeepSeek-V2, DeepSeek-V3, DeepSeekMoE 16B, Llama 3, Mixtral 8x7B, Qwen2.5, DeepSeek-R1, Llama 4, Qwen3, SmolLM2, SmolLM3 | 17 |
| **top-k mixture of experts** | A router sends each token to k of many feed-forward experts. | [Shazeer et al., 2017](https://arxiv.org/abs/1701.06538) | — | GShard, Switch Transformer, DeepSeek-V2, DeepSeek-V3, DeepSeekMoE 16B, Jamba, Mixtral 8x7B, DeepSeek-R1, Kimi Linear, Llama 4, MiniMax-01, Qwen3, Qwen3-Next, Engram-27B, Nemotron 3 Super | 17 |
| **top-1 routing** | Route each token to a single expert. | [Fedus, Zoph, Shazeer, 2021](https://arxiv.org/abs/2101.03961) | Switch Transformer | Switch Transformer | 17 |
| **fine-grained + shared experts** | Many small routed experts plus always-on shared ones. | [Dai et al., 2024](https://arxiv.org/abs/2401.06066) | DeepSeekMoE 16B | DeepSeek-V2, DeepSeek-V3, DeepSeekMoE 16B, DeepSeek-R1, Llama 4, Qwen3-Next, Engram-27B, Nemotron 3 Super | 17 |
| **aux-loss-free balancing** | Balance experts with a bias on the router instead of an extra loss. | [Wang et al., 2024](https://arxiv.org/abs/2408.15664) | — | DeepSeek-V3, DeepSeek-R1, Engram-27B | 17 |
| **latent experts (LatentMoE)** | Routed experts work in a narrower latent width; α× as many of them, same bytes per token. | [Elango et al., 2026](https://arxiv.org/abs/2601.18089) | — | Nemotron 3 Super | 40 |
| **hashed n-gram memory (Engram)** | A table indexed by the last few tokens, read in O(1) and gated by the hidden state. | [Cheng et al. (DeepSeek-AI), 2026](https://arxiv.org/abs/2601.07372) | Engram-27B | Engram-27B | 40 |

### connectivity

| block | what it does | introduced in | first model | used by | module |
|---|---|---|---|---|---|
| **residual connections** | Add each layer's input to its output. | [He et al., 2015](https://arxiv.org/abs/1512.03385) | ResNet | ResNet, AlphaGo Zero, AlphaZero, Transformer, BERT, GPT, GPT-2, MuZero, RoBERTa, T5, DDPM, GPT-3, GShard, Linear Transformer, Vision Transformer (ViT), CLIP, S4, Switch Transformer, BLOOM, GPT-NeoX-20B, InstructGPT, Latent diffusion / Stable Diffusion, PaLM, LLaMA, Llama 2, Mamba, Mistral 7B, Pythia, RetNet, DeepSeek-V2, DeepSeek-V3, DeepSeekMoE 16B, DeltaNet, Gated DeltaNet, Gemma 2, Jamba, Llama 3, Mamba-2, Mixtral 8x7B, Qwen2.5, DeepSeek-R1, Kimi Linear, Llama 4, MiniMax-01, Nemotron-H, Qwen3, Qwen3-Next, SmolLM2, SmolLM3, Engram-27B, Nemotron 3 Super | 11 |
| **dropout** | Randomly zero activations during training. | [Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html) | — | AlexNet, Transformer, BERT, GPT, GPT-2, RoBERTa, T5 | 11 |

### objective

| block | what it does | introduced in | first model | used by | module |
|---|---|---|---|---|---|
| **next-token prediction** | Predict the next token from the ones before. | [Bengio et al., 2003](https://jmlr.org/papers/v3/bengio03a.html) | Neural probabilistic LM | Neural probabilistic LM, RNN encoder-decoder, RNNsearch (Bahdanau attention), seq2seq, Transformer, ELMo, GPT, GPT-2, GPT-3, Linear Transformer, BLOOM, GPT-NeoX-20B, InstructGPT, PaLM, LLaMA, Llama 2, Mamba, Mistral 7B, Pythia, RetNet, DeepSeek-V2, DeepSeek-V3, DeepSeekMoE 16B, DeltaNet, Gated DeltaNet, Gemma 2, Jamba, Llama 3, Mamba-2, Mixtral 8x7B, Qwen2.5, DeepSeek-R1, Kimi Linear, Llama 4, MiniMax-01, Nemotron-H, Qwen3, Qwen3-Next, SmolLM2, SmolLM3, Engram-27B, Nemotron 3 Super | 14 |
| **masked language modeling** | Hide some tokens and predict them from both sides. | [Devlin et al., 2018](https://arxiv.org/abs/1810.04805) | BERT | BERT, RoBERTa | 32 |
| **span corruption** | Replace spans with sentinels; decode the spans. | [Raffel et al., 2019](https://arxiv.org/abs/1910.10683) | T5 | T5, Switch Transformer | — |
| **contrastive image-text** | Pull matching image/caption pairs together, push the rest apart. | [Radford et al., 2021](https://arxiv.org/abs/2103.00020) | CLIP | CLIP | 33 |
| **denoising diffusion** | Learn to remove noise, step by step. | [Ho, Jain, Abbeel, 2020](https://arxiv.org/abs/2006.11239) | DDPM | DDPM, Latent diffusion / Stable Diffusion | 33 |
| **latent diffusion** | Diffuse in an autoencoder's latent space. | [Rombach et al., 2021](https://arxiv.org/abs/2112.10752) | Latent diffusion / Stable Diffusion | Latent diffusion / Stable Diffusion | 33 |
| **multi-token prediction** | Also predict tokens two or more steps ahead. | [Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737) | — | DeepSeek-V3, DeepSeek-R1, Qwen3-Next, Nemotron 3 Super | 17 |
| **deep Q-learning** | Learn action values from rewards with a network. | [Mnih et al., 2013](https://arxiv.org/abs/1312.5602) | DQN | DQN | 34 |
| **self-play + MCTS** | Train policy and value from games against itself, with tree search as the teacher. | [Silver et al., 2017](https://doi.org/10.1038/nature24270) | AlphaGo Zero | AlphaGo Zero, AlphaZero, MuZero | 34 |
| **planning with a learned model** | Search inside a model of the game the network learned itself. | [Schrittwieser et al., 2019](https://arxiv.org/abs/1911.08265) | MuZero | MuZero | 34 |
| **knowledge distillation** | Train on a bigger teacher's next-token distribution instead of (or as well as) the one true token. | [Hinton, Vinyals, Dean, 2015](https://arxiv.org/abs/1503.02531) | — | Gemma 2 | 42 |

### post-training

| block | what it does | introduced in | first model | used by | module |
|---|---|---|---|---|---|
| **RLHF** | Fit a reward model to human preferences, then optimize against it. | [Christiano et al., 2017](https://arxiv.org/abs/1706.03741) | — | InstructGPT, Llama 2 | 35 |
| **DPO** | Optimize preferences directly, no reward model or RL loop. | [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) | — | Llama 3, Qwen2.5, Llama 4, SmolLM2 | 35 |
| **RLVR with GRPO** | RL against verifiable rewards, with group-relative advantages instead of a critic. | [Shao et al., 2024](https://arxiv.org/abs/2402.03300) | — | DeepSeek-V3, Qwen2.5, DeepSeek-R1, Qwen3 | 26 |

## The lineage: models by family

*New to its lineage* lists the blocks a model uses that none of its ancestors use (module 00b's `new_blocks`). That is not the same as inventing them: a block can enter a lineage long after its paper.

### RNN & embeddings

| model | year | by | builds on | new to its lineage | why it matters | module |
|---|---|---|---|---|---|---|
| [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735) | 1997 | Hochreiter & Schmidhuber | — | LSTM | Gated memory that keeps gradients alive over long sequences. | 31 |
| [Neural probabilistic LM](https://jmlr.org/papers/v3/bengio03a.html) | 2003 | Bengio et al. | — | next-token prediction | The first neural language model: learned word vectors predict the next word. | 14 |
| [word2vec](https://arxiv.org/abs/1301.3781) | 2013 | Google | Neural probabilistic LM | skip-gram embeddings | Cheap, good word vectors at scale. | 30 |
| [RNN encoder-decoder](https://arxiv.org/abs/1406.1078) | 2014 | Cho et al. | LSTM | next-token prediction | Encode a sentence to a vector, decode a translation (and the GRU). | 31 |
| [RNNsearch (Bahdanau attention)](https://arxiv.org/abs/1409.0473) | 2014 | Bahdanau, Cho, Bengio | RNN encoder-decoder | additive attention | Attention: the decoder looks back at every encoder state. | 31 |
| [seq2seq](https://arxiv.org/abs/1409.3215) | 2014 | Google | LSTM | next-token prediction | Deep LSTMs translating end to end. | 31 |
| [ELMo](https://arxiv.org/abs/1802.05365) | 2018 | AllenNLP | LSTM, word2vec | — | Contextual word vectors from a bidirectional LSTM language model. | 32 |

### CNN

| model | year | by | builds on | new to its lineage | why it matters | module |
|---|---|---|---|---|---|---|
| [LeNet-5](https://doi.org/10.1109/5.726791) | 1998 | LeCun et al. | — | convolution | Convolutional networks reading handwritten digits. | 12 |
| [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) | 2012 | Krizhevsky, Sutskever, Hinton | LeNet-5 | dropout | A deep CNN on two GPUs wins ImageNet: the deep-learning revival. | 12 |
| [ResNet](https://arxiv.org/abs/1512.03385) | 2015 | Microsoft Research | AlexNet | residual connections, BatchNorm | Residual connections make 100+ layer networks trainable. | 11 |

### RL & games

| model | year | by | builds on | new to its lineage | why it matters | module |
|---|---|---|---|---|---|---|
| [DQN](https://arxiv.org/abs/1312.5602) | 2013 | DeepMind | AlexNet | deep Q-learning | One network learns many Atari games from pixels and score. | 34 |
| [AlphaGo](https://doi.org/10.1038/nature16961) | 2016 | DeepMind | DQN | — | Policy and value networks plus tree search beat a Go champion. | 34 |
| [AlphaGo Zero](https://doi.org/10.1038/nature24270) | 2017 | DeepMind | AlphaGo, ResNet | self-play + MCTS | No human games: self-play with search as the teacher. | 34 |
| [AlphaZero](https://arxiv.org/abs/1712.01815) | 2017 | DeepMind | AlphaGo Zero | — | The same algorithm masters chess, shogi and Go. | 34 |
| [MuZero](https://arxiv.org/abs/1911.08265) | 2019 | DeepMind | AlphaZero | planning with a learned model | Plans without being told the rules. | 34 |

### encoder

| model | year | by | builds on | new to its lineage | why it matters | module |
|---|---|---|---|---|---|---|
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 | Google | Transformer, ELMo, GPT | masked language modeling | Bidirectional pretraining with masked tokens; fine-tune for any task. | 32 |
| [RoBERTa](https://arxiv.org/abs/1907.11692) | 2019 | Facebook AI | BERT | byte-level BPE | BERT trained longer, on more data, without next-sentence prediction. | 32 |
| [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) | 2020 | Google | Transformer, ResNet | patch embeddings, learned absolute positions, pre-norm, GELU | A plain Transformer on image patches matches CNNs at scale. | — |

### encoder-decoder

| model | year | by | builds on | new to its lineage | why it matters | module |
|---|---|---|---|---|---|---|
| [Transformer](https://arxiv.org/abs/1706.03762) | 2017 | Google | RNNsearch (Bahdanau attention) | multi-head self-attention, sinusoidal positions, post-norm, LayerNorm, ReLU feed-forward, residual connections, dropout, BPE subwords | Attention alone, no recurrence: the architecture everything after builds on. | 15 |
| [T5](https://arxiv.org/abs/1910.10683) | 2019 | Google | Transformer, BERT | relative position bias, pre-norm, span corruption, tied input/output embeddings | Every task as text-to-text, with relative position biases. | 32 |
| [GShard](https://arxiv.org/abs/2006.16668) | 2020 | Google | Transformer | top-k mixture of experts | Top-2 mixture-of-experts Transformers sharded across 2,048 TPU cores. | 24 |
| [Switch Transformer](https://arxiv.org/abs/2101.03961) | 2021 | Google | T5, GShard | top-1 routing | Route each token to a single expert: trillion-parameter sparse models. | 17 |

### decoder

| model | year | by | builds on | new to its lineage | why it matters | module |
|---|---|---|---|---|---|---|
| [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | 2018 | OpenAI | Transformer | learned absolute positions, GELU | Generative pretraining of a decoder-only Transformer, then fine-tuning. | 16 |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | OpenAI | GPT | pre-norm, byte-level BPE, tied input/output embeddings | Scale plus byte-level BPE and pre-norm; zero-shot tasks emerge. | 16 |
| [GPT-3](https://arxiv.org/abs/2005.14165) | 2020 | OpenAI | GPT-2 | — | 175B parameters; tasks from a few examples in the prompt. | 20 |
| [BLOOM](https://arxiv.org/abs/2211.05100) | 2022 | BigScience | GPT-3 | ALiBi | Open, multilingual 176B model; ALiBi instead of position embeddings. | — |
| [GPT-NeoX-20B](https://arxiv.org/abs/2204.06745) | 2022 | EleutherAI | GPT-3 | partial RoPE | Open 20B model; rotary positions on a quarter of each head. | — |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | 2022 | OpenAI | GPT-3 | RLHF | Supervised instructions plus RLHF: the recipe behind ChatGPT. | 35 |
| [PaLM](https://arxiv.org/abs/2204.02311) | 2022 | Google | GPT-3, T5 | multi-query attention, RoPE, SwiGLU | 540B parameters with SwiGLU, RoPE and multi-query attention. | — |
| [LLaMA](https://arxiv.org/abs/2302.13971) | 2023 | Meta | GPT-3, PaLM, GPT-NeoX-20B | RMSNorm | Open weights combining pre-norm RMSNorm, SwiGLU and RoPE: the recipe most open models copy. | 17 |
| [Llama 2](https://arxiv.org/abs/2307.09288) | 2023 | Meta | LLaMA | grouped-query attention, RLHF | Longer context, GQA in the 34B/70B models, RLHF-tuned chat models. | 17 |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 2023 | Mistral AI | Llama 2 | sliding-window attention | A 7B model beating Llama 2 13B, with GQA and sliding-window attention. | — |
| [Pythia](https://arxiv.org/abs/2304.01373) | 2023 | EleutherAI | GPT-NeoX-20B | — | A suite of models with every checkpoint public, for studying training. | — |
| [Gemma 2](https://arxiv.org/abs/2408.00118) | 2024 | Google | PaLM, LLaMA | grouped-query attention, sliding-window attention, knowledge distillation | The 2B and 9B were trained by distillation from a larger model, on 50× more tokens than compute-optimal. | 42 |
| [Llama 3](https://arxiv.org/abs/2407.21783) | 2024 | Meta | Llama 2 | DPO | GQA at every size, a 128K vocabulary, RoPE base 500,000, DPO in post-training. | 17 |
| [Qwen2.5](https://arxiv.org/abs/2412.15115) | 2024 | Alibaba | LLaMA | grouped-query attention, DPO, RLVR with GRPO | A strong open family (0.5B–72B): GQA, bias on QKV, DPO then GRPO. | 19 |
| [SmolLM2](https://arxiv.org/abs/2502.02737) | 2025 | Hugging Face | LLaMA | grouped-query attention, DPO | Small models (135M–1.7B) from carefully staged data; the models you run in Modules 18–21. | 18 |
| [SmolLM3](https://huggingface.co/blog/smollm3) | 2025 | Hugging Face | SmolLM2, Llama 3 | RoPE/NoPE hybrid layers | 3B, long context; drops RoPE from every 4th layer. | 17 |

### MoE decoder

| model | year | by | builds on | new to its lineage | why it matters | module |
|---|---|---|---|---|---|---|
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 2024 | DeepSeek | DeepSeekMoE 16B | multi-head latent attention | Multi-head latent attention shrinks the KV cache. | 17 |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 2024 | DeepSeek | DeepSeek-V2 | aux-loss-free balancing, multi-token prediction, RLVR with GRPO | 671B MoE with aux-loss-free balancing and multi-token prediction; you replicated it in miniature. | 17 |
| [DeepSeekMoE 16B](https://arxiv.org/abs/2401.06066) | 2024 | DeepSeek | LLaMA, GShard | fine-grained + shared experts | Many small experts plus always-on shared ones. | 17 |
| [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) | 2024 | Mistral AI | Mistral 7B, GShard | — | Open sparse MoE: 8 experts, 2 per token, 13B of 47B parameters active. | — |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 2025 | DeepSeek | DeepSeek-V3 | — | Reasoning learned largely through RL on verifiable rewards. | 26 |
| [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | 2025 | Meta | Llama 3 | top-k mixture of experts, fine-grained + shared experts, RoPE/NoPE hybrid layers | Meta's first MoE: routed plus shared experts, no RoPE in every 4th layer, natively multimodal. | — |
| [Qwen3](https://arxiv.org/abs/2505.09388) | 2025 | Alibaba | Qwen2.5 | QK-Norm, top-k mixture of experts | Dense and MoE models with QK-Norm and switchable thinking. | — |
| [Engram-27B](https://arxiv.org/abs/2601.07372) | 2026 | DeepSeek | DeepSeek-V3 | hashed n-gram memory (Engram) | A research model (no released weights): part of the expert budget moved into hashed n-gram tables; also uses manifold-constrained hyper-connections. | 40 |

### SSM & hybrid

| model | year | by | builds on | new to its lineage | why it matters | module |
|---|---|---|---|---|---|---|
| [Linear Transformer](https://arxiv.org/abs/2006.16236) | 2020 | Idiap & EPFL | Transformer | linear attention | Attention without the softmax is an RNN: constant memory and time per generated token. | 41 |
| [S4](https://arxiv.org/abs/2111.00396) | 2021 | Stanford | — | structured state space (S4), residual connections | Structured state spaces handle sequences of thousands of steps. | 36 |
| [Mamba](https://arxiv.org/abs/2312.00752) | 2023 | CMU & Princeton | S4 | selective SSM, RMSNorm, pre-norm, next-token prediction | Input-dependent state spaces: attention-quality language modeling in linear time. | 36 |
| [RetNet](https://arxiv.org/abs/2307.08621) | 2023 | Microsoft | Linear Transformer | retention (fixed decay) | Linear attention with a fixed decay per head, trained in parallel and run as an RNN. | 41 |
| [DeltaNet](https://arxiv.org/abs/2406.06484) | 2024 | MIT & IBM | Linear Transformer | delta rule, RMSNorm | The delta rule made trainable at scale with a chunkwise algorithm. | 41 |
| [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | 2024 | NVIDIA & MIT | Mamba-2, DeltaNet | gated delta rule | Mamba-2's decay plus DeltaNet's overwrite: the linear layer most 2025–2026 hybrids use. | 41 |
| [Jamba](https://arxiv.org/abs/2403.19887) | 2024 | AI21 Labs | Mamba, Mixtral 8x7B | attention/SSM hybrid layers, NoPE | Mamba layers with attention every 8th layer and MoE, and no positional encoding. | 36 |
| [Mamba-2](https://arxiv.org/abs/2405.21060) | 2024 | CMU & Princeton | Mamba | state space duality (SSD) | State space duality: SSMs as structured attention, running on tensor cores. | 36 |
| [Kimi Linear](https://arxiv.org/abs/2510.26692) | 2025 | Moonshot AI | Gated DeltaNet, DeepSeek-V3 | Kimi Delta Attention, attention/SSM hybrid layers | Three Kimi Delta Attention layers per MLA layer: up to 75% less KV cache, 6× decoding at 1M tokens. | 41 |
| [MiniMax-01](https://arxiv.org/abs/2501.08313) | 2025 | MiniMax | Linear Transformer, Mixtral 8x7B | attention/SSM hybrid layers | 456B MoE with lightning (linear) attention and a softmax layer after every 7; M2 went back to full attention. | 41 |
| [Nemotron-H](https://arxiv.org/abs/2504.03624) | 2025 | NVIDIA | Mamba-2, Jamba | — | Mamba-2 with about 8% attention layers (4 of 52 in the 8B), up to 3× faster inference. | 41 |
| [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | 2025 | Alibaba | Qwen3, Gated DeltaNet | gated attention, attention/SSM hybrid layers, fine-grained + shared experts, multi-token prediction | Three Gated DeltaNet layers per gated attention layer, 512 experts, 3B of 80B active; Qwen3.5 kept it. | 41 |
| [Nemotron 3 Super](https://arxiv.org/abs/2604.12374) | 2026 | NVIDIA | Jamba, DeepSeek-V3 | state space duality (SSD), latent experts (LatentMoE) | Mamba-2 layers with a few attention layers, and LatentMoE: 512 experts in a 1,024-wide latent, top-22. | 40 |

### vision-language & diffusion

| model | year | by | builds on | new to its lineage | why it matters | module |
|---|---|---|---|---|---|---|
| [DDPM](https://arxiv.org/abs/2006.11239) | 2020 | UC Berkeley | — | denoising diffusion, residual connections | Denoising diffusion produces high-quality images. | 33 |
| [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | OpenAI | Vision Transformer (ViT), GPT-2 | contrastive image-text | Images and captions in one embedding space, trained on 400M pairs. | 33 |
| [Latent diffusion / Stable Diffusion](https://arxiv.org/abs/2112.10752) | 2022 | LMU Munich & Runway | DDPM, CLIP | latent diffusion | Diffusion in a compressed latent space, conditioned on text. | 33 |
