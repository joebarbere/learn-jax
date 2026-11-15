# Seminal papers, by module

The papers behind each module's ideas, with a line on why each matters. Read a module's papers after its lesson, once the ideas have clicked. Every link is checked by `tools/check_papers.py`.

## Part 0 — Setup

### 00 · Your GPU, the container, and the three ROCm settings

- **[Compiling machine learning programs via high-level tracing](https://mlsys.org/Conferences/doc/2018/146.pdf)** — Frostig, Johnson, Leary, 2018. The original JAX paper: trace Python into a graph, compile it with XLA.
- **[Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785)** — Williams, Waterman, Patterson, 2009. Compute-bound vs memory-bound in one picture; the lens for every speed measurement.

## Part 1 — Math in code

### 01 · Arrays: NumPy → jax.numpy, shapes, broadcasting, dtypes

- **[Array programming with NumPy](https://arxiv.org/abs/2006.10256)** — Harris et al., 2020. The array model (shapes, strides, broadcasting) that jax.numpy inherits.

### 02 · Linear algebra as geometry

- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** — Hu et al., 2021. Fine-tune by learning a low-rank update instead of all weights.
- **[Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)** — Aghajanyan, Zettlemoyer, Gupta, 2020. Why low-rank updates are enough: fine-tuning lives in a small subspace.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Vectors (Essence of linear algebra, ch. 1)](https://www.youtube.com/watch?v=fNk_zzaMoSs). Arrows, lists of numbers, and both.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Linear transformations and matrices (ch. 3)](https://www.youtube.com/watch?v=kYB8IZa5AuE). A matrix is where the basis vectors land.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Matrix multiplication as composition (ch. 4)](https://www.youtube.com/watch?v=XkY2DOUCWMU). Why multiplying matrices composes transformations.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [The determinant (ch. 6)](https://www.youtube.com/watch?v=Ip3X9LOh2dk). How much a transformation scales area.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Inverse matrices, column space and null space (ch. 7)](https://www.youtube.com/watch?v=uQhTuRlWMxw). Undoing a transformation, and when you can't.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Dot products and duality (ch. 9)](https://www.youtube.com/watch?v=LyGKycYT2v0). Why the dot product measures alignment.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Eigenvectors and eigenvalues (ch. 14)](https://www.youtube.com/watch?v=PFDu9oVAE-g). Directions a matrix only stretches.

### 03 · Derivatives and autodiff, built from scratch

- **[Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0)** — Rumelhart, Hinton, Williams, 1986. Backpropagation made multi-layer networks trainable.
- **[Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767)** — Baydin et al., 2015. Forward vs reverse mode, dual numbers, tapes: the theory behind jax.grad.
- *Companion ([Andrej Karpathy, *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)):* [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0). The same Value-class autodiff you built, live-coded.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [The essence of calculus (ch. 1)](https://www.youtube.com/watch?v=WUvTyaaNkzM). Derivatives and integrals, visually.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Visualizing the chain rule and product rule (calculus ch. 4)](https://www.youtube.com/watch?v=YG15m2VwSjA). The rule backprop is built on.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Backpropagation, intuitively (Deep learning ch. 3)](https://www.youtube.com/watch?v=Ilg3gGewQ5U). What backprop is really doing.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Backpropagation calculus (Deep learning ch. 4)](https://www.youtube.com/watch?v=tIeHLnjs5U8). The chain rule through a network.

### 04 · Probability, softmax, cross-entropy and JAX random keys

- **[A Mathematical Theory of Communication](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)** — Shannon, 1948. Entropy and cross-entropy start here; a language model's loss is measured in Shannon's bits.
- **[Parallel random numbers: as easy as 1, 2, 3](https://doi.org/10.1145/2063384.2063405)** — Salmon et al., 2011. Counter-based RNGs (Threefry): the design behind JAX's splittable random keys.
- **[Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)** — Jang, Gu, Poole, 2016. The Gumbel-max trick for sampling categories, made differentiable.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Reinventing Entropy (Compression is Intelligence, part 1)](https://www.youtube.com/watch?v=l6DKRf-fAAM). Entropy from first principles.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [But what is cross-entropy? (Compression is Intelligence, part 2)](https://www.youtube.com/watch?v=GlYgs6v2YfU). Cross-entropy as compression: the loss every language model minimizes.

### 05 · Optimization: gradient descent to Adam

- **[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)** — Kingma, Ba, 2014. Momentum + per-parameter step sizes + bias correction.
- **[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)** — Loshchilov, Hutter, 2017. AdamW: why weight decay is not L2 regularization under Adam.
- **[SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)** — Loshchilov, Hutter, 2016. Cosine learning-rate schedules.
- **[On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063)** — Pascanu, Mikolov, Bengio, 2012. Exploding gradients and gradient-norm clipping.
- **[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)** — Liu, Su et al. (Moonshot AI), 2025. Muon (orthogonalized momentum) at LLM scale: weight decay and update-RMS matching make it work.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Gradient descent, how neural networks learn (Deep learning ch. 2)](https://www.youtube.com/watch?v=IHZwWFHWa-w). Cost functions and walking downhill.

## Part 2 — JAX core

### 06 · jit and XLA: tracing, compilation, timing

- **[Compiling machine learning programs via high-level tracing](https://mlsys.org/Conferences/doc/2018/146.pdf)** — Frostig, Johnson, Leary, 2018. The original JAX paper: trace Python into a graph, compile it with XLA.
- **[Operator Fusion in XLA: Analysis and Evaluation](https://arxiv.org/abs/2301.13062)** — Snider, Liang, 2023. What XLA fuses, and why fusion is most of the speedup you measured.

### 07 · vmap, pytrees and functional state

- **[Efficient Per-Example Gradient Computations](https://arxiv.org/abs/1510.01799)** — Goodfellow, 2015. Per-example gradients without a loop, the job vmap(grad) does for free.
- **[Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)** — Abadi et al., 2016. DP-SGD: the best-known use of per-example gradient clipping.

### 08 · Control flow, scan, and mixed precision

- **[Mixed Precision Training](https://arxiv.org/abs/1710.03740)** — Micikevicius et al., 2017. fp16 compute, fp32 master weights, loss scaling: the original recipe.
- **[A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322)** — Kalamkar et al., 2019. Why bfloat16's fp32-sized exponent removes the need for loss scaling.

## Part 3 — Neural networks

### 09 · Linear and logistic regression from scratch

- **[Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://doi.org/10.1080/00401706.1970.10488634)** — Hoerl, Kennard, 1970. The original ridge (L2-regularized) regression.
- **[Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747)** — Xiao, Rasul, Vollgraf, 2017. The dataset used in Part 3.

### 10 · MLPs on Fashion-MNIST, then Flax NNX

- **[Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0)** — Rumelhart, Hinton, Williams, 1986. Backpropagation made multi-layer networks trainable.
- **[Approximation by superpositions of a sigmoidal function](https://doi.org/10.1007/BF02551274)** — Cybenko, 1989. The universal approximation theorem.
- **[Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)** — Hendrycks, Gimpel, 2016. The smooth activation used in GPT-2 and BERT.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [But what is a neural network? (Deep learning ch. 1)](https://www.youtube.com/watch?v=aircAruvnKk). Layers, weights and activations.

### 11 · Training craft: init, norms, schedules, checkpoints

- **[Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)** — Glorot, Bengio, 2010. Xavier initialization and activation statistics.
- **[Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)** — He et al., 2015. He initialization for ReLU nets.
- **[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)** — He et al., 2015. Residual connections: how very deep networks became trainable.
- **[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)** — Ioffe, Szegedy, 2015. Normalize over the batch; the layer that made deep CNNs easy to train (and awkward in JAX).
- **[Layer Normalization](https://arxiv.org/abs/1607.06450)** — Ba, Kiros, Hinton, 2016. Normalize over features, not the batch; standard in Transformers.
- **[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)** — Zhang, Sennrich, 2019. RMSNorm: LayerNorm without the mean; used by Llama and DeepSeek.
- **[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)** — Srivastava et al., 2014. Randomly drop units during training.
- *Companion ([Andrej Karpathy, *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)):* [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc). Activation/gradient statistics in deep nets, and BatchNorm.
- *Companion ([Andrej Karpathy, *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)):* [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI). Backpropagate by hand through cross-entropy, linear, tanh and BatchNorm layers.

### 12 · Convolutional networks

- **[Gradient-based learning applied to document recognition](https://doi.org/10.1109/5.726791)** — LeCun et al., 1998. LeNet: convolution + pooling, trained end to end.
- **[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)** — Krizhevsky, Sutskever, Hinton, 2012. AlexNet: CNNs on GPUs win ImageNet, the start of the deep-learning boom.
- **[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)** — van den Oord et al., 2016. Dilated causal convolutions: exponentially growing context for sequences.
- **[Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)** — Zeiler, Fergus, 2013. What learned filters and feature maps respond to.
- *Companion ([Andrej Karpathy, *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)):* [Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0). Deepening the MLP into a tree of dilated causal convolutions.

## Part 4 — Sequences, attention, transformers

### 13 · Tokenization: build BPE yourself

- **[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)** — Sennrich, Haddow, Birch, 2015. Byte-pair encoding for neural models.
- **[SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)** — Kudo, Richardson, 2018. Tokenization from raw text, no pre-splitting.
- **[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** — Radford et al., 2019. GPT-2: byte-level BPE and a scaled-up decoder-only Transformer.
- *Companion ([Andrej Karpathy, *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)):* [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE). Byte-level BPE from scratch, and the LLM quirks tokenization causes.
- *Companion ([Aditya Makkar, Divya Makkar & Chinmay Jindal, *Jaxformer*](https://jaxformer.com/)):* [Part 1: Tokenization at scale](https://jaxformer.com/tokenization/). Tokenizing FineWeb-EDU with multiprocessing and Ray.

### 14 · Language modeling basics: bigrams to MLP LMs

- **[A Neural Probabilistic Language Model](https://jmlr.org/papers/v3/bengio03a.html)** — Bengio et al., 2003. The MLP language model you built: embeddings + context window.
- **[Prediction and Entropy of Printed English](https://doi.org/10.1002/j.1538-7305.1951.tb01366.x)** — Shannon, 1951. Humans as language models; the entropy of English.
- **[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)** — Mikolov et al., 2013. word2vec: embeddings as a product in their own right.
- *Companion ([Andrej Karpathy, *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)):* [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo). Bigram language model, sampling and negative log-likelihood.
- *Companion ([Andrej Karpathy, *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)):* [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I). Bengio-style MLP language model, learning-rate tuning, splits.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [But what is cross-entropy? (Compression is Intelligence, part 2)](https://www.youtube.com/watch?v=GlYgs6v2YfU). A language model's loss as bits needed to compress text.

### 15 · Attention from scratch

- **[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)** — Bahdanau, Cho, Bengio, 2014. Attention is introduced, as alignment in translation.
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** — Vaswani et al., 2017. The Transformer: self-attention replaces recurrence. Everything in Part 4 builds on it.
- **[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)** — Dao et al., 2022. Attention reorganized around memory traffic, not FLOPs.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Attention in transformers, step-by-step (Deep learning ch. 6)](https://www.youtube.com/watch?v=eMlx5fFNoYc). Queries, keys, values and masking, visualized.

### 16 · Build and train a GPT on your GPU

- **[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)** — Radford et al., 2018. GPT-1: pre-train a decoder-only Transformer, then fine-tune.
- **[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** — Radford et al., 2019. GPT-2: byte-level BPE and a scaled-up decoder-only Transformer.
- **[Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)** — Press, Wolf, 2016. Weight tying between the embedding and the output head.
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** — Kaplan et al., 2020. Loss falls as a power law in parameters, data and compute.
- **[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)** — Hoffmann et al., 2022. Scale tokens with parameters (~20 tokens/param) for a fixed compute budget.
- *Companion ([Andrej Karpathy, *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)):* [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY). A character-level GPT on Shakespeare, built up from a bigram model.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Transformers, the tech behind LLMs (Deep learning ch. 5)](https://www.youtube.com/watch?v=wjZofJX0v4M). The whole GPT pipeline, visually.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs). The big picture in eight minutes.
- *Companion ([Andrej Karpathy, *nanochat*](https://github.com/karpathy/nanochat)):* [nanochat](https://github.com/karpathy/nanochat). nanoGPT's successor: the same GPT, plus tokenizer, SFT, RL, eval and a chat engine.

### 17 · Modern LLM architecture: RoPE, RMSNorm, SwiGLU, MLA, MoE

- **[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)** — Su et al., 2021. RoPE: encode position by rotating query/key feature pairs.
- **[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)** — Shazeer, 2020. SwiGLU feed-forward layers.
- **[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)** — Shazeer, 2019. Multi-query attention: share K/V across heads to shrink the cache.
- **[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)** — Ainslie et al., 2023. Grouped-query attention, the Llama 2/3 middle ground.
- **[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)** — Touvron et al., 2023. The RMSNorm + RoPE + SwiGLU recipe most open models copy.
- **[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)** — Llama Team, AI @ Meta, 2024. GQA, 128K vocabulary, RoPE base 500,000.
- **[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)** — Shazeer et al., 2017. Mixture-of-experts with a learned router.
- **[Switch Transformers](https://arxiv.org/abs/2101.03961)** — Fedus, Zoph, Shazeer, 2021. Top-1 routing and the auxiliary load-balancing loss.
- **[DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)** — Dai et al., 2024. Fine-grained routed experts plus always-on shared experts.
- **[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)** — DeepSeek-AI, 2024. Multi-head latent attention (MLA): a compressed KV cache.
- **[Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)** — Wang et al., 2024. Balance experts with a bias term instead of an extra loss.
- **[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)** — Gloeckle et al., 2024. Predict several future tokens per position.
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)** — DeepSeek-AI, 2024. MLA + DeepSeekMoE + aux-loss-free balancing + MTP: the model you replicated in miniature.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [How might LLMs store facts (Deep learning ch. 7)](https://www.youtube.com/watch?v=9-Jl0dxWQs8). What the MLP / feed-forward blocks (SwiGLU, experts) may be doing.
- *Companion ([Aditya Makkar, Divya Makkar & Chinmay Jindal, *Jaxformer*](https://jaxformer.com/)):* [Part 2: How to write the base model](https://jaxformer.com/base_model/). RMSNorm, RoPE and multi-latent attention in JAX.
- *Companion ([Aditya Makkar, Divya Makkar & Chinmay Jindal, *Jaxformer*](https://jaxformer.com/)):* [Part 6: Implementing mixture-of-experts layers](https://jaxformer.com/moe/). Sigmoid top-k routing, shared experts, load balancing.

## Part 5 — LLMs in practice

### 18 · Inference: KV cache, sampling, quantization

- **[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)** — Holtzman et al., 2019. Why greedy decoding repeats itself; nucleus (top-p) sampling.
- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)** — Leviathan, Kalman, Matias, 2022. A small draft model proposes, the big model verifies.
- **[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)** — Dettmers et al., 2022. Int8 inference and outlier features.
- **[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)** — Frantar et al., 2022. 4-bit weights with little loss, the basis of many local-model formats.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [All about Transformer inference](https://jax-ml.github.io/scaling-book/inference). Prefill vs decode, KV-cache costs, batching, latency.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [Serving LLaMA 3-70B](https://jax-ml.github.io/scaling-book/applied-inference). The inference math applied to a real model.

### 19 · Local serving for coding assistants

- **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)** — Kwon et al., 2023. vLLM: paging the KV cache to serve many requests.
- **[Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)** — Pope et al., 2022. Prefill vs decode, latency vs throughput, and where the memory goes.
- **[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)** — Lin et al., 2023. 4-bit weight quantization guided by activations.

### 20 · Fine-tuning and LoRA

- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** — Hu et al., 2021. Fine-tune by learning a low-rank update instead of all weights.
- **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)** — Dettmers et al., 2023. LoRA on a 4-bit base model: fine-tuning large models on one GPU.
- **[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)** — Houlsby et al., 2019. Adapters: the parameter-efficient fine-tuning idea before LoRA.
- **[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)** — Ouyang et al., 2022. InstructGPT: supervised fine-tuning then RLHF.

### 21 · Evaluating language models

- **[Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)** — Hendrycks et al., 2020. MMLU, the multiple-choice benchmark everyone reports.
- **[Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)** — Liang et al., 2022. HELM: evaluate many metrics across many scenarios.
- **[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)** — Zheng et al., 2023. Using a strong model to grade answers, and its biases.
- *Companion ([Andrej Karpathy, *nanochat*](https://github.com/karpathy/nanochat)):* [nanochat's evaluation (base_eval, chat_eval)](https://github.com/karpathy/nanochat). CORE score, bits per byte, ARC, GSM8K, MMLU in one harness.

## Part 6 — Performance & ROCm

### 22 · Profiling JAX on ROCm

- **[Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785)** — Williams, Waterman, Patterson, 2009. Compute-bound vs memory-bound in one picture; the lens for every speed measurement.
- **[Data Movement Is All You Need: A Case Study on Optimizing Transformers](https://arxiv.org/abs/2007.00072)** — Ivanov et al., 2020. Transformers are often memory-bound; measure data movement, not FLOPs.
- **[Operator Fusion in XLA: Analysis and Evaluation](https://arxiv.org/abs/2301.13062)** — Snider, Liang, 2023. How XLA's fusion decisions show up in a profile.
- *Companion ([Andrej Karpathy, *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)):* [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU). Make GPT training fast step by step: precision, compilation, flash attention, nice numbers.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [All about rooflines](https://jax-ml.github.io/scaling-book/roofline). Compute vs memory vs communication bounds, worked out.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [How to profile TPU programs](https://jax-ml.github.io/scaling-book/profiling). Reading JAX profiler traces (TPU-focused, same ideas on GPU).

### 23 · Kernels: Pallas, HIP and fusion

- **[Triton: an intermediate language and compiler for tiled neural network computations](https://doi.org/10.1145/3315508.3329973)** — Tillet, Kung, Cox, 2019. The tile-based kernel language behind Pallas's GPU backend and XLA's GEMM fusions.
- **[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)** — Dao et al., 2022. Attention reorganized around memory traffic, not FLOPs.
- **[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)** — Dao, 2023. Better work partitioning across the GPU.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [How to think about GPUs](https://jax-ml.github.io/scaling-book/gpus). SMs, memory hierarchy and collectives from the scaling-book view.

## Part 7 — Scale & cloud

### 24 · Parallelism and sharding

- **[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)** — Shoeybi et al., 2019. Tensor parallelism inside Transformer layers.
- **[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)** — Rajbhandari et al., 2019. Shard optimizer state, gradients and params: the idea behind FSDP.
- **[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)** — Lepikhin et al., 2020. Sharding annotations + MoE expert parallelism.
- **[GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663)** — Xu et al., 2021. The XLA partitioner that turns JAX's sharding specs into collectives.
- *Companion ([Aditya Makkar, Divya Makkar & Chinmay Jindal, *Jaxformer*](https://jaxformer.com/)):* [Part 3: Inside the sharded model](https://jaxformer.com/sharded/). Mesh, shard_map, FSDP, GPipe pipelines and tensor parallelism.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [Sharded matrices and how to multiply them](https://jax-ml.github.io/scaling-book/sharding). Which collective each sharded matmul needs.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [Programming TPUs in JAX](https://jax-ml.github.io/scaling-book/jax-stuff). Explicit sharding and shard_map from the JAX side.
- *Companion ([Hugging Face nanotron team, *The Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook)):* [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook). DP, ZeRO/FSDP, TP, PP, context and expert parallelism, measured on GPU clusters.

### 25 · JAX on AWS, and the capstone

- **[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)** — Hoffmann et al., 2022. Scale tokens with parameters (~20 tokens/param) for a fixed compute budget.
- **[Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)** — Pope et al., 2022. Cost per token for serving, the number to compare across hardware.
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)** — DeepSeek-AI, 2024. MLA + DeepSeekMoE + aux-loss-free balancing + MTP: the model you replicated in miniature.
- *Companion ([Aditya Makkar, Divya Makkar & Chinmay Jindal, *Jaxformer*](https://jaxformer.com/)):* [Part 4: Dataset class and config files](https://jaxformer.com/dataset/). Sharded data placement.
- *Companion ([Aditya Makkar, Divya Makkar & Chinmay Jindal, *Jaxformer*](https://jaxformer.com/)):* [Part 5: Distributed training](https://jaxformer.com/distributed_training/). jax.distributed, Orbax checkpoints, the train loop.
- *Companion ([Aditya Makkar, Divya Makkar & Chinmay Jindal, *Jaxformer*](https://jaxformer.com/)):* [Part 7: Final run and training results](https://jaxformer.com/training/). A 949M-param MoE on 32 TPU-v4 chips, multi-host.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [Training LLaMA 3 on TPUs](https://jax-ml.github.io/scaling-book/applied-training). Choosing a parallelism layout for a real run.
- *Companion ([Andrej Karpathy, *nanochat*](https://github.com/karpathy/nanochat)):* [nanochat](https://github.com/karpathy/nanochat). The capstone's blueprint: every stage of a small chat model, end to end.
- *Companion ([Andrej Karpathy, *nanochat*](https://github.com/karpathy/nanochat)):* [Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481). Karpathy's write-up of what made it fast.

## Part 8 — Reasoning

### 26 · Reasoning models: inference scaling and RLVR with GRPO

- **[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)** — Wei et al., 2022. Ask for the steps, get better answers: the start of reasoning prompts.
- **[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)** — Wang et al., 2022. Sample many reasoning paths and majority-vote the answer.
- **[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)** — Cobbe et al., 2021. GSM8K and verifier-reranked sampling: checkable answers as a training signal.
- **[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)** — Lightman et al., 2023. Process vs outcome supervision: rewarding steps instead of only final answers.
- **[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)** — Schulman et al., 2017. The clipped policy-ratio objective GRPO inherits.
- **[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)** — Shao et al., 2024. Introduces GRPO: group-relative advantages, no value network.
- **[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)** — DeepSeek-AI, 2025. RL with verifiable rewards produces long reasoning; R1-Zero, distillation.
- **[s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)** — Muennighoff et al., 2025. Budget forcing: controlling how long a model thinks at inference time.
- *Companion ([Sebastian Raschka, *Build a Reasoning Model (From Scratch)*](https://magazine.sebastianraschka.com/p/building-reasoning-models)):* [Build A Reasoning Model From Scratch 3: The Verifier for Evaluation and RL with Verifiable Rewards](https://www.youtube.com/watch?v=JQJ_8_jSAoY). The checker that is also the reward.
- *Companion ([Sebastian Raschka, *Build a Reasoning Model (From Scratch)*](https://magazine.sebastianraschka.com/p/building-reasoning-models)):* [Build A Reasoning Model From Scratch 4: Inference Scaling 1 (Temperature, Top-p, Self-Consistency)](https://www.youtube.com/watch?v=t5y-kS9nNxU). Sampling and majority voting.
- *Companion ([Sebastian Raschka, *Build a Reasoning Model (From Scratch)*](https://magazine.sebastianraschka.com/p/building-reasoning-models)):* [Build A Reasoning Model From Scratch 5: Inference Scaling 2 (Logprob Scoring, Self-Refinement)](https://www.youtube.com/watch?v=TVMyOJ_3Gxo). Scoring and critique-then-revise.
- *Companion ([Sebastian Raschka, *Build a Reasoning Model (From Scratch)*](https://magazine.sebastianraschka.com/p/building-reasoning-models)):* [Build A Reasoning Model From Scratch 6: Reinforcement Learning 1 (Implementing GRPO for RLVR)](https://www.youtube.com/watch?v=237Hf7Q3lgg). GRPO from scratch.
- *Companion ([Sebastian Raschka, *Build a Reasoning Model (From Scratch)*](https://magazine.sebastianraschka.com/p/building-reasoning-models)):* [Build A Reasoning Model From Scratch 7: Reinforcement Learning 2 (Tracking, KL Terms, And More)](https://www.youtube.com/watch?v=VxfUrUrrczs). Clipping, KL, format rewards, reward hacking.
- *Companion ([Sebastian Raschka, *Build a Reasoning Model (From Scratch)*](https://magazine.sebastianraschka.com/p/building-reasoning-models)):* [Build A Reasoning Model From Scratch 8: Distillation](https://www.youtube.com/watch?v=msp69FR5NZs). Teaching a small model from a reasoning model's answers.
- *Companion ([Andrej Karpathy, *nanochat*](https://github.com/karpathy/nanochat)):* [nanochat's chat_rl stage](https://github.com/karpathy/nanochat). RL on GSM8K after SFT, the same idea as this module at larger scale.

## Part 9 — Using your models

### 27 · Your models in LM Studio: export, convert, serve

- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** — Hu et al., 2021. Fine-tune by learning a low-rank update instead of all weights.
- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)** — Leviathan, Kalman, Matias, 2022. The draft-and-verify scheme behind LM Studio's speculative decoding.
- **[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)** — Frantar et al., 2022. Why 4-bit weights lose so little: the idea behind GGUF's k-quants.
