# Seminal papers, by module

The papers behind each module's ideas, with a line on why each matters. Read a module's papers after its lesson, once the ideas have clicked. Every link is checked by `tools/check_papers.py`.

## Part 0 — Setup

### 00 · Your GPU, the container, and the three ROCm settings

- **[Compiling machine learning programs via high-level tracing](https://mlsys.org/Conferences/doc/2018/146.pdf)** — Frostig, Johnson, Leary, 2018. The original JAX paper: trace Python into a graph, compile it with XLA.
- **[Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785)** — Williams, Waterman, Patterson, 2009. Compute-bound vs memory-bound in one picture; the lens for every speed measurement.

### 00b · A short history of AI, and where each idea lives in this course

- **[A logical calculus of the ideas immanent in nervous activity](https://doi.org/10.1007/BF02478259)** — McCulloch, Pitts, 1943. The first mathematical model of a neuron.
- **[Computing Machinery and Intelligence](https://doi.org/10.1093/mind/LIX.236.433)** — Turing, 1950. Can machines think? The imitation game.
- **[A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence](http://www-formal.stanford.edu/jmc/history/dartmouth/dartmouth.html)** — McCarthy, Minsky, Rochester, Shannon, 1955. The document that named the field.
- **[The perceptron: a probabilistic model for information storage and organization in the brain](https://doi.org/10.1037/h0042519)** — Rosenblatt, 1958. The first learning machine for patterns.
- **[Deep learning](https://doi.org/10.1038/nature14539)** — LeCun, Bengio, Hinton, 2015. The field's own summary of how deep learning came back.
- **[Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)** — Gehring et al., 2017. Family tree: learned absolute position embeddings, later used by GPT and BERT.
- **[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)** — Dosovitskiy et al., 2020. Family tree: ViT, the Transformer on image patches.
- **[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)** — Beltagy, Peters, Cohan, 2020. Family tree: sliding-window attention, later used by Mistral.
- **[Query-Key Normalization for Transformers](https://arxiv.org/abs/2010.04245)** — Henry et al., 2020. Family tree: normalize queries and keys before attention (QK-Norm, used by Qwen3).
- **[GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745)** — Black et al., 2022. Family tree: an open 20B model with partial rotary embeddings.

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
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [All the Transformer math you need to know](https://jax-ml.github.io/scaling-book/transformers). FLOPs, parameter counts and KV-cache sizes — the reference for this module's FLOPs/token and MFU.
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
- **[Transformer Language Models without Positional Encodings Still Learn Positional Information](https://arxiv.org/abs/2203.16634)** — Haviv et al., 2022. NoPE: the causal mask alone leaks position.
- **[The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466)** — Kazemnejad et al., 2023. NoPE vs RoPE, ALiBi and T5's bias on lengths longer than training.
- **[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)** — Press, Smith, Lewis, 2021. ALiBi: a distance penalty on scores instead of rotation.
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)** — DeepSeek-AI, 2024. MLA + DeepSeekMoE + aux-loss-free balancing + MTP: the model you replicated in miniature.
- *Companion ([Grant Sanderson, 3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks)):* [How might LLMs store facts (Deep learning ch. 7)](https://www.youtube.com/watch?v=9-Jl0dxWQs8). What the MLP / feed-forward blocks (SwiGLU, experts) may be doing.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [All the Transformer math you need to know](https://jax-ml.github.io/scaling-book/transformers). FLOPs, parameter counts and KV-cache sizes — including the KV-cache accounting behind GQA and MLA.
- *Companion ([Sebastian Raschka, *Ahead of AI*](https://magazine.sebastianraschka.com/)):* [Understanding Positional Embeddings in Modern LLMs](https://magazine.sebastianraschka.com/p/understanding-positional-embeddings). Learned, sinusoidal, RoPE, partial RoPE, NoPE and hybrids, with a 2026 open-weight model survey.
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
- **[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)** — Brown et al., 2020. GPT-3 and in-context learning: tasks from examples in the prompt, no training (section 13).
- **[Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)** — Ilharco et al., 2022. Task vectors: add fine-tuning deltas to combine skills, subtract them to remove one (section 12).
- **[TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)** — Yadav et al., 2023. Why naive sums interfere, and a fix: trim small changes, resolve sign conflicts, then merge.
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
- **[Decoupling algorithms from schedules for easy optimization of image processing pipelines](https://doi.org/10.1145/2185520.2185528)** — Ragan-Kelley, Adams, Paris et al., 2012. Who decides the schedule: Halide splits *what* a kernel computes from *how* it runs.
- **[Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines](https://doi.org/10.1145/2491956.2462176)** — Ragan-Kelley, Barnes, Adams et al., 2013. The Halide compiler, with schedules searched automatically: the line XLA and Pallas descend from.
- **[TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)** — Chen, Moreau et al., 2018. Halide's algorithm/schedule split applied to deep learning, with learned tuning.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [How to think about GPUs](https://jax-ml.github.io/scaling-book/gpus). SMs, memory hierarchy and collectives from the scaling-book view.
- *Companion ([Csongor Kiss, *Write your own kernel optimiser*](https://blog.csongor.co.uk/cachet-grids-and-tiles/)):* [Write your own kernel optimiser](https://blog.csongor.co.uk/cachet-grids-and-tiles/). A matmul climbing from 1% of OpenBLAS to parity by checked schedule rewrites (the Cachet language) — this module's speed ladder at the loop level.

## Part 7 — Scale & cloud

### 24 · Parallelism and sharding

- **[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)** — Shoeybi et al., 2019. Tensor parallelism inside Transformer layers.
- **[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)** — Rajbhandari et al., 2019. Shard optimizer state, gradients and params: the idea behind FSDP.
- **[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)** — Lepikhin et al., 2020. Sharding annotations + MoE expert parallelism.
- **[GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663)** — Xu et al., 2021. The XLA partitioner that turns JAX's sharding specs into collectives.
- *Companion ([Aditya Makkar, Divya Makkar & Chinmay Jindal, *Jaxformer*](https://jaxformer.com/)):* [Part 3: Inside the sharded model](https://jaxformer.com/sharded/). Mesh, shard_map, FSDP, GPipe pipelines and tensor parallelism.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [Sharded matrices and how to multiply them](https://jax-ml.github.io/scaling-book/sharding). Which collective each sharded matmul needs.
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [How to parallelize a Transformer for training](https://jax-ml.github.io/scaling-book/training). FSDP, Megatron tensor parallelism and pipelines, and when each becomes communication-bound.
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
- *Companion ([Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/)):* [How to think about GPUs](https://jax-ml.github.io/scaling-book/gpus). NVLink, InfiniBand and GPU collectives: the networking behind multi-GPU runs.
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
- **[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)** — Yao et al., 2023. Reasoning as search over partial solutions (section 7).
- **[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)** — Yao et al., 2022. Interleave thinking with tool calls: the agent loop of Module 39.
- **[Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)** — Chen et al., 2021. RL as next-token prediction over (return, state, action) sequences.
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

## Part 10 — Milestones, rebuilt in JAX

### 29 · Perceptron to Hopfield networks (1958–1982)

- **[The perceptron: a probabilistic model for information storage and organization in the brain](https://doi.org/10.1037/h0042519)** — Rosenblatt, 1958. The first learning machine for patterns.
- **[Neural networks and physical systems with emergent collective computational abilities](https://doi.org/10.1073/pnas.79.8.2554)** — Hopfield, 1982. Associative memory as energy minimization; cited by the 2024 physics Nobel.
- **[Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)** — Ramsauer et al., 2020. Modern Hopfield networks: their update rule is attention.

### 30 · Word embeddings: word2vec (2013)

- **[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)** — Mikolov et al., 2013. word2vec's CBOW and skip-gram.
- **[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)** — Mikolov et al., 2013. Negative sampling and the famous analogies.
- **[GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/)** — Pennington, Socher, Manning, 2014. Embeddings from co-occurrence counts.

### 31 · RNNs, LSTMs and seq2seq with attention (1986–2014)

- **[Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1)** — Elman, 1990. The simple recurrent network.
- **[Learning long-term dependencies with gradient descent is difficult](https://doi.org/10.1109/72.279181)** — Bengio, Simard, Frasconi, 1994. Why gradients vanish through time.
- **[Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735)** — Hochreiter, Schmidhuber, 1997. Gated memory cells that let gradients flow.
- **[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)** — Cho et al., 2014. The GRU and the encoder-decoder.
- **[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)** — Sutskever, Vinyals, Le, 2014. seq2seq with LSTMs.
- **[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)** — Bahdanau, Cho, Bengio, 2014. Attention is born here.

### 32 · BERT vs GPT: encoders and decoders (2018)

- **[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)** — Peters et al., 2018. ELMo: context-dependent embeddings.
- **[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)** — Devlin et al., 2018. Masked language modeling with an encoder.
- **[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)** — Liu et al., 2019. BERT, trained properly.
- **[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)** — Raffel et al., 2019. T5: the encoder-decoder answer, every task as text-to-text.
- **[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** — Radford et al., 2019. GPT-2: byte-level BPE and a scaled-up decoder-only Transformer.

### 33 · GANs, VAEs, diffusion and CLIP (2014–2022)

- **[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)** — Goodfellow et al., 2014. Two networks in a game.
- **[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)** — Kingma, Welling, 2013. The VAE.
- **[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)** — Radford, Metz, Chintala, 2015. DCGAN: GANs that train.
- **[Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)** — Sohl-Dickstein et al., 2015. The first diffusion model.
- **[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)** — Ho, Jain, Abbeel, 2020. DDPM: diffusion that works.
- **[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)** — Radford et al., 2021. CLIP: contrastive image-text learning.
- **[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)** — Rombach et al., 2021. Latent diffusion, the basis of Stable Diffusion.

### 34 · Deep RL: DQN to AlphaZero (2013–2019)

- **[Q-learning](https://doi.org/10.1007/BF00992698)** — Watkins, Dayan, 1992. Learning action values from experience.
- **[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)** — Mnih et al., 2013. DQN.
- **[Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236)** — Mnih et al., 2015. DQN at human level on Atari.
- **[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)** — van Hasselt, Guez, Silver, 2016. Double DQN: pick the next action with one network, score it with the other.
- **[Mastering the game of Go with deep neural networks and tree search](https://doi.org/10.1038/nature16961)** — Silver et al., 2016. AlphaGo.
- **[Mastering the game of Go without human knowledge](https://doi.org/10.1038/nature24270)** — Silver et al., 2017. AlphaGo Zero: self-play only.
- **[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)** — Silver et al., 2017. AlphaZero.
- **[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)** — Schrittwieser et al., 2019. MuZero: planning without the rules.
- **[Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning](https://arxiv.org/abs/2303.17503)** — Koyamada et al., 2023. The JAX game environments used here.

### 35 · Learning from preferences: RLHF and DPO (2017–2023)

- **[Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons](https://doi.org/10.2307/2334029)** — Bradley, Terry, 1952. The preference model inside every reward model.
- **[Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)** — Christiano et al., 2017. RLHF begins.
- **[Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)** — Stiennon et al., 2020. RLHF on language models.
- **[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)** — Ouyang et al., 2022. InstructGPT.
- **[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)** — Bai et al., 2022. Preferences from AI feedback.
- **[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)** — Rafailov et al., 2023. DPO: preferences without RL.

### 36 · State-space models and Mamba (2021–2023)

- **[Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)** — Gu, Goel, Ré, 2021. S4.
- **[Resurrecting Recurrent Neural Networks for Long Sequences](https://arxiv.org/abs/2303.06349)** — Orvieto et al., 2023. The linear recurrent unit.
- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)** — Gu, Dao, 2023. Selective SSMs.
- **[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)** — Dao, Gu, 2024. Mamba-2 and the SSM-attention link.
- **[Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)** — Blelloch, 1990. The parallel scan behind jax.lax.associative_scan.
- **[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927)** — Arora et al., 2023. Multi-query associative recall: where fixed-state models fall behind attention.
- **[Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)** — Lieber et al., 2024. Interleave a few attention layers into a Mamba stack (1:7), plus MoE: the hybrid in section 9.

### 37 · Interpretability: induction heads and world models (2020–2023)

- **[Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/)** — Olah et al., 2020. Reading neural networks like circuits.
- **[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)** — Elhage et al., 2021. Attention heads as readable components.
- **[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)** — Olsson et al., 2022. The circuit behind in-context copying.
- **[Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382)** — Li et al., 2022. Othello-GPT learns the board.
- **[Emergent Linear Representations in World Models of Self-Supervised Sequence Models](https://arxiv.org/abs/2309.00941)** — Nanda, Lee, Wattenberg, 2023. The board is linearly readable.
- **[Toy Models of Superposition](https://arxiv.org/abs/2209.10652)** — Elhage et al., 2022. Why neurons are polysemantic.

## Part 11 — Systems around the model: retrieval and agents

### 38 · Retrieval-augmented generation, from BM25 to rerankers (1994–2024)

- **[The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019)** — Robertson, Zaragoza, 2009. BM25 explained by its authors: where idf and term saturation come from.
- **[Reciprocal rank fusion outperforms Condorcet and individual rank learning methods](https://doi.org/10.1145/1571941.1572114)** — Cormack, Clarke, Büttcher, 2009. RRF: fuse rankings with 1/(k + rank).
- **[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)** — Reimers, Gurevych, 2019. One encoder, mean pooling, cosine similarity: the bi-encoder recipe.
- **[Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)** — Nogueira, Cho, 2019. The cross-encoder reranker: read query and passage together.
- **[Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/abs/1906.00300)** — Lee, Chang, Toutanova, 2019. The Inverse Cloze Task: pretrain a retriever with sentences as pseudo-queries.
- **[A Language Modeling Approach to Information Retrieval](https://doi.org/10.1145/290941.291008)** — Ponte, Croft, 1998. Query likelihood: rank a document by how probably it generates the query (the reranker in section 8).
- **[Improving Passage Retrieval with Zero-Shot Question Generation](https://arxiv.org/abs/2204.07496)** — Sachan et al., 2022. UPR: rerank with an LLM's log p(question | passage), no relevance training.
- **[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)** — Karpukhin et al., 2020. DPR: InfoNCE with in-batch negatives beats BM25 on open QA.
- **[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)** — Khattab, Zaharia, 2020. One vector per token: between bi- and cross-encoders.
- **[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)** — Lewis et al., 2020. RAG: retrieve passages, then generate conditioned on them.
- **[Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)** — Johnson, Douze, Jégou, 2017. Faiss: IVF and product quantization on GPUs, the approximate search behind vector databases.
- **[Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)** — Borgeaud et al., 2021. RETRO: retrieval inside the model, a 7.5B model matching 25x larger ones.

### 39 · Agents: tools, memory, workflows and multi-agent systems (2022–)

- **[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)** — Yao et al., 2022. Interleave thoughts with tool calls and observations: the loop this module builds.
- **[Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)** — Schick et al., 2023. A model that learns where an API call helps, from its own annotations.
- **[Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/abs/2102.09690)** — Zhao et al., 2021. Small models lean toward some labels regardless of input; a content-free prompt measures that bias so you can subtract it (the tool-choice fix in section 3).
- **[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)** — Park et al., 2023. A memory stream retrieved by recency, importance and relevance (the score in section 5).
- **[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)** — Shinn et al., 2023. An agent that critiques its failed attempts and retries: evaluator-optimizer.
- **[MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)** — Packer et al., 2023. Context as RAM, external storage as disk: an agent that pages its own memory.
- **[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)** — Wu et al., 2023. Multi-agent conversation as a programming model (the framework, now in maintenance).
- **[Fundamentals of Building Autonomous LLM Agents](https://arxiv.org/abs/2510.09244)** — de Lamo Castrillo et al., 2025. A survey (no experiments): perception, reasoning, memory, execution; useful vocabulary.

## Part 12 — Frontier architectures

### 40 · Latent experts and conditional memory: LatentMoE and Engram (2025–2026)

- **[LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts](https://arxiv.org/abs/2601.18089)** — Elango et al., 2026. Routed experts in a latent width d/α, α× as many: same bytes, better accuracy.
- **[NVIDIA Nemotron 3: Efficient and Open Intelligence](https://arxiv.org/abs/2512.20856)** — NVIDIA, 2025. LatentMoE in a hybrid Mamba/attention model family (Super and Ultra).
- **[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372)** — Cheng et al. (DeepSeek-AI), 2026. Engram: hashed n-gram tables beside the experts, and how to split the sparse budget between them.
- **[Hash Embeddings for Efficient Word Representations](https://arxiv.org/abs/1709.03933)** — Svenstrup, Hansen, Winther, 2017. Several hashes per token, combined, so collisions in one hash don't matter.
- **[Large Memory Layers with Product Keys](https://arxiv.org/abs/1907.05242)** — Lample et al., 2019. An earlier big lookup memory inside a Transformer, chosen by learned keys instead of by hashing.

### 41 · Linear attention and hybrids: from fast weights to Gated DeltaNet (2020–2026)

- **[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)** — Katharopoulos et al., 2020. Drop the softmax and attention becomes a recurrence with a matrix state.
- **[Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)** — Schlag, Irie, Schmidhuber, 2021. Linear attention as fast weights, and a delta-rule write instead of a sum.
- **[Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)** — Sun et al., 2023. A fixed decay per head; parallel, recurrent and chunkwise forms.
- **[Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)** — Yang et al., 2023. Data-dependent decay, and the chunkwise algorithm that made it fast.
- **[Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)** — Yang et al., 2024. DeltaNet: the WY representation turns the delta rule into chunked matmuls.
- **[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)** — Yang, Kautz, Hatamizadeh, 2024. Decay plus overwrite: the linear layer inside Qwen3-Next and Qwen3.5.
- **[Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708)** — Qiu et al., 2025. A sigmoid gate on attention's output; Qwen3-Next uses it.
- **[MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313)** — MiniMax, 2025. Seven linear-attention layers per softmax layer at 456B parameters.
- **[Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models](https://arxiv.org/abs/2504.03624)** — NVIDIA, 2025. Mamba-2 with about 8% attention layers.
- **[Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)** — Kimi Team, 2025. Kimi Delta Attention (a per-channel decay) and MLA at 3 : 1.
- **[Why Did M2 End Up as a Full Attention Model?](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model)** — MiniMax, 2025. The case against hybrids at the frontier, from a lab that shipped one.

### 42 · Small models done well: distillation, pruning and borrowing from big models (2015–2026)

- **[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)** — Hinton, Vinyals, Dean, 2015. Train a small model on a big one's softened predictions; scale the loss by T².
- **[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)** — Sanh et al., 2019. 40% smaller, 60% faster, 97% of BERT's language understanding.
- **[Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118)** — Gemma Team, 2024. The 2B and 9B trained by distillation instead of next-token prediction, far past compute-optimal.
- **[MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)** — Gu et al., 2023. Reverse KL, so the student doesn't spread mass over what the teacher thinks unlikely.
- **[Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)** — Muralidharan et al., 2024. Minitron: prune by activation importance, heal with distillation, up to 40× fewer tokens.
- **[LLM Pruning and Distillation in Practice: The Minitron Approach](https://arxiv.org/abs/2408.11796)** — Sreenivas et al., 2024. Llama 3.1 8B → 4B and Mistral NeMo 12B → 8B, with teacher correction.
- **[ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853)** — Men et al., 2024. Block influence: drop the layers that change their input least.
- **[SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model](https://arxiv.org/abs/2502.02737)** — Allal et al., 2025. 135M–1.7B models trained on 2–11T carefully mixed tokens; this module's teacher.
- **[TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)** — Eldan, Li, 2023. Under 10M parameters writes fluent stories, given data a small model can learn.
- **[Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)** — Gunasekar et al., 2023. phi-1: 1.3B parameters and ~7B tokens of textbook-quality data reach 50.6% on HumanEval.
- **[The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557)** — Penedo et al., 2024. FineWeb-Edu: a classifier trained on LLM ratings picks educational pages.
- **[Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)** — Keller Jordan, 2024. Orthogonalized momentum; the nanoGPT speedrun's optimizer.
- **[Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)** — Yang et al., 2022. µP: tune on a narrow model, reuse the hyperparameters on a wide one.
- **[Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/abs/2212.05055)** — Komatsuzaki et al., 2022. Copy a dense model's MLP into every expert and keep training.

### 43 · From pixels to words: vision transformers and vision-language models (2020–2025)

- **[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)** — Dosovitskiy et al., 2020. ViT: cut the image into patches, treat them as tokens; wins once pretrained on enough data.
- **[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)** — Radford et al., 2021. CLIP: 400M image-text pairs, a contrastive loss, zero-shot classification by text.
- **[Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)** — Zhai et al., 2023. SigLIP: one yes/no question per image-text pair instead of a softmax over the batch.
- **[Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)** — Liu et al., 2023. LLaVA: a frozen CLIP encoder, a projection into an LLM, then instruction tuning.
- **[Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)** — Liu et al., 2023. LLaVA-1.5: a 2-layer MLP projector, higher resolution; the recipe most open VLMs copied.
- **[PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726)** — Beyer et al., 2024. SigLIP + Gemma, trained end to end, image tokens attended bidirectionally.
- **[Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)** — Qwen Team, 2025. Native dynamic resolution: as many image tokens as the image needs, with windowed attention.
- **[SmolVLM: Redefining small and efficient multimodal models](https://arxiv.org/abs/2504.05299)** — Marafioti et al., 2025. SigLIP + SmolLM2 in 256M parameters; pixel shuffle cuts 1,024 patches to 64 tokens.

### 44 · Efficient and sparse attention: reading only part of the past (2019–2026)

- **[Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)** — Child et al., 2019. Strided and local attention patterns: O(T√T) instead of O(T²).
- **[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)** — Beltagy, Peters, Cohan, 2020. Sliding-window attention plus a few global tokens.
- **[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)** — Zaheer et al., 2020. Window + global + random keys, with a proof that it stays as expressive as full attention.
- **[Mistral 7B](https://arxiv.org/abs/2310.06825)** — Jiang et al., 2023. A 4,096-token sliding window and a rolling KV cache.
- **[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)** — Xiao et al., 2023. StreamingLLM: keep the first 4 tokens and a window; pretrained models then run on endless streams.
- **[Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)** — Gemma Team, 2025. 5 local (1,024-token window) : 1 global layers to shrink the KV cache at 128K context.
- **[MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189)** — Lu et al., 2025. Route each query to its top blocks by mean key; used in Kimi's long-context serving.
- **[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)** — Yuan et al., 2025. Compressed, selected and sliding branches with a learned gate, trained sparse from the start.
- **[DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)** — DeepSeek-AI, 2025. DeepSeek Sparse Attention: a lightning indexer picks 2,048 tokens per query.

### 45 · Diffusion language models: writing a whole passage at once (2021–2026)

- **[Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)** — Austin et al., 2021. D3PM: diffusion over discrete tokens; the absorbing [MASK] noise used here is one of its options.
- **[Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834)** — Lou, Meng, Ermon, 2024. SEDD: score entropy for discrete data; ICML 2024 best paper.
- **[Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)** — Sahoo et al., 2024. MDLM: the loss is a weighted mix of masked-LM losses and a bound on the likelihood; this module's recipe.
- **[Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514)** — Nie et al., 2024. Scaling laws for masked diffusion: about 16× the compute of an autoregressive model for the same loss.
- **[Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)** — Nie et al., 2025. LLaDA 8B: masked diffusion from scratch at Llama 3 8B scale; low-confidence remasking, block sampling.
- **[Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)** — Ye et al., 2025. A diffusion LM initialized from an autoregressive model's weights (Qwen2.5-7B).
- **[Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573)** — Arriola et al., 2025. BD3-LM: autoregressive over blocks, diffusion within each, with a KV cache.
- **[Mercury: Ultra-Fast Language Models Based on Diffusion](https://arxiv.org/abs/2506.17298)** — Inception Labs, 2025. A commercial diffusion coder; over 1,000 tokens/s on an H100 by the vendor's measurement.
- **[Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference](https://arxiv.org/abs/2508.02193)** — ByteDance Seed, 2025. A diffusion code model reporting 2,146 tokens/s.

### 46 · Weights by hand: compiling, merging and editing networks (1990s–2026)

- **[Thinking Like Transformers](https://arxiv.org/abs/2106.06981)** — Weiss, Goldberg, Yahav, 2021. RASP: attention as select-and-aggregate, a programming language for what a transformer can compute.
- **[Tracr: Compiled Transformers as a Laboratory for Interpretability](https://arxiv.org/abs/2301.05062)** — Lindner et al., 2023. Compiles RASP programs into real transformer weights; section 2 by hand.
- **[Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)** — Meng, Bau, Andonian, Belinkov, 2022. ROME: causal tracing finds facts in early-middle MLPs; a rank-one edit rewrites one.
- **[Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)** — Meng et al., 2022. MEMIT: thousands of edits at once, spread over several layers.
- **[Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)** — Ilharco et al., 2022. Task vectors: add them to combine skills, subtract them to forget.
- **[TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)** — Yadav, Tam, Choshen, Raffel, Bansal, 2023. Trim, elect sign, disjoint merge: merging when task vectors conflict.
- **[Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)** — Yu et al., 2023. DARE: drop most of a task vector at random and rescale the rest.
- **[Language Models Can Rewrite Their Own Weights](https://arxiv.org/abs/2611.21885)** — Fan et al., 2026. Qwen3.5-27B edits its own weights without gradients; its best edit is one uniform bias change.
- **[WeightBench: Evaluating Language Models as Weight-Space Programmers](https://arxiv.org/abs/2611.18553)** — Marasini, Khadka, Rahimi, 2026. 42 read/fix/write tasks: frontier models score about half and can't yet patch an LM.

### 47 · Reading activations: lenses, sparse autoencoders and steering (2020–2025)

- **[interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)** — nostalgebraist, 2020. Decode every layer with the output head and watch the prediction form.
- **[Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)** — Belrose et al., 2023. One learned affine map per layer fixes the logit lens's early-layer gibberish.
- **[Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762)** — Sun, Chen, Kolter, Liu, 2024. A few huge activations act as fixed biases (attention sinks): why section 2 drops those tokens.
- **[Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features)** — Bricken et al., 2023. A ReLU + L1 sparse autoencoder turns polysemantic neurons into readable features.
- **[Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600)** — Cunningham et al., 2023. The same idea in parallel, on Pythia's residual stream.
- **[Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)** — Gao et al., 2024. TopK SAEs, AuxK for dead latents, and scaling laws: the SAE section 2 trains.
- **[Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2](https://arxiv.org/abs/2408.05147)** — Lieberum et al., 2024. JumpReLU SAEs on every layer of Gemma 2, released for anyone to read.
- **[Automatically Interpreting Millions of Features in Large Language Models](https://arxiv.org/abs/2410.13928)** — Paulo et al., 2024. An LLM explains each feature; detection and fuzzing scores check the explanations.
- **[Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248)** — Turner et al., 2023. ActAdd: one contrast pair ("Love" − "Hate") added to the residual stream steers generation.
- **[Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)** — Rimsky et al., 2023. CAA: average the difference over hundreds of pairs for a cleaner direction.
- **[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)** — Arditi et al., 2024. Directional ablation and weight orthogonalization: the same edit as section 5, applied to refusal.
- **[The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)** — Huh, Cheung, Wang, Isola, 2024. Different models converge on similar representations; mutual nearest neighbours measure it.
- **[Harnessing the Universal Geometry of Embeddings](https://arxiv.org/abs/2505.12540)** — Jha, Zhang, Shmatikov, Morris, 2025. vec2vec: translate between embedding spaces with no paired data at all.

### 48 · Tracing a circuit: patching, transcoders and attribution graphs (2022–2025)

- **[Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)** — Wang et al., 2022. The indirect-object task and its circuit of name movers and S-inhibition heads, found by patching.
- **[Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042)** — Zhang, Nanda, 2023. Logit difference over probabilities, and why the choice of corruption matters.
- **[How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)** — Heimersheim, Nanda, 2024. Denoising vs noising, what a patching result does and doesn't show.
- **[Attribution Patching: Activation Patching At Industrial Scale](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)** — Nanda, 2023. Replace thousands of patching runs with one backward pass: a first-order Taylor estimate.
- **[AtP*: An efficient and scalable method for localizing LLM behaviour to components](https://arxiv.org/abs/2403.00745)** — Kramár et al., 2024. Where attribution patching fails (saturated attention) and how to fix it.
- **[Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997)** — Conmy et al., 2023. ACDC: prune a model's computational graph edge by edge, by patching.
- **[Copy Suppression: Comprehensively Understanding an Attention Head](https://arxiv.org/abs/2310.04625)** — McDougall et al., 2023. The negative name mover: a head that attends to a name and pushes it down.
- **[Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944)** — Dunefsky, Chlenski, Nanda, 2024. Sparse replacements for MLPs, so circuits can be traced through them.
- **[Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647)** — Marks et al., 2024. Circuits of SAE features, with error nodes, found by attribution.
- **[Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)** — Ameisen et al., 2025. Attribution graphs: a frozen linear replacement model, error nodes, pruning by influence.
- **[On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)** — Lindsey et al., 2025. Attribution graphs on Claude 3.5 Haiku: two-hop reasoning, planning, and their limits.

### 49 · Continual learning: forgetting, consolidation and sleep (1989–2026)

- **[Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem](https://doi.org/10.1016/S0079-7421(08)60536-8)** — McCloskey & Cohen, 1989. Train a network on a second task and it can lose the first almost completely: the problem, named.
- **[Why There Are Complementary Learning Systems in the Hippocampus and Neocortex](https://doi.org/10.1037/0033-295X.102.3.419)** — McClelland, McNaughton & O'Reilly, 1995. A fast learner (hippocampus) and a slow one (neocortex), with replay in sleep: the idea every "sleep" method borrows.
- **[Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)** — Kirkpatrick et al., 2017. Elastic weight consolidation: anchor each weight by how much the old task needs it (its Fisher).
- **[Gradient Episodic Memory for Continual Learning](https://arxiv.org/abs/1706.08840)** — Lopez-Paz & Ranzato, 2017. The accuracy matrix and backward transfer, the measurements this module scores every method with.
- **[Physics of Language Models: Part 3.1, Knowledge Storage and Extraction](https://arxiv.org/abs/2309.14316)** — Allen-Zhu & Li, 2023. Invented biographies as a clean test of what a model stores and can retrieve.
- **[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)** — Agarwal et al., 2023. GKD: distil on the student's own samples, with a choice of divergence.
- **[Self-Adapting Language Models](https://arxiv.org/abs/2506.10943)** — Zweiger et al., 2025. SEAL: the model writes its own training data from a passage; RL keeps the writing that helps.
- **[Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)** — Shenfeld et al., 2026. SDFT: the model with a demonstration in its prompt teaches itself without it, and forgets less than SFT.
- **[Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories](https://arxiv.org/abs/2606.03979)** — Behrouz et al., 2026. Sleep = consolidation into new low-rank experts by self-distillation, then dreaming.

### 50 · Uncensored models: removing refusals from the data and from the weights (2023–2024)

- **[Uncensored Models](https://erichartford.com/uncensored-models)** — Hartford, 2023. The data route: drop refusals from an instruction set, fine-tune a base model the same way as before.
- **[Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)** — Taori et al., 2023. Instruction tuning on answers written by an OpenAI model; the prompt template WizardLM and Hartford reused.
- **[WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)** — Xu et al., 2023. Evol-Instruct: ChatGPT rewrites instructions harder, then answers them (with refusals).
- **[Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)** — Zou et al., 2023. AdvBench, the harmful instructions this module uses to find the refusal direction.
- **[XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://arxiv.org/abs/2308.01263)** — Röttger et al., 2023. Safe prompts that look unsafe, with unsafe contrasts: over-refusal and refusal measured together.
- **[Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!](https://arxiv.org/abs/2310.03693)** — Qi et al., 2023. A few fine-tuning examples, even benign ones, undo much of a model's refusal.
- **[LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B](https://arxiv.org/abs/2310.20624)** — Lermen, Rogers-Smith, Ladish, 2023. The data route at its cheapest: a LoRA on one GPU removes refusal.
- **[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)** — Arditi et al., 2024. The weight route: a difference-in-means direction, chosen by bypass/induce/KL, projected out of the weights.

## Books

For a second pass, or a different angle. Most are free to read online from their authors; each entry says so.

- **[Dive into Deep Learning](https://d2l.ai/)** — Zhang, Lipton, Li, Smola, 2023. Free, interactive, and has JAX implementations alongside PyTorch: the closest textbook to this course. *(Modules 01–17.)*
- **[Understanding Deep Learning](https://udlbook.github.io/udlbook/)** — Simon J. D. Prince, 2023. Free. Clear figures and intuition for everything from backprop to transformers and diffusion. *(Modules 03–17, 33.)*
- **[Deep Learning: Foundations and Concepts](https://www.bishopbook.com/)** — Christopher M. Bishop, Hugh Bishop, 2024. Free to read online. The probabilistic view, carefully derived. *(Modules 04–12, 33.)*
- **[Deep Learning](https://www.deeplearningbook.org/)** — Goodfellow, Bengio, Courville, 2016. Free. The 2016 reference: optimization, regularization and the math behind modules 02–12. *(Modules 02–12.)*
- **[Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html)** — Kevin P. Murphy, 2022. Free. Probability and linear models done properly. *(Modules 04, 09.)*
- **[Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html)** — Richard S. Sutton, Andrew G. Barto, 2018. Free. The RL textbook: Q-learning, policy gradients. *(Modules 26, 34, 35.)*
- **[Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)** — Dan Jurafsky, James H. Martin, 2025. Free. N-grams to transformers, from the NLP side. *(Modules 13–16, 30–32.)*
- **[Reinforcement Learning from Human Feedback](https://rlhfbook.com/)** — Nathan Lambert, 2025. Free. Post-training: reward models, RLHF, DPO, RLVR. *(Modules 26, 35.)*
- **[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)** — Sebastian Raschka, 2024. GPT from tokenizer to fine-tuning, in PyTorch; a good second pass over modules 13–20. *(Modules 13–20.)*
- **[AI Engineering](https://huyenchip.com/books/)** — Chip Huyen, 2025. Building products on foundation models: evaluation, RAG, agents, fine-tuning trade-offs. *(Modules 21, 38, 39.)*
- **[Artificial Intelligence: A Modern Approach (4th ed.)](https://aima.cs.berkeley.edu/)** — Stuart Russell, Peter Norvig, 2020. The classic survey of AI beyond deep learning: search, logic, planning. *(Modules 00b, 34.)*
- **[Artificial Intelligence: A Guide for Thinking Humans](https://melaniemitchell.me/aibook/)** — Melanie Mitchell, 2019. A readable history of AI's hype cycles and real limits. *(Modules 00b.)*
