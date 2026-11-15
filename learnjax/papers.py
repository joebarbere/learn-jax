"""Seminal papers for each module: what to read once a lesson's ideas click.

Every link is checked by `tools/check_papers.py` (arXiv titles via the arXiv API, other links by
resolving them). `tools/gen_papers.py` renders docs/PAPERS.md; lessons render their own section with
`papers_md(N)`.
"""
from __future__ import annotations

from typing import NamedTuple


class Paper(NamedTuple):
    title: str
    authors: str
    year: int
    url: str
    why: str


def arxiv(i: str) -> str:
    return f"https://arxiv.org/abs/{i}"


ATTENTION = Paper("Attention Is All You Need", "Vaswani et al.", 2017, arxiv("1706.03762"),
                  "The Transformer: self-attention replaces recurrence. Everything in Part 4 builds on it.")
LORA = Paper("LoRA: Low-Rank Adaptation of Large Language Models", "Hu et al.", 2021, arxiv("2106.09685"),
             "Fine-tune by learning a low-rank update instead of all weights.")
ROOFLINE = Paper("Roofline: An Insightful Visual Performance Model for Multicore Architectures",
                 "Williams, Waterman, Patterson", 2009, "https://doi.org/10.1145/1498765.1498785",
                 "Compute-bound vs memory-bound in one picture; the lens for every speed measurement.")
JAX_PAPER = Paper("Compiling machine learning programs via high-level tracing", "Frostig, Johnson, Leary", 2018,
                  "https://mlsys.org/Conferences/doc/2018/146.pdf",
                  "The original JAX paper: trace Python into a graph, compile it with XLA.")
BACKPROP = Paper("Learning representations by back-propagating errors", "Rumelhart, Hinton, Williams", 1986,
                 "https://doi.org/10.1038/323533a0",
                 "Backpropagation made multi-layer networks trainable.")
FLASH = Paper("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "Dao et al.", 2022,
              arxiv("2205.14135"), "Attention reorganized around memory traffic, not FLOPs.")
DEEPSEEK_V3 = Paper("DeepSeek-V3 Technical Report", "DeepSeek-AI", 2024, arxiv("2412.19437"),
                    "MLA + DeepSeekMoE + aux-loss-free balancing + MTP: the model you replicated in miniature.")
CHINCHILLA = Paper("Training Compute-Optimal Large Language Models", "Hoffmann et al.", 2022, arxiv("2203.15556"),
                   "Scale tokens with parameters (~20 tokens/param) for a fixed compute budget.")
GPT2 = Paper("Language Models are Unsupervised Multitask Learners", "Radford et al.", 2019,
             "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
             "GPT-2: byte-level BPE and a scaled-up decoder-only Transformer.")

PAPERS: dict[int, list[Paper]] = {
    0: [JAX_PAPER, ROOFLINE],
    1: [Paper("Array programming with NumPy", "Harris et al.", 2020, arxiv("2006.10256"),
              "The array model (shapes, strides, broadcasting) that jax.numpy inherits.")],
    2: [LORA,
        Paper("Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning",
              "Aghajanyan, Zettlemoyer, Gupta", 2020, arxiv("2012.13255"),
              "Why low-rank updates are enough: fine-tuning lives in a small subspace.")],
    3: [BACKPROP,
        Paper("Automatic differentiation in machine learning: a survey", "Baydin et al.", 2015, arxiv("1502.05767"),
              "Forward vs reverse mode, dual numbers, tapes: the theory behind jax.grad.")],
    4: [Paper("A Mathematical Theory of Communication", "Shannon", 1948,
              "https://doi.org/10.1002/j.1538-7305.1948.tb01338.x",
              "Entropy and cross-entropy start here; a language model's loss is measured in Shannon's bits."),
        Paper("Parallel random numbers: as easy as 1, 2, 3", "Salmon et al.", 2011,
              "https://doi.org/10.1145/2063384.2063405",
              "Counter-based RNGs (Threefry): the design behind JAX's splittable random keys."),
        Paper("Categorical Reparameterization with Gumbel-Softmax", "Jang, Gu, Poole", 2016, arxiv("1611.01144"),
              "The Gumbel-max trick for sampling categories, made differentiable.")],
    5: [Paper("Adam: A Method for Stochastic Optimization", "Kingma, Ba", 2014, arxiv("1412.6980"),
              "Momentum + per-parameter step sizes + bias correction."),
        Paper("Decoupled Weight Decay Regularization", "Loshchilov, Hutter", 2017, arxiv("1711.05101"),
              "AdamW: why weight decay is not L2 regularization under Adam."),
        Paper("SGDR: Stochastic Gradient Descent with Warm Restarts", "Loshchilov, Hutter", 2016, arxiv("1608.03983"),
              "Cosine learning-rate schedules."),
        Paper("On the difficulty of training Recurrent Neural Networks", "Pascanu, Mikolov, Bengio", 2012,
              arxiv("1211.5063"), "Exploding gradients and gradient-norm clipping."),
        Paper("Muon is Scalable for LLM Training", "Liu, Su et al. (Moonshot AI)", 2025, arxiv("2502.16982"),
              "Muon (orthogonalized momentum) at LLM scale: weight decay and update-RMS matching make it work.")],
    6: [JAX_PAPER,
        Paper("Operator Fusion in XLA: Analysis and Evaluation", "Snider, Liang", 2023, arxiv("2301.13062"),
              "What XLA fuses, and why fusion is most of the speedup you measured.")],
    7: [Paper("Efficient Per-Example Gradient Computations", "Goodfellow", 2015, arxiv("1510.01799"),
              "Per-example gradients without a loop, the job vmap(grad) does for free."),
        Paper("Deep Learning with Differential Privacy", "Abadi et al.", 2016, arxiv("1607.00133"),
              "DP-SGD: the best-known use of per-example gradient clipping.")],
    8: [Paper("Mixed Precision Training", "Micikevicius et al.", 2017, arxiv("1710.03740"),
              "fp16 compute, fp32 master weights, loss scaling: the original recipe."),
        Paper("A Study of BFLOAT16 for Deep Learning Training", "Kalamkar et al.", 2019, arxiv("1905.12322"),
              "Why bfloat16's fp32-sized exponent removes the need for loss scaling.")],
    9: [Paper("Ridge Regression: Biased Estimation for Nonorthogonal Problems", "Hoerl, Kennard", 1970,
              "https://doi.org/10.1080/00401706.1970.10488634", "The original ridge (L2-regularized) regression."),
        Paper("Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms",
              "Xiao, Rasul, Vollgraf", 2017, arxiv("1708.07747"), "The dataset used in Part 3.")],
    10: [BACKPROP,
         Paper("Approximation by superpositions of a sigmoidal function", "Cybenko", 1989,
               "https://doi.org/10.1007/BF02551274", "The universal approximation theorem."),
         Paper("Gaussian Error Linear Units (GELUs)", "Hendrycks, Gimpel", 2016, arxiv("1606.08415"),
               "The smooth activation used in GPT-2 and BERT.")],
    11: [Paper("Understanding the difficulty of training deep feedforward neural networks", "Glorot, Bengio", 2010,
               "https://proceedings.mlr.press/v9/glorot10a.html", "Xavier initialization and activation statistics."),
         Paper("Delving Deep into Rectifiers", "He et al.", 2015, arxiv("1502.01852"), "He initialization for ReLU nets."),
         Paper("Deep Residual Learning for Image Recognition", "He et al.", 2015, arxiv("1512.03385"),
               "Residual connections: how very deep networks became trainable."),
         Paper("Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
               "Ioffe, Szegedy", 2015, arxiv("1502.03167"),
               "Normalize over the batch; the layer that made deep CNNs easy to train (and awkward in JAX)."),
         Paper("Layer Normalization", "Ba, Kiros, Hinton", 2016, arxiv("1607.06450"),
               "Normalize over features, not the batch; standard in Transformers."),
         Paper("Root Mean Square Layer Normalization", "Zhang, Sennrich", 2019, arxiv("1910.07467"),
               "RMSNorm: LayerNorm without the mean; used by Llama and DeepSeek."),
         Paper("Dropout: A Simple Way to Prevent Neural Networks from Overfitting", "Srivastava et al.", 2014,
               "https://jmlr.org/papers/v15/srivastava14a.html", "Randomly drop units during training.")],
    12: [Paper("Gradient-based learning applied to document recognition", "LeCun et al.", 1998,
               "https://doi.org/10.1109/5.726791", "LeNet: convolution + pooling, trained end to end."),
         Paper("ImageNet Classification with Deep Convolutional Neural Networks", "Krizhevsky, Sutskever, Hinton", 2012,
               "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks",
               "AlexNet: CNNs on GPUs win ImageNet, the start of the deep-learning boom."),
         Paper("WaveNet: A Generative Model for Raw Audio", "van den Oord et al.", 2016, arxiv("1609.03499"),
               "Dilated causal convolutions: exponentially growing context for sequences."),
         Paper("Visualizing and Understanding Convolutional Networks", "Zeiler, Fergus", 2013, arxiv("1311.2901"),
               "What learned filters and feature maps respond to.")],
    13: [Paper("Neural Machine Translation of Rare Words with Subword Units", "Sennrich, Haddow, Birch", 2015,
               arxiv("1508.07909"), "Byte-pair encoding for neural models."),
         Paper("SentencePiece: A simple and language independent subword tokenizer", "Kudo, Richardson", 2018,
               arxiv("1808.06226"), "Tokenization from raw text, no pre-splitting."),
         GPT2],
    14: [Paper("A Neural Probabilistic Language Model", "Bengio et al.", 2003,
               "https://jmlr.org/papers/v3/bengio03a.html", "The MLP language model you built: embeddings + context window."),
         Paper("Prediction and Entropy of Printed English", "Shannon", 1951,
               "https://doi.org/10.1002/j.1538-7305.1951.tb01366.x", "Humans as language models; the entropy of English."),
         Paper("Efficient Estimation of Word Representations in Vector Space", "Mikolov et al.", 2013,
               arxiv("1301.3781"), "word2vec: embeddings as a product in their own right.")],
    15: [Paper("Neural Machine Translation by Jointly Learning to Align and Translate", "Bahdanau, Cho, Bengio", 2014,
               arxiv("1409.0473"), "Attention is introduced, as alignment in translation."),
         ATTENTION, FLASH],
    16: [Paper("Improving Language Understanding by Generative Pre-Training", "Radford et al.", 2018,
               "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf",
               "GPT-1: pre-train a decoder-only Transformer, then fine-tune."),
         GPT2,
         Paper("Using the Output Embedding to Improve Language Models", "Press, Wolf", 2016, arxiv("1608.05859"),
               "Weight tying between the embedding and the output head."),
         Paper("Scaling Laws for Neural Language Models", "Kaplan et al.", 2020, arxiv("2001.08361"),
               "Loss falls as a power law in parameters, data and compute."),
         CHINCHILLA],
    17: [Paper("RoFormer: Enhanced Transformer with Rotary Position Embedding", "Su et al.", 2021, arxiv("2104.09864"),
               "RoPE: encode position by rotating query/key feature pairs."),
         Paper("GLU Variants Improve Transformer", "Shazeer", 2020, arxiv("2002.05202"), "SwiGLU feed-forward layers."),
         Paper("Fast Transformer Decoding: One Write-Head is All You Need", "Shazeer", 2019, arxiv("1911.02150"),
               "Multi-query attention: share K/V across heads to shrink the cache."),
         Paper("GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
               "Ainslie et al.", 2023, arxiv("2305.13245"), "Grouped-query attention, the Llama 2/3 middle ground."),
         Paper("LLaMA: Open and Efficient Foundation Language Models", "Touvron et al.", 2023, arxiv("2302.13971"),
               "The RMSNorm + RoPE + SwiGLU recipe most open models copy."),
         Paper("The Llama 3 Herd of Models", "Llama Team, AI @ Meta", 2024, arxiv("2407.21783"),
               "GQA, 128K vocabulary, RoPE base 500,000."),
         Paper("Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", "Shazeer et al.", 2017,
               arxiv("1701.06538"), "Mixture-of-experts with a learned router."),
         Paper("Switch Transformers", "Fedus, Zoph, Shazeer", 2021, arxiv("2101.03961"),
               "Top-1 routing and the auxiliary load-balancing loss."),
         Paper("DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models",
               "Dai et al.", 2024, arxiv("2401.06066"), "Fine-grained routed experts plus always-on shared experts."),
         Paper("DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", "DeepSeek-AI", 2024,
               arxiv("2405.04434"), "Multi-head latent attention (MLA): a compressed KV cache."),
         Paper("Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts", "Wang et al.", 2024,
               arxiv("2408.15664"), "Balance experts with a bias term instead of an extra loss."),
         Paper("Better & Faster Large Language Models via Multi-token Prediction", "Gloeckle et al.", 2024,
               arxiv("2404.19737"), "Predict several future tokens per position."),
         DEEPSEEK_V3],
    18: [Paper("The Curious Case of Neural Text Degeneration", "Holtzman et al.", 2019, arxiv("1904.09751"),
               "Why greedy decoding repeats itself; nucleus (top-p) sampling."),
         Paper("Fast Inference from Transformers via Speculative Decoding", "Leviathan, Kalman, Matias", 2022,
               arxiv("2211.17192"), "A small draft model proposes, the big model verifies."),
         Paper("LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", "Dettmers et al.", 2022,
               arxiv("2208.07339"), "Int8 inference and outlier features."),
         Paper("GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", "Frantar et al.",
               2022, arxiv("2210.17323"), "4-bit weights with little loss, the basis of many local-model formats.")],
    19: [Paper("Efficient Memory Management for Large Language Model Serving with PagedAttention", "Kwon et al.", 2023,
               arxiv("2309.06180"), "vLLM: paging the KV cache to serve many requests."),
         Paper("Efficiently Scaling Transformer Inference", "Pope et al.", 2022, arxiv("2211.05102"),
               "Prefill vs decode, latency vs throughput, and where the memory goes."),
         Paper("AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", "Lin et al.", 2023,
               arxiv("2306.00978"), "4-bit weight quantization guided by activations.")],
    20: [LORA,
         Paper("QLoRA: Efficient Finetuning of Quantized LLMs", "Dettmers et al.", 2023, arxiv("2305.14314"),
               "LoRA on a 4-bit base model: fine-tuning large models on one GPU."),
         Paper("Parameter-Efficient Transfer Learning for NLP", "Houlsby et al.", 2019, arxiv("1902.00751"),
               "Adapters: the parameter-efficient fine-tuning idea before LoRA."),
         Paper("Training language models to follow instructions with human feedback", "Ouyang et al.", 2022,
               arxiv("2203.02155"), "InstructGPT: supervised fine-tuning then RLHF.")],
    21: [Paper("Measuring Massive Multitask Language Understanding", "Hendrycks et al.", 2020, arxiv("2009.03300"),
               "MMLU, the multiple-choice benchmark everyone reports."),
         Paper("Holistic Evaluation of Language Models", "Liang et al.", 2022, arxiv("2211.09110"),
               "HELM: evaluate many metrics across many scenarios."),
         Paper("Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", "Zheng et al.", 2023, arxiv("2306.05685"),
               "Using a strong model to grade answers, and its biases.")],
    22: [ROOFLINE,
         Paper("Data Movement Is All You Need: A Case Study on Optimizing Transformers", "Ivanov et al.", 2020,
               arxiv("2007.00072"), "Transformers are often memory-bound; measure data movement, not FLOPs."),
         Paper("Operator Fusion in XLA: Analysis and Evaluation", "Snider, Liang", 2023, arxiv("2301.13062"),
               "How XLA's fusion decisions show up in a profile.")],
    23: [Paper("Triton: an intermediate language and compiler for tiled neural network computations",
               "Tillet, Kung, Cox", 2019, "https://doi.org/10.1145/3315508.3329973",
               "The tile-based kernel language behind Pallas's GPU backend and XLA's GEMM fusions."),
         FLASH,
         Paper("FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", "Dao", 2023,
               arxiv("2307.08691"), "Better work partitioning across the GPU.")],
    24: [Paper("Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism",
               "Shoeybi et al.", 2019, arxiv("1909.08053"), "Tensor parallelism inside Transformer layers."),
         Paper("ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", "Rajbhandari et al.", 2019,
               arxiv("1910.02054"), "Shard optimizer state, gradients and params: the idea behind FSDP."),
         Paper("GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", "Lepikhin et al.",
               2020, arxiv("2006.16668"), "Sharding annotations + MoE expert parallelism."),
         Paper("GSPMD: General and Scalable Parallelization for ML Computation Graphs", "Xu et al.", 2021,
               arxiv("2105.04663"), "The XLA partitioner that turns JAX's sharding specs into collectives.")],
    26: [Paper("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "Wei et al.", 2022,
               arxiv("2201.11903"), "Ask for the steps, get better answers: the start of reasoning prompts."),
         Paper("Self-Consistency Improves Chain of Thought Reasoning in Language Models", "Wang et al.", 2022,
               arxiv("2203.11171"), "Sample many reasoning paths and majority-vote the answer."),
         Paper("Training Verifiers to Solve Math Word Problems", "Cobbe et al.", 2021, arxiv("2110.14168"),
               "GSM8K and verifier-reranked sampling: checkable answers as a training signal."),
         Paper("Let's Verify Step by Step", "Lightman et al.", 2023, arxiv("2305.20050"),
               "Process vs outcome supervision: rewarding steps instead of only final answers."),
         Paper("Proximal Policy Optimization Algorithms", "Schulman et al.", 2017, arxiv("1707.06347"),
               "The clipped policy-ratio objective GRPO inherits."),
         Paper("DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
               "Shao et al.", 2024, arxiv("2402.03300"),
               "Introduces GRPO: group-relative advantages, no value network."),
         Paper("DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", "DeepSeek-AI", 2025,
               arxiv("2501.12948"), "RL with verifiable rewards produces long reasoning; R1-Zero, distillation."),
         Paper("s1: Simple test-time scaling", "Muennighoff et al.", 2025, arxiv("2501.19393"),
               "Budget forcing: controlling how long a model thinks at inference time.")],
    27: [LORA,
         Paper("Fast Inference from Transformers via Speculative Decoding", "Leviathan, Kalman, Matias", 2022,
               arxiv("2211.17192"), "The draft-and-verify scheme behind LM Studio's speculative decoding."),
         Paper("GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", "Frantar et al.",
               2022, arxiv("2210.17323"), "Why 4-bit weights lose so little: the idea behind GGUF's k-quants.")],
    25: [CHINCHILLA,
         Paper("Efficiently Scaling Transformer Inference", "Pope et al.", 2022, arxiv("2211.05102"),
               "Cost per token for serving, the number to compare across hardware."),
         DEEPSEEK_V3],
}


class Lecture(NamedTuple):
    title: str
    url: str
    length: str
    why: str


def _yt(v: str) -> str:
    return f"https://www.youtube.com/watch?v={v}"


# Companion video lectures, grouped by series. Every YouTube ID is verified by title via YouTube oEmbed.
ZTH = "Andrej Karpathy, *Neural Networks: Zero to Hero* (PyTorch, code-along)"
B3 = "Grant Sanderson, 3Blue1Brown (visual intuition, no code)"
RASCHKA = "Sebastian Raschka, *Build a Reasoning Model (From Scratch)* (PyTorch, code-along)"
JAXFORMER = "Aditya Makkar, Divya Makkar & Chinmay Jindal, *Jaxformer* (written guide, JAX on TPU)"
SCALING = "Austin, Douglas, Frostig, Levskaya et al. (Google DeepMind), *How To Scale Your Model* (written guide, JAX)"
ULTRA = "Hugging Face nanotron team, *The Ultra-Scale Playbook* (written guide, PyTorch)"
NANOCHAT = "Andrej Karpathy, *nanochat* (MIT-licensed repo, a full ChatGPT-style pipeline in PyTorch)"
SERIES = {ZTH: "https://karpathy.ai/zero-to-hero.html",
          JAXFORMER: "https://jaxformer.com/",
          SCALING: "https://jax-ml.github.io/scaling-book/",
          ULTRA: "https://huggingface.co/spaces/nanotron/ultrascale-playbook",
          NANOCHAT: "https://github.com/karpathy/nanochat",
          B3: "https://www.3blue1brown.com/?topic=neural-networks",
          RASCHKA: "https://magazine.sebastianraschka.com/p/building-reasoning-models"}


def _r(url: str, title: str, why: str) -> Lecture:
    """A written chapter or repo page (not a video)."""
    return Lecture(title, url, "", why)


JF, SB = "https://jaxformer.com/", "https://jax-ml.github.io/scaling-book/"
NC = "https://github.com/karpathy/nanochat"
EXTRA: dict[int, dict[str, list[Lecture]]] = {
    13: {JAXFORMER: [_r(JF + "tokenization/", "Part 1: Tokenization at scale", "Tokenizing FineWeb-EDU with multiprocessing and Ray.")]},
    16: {NANOCHAT: [_r(NC, "nanochat", "nanoGPT's successor: the same GPT, plus tokenizer, SFT, RL, eval and a chat engine.")]},
    17: {JAXFORMER: [_r(JF + "base_model/", "Part 2: How to write the base model", "RMSNorm, RoPE and multi-latent attention in JAX."),
                     _r(JF + "moe/", "Part 6: Implementing mixture-of-experts layers", "Sigmoid top-k routing, shared experts, load balancing.")]},
    18: {SCALING: [_r(SB + "inference", "All about Transformer inference", "Prefill vs decode, KV-cache costs, batching, latency."),
                   _r(SB + "applied-inference", "Serving LLaMA 3-70B", "The inference math applied to a real model.")]},
    21: {NANOCHAT: [_r(NC, "nanochat's evaluation (base_eval, chat_eval)", "CORE score, bits per byte, ARC, GSM8K, MMLU in one harness.")]},
    22: {SCALING: [_r(SB + "roofline", "All about rooflines", "Compute vs memory vs communication bounds, worked out."),
                   _r(SB + "profiling", "How to profile TPU programs", "Reading JAX profiler traces (TPU-focused, same ideas on GPU).")]},
    23: {SCALING: [_r(SB + "gpus", "How to think about GPUs", "SMs, memory hierarchy and collectives from the scaling-book view.")]},
    24: {JAXFORMER: [_r(JF + "sharded/", "Part 3: Inside the sharded model", "Mesh, shard_map, FSDP, GPipe pipelines and tensor parallelism.")],
         SCALING: [_r(SB + "sharding", "Sharded matrices and how to multiply them", "Which collective each sharded matmul needs."),
                   _r(SB + "jax-stuff", "Programming TPUs in JAX", "Explicit sharding and shard_map from the JAX side.")],
         ULTRA: [_r("https://huggingface.co/spaces/nanotron/ultrascale-playbook", "The Ultra-Scale Playbook",
                    "DP, ZeRO/FSDP, TP, PP, context and expert parallelism, measured on GPU clusters.")]},
    25: {JAXFORMER: [_r(JF + "dataset/", "Part 4: Dataset class and config files", "Sharded data placement."),
                     _r(JF + "distributed_training/", "Part 5: Distributed training", "jax.distributed, Orbax checkpoints, the train loop."),
                     _r(JF + "training/", "Part 7: Final run and training results", "A 949M-param MoE on 32 TPU-v4 chips, multi-host.")],
         SCALING: [_r(SB + "applied-training", "Training LLaMA 3 on TPUs", "Choosing a parallelism layout for a real run.")],
         NANOCHAT: [_r(NC, "nanochat", "The capstone's blueprint: every stage of a small chat model, end to end."),
                    _r(NC + "/discussions/481", "Beating GPT-2 for <<$100: the nanochat journey", "Karpathy's write-up of what made it fast.")]},
    26: {NANOCHAT: [_r(NC, "nanochat's chat_rl stage", "RL on GSM8K after SFT, the same idea as this module at larger scale.")]},
}


def _b3(v: str, title: str, why: str) -> Lecture:
    return Lecture(title, _yt(v), "", why)


COMPANIONS: dict[int, dict[str, list[Lecture]]] = {
    2: {B3: [_b3("fNk_zzaMoSs", "Vectors (Essence of linear algebra, ch. 1)", "Arrows, lists of numbers, and both."),
             _b3("kYB8IZa5AuE", "Linear transformations and matrices (ch. 3)", "A matrix is where the basis vectors land."),
             _b3("XkY2DOUCWMU", "Matrix multiplication as composition (ch. 4)", "Why multiplying matrices composes transformations."),
             _b3("Ip3X9LOh2dk", "The determinant (ch. 6)", "How much a transformation scales area."),
             _b3("uQhTuRlWMxw", "Inverse matrices, column space and null space (ch. 7)", "Undoing a transformation, and when you can't."),
             _b3("LyGKycYT2v0", "Dot products and duality (ch. 9)", "Why the dot product measures alignment."),
             _b3("PFDu9oVAE-g", "Eigenvectors and eigenvalues (ch. 14)", "Directions a matrix only stretches.")]},
    3: {ZTH: [Lecture("The spelled-out intro to neural networks and backpropagation: building micrograd",
                      _yt("VMj-3S1tku0"), "2h25m", "The same Value-class autodiff you built, live-coded.")],
        B3: [_b3("WUvTyaaNkzM", "The essence of calculus (ch. 1)", "Derivatives and integrals, visually."),
             _b3("YG15m2VwSjA", "Visualizing the chain rule and product rule (calculus ch. 4)", "The rule backprop is built on."),
             _b3("Ilg3gGewQ5U", "Backpropagation, intuitively (Deep learning ch. 3)", "What backprop is really doing."),
             _b3("tIeHLnjs5U8", "Backpropagation calculus (Deep learning ch. 4)", "The chain rule through a network.")]},
    4: {B3: [_b3("l6DKRf-fAAM", "Reinventing Entropy (Compression is Intelligence, part 1)", "Entropy from first principles."),
             _b3("GlYgs6v2YfU", "But what is cross-entropy? (Compression is Intelligence, part 2)",
                 "Cross-entropy as compression: the loss every language model minimizes.")]},
    5: {B3: [_b3("IHZwWFHWa-w", "Gradient descent, how neural networks learn (Deep learning ch. 2)",
                 "Cost functions and walking downhill.")]},
    10: {B3: [_b3("aircAruvnKk", "But what is a neural network? (Deep learning ch. 1)", "Layers, weights and activations.")]},
    11: {ZTH: [Lecture("Building makemore Part 3: Activations & Gradients, BatchNorm", _yt("P6sfmUTpUmc"), "1h55m",
                       "Activation/gradient statistics in deep nets, and BatchNorm."),
               Lecture("Building makemore Part 4: Becoming a Backprop Ninja", _yt("q8SA3rM6ckI"), "1h55m",
                       "Backpropagate by hand through cross-entropy, linear, tanh and BatchNorm layers.")]},
    12: {ZTH: [Lecture("Building makemore Part 5: Building a WaveNet", _yt("t3YJ5hKiMQ0"), "56m",
                       "Deepening the MLP into a tree of dilated causal convolutions.")]},
    13: {ZTH: [Lecture("Let's build the GPT Tokenizer", _yt("zduSFxRajkE"), "2h13m",
                       "Byte-level BPE from scratch, and the LLM quirks tokenization causes.")]},
    14: {ZTH: [Lecture("The spelled-out intro to language modeling: building makemore", _yt("PaCmpygFfXo"), "1h57m",
                       "Bigram language model, sampling and negative log-likelihood."),
               Lecture("Building makemore Part 2: MLP", _yt("TCH_1BHY58I"), "1h15m",
                       "Bengio-style MLP language model, learning-rate tuning, splits.")],
         B3: [_b3("GlYgs6v2YfU", "But what is cross-entropy? (Compression is Intelligence, part 2)",
                  "A language model's loss as bits needed to compress text.")]},
    15: {B3: [_b3("eMlx5fFNoYc", "Attention in transformers, step-by-step (Deep learning ch. 6)",
                  "Queries, keys, values and masking, visualized.")]},
    16: {ZTH: [Lecture("Let's build GPT: from scratch, in code, spelled out.", _yt("kCc8FmEb1nY"), "1h56m",
                       "A character-level GPT on Shakespeare, built up from a bigram model.")],
         B3: [_b3("wjZofJX0v4M", "Transformers, the tech behind LLMs (Deep learning ch. 5)", "The whole GPT pipeline, visually."),
              _b3("LPZh9BOjkQs", "Large Language Models explained briefly", "The big picture in eight minutes.")]},
    17: {B3: [_b3("9-Jl0dxWQs8", "How might LLMs store facts (Deep learning ch. 7)",
                  "What the MLP / feed-forward blocks (SwiGLU, experts) may be doing.")]},
    22: {ZTH: [Lecture("Let's reproduce GPT-2 (124M)", _yt("l8pRSuU81PU"), "~4h",
                       "Make GPT training fast step by step: precision, compilation, flash attention, nice numbers.")]},
    26: {RASCHKA: [Lecture(f"Build A Reasoning Model From Scratch {i}: {t}", _yt(v), "", w) for i, t, v, w in [
        (3, "The Verifier for Evaluation and RL with Verifiable Rewards", "JQJ_8_jSAoY", "The checker that is also the reward."),
        (4, "Inference Scaling 1 (Temperature, Top-p, Self-Consistency)", "t5y-kS9nNxU", "Sampling and majority voting."),
        (5, "Inference Scaling 2 (Logprob Scoring, Self-Refinement)", "TVMyOJ_3Gxo", "Scoring and critique-then-revise."),
        (6, "Reinforcement Learning 1 (Implementing GRPO for RLVR)", "237Hf7Q3lgg", "GRPO from scratch."),
        (7, "Reinforcement Learning 2 (Tracking, KL Terms, And More)", "VxfUrUrrczs", "Clipping, KL, format rewards, reward hacking."),
        (8, "Distillation", "msp69FR5NZs", "Teaching a small model from a reasoning model's answers.")]]},
}


def all_companions(module: int) -> dict[str, list[Lecture]]:
    out = {k: list(v) for k, v in COMPANIONS.get(module, {}).items()}
    for k, v in EXTRA.get(module, {}).items():
        out.setdefault(k, []).extend(v)
    return out


def companions_md(module: int) -> list[str]:
    lines = []
    for series, lectures in all_companions(module).items():
        kind = "Companion reading" if series in (JAXFORMER, SCALING, ULTRA, NANOCHAT) else "Companion lectures"
        lines += ["", f"**{kind}** — [{series.rsplit(' (', 1)[0]}]({SERIES[series]}) ({series.rsplit(' (', 1)[1]}", ""]
        lines += [f"* [{v.title}]({v.url}){f' ({v.length})' if v.length else ''}. {v.why}" for v in lectures]
    return lines


def papers_md(module: int) -> str:
    """Markdown section listing the module's seminal papers and companion lectures."""
    items = PAPERS.get(module, [])
    lines = []
    if items:
        lines += ["## Seminal papers", ""]
        lines += [f"* **[{p.title}]({p.url})** — {p.authors}, {p.year}. {p.why}" for p in items]
    lines += companions_md(module)
    return "\n".join(lines)
