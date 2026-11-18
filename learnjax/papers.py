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
         Paper("Transformer Language Models without Positional Encodings Still Learn Positional Information",
               "Haviv et al.", 2022, arxiv("2203.16634"), "NoPE: the causal mask alone leaks position."),
         Paper("The Impact of Positional Encoding on Length Generalization in Transformers", "Kazemnejad et al.", 2023,
               arxiv("2305.19466"), "NoPE vs RoPE, ALiBi and T5's bias on lengths longer than training."),
         Paper("Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation",
               "Press, Smith, Lewis", 2021, arxiv("2108.12409"), "ALiBi: a distance penalty on scores instead of rotation."),
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
         Paper("Language Models are Few-Shot Learners", "Brown et al.", 2020, arxiv("2005.14165"),
               "GPT-3 and in-context learning: tasks from examples in the prompt, no training (section 13)."),
         Paper("Editing Models with Task Arithmetic", "Ilharco et al.", 2022, arxiv("2212.04089"),
               "Task vectors: add fine-tuning deltas to combine skills, subtract them to remove one (section 12)."),
         Paper("TIES-Merging: Resolving Interference When Merging Models", "Yadav et al.", 2023, arxiv("2306.01708"),
               "Why naive sums interfere, and a fix: trim small changes, resolve sign conflicts, then merge."),
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
               arxiv("2307.08691"), "Better work partitioning across the GPU."),
         Paper("Decoupling algorithms from schedules for easy optimization of image processing pipelines",
               "Ragan-Kelley, Adams, Paris et al.", 2012, "https://doi.org/10.1145/2185520.2185528",
               "Who decides the schedule: Halide splits *what* a kernel computes from *how* it runs."),
         Paper("Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image "
               "processing pipelines", "Ragan-Kelley, Barnes, Adams et al.", 2013,
               "https://doi.org/10.1145/2491956.2462176",
               "The Halide compiler, with schedules searched automatically: the line XLA and Pallas descend from."),
         Paper("TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", "Chen, Moreau et al.", 2018,
               arxiv("1802.04799"), "Halide's algorithm/schedule split applied to deep learning, with learned tuning.")],
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
               "Budget forcing: controlling how long a model thinks at inference time."),
         Paper("Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "Yao et al.", 2023,
               arxiv("2305.10601"), "Reasoning as search over partial solutions (section 7)."),
         Paper("ReAct: Synergizing Reasoning and Acting in Language Models", "Yao et al.", 2022, arxiv("2210.03629"),
               "Interleave thinking with tool calls: the agent loop of Module 39."),
         Paper("Decision Transformer: Reinforcement Learning via Sequence Modeling", "Chen et al.", 2021,
               arxiv("2106.01345"), "RL as next-token prediction over (return, state, action) sequences.")],
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

def _p(title, authors, year, url, why):
    return Paper(title, authors, year, url, why)


DOI = "https://doi.org/"
ROSENBLATT = _p("The perceptron: a probabilistic model for information storage and organization in the brain",
                "Rosenblatt", 1958, DOI + "10.1037/h0042519", "The first learning machine for patterns.")
HOPFIELD = _p("Neural networks and physical systems with emergent collective computational abilities", "Hopfield", 1982,
              DOI + "10.1073/pnas.79.8.2554", "Associative memory as energy minimization; cited by the 2024 physics Nobel.")
PAPERS.update({
    28: [_p("A logical calculus of the ideas immanent in nervous activity", "McCulloch, Pitts", 1943,
            DOI + "10.1007/BF02478259", "The first mathematical model of a neuron."),
         _p("Computing Machinery and Intelligence", "Turing", 1950, DOI + "10.1093/mind/LIX.236.433",
            "Can machines think? The imitation game."),
         _p("A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence", "McCarthy, Minsky, "
            "Rochester, Shannon", 1955, "http://www-formal.stanford.edu/jmc/history/dartmouth/dartmouth.html", "The document that named the field."),
         ROSENBLATT,
         _p("Deep learning", "LeCun, Bengio, Hinton", 2015, DOI + "10.1038/nature14539",
            "The field's own summary of how deep learning came back."),
         _p("Convolutional Sequence to Sequence Learning", "Gehring et al.", 2017, arxiv("1705.03122"),
            "Family tree: learned absolute position embeddings, later used by GPT and BERT."),
         _p("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "Dosovitskiy et al.", 2020,
            arxiv("2010.11929"), "Family tree: ViT, the Transformer on image patches."),
         _p("Longformer: The Long-Document Transformer", "Beltagy, Peters, Cohan", 2020, arxiv("2004.05150"),
            "Family tree: sliding-window attention, later used by Mistral."),
         _p("Query-Key Normalization for Transformers", "Henry et al.", 2020, arxiv("2010.04245"),
            "Family tree: normalize queries and keys before attention (QK-Norm, used by Qwen3)."),
         _p("GPT-NeoX-20B: An Open-Source Autoregressive Language Model", "Black et al.", 2022, arxiv("2204.06745"),
            "Family tree: an open 20B model with partial rotary embeddings.")],
    29: [ROSENBLATT, HOPFIELD,
         _p("Hopfield Networks is All You Need", "Ramsauer et al.", 2020, arxiv("2008.02217"),
            "Modern Hopfield networks: their update rule is attention.")],
    30: [_p("Efficient Estimation of Word Representations in Vector Space", "Mikolov et al.", 2013, arxiv("1301.3781"),
            "word2vec's CBOW and skip-gram."),
         _p("Distributed Representations of Words and Phrases and their Compositionality", "Mikolov et al.", 2013,
            arxiv("1310.4546"), "Negative sampling and the famous analogies."),
         _p("GloVe: Global Vectors for Word Representation", "Pennington, Socher, Manning", 2014,
            "https://aclanthology.org/D14-1162/", "Embeddings from co-occurrence counts.")],
    31: [_p("Finding Structure in Time", "Elman", 1990, DOI + "10.1207/s15516709cog1402_1", "The simple recurrent network."),
         _p("Learning long-term dependencies with gradient descent is difficult", "Bengio, Simard, Frasconi", 1994,
            DOI + "10.1109/72.279181", "Why gradients vanish through time."),
         _p("Long Short-Term Memory", "Hochreiter, Schmidhuber", 1997, DOI + "10.1162/neco.1997.9.8.1735",
            "Gated memory cells that let gradients flow."),
         _p("Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
            "Cho et al.", 2014, arxiv("1406.1078"), "The GRU and the encoder-decoder."),
         _p("Sequence to Sequence Learning with Neural Networks", "Sutskever, Vinyals, Le", 2014, arxiv("1409.3215"),
            "seq2seq with LSTMs."),
         _p("Neural Machine Translation by Jointly Learning to Align and Translate", "Bahdanau, Cho, Bengio", 2014,
            arxiv("1409.0473"), "Attention is born here.")],
    32: [_p("Deep contextualized word representations", "Peters et al.", 2018, arxiv("1802.05365"),
            "ELMo: context-dependent embeddings."),
         _p("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "Devlin et al.", 2018,
            arxiv("1810.04805"), "Masked language modeling with an encoder."),
         _p("RoBERTa: A Robustly Optimized BERT Pretraining Approach", "Liu et al.", 2019, arxiv("1907.11692"),
            "BERT, trained properly."),
         _p("Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", "Raffel et al.", 2019,
            arxiv("1910.10683"), "T5: the encoder-decoder answer, every task as text-to-text."),
         GPT2],
    33: [_p("Generative Adversarial Networks", "Goodfellow et al.", 2014, arxiv("1406.2661"), "Two networks in a game."),
         _p("Auto-Encoding Variational Bayes", "Kingma, Welling", 2013, arxiv("1312.6114"), "The VAE."),
         _p("Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks",
            "Radford, Metz, Chintala", 2015, arxiv("1511.06434"), "DCGAN: GANs that train."),
         _p("Deep Unsupervised Learning using Nonequilibrium Thermodynamics", "Sohl-Dickstein et al.", 2015,
            arxiv("1503.03585"), "The first diffusion model."),
         _p("Denoising Diffusion Probabilistic Models", "Ho, Jain, Abbeel", 2020, arxiv("2006.11239"),
            "DDPM: diffusion that works."),
         _p("Learning Transferable Visual Models From Natural Language Supervision", "Radford et al.", 2021,
            arxiv("2103.00020"), "CLIP: contrastive image-text learning."),
         _p("High-Resolution Image Synthesis with Latent Diffusion Models", "Rombach et al.", 2021, arxiv("2112.10752"),
            "Latent diffusion, the basis of Stable Diffusion.")],
    34: [_p("Q-learning", "Watkins, Dayan", 1992, DOI + "10.1007/BF00992698", "Learning action values from experience."),
         _p("Playing Atari with Deep Reinforcement Learning", "Mnih et al.", 2013, arxiv("1312.5602"), "DQN."),
         _p("Human-level control through deep reinforcement learning", "Mnih et al.", 2015, DOI + "10.1038/nature14236",
            "DQN at human level on Atari."),
         _p("Deep Reinforcement Learning with Double Q-learning", "van Hasselt, Guez, Silver", 2016,
            arxiv("1509.06461"), "Double DQN: pick the next action with one network, score it with the other."),
         _p("Mastering the game of Go with deep neural networks and tree search", "Silver et al.", 2016,
            DOI + "10.1038/nature16961", "AlphaGo."),
         _p("Mastering the game of Go without human knowledge", "Silver et al.", 2017, DOI + "10.1038/nature24270",
            "AlphaGo Zero: self-play only."),
         _p("Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", "Silver et al.",
            2017, arxiv("1712.01815"), "AlphaZero."),
         _p("Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model", "Schrittwieser et al.", 2019,
            arxiv("1911.08265"), "MuZero: planning without the rules."),
         _p("Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning", "Koyamada et al.", 2023,
            arxiv("2303.17503"), "The JAX game environments used here.")],
    35: [_p("Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons", "Bradley, Terry", 1952,
            DOI + "10.2307/2334029", "The preference model inside every reward model."),
         _p("Deep reinforcement learning from human preferences", "Christiano et al.", 2017, arxiv("1706.03741"),
            "RLHF begins."),
         _p("Learning to summarize from human feedback", "Stiennon et al.", 2020, arxiv("2009.01325"),
            "RLHF on language models."),
         _p("Training language models to follow instructions with human feedback", "Ouyang et al.", 2022,
            arxiv("2203.02155"), "InstructGPT."),
         _p("Constitutional AI: Harmlessness from AI Feedback", "Bai et al.", 2022, arxiv("2212.08073"),
            "Preferences from AI feedback."),
         _p("Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "Rafailov et al.", 2023,
            arxiv("2305.18290"), "DPO: preferences without RL.")],
    36: [_p("Efficiently Modeling Long Sequences with Structured State Spaces", "Gu, Goel, Ré", 2021,
            arxiv("2111.00396"), "S4."),
         _p("Resurrecting Recurrent Neural Networks for Long Sequences", "Orvieto et al.", 2023, arxiv("2303.06349"),
            "The linear recurrent unit."),
         _p("Mamba: Linear-Time Sequence Modeling with Selective State Spaces", "Gu, Dao", 2023, arxiv("2312.00752"),
            "Selective SSMs."),
         _p("Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality",
            "Dao, Gu", 2024, arxiv("2405.21060"), "Mamba-2 and the SSM-attention link."),
         _p("Prefix Sums and Their Applications", "Blelloch", 1990, "https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf",
            "The parallel scan behind jax.lax.associative_scan."),
         _p("Zoology: Measuring and Improving Recall in Efficient Language Models", "Arora et al.", 2023,
            arxiv("2312.04927"), "Multi-query associative recall: where fixed-state models fall behind attention."),
         _p("Jamba: A Hybrid Transformer-Mamba Language Model", "Lieber et al.", 2024, arxiv("2403.19887"),
            "Interleave a few attention layers into a Mamba stack (1:7), plus MoE: the hybrid in section 9.")],
    37: [_p("Zoom In: An Introduction to Circuits", "Olah et al.", 2020, "https://distill.pub/2020/circuits/zoom-in/",
            "Reading neural networks like circuits."),
         _p("A Mathematical Framework for Transformer Circuits", "Elhage et al.", 2021,
            "https://transformer-circuits.pub/2021/framework/index.html", "Attention heads as readable components."),
         _p("In-context Learning and Induction Heads", "Olsson et al.", 2022,
            "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html",
            "The circuit behind in-context copying."),
         _p("Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task", "Li et al.", 2022,
            arxiv("2210.13382"), "Othello-GPT learns the board."),
         _p("Emergent Linear Representations in World Models of Self-Supervised Sequence Models", "Nanda, Lee, Wattenberg",
            2023, arxiv("2309.00941"), "The board is linearly readable."),
         _p("Toy Models of Superposition", "Elhage et al.", 2022, arxiv("2209.10652"),
            "Why neurons are polysemantic.")],
    38: [_p("The Probabilistic Relevance Framework: BM25 and Beyond", "Robertson, Zaragoza", 2009,
            DOI + "10.1561/1500000019", "BM25 explained by its authors: where idf and term saturation come from."),
         _p("Reciprocal rank fusion outperforms Condorcet and individual rank learning methods",
            "Cormack, Clarke, Büttcher", 2009, DOI + "10.1145/1571941.1572114", "RRF: fuse rankings with 1/(k + rank)."),
         _p("Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", "Reimers, Gurevych", 2019,
            arxiv("1908.10084"), "One encoder, mean pooling, cosine similarity: the bi-encoder recipe."),
         _p("Passage Re-ranking with BERT", "Nogueira, Cho", 2019, arxiv("1901.04085"),
            "The cross-encoder reranker: read query and passage together."),
         _p("Latent Retrieval for Weakly Supervised Open Domain Question Answering", "Lee, Chang, Toutanova", 2019,
            arxiv("1906.00300"), "The Inverse Cloze Task: pretrain a retriever with sentences as pseudo-queries."),
         _p("A Language Modeling Approach to Information Retrieval", "Ponte, Croft", 1998, DOI + "10.1145/290941.291008",
            "Query likelihood: rank a document by how probably it generates the query (the reranker in section 8)."),
         _p("Improving Passage Retrieval with Zero-Shot Question Generation", "Sachan et al.", 2022, arxiv("2204.07496"),
            "UPR: rerank with an LLM's log p(question | passage), no relevance training."),
         _p("Dense Passage Retrieval for Open-Domain Question Answering", "Karpukhin et al.", 2020, arxiv("2004.04906"),
            "DPR: InfoNCE with in-batch negatives beats BM25 on open QA."),
         _p("ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
            "Khattab, Zaharia", 2020, arxiv("2004.12832"), "One vector per token: between bi- and cross-encoders."),
         _p("Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "Lewis et al.", 2020,
            arxiv("2005.11401"), "RAG: retrieve passages, then generate conditioned on them."),
         _p("Billion-scale similarity search with GPUs", "Johnson, Douze, Jégou", 2017, arxiv("1702.08734"),
            "Faiss: IVF and product quantization on GPUs, the approximate search behind vector databases."),
         _p("Improving language models by retrieving from trillions of tokens", "Borgeaud et al.", 2021,
            arxiv("2112.04426"), "RETRO: retrieval inside the model, a 7.5B model matching 25x larger ones.")],
    39: [_p("ReAct: Synergizing Reasoning and Acting in Language Models", "Yao et al.", 2022, arxiv("2210.03629"),
            "Interleave thoughts with tool calls and observations: the loop this module builds."),
         _p("Toolformer: Language Models Can Teach Themselves to Use Tools", "Schick et al.", 2023,
            arxiv("2302.04761"), "A model that learns where an API call helps, from its own annotations."),
         _p("Calibrate Before Use: Improving Few-Shot Performance of Language Models", "Zhao et al.", 2021,
            arxiv("2102.09690"), "Small models lean toward some labels regardless of input; a content-free prompt "
            "measures that bias so you can subtract it (the tool-choice fix in section 3)."),
         _p("Generative Agents: Interactive Simulacra of Human Behavior", "Park et al.", 2023, arxiv("2304.03442"),
            "A memory stream retrieved by recency, importance and relevance (the score in section 5)."),
         _p("Reflexion: Language Agents with Verbal Reinforcement Learning", "Shinn et al.", 2023,
            arxiv("2303.11366"), "An agent that critiques its failed attempts and retries: evaluator-optimizer."),
         _p("MemGPT: Towards LLMs as Operating Systems", "Packer et al.", 2023, arxiv("2310.08560"),
            "Context as RAM, external storage as disk: an agent that pages its own memory."),
         _p("AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation", "Wu et al.", 2023,
            arxiv("2308.08155"), "Multi-agent conversation as a programming model (the framework, now in maintenance)."),
         _p("Fundamentals of Building Autonomous LLM Agents", "de Lamo Castrillo et al.", 2025, arxiv("2510.09244"),
            "A survey (no experiments): perception, reasoning, memory, execution; useful vocabulary.")],
    40: [_p("LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts", "Elango et al.", 2026,
            arxiv("2601.18089"), "Routed experts in a latent width d/α, α× as many: same bytes, better accuracy."),
         _p("NVIDIA Nemotron 3: Efficient and Open Intelligence", "NVIDIA", 2025, arxiv("2512.20856"),
            "LatentMoE in a hybrid Mamba/attention model family (Super and Ultra)."),
         _p("Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models",
            "Cheng et al. (DeepSeek-AI)", 2026, arxiv("2601.07372"),
            "Engram: hashed n-gram tables beside the experts, and how to split the sparse budget between them."),
         _p("Hash Embeddings for Efficient Word Representations", "Svenstrup, Hansen, Winther", 2017,
            arxiv("1709.03933"), "Several hashes per token, combined, so collisions in one hash don't matter."),
         _p("Large Memory Layers with Product Keys", "Lample et al.", 2019, arxiv("1907.05242"),
            "An earlier big lookup memory inside a Transformer, chosen by learned keys instead of by hashing.")],
})


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
AHEAD = "Sebastian Raschka, *Ahead of AI* (newsletter, PyTorch)"
KISS = "Csongor Kiss, *Write your own kernel optimiser* (blog post, 2026)"
SERIES = {ZTH: "https://karpathy.ai/zero-to-hero.html",
          JAXFORMER: "https://jaxformer.com/",
          SCALING: "https://jax-ml.github.io/scaling-book/",
          ULTRA: "https://huggingface.co/spaces/nanotron/ultrascale-playbook",
          NANOCHAT: "https://github.com/karpathy/nanochat",
          KISS: "https://blog.csongor.co.uk/cachet-grids-and-tiles/",
          B3: "https://www.3blue1brown.com/?topic=neural-networks",
          RASCHKA: "https://magazine.sebastianraschka.com/p/building-reasoning-models",
          AHEAD: "https://magazine.sebastianraschka.com/"}


def _r(url: str, title: str, why: str) -> Lecture:
    """A written chapter or repo page (not a video)."""
    return Lecture(title, url, "", why)


JF, SB = "https://jaxformer.com/", "https://jax-ml.github.io/scaling-book/"
NC = "https://github.com/karpathy/nanochat"
EXTRA: dict[int, dict[str, list[Lecture]]] = {
    13: {JAXFORMER: [_r(JF + "tokenization/", "Part 1: Tokenization at scale", "Tokenizing FineWeb-EDU with multiprocessing and Ray.")]},
    16: {SCALING: [_r(SB + "transformers", "All the Transformer math you need to know", "FLOPs, parameter counts and KV-cache sizes — the reference for this module's FLOPs/token and MFU.")],
         NANOCHAT: [_r(NC, "nanochat", "nanoGPT's successor: the same GPT, plus tokenizer, SFT, RL, eval and a chat engine.")]},
    17: {SCALING: [_r(SB + "transformers", "All the Transformer math you need to know", "FLOPs, parameter counts and KV-cache sizes — including the KV-cache accounting behind GQA and MLA.")],
         AHEAD: [_r("https://magazine.sebastianraschka.com/p/understanding-positional-embeddings",
                    "Understanding Positional Embeddings in Modern LLMs",
                    "Learned, sinusoidal, RoPE, partial RoPE, NoPE and hybrids, with a 2026 open-weight model survey.")],
         JAXFORMER: [_r(JF + "base_model/", "Part 2: How to write the base model", "RMSNorm, RoPE and multi-latent attention in JAX."),
                     _r(JF + "moe/", "Part 6: Implementing mixture-of-experts layers", "Sigmoid top-k routing, shared experts, load balancing.")]},
    18: {SCALING: [_r(SB + "inference", "All about Transformer inference", "Prefill vs decode, KV-cache costs, batching, latency."),
                   _r(SB + "applied-inference", "Serving LLaMA 3-70B", "The inference math applied to a real model.")]},
    21: {NANOCHAT: [_r(NC, "nanochat's evaluation (base_eval, chat_eval)", "CORE score, bits per byte, ARC, GSM8K, MMLU in one harness.")]},
    22: {SCALING: [_r(SB + "roofline", "All about rooflines", "Compute vs memory vs communication bounds, worked out."),
                   _r(SB + "profiling", "How to profile TPU programs", "Reading JAX profiler traces (TPU-focused, same ideas on GPU).")]},
    23: {SCALING: [_r(SB + "gpus", "How to think about GPUs", "SMs, memory hierarchy and collectives from the scaling-book view.")],
         KISS: [_r("https://blog.csongor.co.uk/cachet-grids-and-tiles/", "Write your own kernel optimiser",
                   "A matmul climbing from 1% of OpenBLAS to parity by checked schedule rewrites (the Cachet language)"
                   " — this module's speed ladder at the loop level.")]},
    24: {JAXFORMER: [_r(JF + "sharded/", "Part 3: Inside the sharded model", "Mesh, shard_map, FSDP, GPipe pipelines and tensor parallelism.")],
         SCALING: [_r(SB + "sharding", "Sharded matrices and how to multiply them", "Which collective each sharded matmul needs."),
                   _r(SB + "training", "How to parallelize a Transformer for training",
                      "FSDP, Megatron tensor parallelism and pipelines, and when each becomes communication-bound."),
                   _r(SB + "jax-stuff", "Programming TPUs in JAX", "Explicit sharding and shard_map from the JAX side.")],
         ULTRA: [_r("https://huggingface.co/spaces/nanotron/ultrascale-playbook", "The Ultra-Scale Playbook",
                    "DP, ZeRO/FSDP, TP, PP, context and expert parallelism, measured on GPU clusters.")]},
    25: {JAXFORMER: [_r(JF + "dataset/", "Part 4: Dataset class and config files", "Sharded data placement."),
                     _r(JF + "distributed_training/", "Part 5: Distributed training", "jax.distributed, Orbax checkpoints, the train loop."),
                     _r(JF + "training/", "Part 7: Final run and training results", "A 949M-param MoE on 32 TPU-v4 chips, multi-host.")],
         SCALING: [_r(SB + "applied-training", "Training LLaMA 3 on TPUs", "Choosing a parallelism layout for a real run."),
                   _r(SB + "gpus", "How to think about GPUs", "NVLink, InfiniBand and GPU collectives: the networking behind multi-GPU runs.")],
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
        kind = "Companion reading" if series in (JAXFORMER, SCALING, ULTRA, NANOCHAT, KISS) else "Companion lectures"
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


# ---------------------------------------------------------------------------------------------- books
# Rendered at the end of docs/PAPERS.md. Free official editions first; `modules` = where each one helps most.
class Book(NamedTuple):
    title: str
    authors: str
    year: int
    url: str
    why: str
    modules: str


BOOKS = [
    Book("Dive into Deep Learning", "Zhang, Lipton, Li, Smola", 2023, "https://d2l.ai/",
         "Free, interactive, and has JAX implementations alongside PyTorch: the closest textbook to this course.",
         "01–17"),
    Book("Understanding Deep Learning", "Simon J. D. Prince", 2023, "https://udlbook.github.io/udlbook/",
         "Free. Clear figures and intuition for everything from backprop to transformers and diffusion.", "03–17, 33"),
    Book("Deep Learning: Foundations and Concepts", "Christopher M. Bishop, Hugh Bishop", 2024,
         "https://www.bishopbook.com/", "Free to read online. The probabilistic view, carefully derived.", "04–12, 33"),
    Book("Deep Learning", "Goodfellow, Bengio, Courville", 2016, "https://www.deeplearningbook.org/",
         "Free. The 2016 reference: optimization, regularization and the math behind modules 02–12.", "02–12"),
    Book("Probabilistic Machine Learning: An Introduction", "Kevin P. Murphy", 2022,
         "https://probml.github.io/pml-book/book1.html", "Free. Probability and linear models done properly.", "04, 09"),
    Book("Reinforcement Learning: An Introduction (2nd ed.)", "Richard S. Sutton, Andrew G. Barto", 2018,
         "http://incompleteideas.net/book/the-book-2nd.html", "Free. The RL textbook: Q-learning, policy gradients.",
         "26, 34, 35"),
    Book("Speech and Language Processing (3rd ed. draft)", "Dan Jurafsky, James H. Martin", 2025,
         "https://web.stanford.edu/~jurafsky/slp3/", "Free. N-grams to transformers, from the NLP side.", "13–16, 30–32"),
    Book("Reinforcement Learning from Human Feedback", "Nathan Lambert", 2025, "https://rlhfbook.com/",
         "Free. Post-training: reward models, RLHF, DPO, RLVR.", "26, 35"),
    Book("Build a Large Language Model (From Scratch)", "Sebastian Raschka", 2024,
         "https://www.manning.com/books/build-a-large-language-model-from-scratch",
         "GPT from tokenizer to fine-tuning, in PyTorch; a good second pass over modules 13–20.", "13–20"),
    Book("AI Engineering", "Chip Huyen", 2025, "https://huyenchip.com/books/",
         "Building products on foundation models: evaluation, RAG, agents, fine-tuning trade-offs.", "21, 38, 39"),
    Book("Artificial Intelligence: A Modern Approach (4th ed.)", "Stuart Russell, Peter Norvig", 2020,
         "https://aima.cs.berkeley.edu/", "The classic survey of AI beyond deep learning: search, logic, planning.",
         "00b, 34"),
    Book("Artificial Intelligence: A Guide for Thinking Humans", "Melanie Mitchell", 2019,
         "https://melaniemitchell.me/aibook/", "A readable history of AI's hype cycles and real limits.", "00b"),
]
