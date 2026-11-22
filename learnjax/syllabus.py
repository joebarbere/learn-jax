"""The course outline: (number, directory slug, title, part)."""

PARTS = {
    0: "Setup",
    1: "Math in code",
    2: "JAX core",
    3: "Neural networks",
    4: "Sequences, attention, transformers",
    5: "LLMs in practice",
    6: "Performance & ROCm",
    7: "Scale & cloud",
    8: "Reasoning",
    9: "Using your models",
    10: "Milestones, rebuilt in JAX",
    11: "Systems around the model: retrieval and agents",
    12: "Frontier architectures",
}

MODULES = [
    (0, "00-environment", "Your GPU, the container, and the three ROCm settings", 0),
    (28, "00b-history", "A short history of AI, and where each idea lives in this course", 0),
    (1, "01-arrays", "Arrays: NumPy → jax.numpy, shapes, broadcasting, dtypes", 1),
    (2, "02-linear-algebra", "Linear algebra as geometry", 1),
    (3, "03-autodiff", "Derivatives and autodiff, built from scratch", 1),
    (4, "04-probability", "Probability, softmax, cross-entropy and JAX random keys", 1),
    (5, "05-optimization", "Optimization: gradient descent to Adam", 1),
    (6, "06-jit-xla", "jit and XLA: tracing, compilation, timing", 2),
    (7, "07-vmap-pytrees", "vmap, pytrees and functional state", 2),
    (8, "08-control-flow-numerics", "Control flow, scan, and mixed precision", 2),
    (9, "09-regression", "Linear and logistic regression from scratch", 3),
    (10, "10-mlp", "MLPs on Fashion-MNIST, then Flax NNX", 3),
    (11, "11-training-craft", "Training craft: init, norms, schedules, checkpoints", 3),
    (12, "12-cnns", "Convolutional networks", 3),
    (13, "13-tokenization", "Tokenization: build BPE yourself", 4),
    (14, "14-language-modeling", "Language modeling basics: bigrams to MLP LMs", 4),
    (15, "15-attention", "Attention from scratch", 4),
    (16, "16-gpt", "Build and train a GPT on your GPU", 4),
    (17, "17-modern-llm", "Modern LLM architecture: RoPE, RMSNorm, SwiGLU, MLA, MoE", 4),
    (18, "18-inference", "Inference: KV cache, sampling, quantization", 5),
    (19, "19-local-serving", "Local serving for coding assistants", 5),
    (20, "20-lora", "Fine-tuning and LoRA", 5),
    (21, "21-evaluation", "Evaluating language models", 5),
    (22, "22-profiling", "Profiling JAX on ROCm", 6),
    (23, "23-kernels", "Kernels: Pallas, HIP and fusion", 6),
    (24, "24-parallelism", "Parallelism and sharding", 7),
    (25, "25-aws", "JAX on AWS, and the capstone", 7),
    (26, "26-reasoning", "Reasoning models: inference scaling and RLVR with GRPO", 8),
    (27, "27-lmstudio", "Your models in LM Studio: export, convert, serve", 9),
    (29, "29-perceptron-hopfield", "Perceptron to Hopfield networks (1958–1982)", 10),
    (30, "30-word2vec", "Word embeddings: word2vec (2013)", 10),
    (31, "31-rnn-seq2seq", "RNNs, LSTMs and seq2seq with attention (1986–2014)", 10),
    (32, "32-bert-vs-gpt", "BERT vs GPT: encoders and decoders (2018)", 10),
    (33, "33-generative-images", "GANs, VAEs, diffusion and CLIP (2014–2022)", 10),
    (34, "34-deep-rl-alphazero", "Deep RL: DQN to AlphaZero (2013–2019)", 10),
    (35, "35-rlhf-dpo", "Learning from preferences: RLHF and DPO (2017–2023)", 10),
    (36, "36-mamba-ssm", "State-space models and Mamba (2021–2023)", 10),
    (37, "37-interpretability", "Interpretability: induction heads and world models (2020–2023)", 10),
    (38, "38-rag", "Retrieval-augmented generation, from BM25 to rerankers (1994–2024)", 11),
    (39, "39-agents", "Agents: tools, memory, workflows and multi-agent systems (2022–)", 11),
    (40, "40-latentmoe-engram", "Latent experts and conditional memory: LatentMoE and Engram (2025–2026)", 12),
    (41, "41-linear-attention-hybrids", "Linear attention and hybrids: from fast weights to Gated DeltaNet (2020–2026)",
     12),
    (42, "42-small-models", "Small models done well: distillation, pruning and borrowing from big models (2015–2026)",
     12),
    (43, "43-vision-language", "From pixels to words: vision transformers and vision-language models (2020–2025)", 12),
    (44, "44-sparse-attention", "Efficient and sparse attention: reading only part of the past (2019–2026)", 12),
    (45, "45-diffusion-lm", "Diffusion language models: writing a whole passage at once (2021–2026)", 12),
    (46, "46-weights-by-hand", "Weights by hand: compiling, merging and editing networks (1990s–2026)", 12),
    (47, "47-reading-activations", "Reading activations: lenses, sparse autoencoders and steering (2020–2025)", 12),
    (48, "48-tracing-circuits", "Tracing a circuit: patching, transcoders and attribution graphs (2022–2025)", 12),
    (50, "50-uncensored-models", "Uncensored models: removing refusals from the data and from the weights (2023–2024)",
     12),
]


def slug(n: int) -> str:
    return next(m[1] for m in MODULES if m[0] == n)


def label(slug_: str) -> str:
    """Display label: the slug's prefix ("00", "00b", "29")."""
    return slug_.split("-", 1)[0]


def number(slug_: str) -> int:
    """Registry number for a module directory name (e.g. "00b-history" -> 28)."""
    return next(m[0] for m in MODULES if m[1] == slug_)
