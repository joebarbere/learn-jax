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
}

MODULES = [
    (0, "00-environment", "Your GPU, the container, and the three ROCm settings", 0),
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
]


def slug(n: int) -> str:
    return next(m[1] for m in MODULES if m[0] == n)
