# learn-jax

A hands-on course in **deep learning, JAX, large language models and ROCm**, written for a developer
who is new to ML and owns an AMD GPU. Every lesson is a [marimo](https://marimo.io) notebook that runs
in a container on the GPU (developed on a Radeon RX 7600 XT, 16 GB). Nothing ML-related is installed
on the host.

The course goes from arrays and derivatives to training your own small mixture-of-experts LLM. Along
the way it covers LoRA fine-tuning, local inference for coding assistants, profiling on ROCm, sharding,
and running the same JAX code on NVIDIA GPUs in AWS.

## Setup

You need Linux with the in-kernel `amdgpu` driver (any recent distro), podman, and an RDNA2/3/4 Radeon.
`/dev/kfd` and `/dev/dri/renderD*` must be accessible to your user. On Fedora they are by default. On
other distros, add your user to the `render` and `video` groups.

```bash
make build          # JAX image: rocm/jax (ROCm 10.1, JAX 0.11.1) + marimo, Flax, Optax (~19 GB)
make up             # marimo on http://127.0.0.1:2718 — `make url` prints the link with its token
make progress       # the syllabus, with ✓ for modules whose exercises pass
make help           # everything else
```

`make build-torch` builds the PyTorch image used by the side-by-side notebooks (`make torch-lesson N=10`).

Module 19 runs a local coding model with llama.cpp: `make llama-download` (~12 GB of GGUF models), then
`make llama-up` serves an OpenAI-compatible API on http://127.0.0.1:8080 (Vulkan by default;
`BACKEND=rocm` also works). Lessons reach it automatically once it's up.

**On a card other than the RX 7600 / 7600 XT,** check the three settings in `containers/Containerfile`.
Module 00 explains each one. Your card may not need the gfx override.

## How to take it

Each module in `modules/NN-slug/` has:

- `lesson.py`: read it, run it, move the sliders. `make lesson N=03` opens just that one.
- `exercises.py`: functions for you to implement. The lesson calls them, so cells marked
  **Your turn** come alive as you finish them.
- `test_exercises.py`: run `make test N=03` to grade your work.
- `solutions.py`: reference answers. Try first, peek last.

Each lesson ends with its **seminal papers**: the original sources for the ideas it teaches. All of
them, by module, are in [docs/PAPERS.md](docs/PAPERS.md).

Claude Code in this repo works in **tutor mode** (see `CLAUDE.md`). It gives hints and questions, not
answers, unless you ask for the answer outright.

## Syllabus

| | Module | |
|---|---|---|
| **Part 0 — Setup** | | |
| 00 | Your GPU, the container, and the three ROCm settings | ✅ |
| **Part 1 — Math in code** | | |
| 01 | Arrays: NumPy → jax.numpy, shapes, broadcasting, dtypes | ✅ |
| 02 | Linear algebra as geometry | ✅ |
| 03 | Derivatives and autodiff, built from scratch | ✅ |
| 04 | Probability, softmax, cross-entropy and JAX random keys | ✅ |
| 05 | Optimization: gradient descent to Adam | ✅ |
| **Part 2 — JAX core** | | |
| 06 | jit and XLA: tracing, compilation, timing | ✅ |
| 07 | vmap, pytrees and functional state | ✅ |
| 08 | Control flow, scan, and mixed precision | ✅ |
| **Part 3 — Neural networks** | | |
| 09 | Linear and logistic regression from scratch | ✅ |
| 10 | MLPs on Fashion-MNIST, then Flax NNX *(+ PyTorch)* | ✅ |
| 11 | Training craft: init, norms, schedules, checkpoints | ✅ |
| 12 | Convolutional networks | ✅ |
| **Part 4 — Sequences, attention, transformers** | | |
| 13 | Tokenization: build BPE yourself | ✅ |
| 14 | Language modeling basics: bigrams to MLP LMs | ✅ |
| 15 | Attention from scratch *(+ PyTorch)* | ✅ |
| 16 | Build and train a GPT on your GPU *(+ PyTorch)* | ✅ |
| 17 | Modern LLM architecture: RoPE, RMSNorm, SwiGLU, MLA, MoE | ✅ |
| **Part 5 — LLMs in practice** | | |
| 18 | Inference: KV cache, sampling, quantization | ✅ |
| 19 | Local serving for coding assistants (llama.cpp on ROCm) | ✅ |
| 20 | Fine-tuning and LoRA *(+ PyTorch/PEFT)* | ✅ |
| 21 | Evaluating language models | ✅ |
| **Part 6 — Performance & ROCm** | | |
| 22 | Profiling JAX on ROCm | ✅ |
| 23 | Kernels: Pallas, HIP and fusion | ✅ |
| **Part 7 — Scale & cloud** | | |
| 24 | Parallelism and sharding (on simulated devices) | |
| 25 | JAX on AWS, and the capstone: your own small LLM end to end | |
| **Part 8 — Reasoning** | | |
| 26 | Reasoning models: inference scaling and RLVR with GRPO | |

✅ = written and verified. The rest are being built in phases.

## Credits

This course stands on other people's work. Every lesson cites its sources inline: authors for papers,
creators for videos, posts and docs. Most of all:

- **Vaibhaav Tiwari**, [deepseek-v3-jax](https://github.com/Vaibhaav-Tiwari/deepseek-v3-jax) (MIT). It started
  this course. Module 17 quotes its model code, and Modules 16 and 21 train and evaluate on its published corpus.
- **The JAX team**, the [JAX documentation](https://docs.jax.dev/en/latest/jax.html), including
  [The Training Cookbook](https://docs.jax.dev/en/latest/the-training-cookbook.html). Each lesson links the
  first mention of every JAX API to its reference page.
- **Andrej Karpathy**, [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html),
  **Grant Sanderson**, [3Blue1Brown](https://www.3blue1brown.com/?topic=neural-networks), and
  **Sebastian Raschka**, [Build a Reasoning Model (From Scratch)](https://magazine.sebastianraschka.com/p/building-reasoning-models).
  These are companion lectures, listed in each lesson and in [docs/PAPERS.md](docs/PAPERS.md).
- **Saurabh Alone**, ["Implementing LLaMA3 in 100 Lines of Pure Jax"](https://saurabhalone.com/blog/llama3-jax).
  Its RoPE explanation and two real KV-cache bugs became Module 17's exercises.
- **Models:** [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) by Hugging Face (Apache-2.0)
  and [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF) by the Qwen Team (Apache-2.0),
  served with [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggml-org (MIT).
- **Data:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research (MIT);
  [tiny Shakespeare](https://github.com/karpathy/char-rnn) from Andrej Karpathy's char-rnn; public-domain
  detective novels from Standard Ebooks and Project Gutenberg, via deepseek-v3-jax;
  [ARC](https://allenai.org/data/arc) by the Allen Institute for AI (CC BY-SA 4.0).
- **Papers:** the 90-plus seminal papers are credited to their authors in every lesson and in docs/PAPERS.md.

## Layout

```
containers/   Containerfile (JAX), Containerfile.torch, run.sh (GPU passthrough wrapper)
learnjax/     small shared helpers: exercise loading, timing, GPU info, syllabus, progress
modules/      one directory per module
tools/        lesson_outputs.py: run a lesson and print what each cell shows
docs/         AUTHORING.md: how modules are written and verified
```

`make check` is the course's own test. Every lesson runs headless, and every test passes against the
reference solutions.
