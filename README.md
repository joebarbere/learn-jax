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

Module 27 runs your own exported models in LM Studio, headless in a container: `make lms-build lms-up`, then
`make export-models`.

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
them, by module, are in [docs/PAPERS.md](docs/PAPERS.md), followed by a short list of books.

Everything that went wrong running JAX on this AMD card, and the fix for each, is collected in
[docs/ROCM_GOTCHAS.md](docs/ROCM_GOTCHAS.md): the three environment variables the image sets, GPU hangs,
missing fused attention, slow compiles, profiling, and llama.cpp/LM Studio backends.

Claude Code in this repo works in **tutor mode** (see `CLAUDE.md`). It gives hints and questions, not
answers, unless you ask for the answer outright.

## Ways to read it

| how | command | what you get |
|---|---|---|
| Interactive notebooks | `make up`, or `make lesson N=16` | the real thing: editable code, sliders, live GPU results |
| Read-only app | `make read N=16` | one lesson as a clean page, code hidden, sliders still live |
| Markdown | `make md` | prose and code for every lesson in `markdown/`, no outputs |
| Static site | `make render` (GPU, once), then `make site` | every lesson pre-rendered, light and dark, in `site/`; no GPU needed to read |
| The book | `make render`, then `make pdf` | *Arrays to Agents*: `site/arrays-to-agents.pdf` (light, for print) and `-dark.pdf` (for screens), with contents, running heads, appendices and an index |

The site and book are the *reading edition*: prose, figures and measured results, rendered with the reference
solutions; the code and exercises stay in the notebooks. They're built in a separate, GPU-free image
(`make book-image`: Chromium, Paged.js, KaTeX, fonts). Every term the course uses is defined in
[docs/GLOSSARY.md](docs/GLOSSARY.md), with the module that teaches it. About fifty models, the building blocks
each one uses and who introduced each block are in [docs/LINEAGE.md](docs/LINEAGE.md) (drawn as a graph in
module 00b).

## Syllabus

| | Module | |
|---|---|---|
| **Part 0 — Setup** | | |
| 00 | Your GPU, the container, and the three ROCm settings | ✅ |
| 00b | A short history of AI, and where each idea lives in this course | |
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
| 24 | Parallelism and sharding (on simulated devices) | ✅ |
| 25 | JAX on AWS, and the capstone: your own small LLM end to end | ✅ |
| **Part 8 — Reasoning** | | |
| 26 | Reasoning models: inference scaling and RLVR with GRPO | ✅ |
| **Part 9 — Using your models** | | |
| 27 | Your models in LM Studio: export what you built and trained, convert to GGUF, serve and use | ✅ |
| **Part 10 — Milestones, rebuilt in JAX** | | |
| 29 | Perceptron to Hopfield networks (1958–1982) | |
| 30 | Word embeddings: word2vec (2013) | |
| 31 | RNNs, LSTMs and seq2seq with attention (1986–2014) | |
| 32 | BERT vs GPT: encoders and decoders (2018) | |
| 33 | GANs, VAEs, diffusion and CLIP (2014–2022) | |
| 34 | Deep RL: DQN to AlphaZero (2013–2019) | |
| 35 | Learning from preferences: RLHF and DPO (2017–2023) | |
| 36 | State-space models and Mamba (2021–2023) | |
| 37 | Interpretability: induction heads and world models (2020–2023) | |

✅ = written and verified. Part 10 retells the field's history by rebuilding its milestones; each module says which earlier module to take first.

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
- **Further companion reading:** [Jaxformer](https://jaxformer.com/) by **Aditya Makkar, Divya Makkar and Chinmay
  Jindal**; [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) by **Jacob Austin, Sholto Douglas, Roy
  Frostig, Anselm Levskaya et al.** (Google DeepMind); Hugging Face's
  [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook); and **Andrej Karpathy**'s
  [nanochat](https://github.com/karpathy/nanochat) (MIT), the blueprint for Module 25's capstone. Module 05's Muon
  sidebar follows **Keller Jordan**'s [post](https://kellerjordan.github.io/posts/muon/). Module 17's "Beyond RoPE" section and
  stretch experiment follow **Sebastian Raschka**'s [Understanding Positional Embeddings in Modern LLMs](https://magazine.sebastianraschka.com/p/understanding-positional-embeddings).
- **Saurabh Alone**, ["Implementing LLaMA3 in 100 Lines of Pure Jax"](https://saurabhalone.com/blog/llama3-jax).
  Its RoPE explanation and two real KV-cache bugs became Module 17's exercises.
- **Models:** [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) by Hugging Face (Apache-2.0)
  and [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF) by the Qwen Team (Apache-2.0),
  served with [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggml-org (MIT);
  [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) by the Qwen Team (Apache-2.0) for
  Module 27's tool-calling demo.
- **[LM Studio](https://lmstudio.ai/)** by Element Labs (Module 27), free for personal and internal business use
  under its [app terms](https://lmstudio.ai/app-terms). The course fetches it only inside a container build.
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
docs/         AUTHORING.md (how modules are written and verified), PAPERS.md, LINEAGE.md, ROCM_GOTCHAS.md
```

`make check` is the course's own test. Every lesson runs headless, and every test passes against the
reference solutions.
