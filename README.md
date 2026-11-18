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
make build          # JAX image: rocm/jax (ROCm 10.1, JAX 0.11.1) + marimo, Flax, Optax (~9 GB download, ~19 GB on disk)
make up             # marimo on http://127.0.0.1:2718 — `make url` prints the link with its token
make progress       # the syllabus, with ✓ for modules whose exercises pass
make help           # everything else
```

Python packages are installed with [uv](https://docs.astral.sh/uv/) from lockfiles
(`containers/requirements*.lock`). Every package each image adds is pinned there, resolved inside the ROCm
base image so the base's JAX and ROCm wheels never move. To change a dependency, edit
`containers/requirements*.txt`, then `make lock` and `make build`. On the host you only need uv if you want
`make glossary` or ruff in your editor (`uv sync`). Nothing ML-related is installed there.

`make build-torch` builds the PyTorch image used by the side-by-side notebooks (`make torch-lesson N=10`);
it's optional and large, see *Disk space* below.

Module 27 runs your own exported models in LM Studio, headless in a container: `make lms-build lms-up`, then
`make export-models`.

Module 19 runs a local coding model with llama.cpp: `make llama-download` (~12 GB of GGUF models, listed under
*Disk space*), then `make llama-up` serves an OpenAI-compatible API on http://127.0.0.1:8080 (Vulkan by default;
`BACKEND=rocm` also works; its image is a 7.7 GB download, 23 GB on disk, pulled on first use). Lessons reach it
automatically once it's up.

### Disk space

**What you're getting into.** The minimum is the JAX image: about **9 GB to download, 19 GB on disk**, plus a
couple of GB of datasets and weights that lessons fetch into the `learn-jax-cache` volume as you go. Everything
else is optional and per-module. Taking the whole course with every extra is roughly **50 GB to download and
100 GB on disk**. Download sizes are compressed (what actually crosses the network), and disk sizes are after
podman unpacks them.

The course's container images are big, but every one can be removed and restored exactly: images are built
from the lockfiles or pulled by pinned digest. Your data lives elsewhere (the `learn-jax-cache` volume and
`runs/`), so removing an image loses nothing.

| image | download | on disk | needed for | remove | restore |
|---|---|---|---|---|---|
| `localhost/learn-jax` | ~9 GB (8.3 GB `rocm/jax` base + packages) | ~19 GB | everything | — | `make build` |
| `localhost/learn-jax-torch` (+ its `rocm/pytorch` base) | ~12 GB (11.8 GB base + packages) | ~26 GB | PyTorch side-by-side notebooks, `make export-models`' check | `podman rmi localhost/learn-jax-torch docker.io/rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0` | `make build-torch` |
| llama.cpp `server-rocm` | 7.7 GB | 23 GB | `BACKEND=rocm` in Module 19 | `podman rmi ghcr.io/ggml-org/llama.cpp:server-rocm-b12375` | automatic on next `BACKEND=rocm make llama-up` |
| llama.cpp `server-vulkan`, `full-vulkan` | 0.3 + 1.1 GB | 0.9 + 2.7 GB | Module 19 (default), Module 27's GGUF conversion | — | automatic on next use |
| `localhost/learn-jax-lmstudio` | LM Studio's binaries, fetched at build | ~10 GB | Module 27 | `podman rmi localhost/learn-jax-lmstudio` | `make lms-build` |
| `localhost/learn-jax-book` | under 1.4 GB (`python:3.12-slim` + Chromium, fonts, KaTeX) | ~1.4 GB | `make site`, `make pdf` | `podman rmi localhost/learn-jax-book` | `make book-image` |

Models and data download into the `learn-jax-cache` volume, once, the first time something needs them (download
and disk size are the same, since GGUF and weight files don't compress):

| what | size | needed for | fetched by |
|---|---|---|---|
| Qwen2.5-Coder-14B-Instruct Q4_K_M (the default served model) | 9.0 GB | Module 19 | `make llama-download` |
| Qwen2.5-Coder-1.5B-Instruct Q4_K_M and Q8_0 | 1.1 + 1.9 GB | Module 19 (small/fast and quantization comparisons) | `make llama-download` |
| Qwen2.5-1.5B-Instruct Q4_K_M | 1.1 GB | Module 27 (tool calling: the Coder models don't emit the tags LM Studio parses) | `make export-models` |
| Hugging Face weights and tokenizers (GPT-2, SmolLM2-360M, …) and datasets (Fashion-MNIST, Tiny Shakespeare, the detective corpus, …) | ~2 GB | the lessons that use them | automatic, on first use |

All four GGUF models are Apache-2.0, from Qwen's official GGUF repos on Hugging Face.

If an image is missing, `containers/run.sh` stops and prints the command that restores it, and `make check`
skips the PyTorch notebooks instead of failing. Don't prune podman volumes: that would delete
`learn-jax-cache`, which holds the downloaded weights, datasets and GGUF models (about 15 GB to download again).

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
module 00b). Every math operation the course introduces is written five ways (the formula as papers
write it, plain Python loops, NumPy, JAX and PyTorch, all tested to agree) in [docs/ROSETTA.md](docs/ROSETTA.md),
with a guide to reading paper notation and to translating NumPy into JAX. The lessons show each one as tabs where
it's first taught; `make rosetta-browse` opens them all in one notebook, and the book has them as Appendix E.

## Syllabus

| | Module | |
|---|---|---|
| **Part 0 — Setup** | | |
| 00 | Your GPU, the container, and the three ROCm settings | ✅ |
| 00b | A short history of AI, and where each idea lives in this course | ✅ |
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
| 29 | Perceptron to Hopfield networks (1958–1982) | ✅ |
| 30 | Word embeddings: word2vec (2013) | ✅ |
| 31 | RNNs, LSTMs and seq2seq with attention (1986–2014) | ✅ |
| 32 | BERT vs GPT: encoders and decoders (2018) | ✅ |
| 33 | GANs, VAEs, diffusion and CLIP (2014–2022) | ✅ |
| 34 | Deep RL: DQN to AlphaZero (2013–2019) | ✅ |
| 35 | Learning from preferences: RLHF and DPO (2017–2023) | ✅ |
| 36 | State-space models and Mamba (2021–2023) | ✅ |
| 37 | Interpretability: induction heads and world models (2020–2023) | ✅ |
| **Part 11 — Systems around the model: retrieval and agents** | | |
| 38 | Retrieval-augmented generation, from BM25 to rerankers (1994–2024) | ✅ |
| 39 | Agents: tools, memory, workflows and multi-agent systems (2022–) | ✅ |
| **Part 12 — Frontier architectures** | | |
| 40 | Latent experts and conditional memory: LatentMoE and Engram (2025–2026) | ✅ |

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
- **DeepSeek-AI**, [Engram](https://github.com/deepseek-ai/Engram) (Apache-2.0). Module 40's n-gram hashing and
  tokenizer compression follow its `engram_demo_v1.py` and are checked against it bit for bit. Module 40's
  LatentMoE follows **Venmugil Elango et al.** (NVIDIA), [LatentMoE](https://arxiv.org/abs/2601.18089).
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
  [ARC](https://allenai.org/data/arc) by the Allen Institute for AI (CC BY-SA 4.0); 49,950 arXiv abstracts
  from the [aws-ai](https://github.com/joebarbere/aws-ai) `local-kb` benchmark (Module 38).
- **Papers:** the 90-plus seminal papers are credited to their authors in every lesson and in docs/PAPERS.md.

## License

- **Code** (Python, notebook code cells, the Makefile, containers, tools, infra): [MIT](LICENSE).
- **Prose** (the lessons' explanatory text, docs/, diagrams, and the book *Arrays to Agents*):
  [CC BY 4.0](LICENSE-CONTENT). Share and adapt it freely; credit "Joe Barbere, learn-jax" and link the repo.

Third-party material keeps its own license: code quoted from deepseek-v3-jax (MIT, Vaibhaav Tiwari), and the
models, datasets and tools listed under Credits.

## Layout

```
containers/   Containerfile (JAX), Containerfile.torch, run.sh (GPU passthrough wrapper)
learnjax/     small shared helpers: exercise loading, timing, GPU info, syllabus, progress; rosetta/ (math ↔ code)
modules/      one directory per module
tests/        test_rosetta.py: every math ↔ code entry agrees across versions and is placed in a lesson
tools/        lesson_outputs.py: run a lesson and print what each cell shows
docs/         AUTHORING.md (how modules are written and verified), PAPERS.md, LINEAGE.md, ROSETTA.md, ROCM_GOTCHAS.md
```

`make check` is the course's own test. Every lesson runs headless, and every test passes against the
reference solutions.
