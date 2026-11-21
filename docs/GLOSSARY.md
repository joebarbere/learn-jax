# Glossary

597 terms from the course, each with the module that teaches it and the others where it matters; read a term's module section for the full story.

[#](#symbols) · [A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [J](#j) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w) · [X](#x)

<a id="symbols"></a>

## Numbers & symbols

- **6N rule** — Training costs about 6 FLOPs per parameter per token: 2 for the forward pass and 4 for the backward. *(Module 16.)* Origin: Kaplan et al., 2020.
- **µP** — A way to scale initialization and learning rates with width so the best hyperparameters found on a narrow model still work on a wide one. *(Module 42.)* Origin: Yang et al., 2022.
- **ε-greedy** — Take a random action with probability ε, otherwise the best-known one: the simplest balance of exploration and exploitation. *(Module 34.)*

<a id="a"></a>

## A

- **ablation** — Switching off a component (e.g. zeroing a head's output) and measuring what breaks, to test its role. *(Module 37; also 34.)*
- **acceptance rate** — In speculative decoding, the fraction of draft tokens the target model accepts; it decides the speed-up. *(Module 18.)*
- **accumulation** — Summing many products into one result. Doing it in float32 even when inputs are bfloat16 (`preferred_element_type=float32`) avoids losing small terms. *(Module 08; also 23.)*
- **accuracy** — The fraction of predictions that are exactly right. *(Module 09; also 10, 12, 15, 21, 26.)*
- **activation function** — The elementwise nonlinear function between layers. Without it, stacked linear layers collapse into one linear map. *(Module 10; also 31.)*
- **activations** — The intermediate values a network computes in the forward pass. They must be stored for the backward pass, which often dominates training memory. *(Module 10; also 11, 17, 20, 25, 37.)*
- **Adam** — Momentum plus RMSProp, with bias correction for the early steps. The default optimizer for deep learning. *(Module 05; also 09, 12, 14, 20, 24.)* Origin: Kingma & Ba, 2014.
- **AdamW** — Adam with decoupled weight decay: shrink the weights directly each step instead of adding an L2 term to the loss, which Adam's per-parameter scaling would distort. The standard optimizer for transformers. *(Module 05; also 10, 16, 20, 22, 25.)* Origin: Loshchilov & Hutter, 2017.
- **adapter** — A small set of trainable weights added to a frozen model to specialize it, like a LoRA update. Can be shipped separately and swapped. *(Module 20; also 26, 27, 35.)* Origin: Houlsby et al., 2019.
- **additive attention** — The first attention: score each encoder state against the decoder state with a small MLP, softmax, and average. Added to seq2seq for translation. *(Module 31.)* Origin: Bahdanau, Cho, Bengio, 2014.
- **advantage** — How much better an action's outcome was than a baseline expectation; positive advantages are reinforced, negative ones suppressed. *(Module 26.)*
- **agent** — An LLM that directs its own process: in a loop it decides which tool to call, reads the result and continues until done. *(Module 39; also 19, 26, 34, 38.)*
- **agent loop** — The core of an agent: send the conversation to the model, execute any tool call it makes, append the result, repeat until it answers (or a turn limit hits). *(Module 39.)*
- **AI winter** — A period when funding and interest in AI collapsed after over-promising: roughly the mid-1970s and again the late 1980s to early 1990s. *(Module 00b; also 29.)*
- **AlexNet** — The convolutional network that won the 2012 ImageNet competition by a wide margin, trained on two GPUs. It is the usual start date of the deep-learning era. *(Module 00b.)* Origin: Krizhevsky, Sutskever, Hinton, 2012.
- **ALiBi** — Attention with Linear Biases: no position embeddings; each head subtracts a penalty proportional to the query-key distance from its attention scores. *(Module 17; also 00b.)* Origin: Press, Smith, Lewis, 2021.
- **all-gather** — Every device contributes its shard and every device ends up with the full array. *(Module 24; also 25.)*
- **all-reduce** — Every device contributes a value and every device gets the sum. Data parallelism uses it to average gradients. *(Module 24; also 25.)*
- **all-to-all** — Every device sends a different piece to every other device; used to route tokens to experts in expert parallelism. *(Module 24; also 40.)*
- **AlphaGo** — DeepMind's Go program that beat Lee Sedol in 2016, combining neural networks with MCTS; AlphaGo Zero (2017) learned from self-play alone. *(Module 34; also 00b.)* Origin: Silver et al., 2016, 2017.
- **AlphaZero** — One algorithm for Go, chess and shogi: a network with policy and value heads guides MCTS, and is trained on self-play to match MCTS's move choices and the game outcomes. *(Module 34; also 00b.)* Origin: Silver et al., 2017.
- **Anthropic-compatible API** — A server endpoint speaking Anthropic's Messages API, so tools built for Claude (like Claude Code) can talk to a local model. *(Module 27.)*
- **approximate nearest neighbour** — Finding vectors close to a query without comparing against every one, trading a little recall for large speed-ups; what vector databases do. *(Module 38.)*
- **arithmetic intensity** — FLOPs performed per byte moved to or from memory. Below the card's ridge point a kernel is memory-bound; above it, compute-bound. *(Module 22.)* Origin: Williams, Waterman, Patterson, 2009.
- **array** — A grid of numbers of one dtype with a fixed shape; the only data structure JAX computes on. JAX arrays are immutable and may live on the GPU. *(Module 01; also 02, 03, 07, 24, 34.)*
- **artificial neuron** — A unit that sums weighted inputs and fires if the total crosses a threshold; McCulloch and Pitts' 1943 model of a neuron. *(Module 29.)* Origin: McCulloch & Pitts, 1943.
- **associative recall** — A synthetic task: given key–value pairs in context, return the value for a queried key. Attention solves it easily; it probes what sequence models can store and retrieve. *(Module 15; also 36, 41.)*
- **associative scan** — A parallel prefix scan for any associative combine; JAX's `lax.associative_scan` runs a linear recurrence in log depth on the GPU. *(Module 36.)* Origin: Blelloch, 1990.
- **asynchronous dispatch** — JAX returns from a call as soon as the work is queued on the GPU, not when it finishes. Timing must call `block_until_ready()` or you measure only the launch. *(Module 06; also 10, 16, 18, 20, 33, 40.)*
- **attention** — A soft dictionary lookup: each position scores every position it may see, softmaxes the scores into weights, and takes the weighted average of their values. Lets any token read from any other. *(Module 15; also 17, 23, 31, 36, 37.)* Origin: Bahdanau et al., 2014; Vaswani et al., 2017.
- **attention pattern** — The matrix of attention weights from each query to each key, the first thing to look at in a head. *(Module 37.)*
- **attention sink** — The first token(s) of a sequence, on which a trained model parks attention it doesn't need (softmax weights must sum to 1). Dropping them from a sliding window breaks a pretrained model; keeping four fixes it. *(Module 44.)* Origin: Xiao et al., 2023.
- **autoencoder** — A network trained to reconstruct its input through a narrow bottleneck, so the bottleneck learns a compressed code. *(Module 33; also 37.)*
- **automatic differentiation** — Computing exact derivatives of a program by applying the chain rule to each primitive operation it executes. Neither symbolic algebra nor finite differences. *(Module 03; also 00b.)* Origin: Baydin et al., 2015 (survey).
- **autoregressive** — Generating one token at a time, each conditioned on all earlier ones, by feeding outputs back in as inputs. *(Module 18.)*
- **axis** — One dimension of an array. Reductions such as `sum(axis=0)` collapse an axis; a batch axis holds independent examples. *(Module 01; also 07, 12, 17, 24, 33.)*

<a id="b"></a>

## B

- **back-of-the-envelope estimate** — Predicting a training run's memory, time, communication and cost from a few measured numbers before paying for it. *(Module 25.)*
- **backpropagation** — Reverse-mode autodiff applied to a neural network: run forward, then propagate the loss's derivative backwards through each layer to get every parameter's gradient in one pass. *(Module 03; also 00b, 10, 11, 29, 31.)* Origin: Rumelhart, Hinton, Williams, 1986.
- **backpropagation through time** — Backpropagation on an RNN unrolled over the sequence; gradients pass back through every step, so they can vanish or explode. *(Module 31.)* Origin: Werbos, 1990.
- **base model** — A model trained only on next-token prediction over raw text. It continues text but doesn't follow instructions until fine-tuned. *(Module 20; also 18, 21, 25, 26, 35.)*
- **batching** — Serving several sequences in one forward pass, so each weight read from memory is used many times. Throughput rises almost for free until compute becomes the limit. *(Module 18; also 27.)*
- **BatchNorm** — Normalize each feature across the examples in a batch. Needs running averages for inference, so train and inference behave differently; the CNN standard. *(Module 11.)* Origin: Ioffe & Szegedy, 2015.
- **BCOO** — JAX's sparse-matrix format (batched coordinate); the course computes BM25 for all documents as one sparse × dense matmul. *(Module 38.)*
- **Bellman equation** — A value equals the immediate reward plus the discounted value of where you land: Q(s, a) = r + γ max Q(s′, a′). Q-learning makes its estimates satisfy it. *(Module 34.)* Origin: Bellman, 1957.
- **benchmark** — A fixed dataset and scoring rule used to compare models. *(Module 21; also 25, 32, 36, 38.)*
- **BERT** — A bidirectional encoder-only transformer pretrained by predicting masked tokens, then fine-tuned for classification or retrieval. Encoders still power embeddings and rerankers. *(Module 32; also 00b, 30, 38.)* Origin: Devlin et al., 2018.
- **best-of-n** — Generate n answers and keep the one a verifier or reward model scores highest. Rejection sampling keeps only the good ones, e.g. as training data. *(Module 35.)*
- **bfloat16** — 16-bit float with float32's 8 exponent bits but only 7 mantissa bits: same range, much less precision. Halves memory and is much faster on the GPU, which is why training uses it for compute. *(Module 08; also 06, 18, 22, 25, 27.)* Origin: Google Brain.
- **bi-encoder** — A retriever that embeds query and document separately, so document vectors can be precomputed and searched fast. *(Module 38.)*
- **bias** — The additive constant in a layer, y = Wx + b, which shifts the output independently of the input. *(Module 10; also 05, 11, 16, 17, 21.)*
- **bigram model** — A language model that predicts the next token from only the current one, i.e. a table of counts of adjacent pairs. *(Module 14.)*
- **binary cross-entropy** — Cross-entropy for a yes/no prediction: −[y log p + (1 − y) log(1 − p)]. *(Module 09.)*
- **bits per byte** — Total cross-entropy in bits divided by the number of UTF-8 bytes of text. Unlike perplexity, it compares models with different tokenizers. *(Module 21; also 25.)*
- **block influence** — How much a layer changes its input: 1 − the mean cosine similarity of each token's hidden state before and after it. Layers with low block influence are the first to remove when pruning depth. *(Module 42.)* Origin: Men et al., 2024 (ShortGPT).
- **BM25** — The standard keyword-search scoring function: TF-IDF with saturating term frequency and document-length normalization. *(Module 38; also 39.)* Origin: Robertson et al., 1994.
- **bootstrap** — Estimating uncertainty by recomputing a metric on many resamples (with replacement) of the test items; the spread gives a confidence interval. *(Module 21; also 38.)* Origin: Efron, 1979.
- **Bradley–Terry model** — The probability that A beats B is σ(score_A − score_B). Training a reward model is maximizing this likelihood on preference pairs. *(Module 35.)* Origin: Bradley & Terry, 1952.
- **broadcasting** — The rule that lets arrays of different shapes combine elementwise: shapes are right-aligned, and any axis of size 1 (or missing) is stretched to match. It replaces most Python loops. *(Module 01.)*
- **bucketing** — Padding variable-length inputs up to one of a few fixed sizes so jit compiles only a few programs. *(Module 06.)*
- **byte-level BPE** — BPE starting from the 256 byte values, so any text can be encoded with no unknown tokens. Used by GPT-2 and most modern LLMs. *(Module 13; also 40.)* Origin: Radford et al., 2019.
- **byte-pair encoding** — A tokenizer trained by repeatedly merging the most frequent adjacent pair of tokens into a new token, starting from bytes or characters. The merges are applied in order to tokenize. *(Module 13; also 27, 30.)* Origin: Gage, 1994; Sennrich et al., 2016.

<a id="c"></a>

## C

- **[CLS] token** — A special token whose final vector serves as a summary of the whole input for classification. *(Module 32.)* Origin: Devlin et al., 2018.
- **calibration** — Correcting a model's built-in lean toward some options regardless of the input, e.g. by subtracting each option's score on a content-free question, so the choice reflects what this question adds. *(Module 39; also 27.)* Origin: Zhao et al., 2021.
- **capacity factor** — How many token slots an expert (or a device's experts) accepts per batch, as a multiple of the even share. Slots past capacity are dropped; 1.25 leaves room for uneven routing. *(Module 40; also 24.)* Origin: Lepikhin et al., 2020 (GShard).
- **capstone** — The course's final project: specialize, evaluate and serve a model of your own, end to end, with optional cloud scale-up. *(Module 25.)*
- **carry** — The state lax.scan passes from one step to the next (an RNN hidden state, a running sum, the train state). *(Module 08; also 05, 09, 10, 14, 24.)*
- **causal mask** — Setting attention scores for future positions to −∞ before softmax, so each token sees only itself and the past. Required for next-token training. *(Module 15; also 32.)*
- **causal tracing** — Finding where a model computes something by corrupting the input and restoring one layer's activations at a time: the layers whose restoration brings the answer back are where it lives. *(Module 46.)* Origin: Meng et al., 2022.
- **chain of thought** — Having a model write out intermediate reasoning steps before its final answer, which improves accuracy on multi-step problems. *(Module 26.)* Origin: Wei et al., 2022.
- **chain rule** — The derivative of a composition is the product of the derivatives of its parts. Backpropagation is the chain rule applied systematically over a computation graph. *(Module 03.)*
- **channel** — One feature map in a convolutional layer, like the R, G, B of an image or one learned detector's output. *(Module 12; also 01.)*
- **chat template** — The rule (usually a Jinja template shipped with the model) that turns a list of chat messages into the exact token string, with role markers, that the model was fine-tuned on. *(Module 27.)*
- **ChatML** — A chat format that wraps each message as <|im_start|>role … <|im_end|>; used by SmolLM2 and Qwen. *(Module 27; also 20, 26.)* Origin: OpenAI, 2023.
- **checkpoint** — A saved snapshot of parameters and optimizer state so training can resume or the model be reused. The course uses Orbax, JAX's checkpointing library. *(Module 11; also 10, 16, 26, 39.)*
- **Chinchilla scaling** — The finding that, for a fixed compute budget, parameters and training tokens should grow together, roughly 20 tokens per parameter. Many earlier large models were under-trained. *(Module 16; also 42.)* Origin: Hoffmann et al., 2022.
- **chunking** — Splitting documents into passages before indexing. Chunk size trades context per hit against retrieval precision. *(Module 38.)*
- **circuit** — A subnetwork of heads and neurons that together implement a specific behaviour. *(Module 37.)* Origin: Olah et al., 2020.
- **classifier-free guidance** — Training a diffusion model with and without its conditioning, then extrapolating from the unconditional toward the conditional prediction to follow the prompt more strongly. *(Module 33.)* Origin: Ho & Salimans, 2022.
- **CLIP** — Contrastive Language–Image Pretraining: an image encoder and a text encoder trained so matching image–caption pairs have high cosine similarity. Enables zero-shot classification and guides image generation. *(Module 33; also 38, 43.)* Origin: Radford et al., 2021.
- **clipped objective** — PPO's trick: cap how far the probability ratio between new and old policy can move in one update, so a step can't change the policy too much. *(Module 26.)* Origin: Schulman et al., 2017.
- **closed book** — Answering questions from the model's parameters alone, with no retrieved sources. *(Module 38.)*
- **coding agent** — An LLM in a loop that reads code, proposes edits and runs tools. Module 19 builds a tiny one against a local server. *(Module 19.)*
- **ColBERT** — A retriever keeping one vector per token and scoring by summed max-similarity: between a bi-encoder and a cross-encoder in cost and accuracy. *(Module 38.)* Origin: Khattab & Zaharia, 2020.
- **collective** — A communication operation involving all devices along an axis, such as all-reduce or all-gather. *(Module 24; also 25.)*
- **command buffer** — Recording a sequence of kernel launches once and replaying it, to cut per-launch CPU overhead (CUDA/HIP graphs). *(Module 22.)*
- **communication-to-compute ratio** — In data parallelism, time spent all-reducing gradients divided by time spent computing. Model size cancels, so it depends on tokens per device per step and on the hardware's FLOP/s-to-bandwidth ratio. *(Module 25; also 24.)*
- **compilation cache** — Storing compiled XLA programs on disk so a restarted process skips recompiling them. *(Module 16.)*
- **compile time** — The one-off cost of tracing and compiling a jitted function, separate from the per-call run time. Unrolled Python loops make it grow with the loop length. *(Module 06.)*
- **computation graph** — A record of a computation as nodes (operations) and edges (values). Autodiff walks it backwards to compute gradients; JAX's jaxpr is such a graph. *(Module 03.)*
- **compute unit** — AMD's basic GPU building block (like an NVIDIA SM): it runs workgroups and has its own LDS. The RX 7600 XT has 32. *(Module 23.)*
- **compute-bound** — When the GPU's arithmetic units are the bottleneck: the work does many FLOPs per byte moved. Large matmuls are compute-bound. *(Module 10; also 19.)*
- **condition number** — The ratio of the largest to smallest curvature of a loss (or singular values of a matrix). A large one means a long narrow valley where plain gradient descent zig-zags slowly. *(Module 05.)*
- **conditional memory** — A lookup table read at a few layers, indexed by the last few tokens (hashed) and gated by the hidden state: knowledge fetched in O(1) instead of rebuilt by computation. DeepSeek's Engram. *(Module 40.)* Origin: Cheng et al. (DeepSeek-AI), 2026.
- **confusion matrix** — A table counting, for each true class, how often the model predicted each class. Shows which classes get mixed up. *(Module 09.)*
- **constrained decoding** — Masking the logits at each step so only tokens allowed by a grammar or JSON schema can be generated, guaranteeing well-formed output (e.g. tool calls) without retraining. *(Module 39.)*
- **container** — A packaged Linux userland (here a Podman image) that holds ROCm, JAX and every library, so nothing GPU-related is installed on the host. `containers/run.sh` starts it with the GPU device files passed through. *(Module 00.)*
- **contamination** — Test items leaking into training data, inflating scores. Detected by n-gram overlap and by comparing exact vs perturbed items. *(Module 21.)*
- **content-addressable memory** — Memory retrieved by giving part of its content rather than an address; a Hopfield network is one. *(Module 29.)*
- **context window** — How many previous tokens a model can see when predicting the next one. *(Module 14; also 13, 27, 36, 39.)*
- **contextual embedding** — A vector for a word that depends on its sentence, as produced by a transformer, unlike word2vec's single vector per word. *(Module 32; also 30, 38.)*
- **continued pretraining** — Keep training an already-pretrained model with the same next-token objective on new data (a domain, a language), instead of starting from random weights. *(Module 42.)*
- **convolution** — Sliding a small kernel of weights across the input and taking a dot product at every position. The same weights look for the same pattern everywhere. *(Module 12; also 00b, 33, 36.)*
- **convolutional neural network** — A network built from convolution layers (plus pooling and a final classifier). The standard image model before vision transformers. *(Module 12; also 33.)* Origin: LeCun et al., 1989.
- **corpus** — The collection of documents searched or trained on. *(Module 38; also 13, 21, 30, 32.)*
- **cosine decay** — Lowering the learning rate along a half cosine curve from its peak to near zero by the end of training. *(Module 05.)* Origin: Loshchilov & Hutter, 2016.
- **cosine similarity** — The dot product of two vectors divided by both their lengths: the cosine of the angle between them, from -1 to 1. The standard way to compare embeddings. *(Module 02.)*
- **cross-encoder** — A model that reads query and document together and outputs a relevance score. More accurate than a bi-encoder but too slow for the whole corpus, so used to rerank. *(Module 38.)*
- **cross-entropy** — The average surprise when outcomes come from the true distribution but you predicted with another: −Σ p log q. With one-hot targets it is −log of the probability given to the right answer, the standard classification and language-model loss. *(Module 04; also 08, 09, 11, 14, 16.)* Origin: Shannon, 1948.
- **CUDA** — NVIDIA's GPU programming platform (2007), which made GPUs general-purpose compute devices. ROCm/HIP is AMD's counterpart. *(Module 00b; also 23, 25.)*

<a id="d"></a>

## D

- **DALL·E** — OpenAI's text-to-image models (2021–); with CLIP they launched the text-to-image era. *(Module 33.)* Origin: Ramesh et al., 2021.
- **Dartmouth workshop** — The 1956 summer workshop, proposed by McCarthy, Minsky, Rochester and Shannon, that named the field "artificial intelligence". *(Module 00b.)* Origin: McCarthy et al., 1955 proposal.
- **data parallelism** — Each device holds a full copy of the model and processes different examples; gradients are all-reduced. *(Module 24; also 25.)*
- **DCGAN** — A convolutional GAN architecture with design rules that made GAN training stable enough for images. *(Module 33.)* Origin: Radford, Metz, Chintala, 2015.
- **DDIM** — A deterministic diffusion sampler that skips steps, generating in tens instead of hundreds of steps from the same trained model. *(Module 33.)* Origin: Song, Meng, Ermon, 2020.
- **DDPM** — Denoising Diffusion Probabilistic Models: the 2020 formulation that trains a network to predict the added noise with a simple squared-error loss. *(Module 33.)* Origin: Ho, Jain, Abbeel, 2020.
- **decay mask** — The lower-triangular matrix of products a_{s+1}···a_t that turns decayed linear attention into masked attention, (QKᵀ ⊙ L)V. Built in log space from differences of a cumulative sum. *(Module 41; also 36.)* Origin: Dao & Gu, 2024.
- **decode** — The second phase of generation: produce one token per step, each reading all the weights and the KV cache. Memory-bandwidth-bound. *(Module 18; also 17, 19, 41.)*
- **decoder-only** — A transformer with only causally masked self-attention, trained to predict the next token. The architecture of GPT and nearly every modern LLM. *(Module 16.)*
- **deep learning** — Machine learning with neural networks of many layers, trained end to end by gradient descent on large data. It took off in 2012, when AlexNet won ImageNet on GPUs. *(Module 00b; also 06, 08, 12, 22.)*
- **DeepSeek Sparse Attention** — DeepSeek V3.2's attention: a small ReLU-scored indexer ranks every past token and each query reads only the top k (2,048); the indexer is trained to match the main attention's weights. *(Module 44.)* Origin: DeepSeek-AI, 2025.
- **DeepSeek-V3** — DeepSeek's 2024 MoE LLM combining MLA, DeepSeekMoE, aux-loss-free balancing and MTP. The owner's deepseek-v3-jax replication is a recurring reference in this course. *(Module 17; also 05, 08, 11, 13, 16.)* Origin: DeepSeek-AI, 2024.
- **delta rule** — Update a memory by writing only the error: S ← S + β k (v − Sᵀk)ᵀ, one gradient step on ‖Sᵀk − v‖². In a linear-attention layer it overwrites a key's old value instead of adding to it. *(Module 41.)* Origin: Widrow & Hoff, 1960; Schlag et al., 2021.
- **dense retrieval** — Search by embedding query and documents as dense vectors and finding the nearest by cosine similarity, so matches need not share words. *(Module 38.)* Origin: Karpukhin et al., 2020 (DPR).
- **depthwise convolution** — A convolution applied to each channel separately; Mamba blocks use a short causal one before the SSM. *(Module 36; also 40.)*
- **derivative** — The slope of a function at a point: how much the output changes per unit change in the input. *(Module 03; also 10.)*
- **determinant** — How much a matrix scales area (or volume); zero means it squashes space flat and cannot be undone, negative means it flips orientation. *(Module 02.)*
- **device** — A piece of hardware JAX can place arrays on and run computation on, such as `rocm:0` or the CPU. `jax.device_put` moves data there. *(Module 01; also 00, 08, 22, 24, 25.)*
- **diffusion model** — Learn to reverse a process that gradually adds noise to data: a network predicts the noise at each step, and generation starts from pure noise and denoises step by step. *(Module 33; also 00b, 45.)* Origin: Sohl-Dickstein et al., 2015; Ho et al., 2020.
- **dilated convolution** — A convolution whose kernel taps are spaced apart, so the receptive field doubles per layer with dilations 1-2-4-8. WaveNet stacks them causally. *(Module 12; also 40.)* Origin: van den Oord et al., 2016 (WaveNet).
- **discounted return** — The total (usually discounted) reward from a point onward; what an RL agent maximizes. *(Module 34.)*
- **discretization** — Converting a continuous-time state-space model into a step-by-step update with a step size Δ; zero-order hold assumes the input is constant within each step. *(Module 36.)*
- **distillation** — Training a smaller student model on a larger teacher's outputs (or reasoning traces) so it inherits the teacher's ability. *(Module 26; also 12, 31, 42.)* Origin: Hinton, Vinyals, Dean, 2015.
- **distributional hypothesis** — "You shall know a word by the company it keeps": words in similar contexts have similar meanings. The idea behind all learned embeddings. *(Module 30.)* Origin: Harris, 1954; Firth, 1957.
- **divergence** — Training going unstable: the loss shoots up or becomes NaN, usually from too large a learning rate. *(Module 05; also 04.)*
- **domain shift** — When test data differs in kind from training data, so in-distribution scores overstate real performance. *(Module 21.)*
- **donation** — Telling jit that an input buffer won't be used again, so XLA can write the output into it. Halves peak memory for parameters and optimizer state in a train step. *(Module 16; also 22, 25.)*
- **dot product** — Multiply two vectors elementwise and sum. It measures alignment: large and positive when they point the same way, zero when perpendicular. Attention scores are dot products. *(Module 02; also 08, 12, 17.)*
- **Double DQN** — Pick the next action with the online network but evaluate it with the target network, reducing Q-learning's overestimation. *(Module 34.)* Origin: van Hasselt et al., 2015.
- **DPO** — Preference learning without a reward model or RL: a classification-style loss raises the policy's log-probability ratio (vs a reference model) for chosen responses over rejected ones. *(Module 35.)* Origin: Rafailov et al., 2023.
- **DPR** — Dense Passage Retrieval: a BERT bi-encoder trained with in-batch and BM25 hard negatives that beat BM25 for open-domain question answering. *(Module 38.)* Origin: Karpukhin et al., 2020.
- **DQN** — Q-learning with a neural network for Q-values, made stable by experience replay and a target network; learned Atari games from pixels. *(Module 34; also 00b.)* Origin: Mnih et al., 2013, 2015.
- **dropout** — During training, randomly zero a fraction of activations and scale up the rest, so the network can't rely on any one unit. Turned off at inference. *(Module 11.)* Origin: Srivastava et al., 2014.
- **dtype** — The number format of an array's elements, such as float32, bfloat16 or int32. It sets precision, range and memory per element. *(Module 01; also 06, 08, 10, 23, 32.)*

<a id="e"></a>

## E

- **eigenvector** — A direction a matrix only stretches, never turns; the stretch factor is its eigenvalue. They explain repeated application, e.g. why some directions blow up or vanish in an RNN. *(Module 02.)*
- **einsum** — A one-line notation for products and sums over named axes, e.g. `einsum('bij,bjk->bik', a, b)` for a batched matmul. Indices that appear in the inputs but not the output are summed. *(Module 01; also 15, 17, 31, 36.)*
- **ELBO** — A lower bound on the log-likelihood of the data: reconstruction quality minus the KL divergence of the encoder's latent distribution from the prior. VAEs maximize it. *(Module 33.)* Origin: Kingma & Welling, 2013.
- **Elman network** — The simple RNN: h_t = tanh(W h_{t-1} + U x_t + b). *(Module 31.)* Origin: Elman, 1990.
- **embedding** — A learned vector for each token id, looked up from a table (V × d). Similar tokens end up with nearby vectors. *(Module 14; also 15, 16, 27, 30, 38.)* Origin: Bengio et al., 2003.
- **embedding model** — A model whose output is a single vector per text, trained so related texts are close; the encoder side of dense retrieval. *(Module 38; also 27.)*
- **encoder-decoder transformer** — A transformer with an encoder reading the input and a decoder generating with cross-attention to it, as in the original Transformer and T5. *(Module 32.)* Origin: Vaswani et al., 2017; Raffel et al., 2019.
- **encoder-only** — A transformer without a causal mask, so every token sees the whole input; produces representations rather than generating text. *(Module 32.)*
- **energy function** — A scalar a Hopfield network's updates never increase, so the network rolls downhill into stored patterns (local minima). *(Module 29.)* Origin: Hopfield, 1982.
- **entropy** — The average surprise of a distribution, −Σ p log p: how uncertain it is. Measured in bits (log base 2) or nats (natural log). *(Module 04; also 33, 37.)* Origin: Shannon, 1948.
- **epoch** — One full pass over the training set. *(Module 10; also 09, 11, 12, 29, 33.)*
- **evaluator-optimizer** — A workflow in which one call proposes an answer and another critiques it, repeating until the critic is satisfied. *(Module 39.)* Origin: Anthropic, 2024; Shinn et al., 2023.
- **exact match** — Scoring a generated answer as right only if it equals the reference after normalization. *(Module 21; also 20.)*
- **expectation** — The probability-weighted average of a random variable: its long-run mean. *(Module 04.)*
- **experience replay** — Storing past transitions and training on random samples of them, which breaks correlations between consecutive steps and reuses data. *(Module 34.)* Origin: Lin, 1992; Mnih et al., 2015.
- **expert parallelism** — Placing different MoE experts on different devices and using all-to-all to send each token to its experts' devices. *(Module 24; also 40.)*
- **expert system** — AI built from hand-written if-then rules elicited from human experts. Commercially popular in the 1980s, they were brittle outside their rules, which helped bring on the second AI winter. *(Module 00b.)*
- **explicit sharding** — JAX mode in which an array's sharding is part of its type and visible at trace time, so ambiguous cases are errors rather than compiler guesses. *(Module 24.)*
- **exponent bits** — The bits of a float that set its scale. bfloat16 keeps float32's 8, so it has the same range. *(Module 08.)*
- **exponential moving average** — A running average where each new value gets weight (1 − β) and the old average β, so recent values count most. Used by momentum, Adam, BatchNorm and target networks. *(Module 08; also 05.)*

<a id="f"></a>

## F

- **family tree** — The course's registry of about fifty models, the models each one builds on, and the building blocks each uses, filed by slot (position, normalization, attention, …). *(Module 00b; also 16, 17.)*
- **fan-in** — The number of inputs (fan-in) or outputs (fan-out) of a layer, used to scale initial weights and some optimizer updates. *(Module 11; also 05.)*
- **Fashion-MNIST** — 70,000 28×28 grayscale images of clothing in 10 classes: the course's image benchmark for regression, MLPs and CNNs. *(Module 09; also 07, 10, 11, 12, 33.)* Origin: Xiao, Rasul, Vollgraf, 2017.
- **fast weights** — A weight matrix written by the network itself while it reads its input, rather than by training. A linear-attention state is one: a key→value map built from the tokens seen so far. *(Module 41.)* Origin: Schmidhuber, 1992.
- **feature** — One input variable to a model. In interpretability, a direction in activation space that represents a concept. *(Module 09; also 11, 12, 15, 17, 37.)*
- **FFT convolution** — Computing a long convolution in O(L log L) via the fast Fourier transform instead of O(L²). *(Module 36.)*
- **filter** — The small weight array a convolution slides over the input; each learns to detect one pattern, producing one output channel. *(Module 12.)*
- **fine-tuning** — Continuing to train a pretrained model on a smaller, task-specific dataset. *(Module 20; also 26, 32, 35, 38.)*
- **finite differences** — Estimating a derivative by nudging an input and measuring the change in output. Simple but needs one evaluation per input and suffers truncation and round-off error, so it is used only to check gradients. *(Module 03.)*
- **FlashAttention** — Exact attention computed tile by tile with an online softmax, never writing the L × L score matrix to memory. Fast because it cuts memory traffic, not FLOPs. *(Module 23; also 15, 19.)* Origin: Dao et al., 2022.
- **Flax NNX** — JAX's neural-network library in which layers are Python objects holding their parameters, yet still convertible to pytrees so jit and grad apply. *(Module 10; also 11, 12.)*
- **float16** — 16-bit IEEE float with 5 exponent bits: more precision than bfloat16 but a tiny range, so training in it needs loss scaling. Common for storing inference weights. *(Module 08; also 19, 23, 27.)*
- **float32** — 32-bit float: about 7 significant digits. The safe default, and the format for master weights and accumulations. *(Module 08; also 00, 01, 06, 23, 27.)*
- **floating point** — Numbers stored as sign × mantissa × 2^exponent. Exponent bits set the range, mantissa bits the precision. *(Module 08.)*
- **forward mode** — Autodiff that pushes derivatives forward alongside the values, one input direction per pass. Cheap when there are few inputs and many outputs. *(Module 03.)*
- **forward process** — In diffusion, the fixed procedure that adds a scheduled amount of Gaussian noise to the data at each step until only noise remains. *(Module 33.)*
- **freezing** — Keeping some parameters fixed during training, either by differentiating only the trainable part or masking the optimizer. *(Module 20.)*
- **FSDP** — Data parallelism with parameters and optimizer state sharded across devices and all-gathered just before use, so memory per device shrinks. *(Module 24; also 25.)* Origin: Rajbhandari et al., 2019 (ZeRO).
- **functional state** — Keeping parameters, optimizer state and RNG keys as values passed into and returned from pure functions, instead of mutating objects. JAX's way of training. *(Module 07.)*

<a id="g"></a>

## G

- **GAN** — A generator makes fakes from noise while a discriminator learns to tell them from real data; each trains against the other. *(Module 33; also 00b.)* Origin: Goodfellow et al., 2014.
- **gated attention** — Softmax attention whose output is multiplied by a sigmoid gate computed from the layer input; used in Qwen3-Next's attention layers. *(Module 41.)* Origin: Qiu et al., 2025.
- **Gated DeltaNet** — A linear-attention layer combining Mamba-2's per-step decay with the delta rule: S_t = a_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ. Three of every four layers of Qwen3-Next and Qwen3.5. *(Module 41.)* Origin: Yang, Kautz & Hatamizadeh, 2025.
- **gates** — A sigmoid-valued vector that multiplies another, controlling how much passes through: what to keep, forget or emit. *(Module 31; also 36.)*
- **Gaussian** — The bell-curve distribution described by a mean and variance. Used for weight initialization, VAE latents and diffusion noise. *(Module 33.)*
- **GELU** — A smooth version of ReLU, x·Φ(x), used in GPT-2 and BERT. *(Module 10; also 22, 23.)* Origin: Hendrycks & Gimpel, 2016.
- **GEMM** — GEneral Matrix Multiply: the C = A·B (+ C) operation that dominates deep-learning compute. Kernels and libraries are judged mostly by how fast their GEMMs run. *(Module 22.)*
- **generalization** — How well a model performs on data it was not trained on. The real goal of training. *(Module 10; also 11.)*
- **generative model** — A model that learns a data distribution so it can produce new samples resembling the training data. *(Module 33.)*
- **generative perplexity** — A score for generated text: a separate autoregressive model's loss on it. Easy to game with dull, repetitive text, so it is read alongside diversity measures such as entropy and repeated n-grams. *(Module 45.)* Origin: Lou et al., 2024; Sahoo et al., 2024.
- **gfx1102** — The instruction-set name of the RX 7600 XT (Navi 33, RDNA3). JAX's ROCm plugin only accepts a fixed list of architectures, so the card is made to report gfx1100 (the RX 7900's) instead. *(Module 00; also 22, 23.)*
- **GGUF** — The single-file model format of llama.cpp: weights (often quantized), tokenizer, chat template and metadata together. *(Module 19; also 27.)*
- **glitch tokens** — Tokens that exist in the vocabulary but were almost never seen in training, so their embeddings are untrained and the model behaves strangely on them. *(Module 13.)*
- **Goodhart's law** — "When a measure becomes a target, it ceases to be a good measure." Optimizing a proxy reward too hard makes the true quality fall while the proxy keeps climbing. *(Module 35.)* Origin: Goodhart, 1975.
- **GPT** — A decoder-only transformer trained on next-token prediction. GPT-2 (2019) is the template the course's model follows. *(Module 16; also 13, 21, 25, 27, 32.)* Origin: Radford et al., 2018, 2019.
- **gradient** — The vector of all partial derivatives of a scalar loss with respect to the parameters. It points in the direction of steepest increase, so training steps the other way. *(Module 03; also 05, 07, 11, 20, 31.)*
- **gradient accumulation** — Summing gradients over several micro-batches before one optimizer step: a larger effective batch without more memory, and fewer all-reduces per token. *(Module 25.)*
- **gradient clipping** — If the gradient's overall norm exceeds a threshold, scale it down to the threshold. Prevents one bad batch from wrecking the weights. *(Module 05; also 02, 07, 11, 16.)*
- **gradient descent** — Repeatedly step the parameters a small amount against the gradient: θ ← θ − η·∇L. *(Module 05; also 09, 14.)* Origin: Cauchy, 1847.
- **greedy decoding** — Always picking the most probable next token. Deterministic and often repetitive. *(Module 18; also 20, 26, 27, 35.)*
- **group-wise quantization** — Quantizing with one scale per row/channel or per small group of weights, so one outlier can't wreck the precision of everything else. *(Module 18.)*
- **grouped-query attention** — Attention in which several query heads share each key/value head, shrinking the KV cache by the group factor with little quality loss. *(Module 17; also 18, 19.)* Origin: Ainslie et al., 2023.
- **GRPO** — Policy-gradient RL for LLMs: sample a group of answers per prompt, score them, use each answer's reward minus the group mean (over the std) as its advantage, with PPO-style clipping and a KL penalty. No value network needed. *(Module 26; also 27, 34, 35.)* Origin: Shao et al., 2024 (DeepSeekMath).
- **GRU** — A simpler gated RNN cell than the LSTM, with reset and update gates and no separate cell state. *(Module 31.)* Origin: Cho et al., 2014.
- **guardrails** — Checks around an agent: input rules, tool allowlists, turn limits and human approval for high-risk actions. *(Module 39.)*
- **Gumbel-max trick** — Sample from a softmax by adding Gumbel noise to the logits and taking the argmax. Makes sampling a deterministic function of the key, and reappears in Gumbel MuZero. *(Module 04.)* Origin: Gumbel, 1954.

<a id="h"></a>

## H

- **handoff** — A one-way transfer of a conversation from one agent to another, which then owns the reply. *(Module 39.)* Origin: OpenAI Agents SDK.
- **hard negative** — A wrong document that looks right (e.g. one BM25 ranks highly); training against them makes a retriever sharper. *(Module 38.)* Origin: Karpukhin et al., 2020.
- **hashing trick** — Map a huge or open-ended set of keys (words, n-grams) to rows of a fixed-size table with a hash function, accepting that some keys share a row. Several independent hashes make a shared row harmless. *(Module 40.)* Origin: Weinberger et al., 2009.
- **He initialization** — Draw initial weights with variance 2/fan_in so activations keep their scale through ReLU layers. *(Module 10; also 11.)* Origin: He et al., 2015.
- **Hebbian learning** — "Cells that fire together wire together": strengthen a weight when both units are active. The Hopfield network stores patterns this way. *(Module 29.)* Origin: Hebb, 1949.
- **Hessian** — The matrix of second derivatives of a scalar function: how the gradient itself changes. It describes curvature. *(Module 03.)*
- **hidden layer** — A layer between input and output whose values (activations) are learned features rather than given data or predictions. *(Module 10; also 29, 30.)*
- **hidden state** — The vector an RNN or state-space model carries from step to step, summarizing everything read so far. *(Module 31.)*
- **HIP** — AMD's C++ GPU programming interface and runtime, deliberately close to CUDA so code ports almost line for line. A hand-written HIP kernel is the low-level alternative to letting XLA or Pallas generate one. *(Module 23; also 00b, 19, 22.)*
- **hipBLASLt** — AMD's tuned matrix-multiply libraries. With Triton GEMM disabled, XLA hands every large matmul (a GEMM) to hipBLASLt or rocBLAS, which is both safe and fast on this card. *(Module 00; also 22, 23.)*
- **HiPPO** — A principled initialization of an SSM's state matrix so its state optimally summarizes the input history. *(Module 36.)* Origin: Gu et al., 2020.
- **HLO** — XLA's High-Level Operations IR: the program JAX hands the compiler (as StableHLO), and the optimized, fused program XLA produces. Reading it shows what actually runs. *(Module 06; also 22, 23, 24.)*
- **HNSW** — A graph-based approximate nearest-neighbour index used by most vector databases. *(Module 38.)* Origin: Malkov & Yashunin, 2016.
- **Hopfield capacity** — A classic Hopfield network reliably stores only about 0.14 patterns per neuron before recall breaks down. *(Module 29.)* Origin: Amit, Gutfreund, Sompolinsky, 1985.
- **Hopfield network** — A recurrent network of binary units with symmetric weights that stores patterns as minima of an energy; started from a corrupted pattern, it settles into the stored one. *(Module 29; also 00b.)* Origin: Hopfield, 1982.
- **HSA_OVERRIDE_GFX_VERSION** — An environment variable that makes the ROCm runtime report a different GPU architecture. Set to 11.0.0 here so the RX 7600 XT (gfx1102) is accepted as gfx1100; the instruction sets are close enough that the code runs. *(Module 00.)*
- **Huber loss** — Squared error for small errors, absolute error for large ones, so outliers don't produce huge gradients. *(Module 34.)* Origin: Huber, 1964.
- **Hugging Face** — The main hub for open models and datasets, and the `transformers` library that loads them. *(Module 27; also 13, 18, 26, 35.)*
- **hybrid model** — An architecture interleaving attention layers with SSM (Mamba) layers to get most of attention's recall at a fraction of its cost; Jamba is one. *(Module 36; also 40, 41.)* Origin: Lieber et al., 2024 (Jamba).
- **hybrid search** — Combining keyword (BM25) and dense retrieval results, which usually beats either alone. *(Module 38.)*
- **hyperparameter** — A setting chosen by you rather than learned: learning rate, batch size, depth, weight decay. *(Module 09.)*

<a id="i"></a>

## I

- **im2col** — Unfolding every input patch into a column so a convolution becomes one big matrix multiply, which GPUs do fast. *(Module 12.)*
- **ImageNet** — A dataset of over a million labelled photos in 1,000 classes whose yearly competition became deep learning's proving ground. *(Module 00b.)* Origin: Deng et al., 2009.
- **immutability** — JAX arrays cannot be changed in place. To update an element you write `x.at[i].set(v)`, which returns a new array (and XLA usually makes it an in-place update under jit). *(Module 01.)*
- **implicit reward** — In DPO, β × the log-ratio of policy to reference probability for a response; the reward the policy implicitly optimizes. *(Module 35.)* Origin: Rafailov et al., 2023.
- **in-context learning** — A model doing a new task from examples given in the prompt, with no weight updates. *(Module 20; also 33, 37.)* Origin: Brown et al., 2020 (GPT-3).
- **in_axes** — Arguments to vmap saying which axis of each input (or None for no axis) is the batch axis to map over. *(Module 07.)*
- **induction head** — A two-layer attention circuit that completes repeated patterns ([A][B] … [A] → [B]): a previous-token head plus a head that looks for the token after the last copy. The mechanism behind much in-context learning. *(Module 37.)* Origin: Elhage et al., 2021; Olsson et al., 2022.
- **inductive bias** — The assumptions an architecture builds in, such as locality for CNNs or order for RNNs. Good biases trade flexibility for data efficiency. *(Module 12.)*
- **inference-time scaling** — Getting better answers by spending more compute at generation time: more samples, longer reasoning, voting or search. *(Module 26.)* Origin: Snell et al., 2024.
- **InfoNCE** — A contrastive loss: treat each matching pair in a batch as the right answer of a softmax classification against all the other items in the batch (the in-batch negatives). *(Module 33; also 38.)* Origin: van den Oord et al., 2018.
- **information retrieval** — Finding the documents most relevant to a query from a large collection. *(Module 38.)*
- **initialization** — Choosing a network's starting weights. Scale them so signals neither explode nor vanish layer to layer. *(Module 11; also 05, 07, 10, 16, 20.)*
- **instruction tuning** — Fine-tuning a base model on instruction–response examples so it follows requests; "-Instruct" models have had it. *(Module 20; also 21.)* Origin: Wei et al., 2021; Ouyang et al., 2022.
- **interpret mode** — Running a Pallas kernel as ordinary JAX on the CPU, for debugging before it touches the GPU. *(Module 23.)*
- **intrinsic dimension** — The finding that fine-tuning can succeed within a very low-dimensional subspace of the weights, which is why LoRA works. *(Module 20.)* Origin: Aghajanyan et al., 2020.
- **IVF** — Inverted-file vector index: cluster the vectors (k-means) and search only the few clusters nearest the query. *(Module 38.)* Origin: Jégou, Douze, Schmid, 2011.

<a id="j"></a>

## J

- **Jacobian** — The matrix of all partial derivatives of a vector-valued function: one row per output, one column per input. *(Module 03.)*
- **JAX** — A Python library with a NumPy-like API whose functions can be transformed: differentiated (`grad`), compiled (`jit`), vectorized (`vmap`) and parallelized. It works by tracing Python into a graph that XLA compiles. *(Module 00; also 03, 06, 07, 23, 24.)* Origin: Frostig, Johnson, Leary, 2018.
- **jax.grad** — JAX's transformation that turns a scalar-valued function into one that returns its gradient (reverse mode). `value_and_grad` returns the loss too. *(Module 03; also 07, 08, 10, 11, 20.)*
- **jaxpr** — JAX's intermediate representation: the traced recording of a function as a list of primitive operations. `jax.make_jaxpr` prints it. *(Module 06; also 07.)*
- **jit** — Just-in-time compilation: the first call traces the function into a graph, XLA compiles it, and later calls with the same shapes and dtypes reuse the compiled program. *(Module 06; also 07, 08, 10, 16, 23.)*
- **JVP** — Jacobian-vector product: the change in a function's outputs for a given input direction, computed by forward mode without building the Jacobian. *(Module 03.)*

<a id="k"></a>

## K

- **k-means** — Cluster points by alternating: assign each to its nearest centre, move each centre to its points' mean. *(Module 38.)* Origin: Lloyd, 1957/1982.
- **K-quants** — llama.cpp's quantization types: Q8_0 is 8-bit (~8.5 bits/weight, nearly lossless), Q4_K_M a mixed ~4.8 bits/weight that costs a little perplexity. *(Module 19; also 18, 27.)*
- **kernel** — One function launched on the GPU over a grid of threads. Every jitted program runs as a series of kernels. *(Module 23; also 00, 06, 12, 22, 36.)*
- **kernel fusion** — XLA combining several elementwise operations into one GPU kernel so intermediates stay in registers instead of round-tripping through memory. Often the biggest speed-up jit gives. *(Module 06; also 15, 22, 23, 27, 38.)*
- **KL divergence** — How many extra bits you pay by using distribution q when the truth is p: cross-entropy minus entropy. Always ≥ 0 and not symmetric; used as a leash in RLHF and in the VAE loss. *(Module 04; also 26, 33, 35.)* Origin: Kullback & Leibler, 1951.
- **KL penalty** — A term that penalizes the trained policy's divergence from a frozen reference model, keeping RL fine-tuning from drifting into gibberish or exploits. *(Module 26; also 35.)*
- **KV cache** — During generation, store each layer's keys and values for past tokens so each new token computes only its own, instead of re-running the whole prefix. Its size often limits context length and batch. *(Module 17; also 18, 19, 25, 36, 39, 41.)*

<a id="l"></a>

## L

- **L2 regularization** — Adding λ·‖w‖² to the loss to penalize large weights. Equivalent to weight decay for plain SGD but not for Adam. *(Module 05.)*
- **LangGraph** — A framework for building agents and workflows as graphs of LLM calls and tools. *(Module 39.)*
- **language model** — A model that assigns a probability to the next token given the previous ones; chained, that gives a probability for any text and a way to generate it. *(Module 14; also 04, 15, 31, 34, 37.)*
- **language-model head** — The final linear layer that maps each token's hidden vector to logits over the vocabulary. *(Module 16.)*
- **latency** — Time from request to response (or to the first token). Batching improves throughput at some cost in latency. *(Module 18.)*
- **latent diffusion** — Diffusion run in an autoencoder's compact latent space instead of on pixels, which made high-resolution text-to-image generation cheap (Stable Diffusion). *(Module 33.)* Origin: Rombach et al., 2021.
- **latent space** — A lower-dimensional space of hidden codes from which a generator or decoder produces data. *(Module 33.)*
- **LatentMoE** — A mixture-of-experts layer whose routed experts work in a narrower latent width ℓ = d/α: tokens are projected down before they're sent to experts and up after. The savings buy α× as many experts. *(Module 40.)* Origin: Elango et al. (NVIDIA), 2026.
- **lax.cond** — Traced control flow: `lax.cond` picks one of two branches based on a traced boolean and compiles both; `lax.switch` picks from many by index; `jnp.where` selects elementwise. *(Module 08; also 06, 17, 29.)*
- **lax.fori_loop** — A compiled loop whose body is traced once, so compile time doesn't grow with the iteration count. `while_loop` stops on a condition but can't be reverse-differentiated. *(Module 08; also 18, 23.)*
- **lax.scan** — A compiled loop that threads a carry through steps and stacks each step's output. It is how JAX writes RNNs, training loops and decoding without unrolling. *(Module 08; also 05, 10, 14, 16, 36.)*
- **LayerNorm** — Normalize each example's feature vector to mean 0 and variance 1, then apply a learned scale and shift. The transformer's standard normalization. *(Module 11; also 16, 17.)* Origin: Ba, Kiros, Hinton, 2016.
- **LDS** — Local data share: fast on-chip memory shared by a workgroup (CUDA's shared memory), 64 KiB per workgroup here. Tiled kernels stage data in it. *(Module 23; also 22.)*
- **leaf** — An array (or other non-container value) at the bottom of a pytree. *(Module 07; also 08.)*
- **learning rate** — The multiplier η on each gradient step. Too small is slow, too large oscillates or diverges; it is the most important hyperparameter. *(Module 05; also 09, 11, 16, 20, 26.)*
- **learning-rate schedule** — A rule that changes the learning rate over training, typically a warmup followed by a decay. *(Module 05.)*
- **length extrapolation** — Running a model on sequences longer than any it trained on. Position schemes differ in how gracefully they degrade. *(Module 17.)*
- **length normalization** — Dividing an answer's log-likelihood by its length (in tokens or bytes) so longer options aren't unfairly penalized. *(Module 21.)*
- **linear attention** — Attention without the softmax: o_t = Σ_{s≤t} (q_t·k_s) v_s, which regroups into a recurrence on a fixed d_k × d_v state, S_t = S_{t−1} + k_t v_tᵀ. Constant memory and time per generated token. *(Module 41; also 36.)* Origin: Katharopoulos et al., 2020.
- **linear probe** — A linear classifier trained on a model's frozen activations to test whether some property (like a board square's state) is linearly readable from them. *(Module 37.)* Origin: Alain & Bengio, 2016.
- **linear regression** — Fitting a straight line (or hyperplane) y ≈ Xw + b by minimizing squared error. Solvable in closed form or by gradient descent. *(Module 09.)*
- **linear time-invariant** — A system whose update matrices are the same at every step and independent of the input, so it can be computed as one long convolution. *(Module 36.)*
- **linearly separable** — Two classes that a single straight line (or hyperplane) can separate. XOR is not. *(Module 29.)*
- **Llama** — Meta's family of open-weight LLMs; their architecture (RMSNorm, RoPE, SwiGLU, GQA) became the standard modern recipe. *(Module 17.)* Origin: Touvron et al., 2023.
- **llama.cpp** — A C/C++ LLM inference engine that runs GGUF models on CPUs and GPUs (via ROCm/HIP or Vulkan) and serves an OpenAI-compatible API. *(Module 19; also 18, 21, 27, 39.)*
- **LLM-as-judge** — Using a language model to grade other models' outputs. Cheap and scalable, but biased (e.g. position bias) and must be checked. *(Module 21; also 38.)* Origin: Zheng et al., 2023.
- **LM Studio** — A desktop app and server for running local LLMs (via llama.cpp engines), with OpenAI-, Anthropic-compatible and native APIs. *(Module 27; also 26, 38, 39.)*
- **load balancing** — Keeping MoE experts evenly used. Classic MoE adds an auxiliary loss; DeepSeek-V3 instead nudges a per-expert bias in the router after each step (aux-loss-free). *(Module 17; also 40.)* Origin: DeepSeek-AI, 2024 (V3).
- **local-global attention** — Mixing sliding-window layers with a few full-attention layers (Gemma 2 1 : 1, Gemma 3 5 : 1): most of the KV-cache saving of a window, with a path for long-range information. *(Module 44.)* Origin: Gemma Team, 2024.
- **log-sum-exp** — log(Σ exp xᵢ), computed safely by subtracting the max first so nothing overflows. It is the normalizer inside log-softmax and cross-entropy. *(Module 04; also 08, 11, 39.)*
- **logistic regression** — A linear model followed by a sigmoid, giving the probability of the positive class; trained with binary cross-entropy. *(Module 09.)*
- **logit** — An unnormalized score, one per class or token, that softmax turns into probabilities. Logits are log-probabilities up to a shared constant. *(Module 04; also 09, 11, 14, 18, 27.)*
- **long-term memory** — Information an agent stores across sessions and retrieves when relevant, often with RAG over past episodes. *(Module 39.)*
- **LoRA** — Fine-tuning by freezing the base weights W and learning a low-rank update B·A (rank r) added to them, scaled by α/r. B starts at zero, so training starts from the base model, and only a tiny adapter is trained and shipped. *(Module 20; also 02, 26, 27, 35.)* Origin: Hu et al., 2021.
- **loss** — A single number measuring how wrong the model is on data; training adjusts parameters to make it small. *(Module 05; also 04, 10, 11, 14, 16.)*
- **loss at init** — A sanity check: a freshly initialized classifier should predict near-uniformly, so its cross-entropy should be about ln V for V classes. *(Module 16.)*
- **loss landscape** — The loss pictured as a surface over parameter space. Training walks downhill on it; narrow valleys, plateaus and saddle points make the walk hard. *(Module 05.)*
- **loss spike** — A sudden jump in training loss, often from a bad batch, too high a learning rate or a numerical problem. Triage: check data, gradient norm, learning rate, precision. *(Module 11.)*
- **low-confidence remasking** — LLaDA's sampling rule for diffusion LMs: at each step keep the proposals the model is surest of and leave the rest masked for later. *(Module 45.)* Origin: Nie et al., 2025.
- **low-rank approximation** — Keep only the top few singular values of a matrix and drop the rest; the best approximation of that rank. LoRA and MLA both rely on weight changes or activations being nearly low-rank. *(Module 02; also 20.)* Origin: Eckart & Young, 1936.
- **LSTM** — An RNN cell with a separate cell state and input, forget and output gates, so information (and gradients) can be kept unchanged over long spans. *(Module 31; also 00b, 36.)* Origin: Hochreiter & Schmidhuber, 1997.

<a id="m"></a>

## M

- **Mamba** — A selective state-space model with a gated block structure and a hardware-aware scan; linear in sequence length with a constant-size state for generation. *(Module 36; also 00b, 41.)* Origin: Gu & Dao, 2023.
- **mantissa** — The bits of a float that hold its significant digits. bfloat16 keeps only 7 (about 2–3 decimal digits). *(Module 08.)*
- **masked diffusion language model** — A language model that generates by filling in blanks: trained to predict tokens hidden at every masking rate (weighted by 1/t, a bound on the likelihood), it starts from an all-[MASK] passage and reveals several tokens per step, in any order. *(Module 45; also 33.)* Origin: Sahoo et al., 2024; Nie et al., 2025.
- **masked language modeling** — BERT's training task: hide ~15% of tokens (80% replaced by [MASK], 10% random, 10% unchanged) and predict them from both sides. *(Module 32; also 45.)* Origin: Devlin et al., 2018.
- **master weights** — The float32 copy of the parameters the optimizer updates. Tiny updates would be rounded away if added to bfloat16 weights. *(Module 08.)*
- **matrix** — A 2-D array, best understood as a linear transformation of space: each column says where a basis vector lands. A neural-network layer is mostly a matrix. *(Module 02; also 00, 05, 07, 12, 22.)*
- **matrix multiplication** — Combining two matrices so the result applies one transformation after the other (right to left). Order matters. It is the operation GPUs are built to do fast. *(Module 02; also 08, 18, 22, 23, 24.)*
- **MCP** — Model Context Protocol: an open standard for exposing tools and data to LLM applications through a common server interface. *(Module 39.)* Origin: Anthropic, 2024.
- **mean squared error** — The average squared difference between prediction and target: the standard regression loss. *(Module 09; also 10, 33, 34.)*
- **mechanistic interpretability** — Reverse-engineering what a trained network computes, finding the features and circuits inside it. *(Module 37.)*
- **memory bandwidth** — How fast the GPU can read and write its memory, 288 GB/s on the RX 7600 XT. Sets the speed limit for memory-bound work. *(Module 18; also 06.)*
- **memory profile** — Measurements of how much GPU memory is live and at peak, used to see the effect of donation and precision. *(Module 22.)*
- **memory-bandwidth-bound** — When time is set by bytes moved from GPU memory, not arithmetic. Single-sequence decoding is, so its speed ≈ bandwidth ÷ model size. *(Module 18; also 19, 22.)*
- **merge** — One BPE rule, "replace pair (a, b) with new token c", learned in frequency order. *(Module 13.)*
- **merging** — Folding a LoRA update into the base weights (W + scale·B·A) so inference needs no extra computation. *(Module 20; also 27.)*
- **mesh** — A grid of devices with named axes (e.g. 'data', 'model') that sharding specs refer to. *(Module 24; also 25.)*
- **min-max fusion** — Rescale each retriever's scores to [0, 1] over the candidates and mix them with a weight. *(Module 38.)*
- **minibatch** — A small random subset of the training data used for one gradient step. *(Module 05; also 11, 18.)*
- **minimax** — Classic game search: assume both players play optimally, maximize your worst case; alpha-beta pruning skips branches that can't matter. Deep Blue's method. *(Module 34.)*
- **mixed precision** — Doing the heavy matmuls in bfloat16 while keeping master weights, optimizer state and sensitive reductions in float32. Most of the speed, little of the risk. *(Module 08; also 16.)* Origin: Micikevicius et al., 2017.
- **mixture of experts** — Replace one big MLP with many expert MLPs and a router that sends each token to a few of them. Parameters grow without per-token compute growing. *(Module 17; also 00b, 16, 19, 24, 40.)* Origin: Shazeer et al., 2017.
- **MoBA** — Block-sparse attention where each query reads its own block plus the earlier blocks whose mean key scores highest against it: mixture-of-experts routing over blocks of the context. *(Module 44; also 17.)* Origin: Lu et al., 2025.
- **mode collapse** — A GAN failure in which the generator produces only a few kinds of output, ignoring the variety of the data. *(Module 33.)*
- **model FLOPs utilization** — Achieved training FLOP/s (≈ 6 × parameters × tokens per second) divided by the GPU's peak. A single number for how well you use the hardware. *(Module 16; also 22, 25, 42.)* Origin: Chowdhery et al., 2022 (PaLM).
- **model merging** — Combining fine-tunes of one base model by arithmetic on their weights, with no further training. TIES trims small changes and resolves sign conflicts; DARE drops most of each change at random and rescales. *(Module 46; also 20.)* Origin: Yadav et al., 2023; Yu et al., 2023.
- **modern Hopfield network** — Hopfield networks with sharper energy functions that store exponentially many patterns; their update rule is softmax attention. *(Module 29.)* Origin: Ramsauer et al., 2020.
- **momentum** — Keep a running average of past gradients and step along it, like a ball rolling downhill. It damps zig-zagging and speeds travel along consistent directions; Nesterov evaluates the gradient at the look-ahead point. *(Module 05; also 11.)* Origin: Polyak, 1964.
- **Monte Carlo** — Estimating an expectation by averaging random samples. The error shrinks like 1/√n, so 100× more samples buys only 10× more accuracy. *(Module 04; also 34.)*
- **Monte Carlo tree search** — Plan by repeatedly simulating from the current position down a search tree, choosing moves that balance promise and uncertainty, and backing up the results. *(Module 34; also 00b.)* Origin: Coulom, 2006; Kocsis & Szepesvári, 2006.
- **MRR** — Mean reciprocal rank: average of 1/(rank of the first relevant result) over queries. *(Module 38.)*
- **multi-agent system** — Several agents cooperating: a manager calling specialists as tools, or agents handing off to one another. Worth it only when one agent's prompt or toolset gets unwieldy. *(Module 39.)* Origin: OpenAI, 2025 (A practical guide to building agents).
- **multi-head attention** — Running several attention operations in parallel on smaller slices of the vector, each free to look for a different relation, then concatenating the results. *(Module 15; also 17, 18.)* Origin: Vaswani et al., 2017.
- **multi-head latent attention** — DeepSeek's attention: compress keys and values into a small shared latent vector and cache only that, with a separate small RoPE part. Much smaller KV cache than GQA at similar quality. *(Module 17; also 15, 16, 18.)* Origin: DeepSeek-AI, 2024 (V2).
- **multi-host** — Running one JAX program across several machines, each with its own devices, coordinated by `jax.distributed`. *(Module 24; also 25.)*
- **multi-query attention** — The extreme of GQA: all query heads share a single key/value head. *(Module 17.)* Origin: Shazeer, 2019.
- **multi-token prediction** — An extra training head that predicts the token after next, densifying the training signal; its predictions can also drive speculative decoding. *(Module 17.)* Origin: Gloeckle et al., 2024; DeepSeek-AI, 2024.
- **multilayer perceptron** — Layers of linear maps with a nonlinearity between them. The basic neural network, and the per-token feed-forward block inside every transformer. *(Module 10; also 11, 12, 14, 16, 31.)* Origin: Rumelhart, Hinton, Williams, 1986.
- **multiple-choice evaluation** — Scoring a model on a question by which answer option it gives the highest (often length-normalized) log-likelihood, without generating text. *(Module 21; also 39.)*
- **Muon** — An optimizer for hidden 2-D weight matrices: take the momentum, orthogonalize it with Newton–Schulz iterations (set all its singular values to ≈1), and step along that. Embeddings, the output head and vector parameters stay on AdamW. *(Module 05; also 25, 42.)* Origin: Keller Jordan, 2024.
- **MuZero** — AlphaZero with a learned model of the game's dynamics, so it plans without knowing the rules. `mctx` is DeepMind's JAX MCTS library built around it. *(Module 34.)* Origin: Schrittwieser et al., 2019.

<a id="n"></a>

## N

- **n-gram** — A run of n consecutive tokens; an n-gram model predicts from the previous n − 1. Also used to measure text overlap in evaluation and contamination checks. *(Module 14; also 21, 40.)* Origin: Shannon, 1948.
- **naive Bayes** — A classifier that assumes features are independent given the class; for binary features its log-probabilities are a linear layer whose weights are log-odds counted from data, not trained. *(Module 46.)*
- **NaN** — Not-a-Number: the result of invalid float math like 0/0 or inf − inf. One NaN poisons everything it touches, so catching its source is a key debugging skill. *(Module 11; also 05, 09, 25, 35, 39.)*
- **Native Sparse Attention** — DeepSeek's trainable sparse attention: compressed block summaries, a few selected blocks and a sliding window, combined by a learned gate and laid out for GPU kernels. *(Module 44.)* Origin: Yuan et al., 2025.
- **nats** — Units of information: a nat uses the natural log, a bit uses log base 2 (1 nat ≈ 1.44 bits). Losses in JAX are usually in nats. *(Module 04; also 14, 32, 33, 35, 37.)*
- **nDCG** — Normalized discounted cumulative gain: rewards relevant results more the higher they rank (with a log discount), scaled so a perfect ranking scores 1. *(Module 38.)* Origin: Järvelin & Kekäläinen, 2002.
- **negative log-likelihood** — −log of the probability a model assigns to the observed data. Minimizing it is maximum-likelihood training and equals minimizing cross-entropy. *(Module 14; also 21.)*
- **negative sampling** — Instead of a softmax over the whole vocabulary, train the model to tell real (word, context) pairs from a few random fake ones, using a sigmoid each. *(Module 30.)* Origin: Mikolov et al., 2013.
- **neural probabilistic language model** — Bengio's 2003 model: embed the previous few tokens, concatenate, and predict the next with an MLP. It introduced learned word embeddings. *(Module 14.)* Origin: Bengio et al., 2003.
- **Newton–Schulz iteration** — A few matrix-multiply-only iterations that push a matrix's singular values toward 1, approximating its polar factor (the nearest orthogonal matrix). It is the core of Muon and runs well on GPUs. *(Module 05.)*
- **Newton–Schulz iteration** — A matmul-only iteration that pushes a matrix's singular values toward 1 while keeping its singular vectors: an approximate orthogonalization. Muon runs five steps of a tuned quintic version. *(Module 42; also 05.)* Origin: Keller Jordan, 2024 (Muon's coefficients).
- **next-token prediction** — The training task of language models: predict token t+1 from tokens up to t. Inputs and targets are the same sequence shifted by one. *(Module 14; also 04, 18, 32, 35, 37.)*
- **NHWC** — Axis-order conventions for images (batch, height, width, channels) and conv kernels (height, width, in, out) used by JAX's convolution API. *(Module 12.)*
- **non-saturating loss** — The practical GAN generator loss, maximizing log D(fake) instead of minimizing log(1 − D(fake)), so it gets strong gradients early on. *(Module 33.)* Origin: Goodfellow et al., 2014.
- **NoPE** — No positional encoding at all. The causal mask still leaks order (a token at position p averages over p + 1 values), so decoders can learn positions without being given them. *(Module 17; also 00b, 36.)* Origin: Haviv et al., 2022.
- **norm** — A measure of a vector's or matrix's size, most often the L2 (Euclidean) length. Used for regularization, clipping and diagnosing training. *(Module 02; also 05, 07, 08, 11, 12.)*
- **normal equations** — The direct least-squares solution w = (XᵀX)⁻¹Xᵀy, found by setting the gradient to zero instead of iterating. *(Module 09; also 16.)*
- **normalization layer** — A layer that re-centres and re-scales activations to a stable range, so deep networks train reliably. *(Module 11; also 33.)*

<a id="o"></a>

## O

- **occupancy** — How many wavefronts are resident per SIMD. More lets the GPU hide memory latency by switching waves; if not even one workgroup fits, the launch fails. *(Module 23.)*
- **offloading** — Putting some layers on the GPU and the rest on the CPU when a model doesn't fit in VRAM. Each CPU layer slows generation a lot. *(Module 19.)*
- **one-hot** — A vector of zeros with a single 1 at the index of a category. Multiplying a one-hot by a matrix just selects a row, which is what an embedding lookup does. *(Module 01; also 14, 15, 30.)*
- **online softmax** — Computing softmax in one pass over blocks by keeping a running max and sum and rescaling as a new max appears, so the full row never has to be in memory. *(Module 23.)* Origin: Milakov & Gimelshein, 2018.
- **OpenAI-compatible API** — The de facto HTTP API for LLM servers (`/v1/chat/completions`), so any client written for OpenAI can point at a local server instead. *(Module 19; also 27, 39.)*
- **optax** — JAX's optimizer library: optimizers are composable gradient transformations (`init` and `update`) chained together, e.g. clip then AdamW. *(Module 05; also 09, 10, 14, 16, 20.)*
- **orchestrator-workers** — A workflow in which one model breaks the task into sub-tasks at run time, worker calls do them, and the results are combined. *(Module 39.)* Origin: Anthropic, 2024.
- **orthogonal matrix** — A matrix whose columns are perpendicular unit vectors; it rotates or reflects without stretching. Gram–Schmidt builds one from any set of vectors. *(Module 02.)*
- **Othello-GPT** — A GPT trained only on legal Othello move sequences, which nonetheless develops an internal board representation detectable with probes. *(Module 37.)* Origin: Li et al., 2022; Nanda et al., 2023.
- **overfit one batch** — A debugging check: a correct model and training loop should drive the loss to near zero on one small batch. If it can't, there's a bug. *(Module 11.)*
- **overfitting** — Fitting the training data so closely, noise included, that performance on new data gets worse. The gap between training and validation loss shows it. *(Module 09; also 10, 11, 35.)*
- **overhead-bound** — When a program's time goes to Python, dispatch and kernel-launch overhead rather than arithmetic or memory. Small models on a big GPU are usually overhead-bound; jit and scan fix it. *(Module 10.)*
- **overtraining** — Training a model on many more tokens than the compute-optimal ~20 per parameter. It costs more to train but gives a better model at a size that's cheap to run; SmolLM2-135M saw ~15,000 tokens per parameter. *(Module 42.)*

<a id="p"></a>

## P

- **padding** — Adding zeros around the input so a convolution's output keeps a chosen size; also padding sequences to a common length. *(Module 12; also 06, 16, 22.)*
- **Pallas** — JAX's kernel language: write a kernel in Python on blocks of arrays (refs), and it is compiled via Triton (GPU) or Mosaic (TPU). Has an interpret mode for debugging on CPU. *(Module 23.)*
- **Pallas grid** — In Pallas, the set of kernel instances to launch; each instance gets a block of the inputs chosen by its BlockSpec. *(Module 23.)*
- **Pallas ref** — In Pallas (and JAX state), a mutable reference to a block of memory that a kernel reads with `x_ref[...]` and writes in place. *(Module 23.)*
- **parallelization** — A workflow running several LLM calls at once and combining them, either by voting on repeated attempts or by splitting the task into sections. *(Module 39.)* Origin: Anthropic, 2024.
- **parameter-efficient fine-tuning** — Fine-tuning methods that train only a small number of extra or selected parameters, like LoRA. *(Module 20.)*
- **parameters** — The learned numbers of a model (weights and biases). Training is the search for good parameter values. *(Module 10; also 03, 05, 12, 16, 20.)*
- **partial derivative** — The derivative with respect to one input while holding the others fixed. *(Module 03.)*
- **partial RoPE** — RoPE applied to only some of each head's features; the rest stay unrotated and match on content regardless of distance. *(Module 17; also 00b.)* Origin: Black et al., 2022 (GPT-NeoX-20B).
- **PartitionSpec** — A tuple naming which mesh axis, if any, each array dimension is split over, e.g. P('data', None). *(Module 24.)*
- **pass@k** — The probability that at least one of k sampled answers is correct; an upper bound for what voting or verification can reach. *(Module 26.)* Origin: Chen et al., 2021 (Codex).
- **patch embedding** — A ViT's first layer: cut the image into p × p patches, flatten each, and multiply by one matrix (the same as a p × p convolution with stride p). Each patch becomes a token. *(Module 43.)* Origin: Dosovitskiy et al., 2020.
- **per-example gradients** — A separate gradient for every example in a batch, obtained as `vmap(grad(loss))`. Used in differential privacy and data diagnostics. *(Module 07.)*
- **perceptron** — Rosenblatt's 1958 trainable linear threshold unit and its learning rule: when it misclassifies, add (or subtract) the input to the weights. Converges whenever the classes are linearly separable. *(Module 29; also 00b, 10.)* Origin: Rosenblatt, 1958.
- **perceptron convergence theorem** — If the data are linearly separable, the perceptron rule finds a separating line in a finite number of mistakes. *(Module 29.)* Origin: Novikoff, 1962.
- **Perceptrons (Minsky & Papert)** — Minsky and Papert's 1969 book proving the limits of single-layer perceptrons (e.g. XOR), widely blamed for the first AI winter. *(Module 29; also 00b.)* Origin: Minsky & Papert, 1969.
- **permutation equivariance** — Shuffling the inputs just shuffles the outputs the same way. Plain attention has it, so it can't tell word order without positional information. *(Module 15.)*
- **perplexity** — exp of the average per-token cross-entropy (in nats): the effective number of choices the model is torn between at each step. Lower is better; only comparable across models with the same tokenizer. *(Module 04; also 14, 18, 21, 27.)*
- **perplexity filtering** — Choosing training data by how predictable a reference model finds it: keep the text it scores well, drop the rest. Cheap, but it favors easy, repetitive text as well as clean text. *(Module 42.)*
- **pinned memory** — Host RAM the operating system promises not to move, so the GPU can copy to and from it directly (DMA) without a staging copy. JAX exposes it as the memory kind `pinned_host`. *(Module 40.)*
- **pipeline parallelism** — Putting different layers on different devices and streaming microbatches through, like an assembly line. Startup and drain leave idle time, the bubble. *(Module 24.)* Origin: Huang et al., 2018 (GPipe).
- **pixel shuffle** — Merging each r × r neighbourhood of image tokens into one wider token: r² fewer tokens for the language model to read. SmolVLM uses it to cut 1,024 patches to 64 tokens. *(Module 43.)* Origin: Shi et al., 2016; Marafioti et al., 2025.
- **PJRT** — The plugin interface between JAX and a hardware backend. On this machine the `jax-rocm` PJRT plugin is the bridge from JAX to the AMD GPU. *(Module 00.)*
- **policy** — In RL, the rule (here the model's token distribution) that maps a state to actions. Fine-tuning an LLM with RL treats the model as the policy. *(Module 34; also 20, 26, 35, 39.)*
- **policy gradient** — Improve a policy by raising the log-probability of actions in proportion to how much better than expected they turned out (the advantage). *(Module 34; also 26.)* Origin: Williams, 1992 (REINFORCE).
- **polysemantic** — A neuron that responds to several unrelated features, a consequence of superposition. *(Module 37.)*
- **pooling** — Downsampling by taking the max or mean over small windows, which shrinks feature maps and adds tolerance to small shifts. *(Module 12.)*
- **position bias** — A judge preferring whichever answer comes first (or second) regardless of content. Countered by scoring both orders. *(Module 21.)*
- **positional encoding** — Information about each token's position added to (or rotated into) its representation so attention can use word order. *(Module 15; also 16, 17.)* Origin: Vaswani et al., 2017.
- **post-training** — Everything done to a pretrained base model to make it a useful assistant: SFT, preference learning, RL with verifiable rewards. *(Module 26; also 39.)*
- **PPO** — The policy-gradient algorithm classic RLHF uses: clip the policy-probability ratio so each update stays small, with a learned value function as the baseline. *(Module 35; also 26, 34.)* Origin: Schulman et al., 2017.
- **pre-norm** — Applying normalization at the start of each residual branch, x + f(norm(x)), rather than after the addition. More stable for deep transformers; used by GPT-2 onward. *(Module 11; also 17.)*
- **pre-tokenization** — Splitting text into chunks (words, numbers, spaces) with a regex before BPE, so merges never cross those boundaries. *(Module 13; also 27.)*
- **preallocation** — By default JAX grabs 75% of GPU memory at startup so later allocations are fast. The course turns it off (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) so several notebooks can share one card. *(Module 00.)*
- **preference data** — Pairs of responses to the same prompt with a label saying which is better; the training data for reward models and DPO. *(Module 35.)*
- **prefetch** — Start loading data you'll need next while the current work runs, so the wait hides behind it. Possible whenever the addresses are known in advance. *(Module 40.)*
- **prefill** — The first phase of generation: run the whole prompt through the model at once, filling the KV cache. Compute-bound. *(Module 18; also 17, 19, 39, 41.)*
- **prefix scan** — Computing all running results (running sums, running products) of a sequence; with an associative operator it parallelizes in O(log L) steps. *(Module 36.)* Origin: Blelloch, 1990.
- **previous-token head** — An attention head that mostly attends to the immediately preceding token; the first half of an induction circuit. *(Module 37.)*
- **primitive** — One of JAX's built-in elementary operations (add, dot_general, exp, …) that every function is traced down to and that each transformation knows how to handle. *(Module 06.)*
- **PRNG key** — JAX's explicit random state: a key you pass to every random function and `split` into fresh subkeys. Same key in, same numbers out, so randomness is reproducible and safe under jit and vmap. *(Module 04; also 11, 16, 17, 31, 34.)*
- **probability distribution** — An assignment of probabilities to outcomes that sum (or integrate) to 1. A language model outputs one distribution over the vocabulary at each position. *(Module 04.)*
- **profiler trace** — A timeline of every kernel and host event during a run, captured with `jax.profiler` and viewed in Perfetto. Shows which kernels dominate and where the GPU sits idle. *(Module 22; also 25.)*
- **projection** — Dropping a vector onto a line or subspace: the closest point there. The dot product with a unit vector gives the projection's length. *(Module 02; also 15, 20.)*
- **prompt caching** — Reusing the KV cache for a prompt prefix shared across requests (like a long system prompt), so it is prefilled once. *(Module 39.)*
- **prompt chaining** — A workflow of fixed LLM steps, each feeding the next, with programmatic checks (gates) between them. *(Module 39.)* Origin: Anthropic, 2024.
- **prompt format** — The exact wording and layout of a prompt. Changing it can move benchmark scores by several points, so it is part of the result. *(Module 21.)*
- **PUCT** — The rule AlphaZero's MCTS uses to pick which move to explore: value estimate plus a bonus proportional to the network's prior for the move, shrinking as it is visited more. *(Module 34.)* Origin: Silver et al., 2017.
- **pure function** — A function whose output depends only on its inputs and which changes nothing outside itself. jit and the other transformations assume purity; side effects run only once, at trace time. *(Module 06; also 05, 07, 10.)*
- **PyTorch** — The most widely used deep-learning framework, imperative and object-oriented. The course compares it with JAX in side notes, using a separate ROCm PyTorch image. *(Module 10; also 03, 15, 22, 23, 26.)*
- **pytree** — Any nested structure of lists, tuples, dicts (or registered classes) with arrays at the leaves. JAX transformations and optax accept pytrees, so model parameters are just a dict of arrays. *(Module 07; also 05, 08, 10, 16, 20.)*

<a id="q"></a>

## Q

- **Q-learning** — Learn Q-values by nudging each estimate toward reward + discounted best next Q, from experience, without a model of the environment. *(Module 34.)* Origin: Watkins, 1989.
- **Q-value** — The expected return of taking a given action in a given state, then acting well. *(Module 34.)*
- **QK-Norm** — Normalizing queries and keys before their dot product, which keeps attention logits from blowing up and stabilizes training. *(Module 00b.)* Origin: Henry et al., 2020.
- **quadratic cost** — Attention compares every pair of positions, so compute and the score matrix grow with the square of the sequence length. *(Module 15.)*
- **quantization** — Storing weights in fewer bits (int8, int4) with a scale per group or channel. Smaller models load faster and decoding speeds up, at a small accuracy cost. *(Module 18; also 01, 19, 27.)*
- **query likelihood** — Ranking documents by the probability a language model would generate the query from them. *(Module 38.)* Origin: Ponte & Croft, 1998.
- **query, key, value** — The three projections in attention: a query asks what a position is looking for, keys advertise what each position has, values are what gets copied. Score = query · key. *(Module 15; also 23.)* Origin: Vaswani et al., 2017.
- **Qwen** — Alibaba's family of open-weight LLMs, used in the course as capable local models in LM Studio and for embeddings. *(Module 27; also 19, 26, 38.)*

<a id="r"></a>

## R

- **random variable** — A quantity whose value is drawn from a probability distribution, like a die roll or the next token. *(Module 04.)*
- **rank** — The number of independent directions a matrix can output: the count of its nonzero singular values. An (m, n) matrix written as A·B with inner size r has rank at most r. *(Module 02; also 05, 17, 20.)*
- **RASP** — A small programming language whose select and aggregate operations are what attention can do; Tracr compiles RASP programs into real transformer weights. *(Module 46; also 37.)* Origin: Weiss et al., 2021; Lindner et al., 2023.
- **RDNA3** — The AMD GPU architecture of the RX 7600 XT, with 32-thread wavefronts and compute units grouped into SIMDs. *(Module 23; also 22.)*
- **ReAct** — Prompting a model to interleave reasoning (Thought), tool calls (Action) and their results (Observation). *(Module 39; also 26.)* Origin: Yao et al., 2022.
- **reasoning model** — An LLM trained, typically with RL on verifiable rewards, to produce long chains of thought before answering. *(Module 26; also 27.)* Origin: DeepSeek-AI, 2025.
- **recall@k** — The fraction of relevant documents that appear in the top k results. *(Module 38.)*
- **receptive field** — The region of the input that can influence one output unit. Stacking and dilating convolutions grows it. *(Module 12.)*
- **reciprocal rank fusion** — Merge ranked lists by summing 1/(k + rank) for each document across lists (k = 60 by convention); needs no score calibration. *(Module 38.)* Origin: Cormack, Clarke, Buettcher, 2009.
- **recompilation** — jit compiles again whenever a call brings a new input shape, dtype or static value. Varying shapes cause a recompile storm; padding to a few bucket sizes avoids it. *(Module 06.)*
- **recurrent neural network** — A network that reads a sequence one step at a time, updating a hidden state from the previous state and the current input with the same weights at every step. *(Module 31; also 08, 36.)* Origin: Elman, 1990.
- **reduce-scatter** — Sum across devices, but each device keeps only its own slice of the result; an all-reduce is a reduce-scatter plus an all-gather. *(Module 24; also 25.)*
- **reduction** — An operation that collapses an axis into one value per remaining position: sum, mean, max, argmax. *(Module 01; also 08, 24.)*
- **regularization** — Anything that discourages fitting noise and so improves generalization: weight decay, dropout, data augmentation, early stopping. *(Module 11; also 05, 09.)*
- **reinforcement learning** — Learning to act by trial and error: an agent observes the state of an environment, takes actions, and adjusts its policy to maximize total reward. *(Module 34; also 25, 26, 35, 39.)*
- **relative position bias** — A learned number per attention head and distance bucket, added to the attention scores: T5's position scheme. *(Module 17; also 32, 00b.)* Origin: Raffel et al., 2019.
- **ReLU** — max(0, x): the simplest common nonlinearity. Networks of ReLUs compute piecewise-linear functions. *(Module 10; also 11, 12, 17, 33, 37.)* Origin: Nair & Hinton, 2010.
- **reparameterization trick** — Write a sample as μ + σ·ε with ε random noise, so gradients can flow through the sampling step to μ and σ. *(Module 33.)* Origin: Kingma & Welling, 2013.
- **repetition penalty** — Lowering the logits of tokens already generated to discourage loops. *(Module 18.)* Origin: Keskar et al., 2019.
- **reranker** — A second stage that rescores the top candidates from a fast retriever with a slower, more accurate model. *(Module 38.)*
- **residual connection** — Adding a block's input to its output, x + f(x). Gradients get a direct path back through the network, which is what makes very deep networks trainable. *(Module 11; also 31.)* Origin: He et al., 2015.
- **residual stream** — The vector each token carries through the network; every attention and MLP block reads from it and adds its output back. A useful mental model for interpretability. *(Module 16; also 37.)* Origin: Elhage et al., 2021.
- **ResNet** — The 2015 deep convolutional network built from residual blocks, which made 100+ layer networks trainable. *(Module 11.)* Origin: He et al., 2015.
- **retrieval-augmented generation** — Retrieve relevant documents for a question and put them in the prompt, so the model answers from sources instead of memory alone. *(Module 38; also 32, 39.)* Origin: Lewis et al., 2020.
- **reverse mode** — Autodiff that runs forward, then sweeps backwards from the output, getting the gradient for all inputs in one pass. Ideal for one scalar loss and millions of parameters, which is why training uses it. *(Module 03; also 08.)*
- **reverse process** — In diffusion, the learned procedure that removes noise step by step, turning random noise into a sample. *(Module 33.)*
- **reward** — The scalar feedback an RL environment (or a reward model or verifier) gives for an action or outcome. *(Module 34; also 26, 35.)*
- **reward hacking** — A policy maximizing the reward in a way that defeats its purpose, such as gaming a format check, because the reward measures the goal imperfectly. *(Module 26.)*
- **reward model** — A model trained to score a response so preferred responses score higher; it stands in for human judgment during RL. *(Module 35.)* Origin: Christiano et al., 2017.
- **ridge point** — The arithmetic intensity at which a roofline's memory and compute roofs meet: peak FLOP/s ÷ bandwidth. *(Module 22.)*
- **ridge regression** — Linear regression with an L2 penalty on the weights, which tames overfitting and makes the solution stable. *(Module 09; also 22.)* Origin: Hoerl & Kennard, 1970.
- **RLAIF** — Using an AI model's judgments instead of human labels to produce preference data. *(Module 35.)* Origin: Bai et al., 2022.
- **RLHF** — Train a reward model on human preference comparisons, then fine-tune the LM with RL to maximize that reward while a KL penalty keeps it near the original. How ChatGPT-style assistants were aligned. *(Module 35; also 34.)* Origin: Christiano et al., 2017; Ouyang et al., 2022.
- **RLVR** — Reinforcement learning whose reward comes from an automatic verifier (correct answer, passing tests) rather than a learned reward model. How DeepSeek-R1-style reasoning is trained. *(Module 26.)* Origin: Lambert et al., 2024; DeepSeek-AI, 2025.
- **RMSNorm** — LayerNorm without the mean subtraction: divide by the root-mean-square, then scale. Cheaper and works as well; used by Llama and DeepSeek. *(Module 11; also 16, 17.)* Origin: Zhang & Sennrich, 2019.
- **RMSProp** — Divide each parameter's step by a running root-mean-square of its recent gradients, so steep directions take smaller steps and flat ones larger. *(Module 05.)* Origin: Hinton, 2012 (lecture).
- **ROCm** — AMD's open GPU software stack: driver interface, the HIP runtime and tuned math libraries. It plays the role CUDA plays on NVIDIA cards, and in this course it lives only inside the container. *(Module 00; also 00b, 19, 22, 23, 27.)*
- **rocprof** — AMD's command-line GPU profiler; it reports every kernel launch with its resources, which is how the Triton-GEMM failure is diagnosed. *(Module 22.)*
- **ROME** — Rank-One Model Editing: rewrite one fact by changing one MLP output matrix so the subject's key produces a new value, sized by the covariance of the layer's usual inputs. MEMIT does thousands at once. *(Module 46.)* Origin: Meng et al., 2022.
- **roofline** — A plot of achievable FLOP/s against arithmetic intensity: a sloped memory roof for low intensity and a flat compute roof for high. Tells you whether a kernel is memory- or compute-bound and how far from peak. *(Module 22; also 18.)* Origin: Williams, Waterman, Patterson, 2009.
- **RoPE** — Rotary position embedding: rotate pairs of query and key dimensions by angles proportional to position, so their dot product depends only on relative position. The base sets the rotation frequencies. *(Module 17; also 16.)* Origin: Su et al., 2021.
- **round-off error** — The error from storing a number with finite precision. It compounds in long sums and is worst in low-precision formats. *(Module 08; also 03, 18, 27.)*
- **router** — In MoE, the small layer that scores the experts for each token and picks the top few. *(Module 17.)*

<a id="s"></a>

## S

- **S4** — Structured State Space sequence model: an LTI SSM with a special (HiPPO-initialized) state matrix, computed by FFT convolution, that handled very long sequences. *(Module 36.)* Origin: Gu, Goel, Ré, 2021.
- **safetensors** — A simple, safe file format for model weights (a header plus raw tensors), standard on Hugging Face. *(Module 27.)*
- **sampling** — Generating text by drawing each next token from the model's predicted distribution, rather than always taking the most likely. *(Module 04; also 18, 21, 26, 33, 35.)*
- **scaled dot-product attention** — Attention whose scores are divided by √d before softmax, because dot products of random d-dimensional vectors have variance d and would otherwise saturate the softmax. *(Module 15.)* Origin: Vaswani et al., 2017.
- **scaling laws** — Empirical power laws relating a model's loss to its parameter count, data and compute. They let you predict how much bigger and better a model gets for a given budget. *(Module 00b; also 42.)* Origin: Kaplan et al., 2020; Hoffmann et al., 2022.
- **seatbelt** — The course's AWS safety rails: budgets, quotas, automatic stop of idle instances, and never applying infrastructure changes without the owner's go-ahead. *(Module 25.)*
- **selective scan** — Mamba's core: a selective, input-dependent SSM recurrence computed with a scan rather than a convolution. *(Module 36.)* Origin: Gu & Dao, 2023.
- **selectivity** — Making an SSM's step size and B, C matrices depend on the current input, so it can choose what to store or ignore. Breaks time-invariance (no more convolution) but enables content-based recall. *(Module 36.)* Origin: Gu & Dao, 2023.
- **self-attention** — Attention in which queries, keys and values all come from the same sequence, so tokens attend to each other. *(Module 15; also 31.)* Origin: Vaswani et al., 2017.
- **self-consistency** — Sample several reasoning chains and take the most common final answer. *(Module 26.)* Origin: Wang et al., 2022.
- **self-play** — Training a game agent on games it plays against itself, generating its own ever-harder data. *(Module 34.)* Origin: Samuel, 1959; Tesauro, 1992.
- **self-refinement** — Asking a model to critique and revise its own answer. *(Module 26.)* Origin: Madaan et al., 2023.
- **semi-autoregressive sampling** — Generate in blocks left to right, filling each block by diffusion: a middle ground between autoregressive and fully parallel decoding. *(Module 45.)* Origin: Arriola et al., 2025; Nie et al., 2025.
- **seq2seq** — An encoder RNN reads the input into one fixed-size vector and a decoder RNN generates the output from it; the 2014 recipe for translation. That single vector is a bottleneck, which attention removed. *(Module 31; also 00b.)* Origin: Sutskever, Vinyals, Le, 2014.
- **shape** — The tuple of an array's sizes along each axis, e.g. (batch, seq, d). Most deep-learning bugs are shape bugs, so reading shapes is the first skill of the course. *(Module 01; also 06, 07, 12, 17, 23.)*
- **shard_map** — A JAX transformation in which you write the per-device program and its communication explicitly, instead of letting the compiler partition a global one. *(Module 24.)*
- **sharding** — How an array is split (or replicated) across the devices of a mesh. With sharded inputs, jit produces a program that runs on all of them. *(Module 24; also 25.)*
- **shared experts** — DeepSeekMoE experts that every token always uses, alongside the routed ones the router chooses; they hold common knowledge. *(Module 17.)* Origin: Dai et al., 2024 (DeepSeekMoE).
- **short-term memory** — What an agent keeps of the current conversation: the full history, a sliding window of recent turns, a running summary or a structured ledger. *(Module 39; also 21.)*
- **SigLIP** — A CLIP-style image-text model trained with a sigmoid loss: each image-text pair in the batch is its own yes/no question, with no softmax across the batch. The vision encoder of most open VLMs. *(Module 43.)* Origin: Zhai et al., 2023.
- **sigmoid** — σ(x) = 1/(1 + e⁻ˣ): squashes any number into (0, 1). The two-class special case of softmax. *(Module 09; also 10, 33.)*
- **SIMD** — Single instruction, multiple data: a hardware unit applying one instruction to a whole wavefront at once. *(Module 23.)*
- **skip-gram** — word2vec's objective: from a centre word, predict the words within a small window around it. *(Module 30.)* Origin: Mikolov et al., 2013.
- **sliding-window attention** — Each token attends only to the previous W tokens, so compute and the KV cache grow with W instead of the context length. *(Module 44; also 00b.)* Origin: Beltagy, Peters, Cohan, 2020 (Longformer).
- **SmolLM2** — Hugging Face's small open LLMs (135M–1.7B); the real pretrained weights the course loads for inference, LoRA, evaluation, RAG and agents. *(Module 18; also 20, 21, 27, 35, 38.)* Origin: Allal et al., 2025.
- **smoothing** — Adding a small count to every possibility so unseen pairs don't get probability zero (and infinite loss). *(Module 14.)*
- **softmax** — Exponentiate a vector of logits and divide by the total, turning arbitrary scores into a probability distribution. Large logits dominate. *(Module 04; also 08, 15, 18, 22, 23.)*
- **softmax regression** — Logistic regression for many classes: one linear score per class, then softmax and cross-entropy. *(Module 09; also 10.)*
- **sparse activation** — Only part of the model runs per token (the chosen experts), so active parameters are far fewer than total parameters. *(Module 17.)*
- **sparse autoencoder** — An autoencoder with a sparsity penalty trained on a model's activations, to pull superposed features apart into separate, more interpretable units. *(Module 37.)* Origin: Bricken et al., 2023.
- **sparse retrieval** — Search by exact word overlap, using sparse vectors with one dimension per vocabulary word (TF-IDF, BM25). *(Module 38.)*
- **sparse upcycling** — Turn a trained dense model into a mixture of experts by copying its feed-forward block into every expert, adding a fresh router, and training on. *(Module 42; also 17.)* Origin: Komatsuzaki et al., 2022.
- **sparsity** — Having many parameters but using only a few for each input: an MoE's unpicked experts, an Engram table's unread rows. Capacity grows without per-token compute growing. *(Module 40; also 17.)*
- **special tokens** — Reserved tokens with structural meaning, like end-of-text or chat-role markers, that never come from ordinary text. *(Module 13.)*
- **speculative decoding** — A small draft model guesses several tokens, the big target model checks them all in one forward pass, and every accepted guess is a token gained. The output distribution is unchanged. *(Module 18; also 17, 27.)* Origin: Leviathan et al., 2022; Chen et al., 2023.
- **speed ladder** — Module 22's sequence of changes, each timed on the same model: eager → jit → mixed precision → donation → scanned steps. *(Module 22.)*
- **SPMD** — Single program, multiple data: every device runs the same program on its own shard. XLA's SPMD partitioner rewrites one global program into that form, inserting communication. *(Module 24.)*
- **spurious memories** — Stable states of a Hopfield network that weren't stored, such as mixtures or negatives of stored patterns. *(Module 29.)*
- **state space duality** — Mamba-2's observation that a selective SSM with a scalar decay is decayed linear attention: the same function computed as a recurrence or as masked attention, and fastest as chunks of both. *(Module 41; also 36.)* Origin: Dao & Gu, 2024.
- **state-space model** — A sequence model that keeps a fixed-size state updated linearly at each step, h_t = A h_{t−1} + B x_t, y_t = C h_t. Runs as a recurrence, a convolution or a scan. *(Module 36.)* Origin: Kalman, 1960; Gu et al., 2021.
- **static argument** — A jit argument treated as a compile-time constant rather than traced. Python control flow may depend on it, but each new value triggers a recompile. *(Module 06; also 08.)*
- **step bubble** — Idle GPU time between training steps while the host does Python work or waits. Scanning many steps per call removes it. *(Module 22.)*
- **stochastic gradient descent** — Gradient descent on a random minibatch instead of all the data each step: noisier but far cheaper, and the noise can even help. *(Module 05.)* Origin: Robbins & Monro, 1951.
- **stop_gradient** — Marks a value as a constant for differentiation: it passes through forward unchanged, but no gradient flows back through it. Used for targets, frozen parts and bias updates that shouldn't be learned by backprop. *(Module 17.)*
- **streaming** — Sending generated tokens to the client as they are produced rather than all at the end. *(Module 19; also 23.)*
- **stride** — How far a convolution or pooling window moves between positions; stride 2 halves the output size. *(Module 12.)*
- **structured output** — Constraining a model's output to match a JSON schema, typically enforced by constrained decoding on the server. *(Module 27.)*
- **structured pruning** — Make a trained model smaller by removing whole pieces — layers, attention heads, feed-forward neurons, embedding channels — chosen by an importance score, then retrain briefly (ideally by distillation). *(Module 42.)* Origin: Muralidharan et al., 2024 (Minitron).
- **subsampling** — Randomly dropping very frequent words (the, of) from training pairs so rarer words get more updates. *(Module 30.)* Origin: Mikolov et al., 2013.
- **subword** — A token that is part of a word: common words stay whole while rare ones split into pieces, balancing vocabulary size and sequence length. *(Module 13.)*
- **superposition** — Networks representing more features than they have dimensions by overlapping them in nearly orthogonal directions, which works when features are sparse. *(Module 37.)* Origin: Elhage et al., 2022.
- **supervised fine-tuning** — The first post-training step: fine-tune on demonstration responses by ordinary next-token prediction. *(Module 35; also 25, 26.)* Origin: Ouyang et al., 2022.
- **surprise** — −log p of an outcome: rare events are very surprising, certain ones not at all. Entropy is the expected surprise. *(Module 04; also 21.)* Origin: Shannon, 1948.
- **SVD** — Any matrix factors as rotate, scale along axes, rotate: A = U Σ Vᵀ. The scale factors (singular values) rank the directions by importance. *(Module 02; also 05, 30.)*
- **SwiGLU** — A gated MLP: one projection passed through SiLU (swish) multiplies another elementwise before the down-projection. Used by Llama and DeepSeek in place of a GELU MLP. *(Module 17; also 10.)* Origin: Shazeer, 2020.
- **symbolic AI** — The tradition of building intelligence from explicit rules, logic and symbols, as opposed to learning from examples. The course's history module contrasts these two traditions. *(Module 00b; also 29.)*
- **system prompt** — The first message in a chat, setting the assistant's role and rules before the user speaks. *(Module 27; also 39.)*

<a id="t"></a>

## T

- **tanh** — A sigmoid-shaped nonlinearity with outputs in (−1, 1). Its flat tails cause vanishing gradients in deep or recurrent networks. *(Module 10; also 03, 06, 07, 11, 31.)*
- **target network** — A slowly updated copy of the Q-network used to compute training targets, so the targets don't chase every update. *(Module 34.)* Origin: Mnih et al., 2015.
- **task arithmetic** — Treating the weight change from fine-tuning as a vector you can add, subtract and scale to combine or remove skills. *(Module 20; also 46.)* Origin: Ilharco et al., 2022.
- **temperature** — Divide logits by T before softmax: T < 1 sharpens the distribution toward the top choice, T > 1 flattens it, T → 0 is greedy decoding. *(Module 04; also 14, 16, 18, 19, 26, 42.)*
- **temporal-difference learning** — Updating value estimates toward a target built from the next step's estimate, learning from incomplete episodes. TD-Gammon used it to play backgammon in 1992. *(Module 34.)* Origin: Sutton, 1988; Tesauro, 1992.
- **tensor parallelism** — Splitting each weight matrix across devices (column- or row-wise) so each computes part of every layer, with collectives in between. *(Module 24.)* Origin: Shoeybi et al., 2019 (Megatron-LM).
- **Terraform** — An infrastructure-as-code tool that creates cloud resources from configuration files; `apply` is the step that actually provisions and costs money. *(Module 25.)*
- **TF-IDF** — Scoring a term in a document by how often it appears there (term frequency) times how rare it is across documents (inverse document frequency). *(Module 38.)* Origin: Spärck Jones, 1972.
- **TFLOP/s** — Floating-point operations per second, in trillions: the speed unit for GPU arithmetic. A matrix multiply of (m,k)·(k,n) costs about 2·m·k·n FLOPs, so time × TFLOP/s tells you whether the card is busy. *(Module 00; also 06, 16, 18, 22, 25.)*
- **throughput** — Work done per unit time, e.g. tokens per second; distinct from latency, the time for one request. *(Module 18; also 06, 16, 19, 22, 25.)*
- **tiling** — Splitting a big computation into blocks that fit in registers or LDS, so each piece of data is loaded from memory once and reused many times. *(Module 23; also 24.)*
- **token** — One unit of text as a model sees it, an integer id for a character, byte, word piece or word. *(Module 13; also 16, 17, 18, 21, 32.)*
- **tokenizer** — The component that turns text into a sequence of token ids and back. It is fixed before training and shapes cost and behaviour. *(Module 13; also 16, 18, 21, 27, 32.)*
- **tokenizer compression** — Map token ids that mean the same thing after normalization (case, accents, spaces) to one canonical id. Engram indexes its tables this way, so variants share rows. *(Module 40; also 13.)* Origin: Cheng et al. (DeepSeek-AI), 2026.
- **tool calling** — A model emitting a structured request (tool name + JSON arguments) that the host program executes, feeding the result back as a message. *(Module 27; also 25, 39.)* Origin: Schick et al., 2023 (Toolformer).
- **top-k sampling** — Sample only from the k most probable tokens, renormalized. *(Module 18.)* Origin: Fan et al., 2018.
- **top-p sampling** — Sample from the smallest set of tokens whose probabilities add up to p, so the candidate set adapts to how confident the model is. *(Module 18.)* Origin: Holtzman et al., 2019.
- **TPU** — Google's Tensor Processing Unit, a chip built for matrix multiplies. XLA and JAX grew up targeting it, which is why JAX's programming model is compiler-first. *(Module 00b; also 25.)*
- **tracer** — The placeholder value JAX passes through your function while tracing. It knows its shape and dtype but not its numbers, which is why Python `if` on it fails. *(Module 06.)*
- **tracing** — Running a Python function with abstract stand-in values (tracers) that record each array operation instead of computing it. The recording is what JAX transforms and compiles. *(Module 06; also 08.)* Origin: Frostig, Johnson, Leary, 2018.
- **train/validation/test split** — Training data fits the parameters, validation data picks hyperparameters and detects overfitting, and the test set is touched once at the end for an honest estimate. *(Module 09; also 10, 13, 21, 35, 37.)*
- **Transformer** — The architecture of stacked blocks of self-attention plus a per-token MLP, each wrapped in a residual connection and normalization. The basis of all modern LLMs. *(Module 15; also 00b, 17, 31, 32, 37.)* Origin: Vaswani et al., 2017.
- **translation equivariance** — Shifting the input shifts the output the same way. Convolutions have it built in; attention is instead permutation-equivariant. *(Module 12.)*
- **Tree of Thoughts** — Search over partial chains of thought, evaluating and expanding the most promising branches rather than following one chain. *(Module 26.)* Origin: Yao et al., 2023.
- **Triton** — A Python-embedded language and compiler for GPU kernels. XLA can use it to generate its own matmul kernels; on this card some of those request more registers or shared memory than exist and the launch fails, so the course disables them. *(Module 22; also 00, 23.)* Origin: Tillet et al., 2019.
- **type promotion** — The rule for the result dtype when you combine arrays of different dtypes. Python scalars are weakly typed in JAX, so `x * 2.0` keeps x's dtype instead of upcasting. *(Module 01.)*

<a id="u"></a>

## U

- **U-Net** — A convolutional encoder-decoder with skip connections between matching resolutions, the standard denoiser in image diffusion. *(Module 33.)* Origin: Ronneberger et al., 2015.
- **underfitting** — A model too simple, or trained too little, to capture the pattern even on the training data. *(Module 09; also 11.)*
- **universal approximation** — One hidden layer with enough units can approximate any continuous function. It says nothing about how many units or how to find the weights. *(Module 10.)* Origin: Cybenko, 1989; Hornik, 1991.
- **unrolling** — Turning a Python loop into a long straight-line program during tracing. It makes compile time grow with the loop length; lax.scan avoids it. *(Module 08.)*
- **UTF-8** — The variable-length encoding of Unicode text as bytes (1 to 4 per character). Byte-level tokenizers start from these 256 values. *(Module 13; also 21.)*

<a id="v"></a>

## V

- **VAE** — An autoencoder whose encoder outputs a Gaussian over latents; trained on the ELBO, so the latent space is smooth enough to sample new data from. *(Module 33.)* Origin: Kingma & Welling, 2013.
- **value function** — The expected return from a state under a policy. AlphaZero's network predicts it to judge positions. *(Module 34; also 17.)*
- **vanishing gradients** — Gradients shrinking (or growing) exponentially as they are multiplied back through many layers or time steps, so early layers stop learning (or blow up). *(Module 31.)* Origin: Hochreiter, 1991; Bengio et al., 1994.
- **variance** — The expected squared distance of a random variable from its mean: how spread out it is. The standard deviation is its square root. *(Module 04; also 10, 11, 15, 33.)*
- **vector** — An ordered list of numbers, best pictured as an arrow in space. Embeddings, gradients and hidden states are all vectors. *(Module 02; also 15, 17, 30, 32, 38.)*
- **vector database** — A database that stores embeddings and answers nearest-neighbour queries, usually with an ANN index; pgvector and OpenSearch are two. *(Module 38; also 39.)*
- **verifier** — A program that checks an answer automatically (e.g. exact arithmetic). Its pass/fail becomes the reward, so no learned reward model is needed. *(Module 26; also 35, 39.)*
- **VGPR** — Vector general-purpose registers: each thread's private fast storage. A SIMD's register file is shared by its resident waves, so more VGPRs per thread means fewer waves fit. *(Module 23.)*
- **vision transformer** — A Transformer encoder applied to an image cut into patches, one token per patch. Weaker than a CNN on small datasets, stronger once pretrained on enough data. *(Module 43; also 12.)* Origin: Dosovitskiy et al., 2020.
- **vision-language model** — A language model that reads images: a vision encoder's patch tokens, mapped by a small projector into the LLM's embedding space, are placed in the prompt like words (LLaVA's recipe). *(Module 43; also 33.)* Origin: Liu et al., 2023.
- **VJP** — Vector-Jacobian product: pull a weighting of the outputs back to the inputs, computed by reverse mode. `grad` is a VJP with the vector 1. *(Module 03.)*
- **vmap** — JAX's automatic vectorization: write a function for one example, and `vmap` turns it into one that runs on a whole batch, as if the loop were written by hand. *(Module 07; also 03, 04, 14, 16, 34.)*
- **vocabulary** — The set of tokens a tokenizer can produce. Its size V fixes the embedding table and output layer, which cost V × d parameters each. *(Module 13; also 14, 16, 17, 22, 31.)*
- **VRAM** — The GPU's own memory, 16 GB on the RX 7600 XT. Weights + KV cache + overhead must fit for full-speed inference. *(Module 19; also 00.)*
- **Vulkan** — A cross-vendor graphics and compute API; llama.cpp's Vulkan backend is an alternative to ROCm/HIP and is sometimes faster at generation on this card. *(Module 19; also 27.)*

<a id="w"></a>

## W

- **warmup** — Ramping the learning rate up from near zero over the first steps, so the early, badly scaled updates (and Adam's noisy statistics) don't blow the model up. *(Module 05; also 11, 16.)*
- **wavefront** — A group of threads (32 on RDNA3) that execute in lock-step on one SIMD; AMD's name for a warp. *(Module 23.)*
- **WaveNet** — DeepMind's 2016 audio model built from dilated causal convolutions; an early autoregressive sequence model without recurrence. *(Module 12.)* Origin: van den Oord et al., 2016.
- **weight decay** — Shrinking every weight slightly toward zero each step, which discourages large weights and reduces overfitting. Decoupled from the gradient in AdamW. *(Module 05; also 11.)*
- **weight sharing** — Using the same parameters at every position (convolution) or every time step (RNN), so the model generalizes across position with far fewer parameters. *(Module 12.)*
- **weight tying** — Using the same matrix for the input embedding and the output projection to logits, saving V × d parameters. *(Module 16.)* Origin: Press & Wolf, 2016.
- **weight-space programming** — An LLM reading and editing another network's raw weights by reasoning, with a few forward passes and no gradients (Takehi et al., 2026); later work had models edit their own weights. *(Module 46.)* Origin: Takehi et al., 2026.
- **win rate** — The fraction of comparisons in which one model's answer is preferred over another's (by people, a judge or a reward model). *(Module 35.)*
- **word analogy** — Solving "a is to b as c is to ?" by vector arithmetic b − a + c and finding the nearest word. *(Module 30.)* Origin: Mikolov et al., 2013.
- **word2vec** — Mikolov's 2013 method for learning word embeddings by training a shallow model to predict context words; it showed vectors capture meaning (king − man + woman ≈ queen). *(Module 30; also 00b, 32.)* Origin: Mikolov et al., 2013.
- **workflow** — LLMs and tools orchestrated through predefined code paths, as opposed to an agent that directs itself. Anthropic's five patterns are prompt chaining, routing, parallelization, orchestrator-workers and evaluator-optimizer. *(Module 39.)* Origin: Anthropic, 2024 (Building effective agents).
- **workgroup** — Threads that run on one compute unit and can share LDS and synchronize; at most 1024 threads. *(Module 23; also 22.)*
- **world model** — An internal representation of the state of the world behind the data (like the board behind the moves) that a model builds and uses. *(Module 37.)*
- **WY representation** — A compact form for a product of (I − β k kᵀ) factors. It lets the delta rule run chunk by chunk with one triangular solve and a few matmuls per chunk. *(Module 41.)* Origin: Yang et al., 2024 (DeltaNet).

<a id="x"></a>

## X

- **Xavier initialization** — Initial weight variance 2/(fan_in + fan_out), designed for tanh-like activations. *(Module 11.)* Origin: Glorot & Bengio, 2010.
- **XLA** — The compiler underneath JAX: it takes the traced graph of array operations, fuses and optimizes it, and generates GPU kernels. *(Module 06; also 00, 08, 16, 22, 23.)*
- **XLA dump** — Asking XLA to write its intermediate programs (before and after optimization) to files so you can see what it fused. *(Module 22.)*
- **XLA_FLAGS** — Environment variable that passes options to the XLA compiler. This course sets `--xla_gpu_enable_triton_gemm=false` so matrix multiplies go to AMD's tuned libraries instead of Triton-generated kernels that crash this card at full size. *(Module 00; also 22.)*
- **XOR** — The exclusive-or function: true when exactly one input is true. No straight line separates its classes, the classic thing a single linear layer or perceptron cannot learn. *(Module 10; also 29.)* Origin: Minsky & Papert, 1969.
