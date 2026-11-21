"""Glossary: every technical term the course uses, where it is taught, and where else it matters.

Each `Term` is written for a strong developer who is new to ML: intuition first, no unexplained jargon.
`module` is the display label of the lesson that teaches the term ("00b", "17"); `also` lists other modules
where it matters. `aliases` are other spellings used to find the term in lesson text (case-insensitive).
`tools/gen_glossary.py` renders docs/GLOSSARY.md.
"""
from __future__ import annotations

from typing import NamedTuple


class Term(NamedTuple):
    term: str  # canonical name as the course writes it
    aliases: tuple[str, ...]  # other spellings found in lessons (case-insensitive)
    definition: str  # 1-3 plain-language sentences, intuition first
    module: str  # display label of the module that teaches it
    also: tuple[str, ...]  # other modules where it matters (display labels)
    origin: str  # paper/person/year it comes from, or ""


TERMS: list[Term] = [
    Term("6N rule", ("6 FLOPs per parameter per token", "6N"),
         "Training costs about 6 FLOPs per parameter per token: 2 for the forward pass and 4 for the "
         "backward.",
         "16", (), "Kaplan et al., 2020"),
    Term("[CLS] token", ("cls_pool",),
         "A special token whose final vector serves as a summary of the whole input for classification.",
         "32", (), "Devlin et al., 2018"),
    Term("ablation", ("ablate", "ablate_head", "ablating"),
         "Switching off a component (e.g. zeroing a head's output) and measuring what breaks, to test its "
         "role.",
         "37", ("34",), ""),
    Term("acceptance rate", ("accepted tokens",),
         "In speculative decoding, the fraction of draft tokens the target model accepts; it decides the "
         "speed-up.",
         "18", (), ""),
    Term("accumulation", ("preferred_element_type",),
         "Summing many products into one result. Doing it in float32 even when inputs are bfloat16 "
         "(`preferred_element_type=float32`) avoids losing small terms.",
         "08", ("23",), ""),
    Term("accuracy", ("top-1 accuracy",),
         "The fraction of predictions that are exactly right.",
         "09", ("10", "12", "15", "21", "26"), ""),
    Term("activation function", ("nonlinearity",),
         "The elementwise nonlinear function between layers. Without it, stacked linear layers collapse "
         "into one linear map.",
         "10", ("31",), ""),
    Term("activations", ("activation memory",),
         "The intermediate values a network computes in the forward pass. They must be stored for the "
         "backward pass, which often dominates training memory.",
         "10", ("11", "17", "20", "25", "37"), ""),
    Term("Adam", (),
         "Momentum plus RMSProp, with bias correction for the early steps. The default optimizer for deep "
         "learning.",
         "05", ("09", "12", "14", "20", "24"), "Kingma & Ba, 2014"),
    Term("AdamW", (),
         "Adam with decoupled weight decay: shrink the weights directly each step instead of adding an L2 "
         "term to the loss, which Adam's per-parameter scaling would distort. The standard optimizer for "
         "transformers.",
         "05", ("10", "16", "20", "22", "25"), "Loshchilov & Hutter, 2017"),
    Term("adapter", ("adapters",),
         "A small set of trainable weights added to a frozen model to specialize it, like a LoRA update. "
         "Can be shipped separately and swapped.",
         "20", ("26", "27", "35"), "Houlsby et al., 2019"),
    Term("additive attention", ("Bahdanau attention", "additive_attention"),
         "The first attention: score each encoder state against the decoder state with a small MLP, "
         "softmax, and average. Added to seq2seq for translation.",
         "31", (), "Bahdanau, Cho, Bengio, 2014"),
    Term("advantage", ("advantages",),
         "How much better an action's outcome was than a baseline expectation; positive advantages are "
         "reinforced, negative ones suppressed.",
         "26", (), ""),
    Term("agent", ("agents", "LLM agent", "agentic"),
         "An LLM that directs its own process: in a loop it decides which tool to call, reads the result "
         "and continues until done.",
         "39", ("19", "26", "34", "38"), ""),
    Term("agent loop", ("tool loop", "exit condition"),
         "The core of an agent: send the conversation to the model, execute any tool call it makes, append "
         "the result, repeat until it answers (or a turn limit hits).",
         "39", (), ""),
    Term("AI winter", ("AI winters",),
         "A period when funding and interest in AI collapsed after over-promising: roughly the mid-1970s "
         "and again the late 1980s to early 1990s.",
         "00b", ("29",), ""),
    Term("AlexNet", (),
         "The convolutional network that won the 2012 ImageNet competition by a wide margin, trained on "
         "two GPUs. It is the usual start date of the deep-learning era.",
         "00b", (), "Krizhevsky, Sutskever, Hinton, 2012"),
    Term("ALiBi", ("attention with linear biases",),
         "Attention with Linear Biases: no position embeddings; each head subtracts a penalty proportional "
         "to the query-key distance from its attention scores.",
         "17", ("00b",), "Press, Smith, Lewis, 2021"),
    Term("all-gather", ("allgather",),
         "Every device contributes its shard and every device ends up with the full array.",
         "24", ("25",), ""),
    Term("all-reduce", ("allreduce", "psum"),
         "Every device contributes a value and every device gets the sum. Data parallelism uses it to "
         "average gradients.",
         "24", ("25",), ""),
    Term("all-to-all", ("all to all",),
         "Every device sends a different piece to every other device; used to route tokens to experts in "
         "expert parallelism.",
         "24", ("40",), ""),
    Term("AlphaGo", ("AlphaGo Zero",),
         "DeepMind's Go program that beat Lee Sedol in 2016, combining neural networks with MCTS; AlphaGo "
         "Zero (2017) learned from self-play alone.",
         "34", ("00b",), "Silver et al., 2016, 2017"),
    Term("AlphaZero", (),
         "One algorithm for Go, chess and shogi: a network with policy and value heads guides MCTS, and is "
         "trained on self-play to match MCTS's move choices and the game outcomes.",
         "34", ("00b",), "Silver et al., 2017"),
    Term("Anthropic-compatible API", ("Anthropic-compatible", "/v1/messages", "Messages API"),
         "A server endpoint speaking Anthropic's Messages API, so tools built for Claude (like Claude "
         "Code) can talk to a local model.",
         "27", (), ""),
    Term("approximate nearest neighbour", ("ANN", "approximate nearest neighbor"),
         "Finding vectors close to a query without comparing against every one, trading a little recall "
         "for large speed-ups; what vector databases do.",
         "38", (), ""),
    Term("arithmetic intensity", ("operational intensity", "FLOPs per byte"),
         "FLOPs performed per byte moved to or from memory. Below the card's ridge point a kernel is "
         "memory-bound; above it, compute-bound.",
         "22", (), "Williams, Waterman, Patterson, 2009"),
    Term("array", ("ndarray", "jax.Array"),
         "A grid of numbers of one dtype with a fixed shape; the only data structure JAX computes on. JAX "
         "arrays are immutable and may live on the GPU.",
         "01", ("02", "03", "07", "24", "34"), ""),
    Term("artificial neuron", ("McCulloch-Pitts", "McCulloch–Pitts", "threshold unit"),
         "A unit that sums weighted inputs and fires if the total crosses a threshold; McCulloch and "
         "Pitts' 1943 model of a neuron.",
         "29", (), "McCulloch & Pitts, 1943"),
    Term("associative recall", ("multi-query associative recall", "MQAR"),
         "A synthetic task: given key–value pairs in context, return the value for a queried key. "
         "Attention solves it easily; it probes what sequence models can store and retrieve.",
         "15", ("36", "41"), ""),
    Term("associative scan", ("lax.associative_scan", "associative_scan"),
         "A parallel prefix scan for any associative combine; JAX's `lax.associative_scan` runs a linear "
         "recurrence in log depth on the GPU.",
         "36", (), "Blelloch, 1990"),
    Term("asynchronous dispatch", ("dispatches asynchronously", "async dispatch", "block_until_ready"),
         "JAX returns from a call as soon as the work is queued on the GPU, not when it finishes. Timing "
         "must call `block_until_ready()` or you measure only the launch.",
         "06", ("10", "16", "18", "20", "33", "40"), ""),
    Term("attention", ("attention mechanism",),
         "A soft dictionary lookup: each position scores every position it may see, softmaxes the scores "
         "into weights, and takes the weighted average of their values. Lets any token read from any "
         "other.",
         "15", ("17", "23", "31", "36", "37"), "Bahdanau et al., 2014; Vaswani et al., 2017"),
    Term("attention pattern", ("attention patterns", "attention map", "attention_entropy"),
         "The matrix of attention weights from each query to each key, the first thing to look at in a "
         "head.",
         "37", (), ""),
    Term("attention sink", ("attention sinks", "sink token", "sinks", "StreamingLLM"),
         "The first token(s) of a sequence, on which a trained model parks attention it doesn't need "
         "(softmax weights must sum to 1). Dropping them from a sliding window breaks a pretrained model; "
         "keeping four fixes it.",
         "44", (), "Xiao et al., 2023"),
    Term("autoencoder", ("autoencoders", "encoder and decoder"),
         "A network trained to reconstruct its input through a narrow bottleneck, so the bottleneck learns "
         "a compressed code.",
         "33", ("37",), ""),
    Term("automatic differentiation", ("autodiff", "autograd"),
         "Computing exact derivatives of a program by applying the chain rule to each primitive operation "
         "it executes. Neither symbolic algebra nor finite differences.",
         "03", ("00b",), "Baydin et al., 2015 (survey)"),
    Term("autoregressive", ("autoregressively",),
         "Generating one token at a time, each conditioned on all earlier ones, by feeding outputs back in "
         "as inputs.",
         "18", (), ""),
    Term("axis", ("axes",),
         "One dimension of an array. Reductions such as `sum(axis=0)` collapse an axis; a batch axis holds "
         "independent examples.",
         "01", ("07", "12", "17", "24", "33"), ""),
    Term("back-of-the-envelope estimate", ("back of the envelope", "estimate_run"),
         "Predicting a training run's memory, time, communication and cost from a few measured numbers "
         "before paying for it.",
         "25", (), ""),
    Term("backpropagation", ("backprop", "back-propagation", "backpropagated"),
         "Reverse-mode autodiff applied to a neural network: run forward, then propagate the loss's "
         "derivative backwards through each layer to get every parameter's gradient in one pass.",
         "03", ("00b", "10", "11", "29", "31"), "Rumelhart, Hinton, Williams, 1986"),
    Term("backpropagation through time", ("BPTT",),
         "Backpropagation on an RNN unrolled over the sequence; gradients pass back through every step, so "
         "they can vanish or explode.",
         "31", (), "Werbos, 1990"),
    Term("base model", ("base models", "pretrained model", "pretraining", "pre-training"),
         "A model trained only on next-token prediction over raw text. It continues text but doesn't "
         "follow instructions until fine-tuned.",
         "20", ("18", "21", "25", "26", "35"), ""),
    Term("batching", ("batched inference", "continuous batching"),
         "Serving several sequences in one forward pass, so each weight read from memory is used many "
         "times. Throughput rises almost for free until compute becomes the limit.",
         "18", ("27",), ""),
    Term("BatchNorm", ("batch normalization", "batch norm"),
         "Normalize each feature across the examples in a batch. Needs running averages for inference, so "
         "train and inference behave differently; the CNN standard.",
         "11", (), "Ioffe & Szegedy, 2015"),
    Term("BCOO", ("sparse matrix", "jax.experimental.sparse"),
         "JAX's sparse-matrix format (batched coordinate); the course computes BM25 for all documents as "
         "one sparse × dense matmul.",
         "38", (), ""),
    Term("Bellman equation", ("Bellman",),
         "A value equals the immediate reward plus the discounted value of where you land: Q(s, a) = r + γ "
         "max Q(s′, a′). Q-learning makes its estimates satisfy it.",
         "34", (), "Bellman, 1957"),
    Term("benchmark", ("benchmarks",),
         "A fixed dataset and scoring rule used to compare models.",
         "21", ("25", "32", "36", "38"), ""),
    Term("BERT", (),
         "A bidirectional encoder-only transformer pretrained by predicting masked tokens, then fine-tuned "
         "for classification or retrieval. Encoders still power embeddings and rerankers.",
         "32", ("00b", "30", "38"), "Devlin et al., 2018"),
    Term("best-of-n", ("best of n", "best_of_n"),
         "Generate n answers and keep the one a verifier or reward model scores highest. Rejection "
         "sampling keeps only the good ones, e.g. as training data.",
         "35", (), ""),
    Term("bfloat16", ("bf16",),
         "16-bit float with float32's 8 exponent bits but only 7 mantissa bits: same range, much less "
         "precision. Halves memory and is much faster on the GPU, which is why training uses it for "
         "compute.",
         "08", ("06", "18", "22", "25", "27"), "Google Brain"),
    Term("bi-encoder", ("bi-encoders", "dual encoder", "two-tower"),
         "A retriever that embeds query and document separately, so document vectors can be precomputed "
         "and searched fast.",
         "38", (), ""),
    Term("bias", ("biases",),
         "The additive constant in a layer, y = Wx + b, which shifts the output independently of the "
         "input.",
         "10", ("05", "11", "16", "17", "21"), ""),
    Term("bigram model", ("bigram", "bigrams"),
         "A language model that predicts the next token from only the current one, i.e. a table of counts "
         "of adjacent pairs.",
         "14", (), ""),
    Term("binary cross-entropy", ("BCE", "log loss"),
         "Cross-entropy for a yes/no prediction: −[y log p + (1 − y) log(1 − p)].",
         "09", (), ""),
    Term("bits per byte", ("bits-per-byte", "BPB"),
         "Total cross-entropy in bits divided by the number of UTF-8 bytes of text. Unlike perplexity, it "
         "compares models with different tokenizers.",
         "21", ("25",), ""),
    Term("BM25", ("Okapi BM25",),
         "The standard keyword-search scoring function: TF-IDF with saturating term frequency and "
         "document-length normalization.",
         "38", ("39",), "Robertson et al., 1994"),
    Term("bootstrap",
         ("bootstrap confidence interval", "bootstrapping", "confidence interval", "confidence intervals"),
         "Estimating uncertainty by recomputing a metric on many resamples (with replacement) of the test "
         "items; the spread gives a confidence interval.",
         "21", ("38",), "Efron, 1979"),
    Term("Bradley–Terry model", ("Bradley-Terry", "Bradley–Terry", "bradley_terry_loss"),
         "The probability that A beats B is σ(score_A − score_B). Training a reward model is maximizing "
         "this likelihood on preference pairs.",
         "35", (), "Bradley & Terry, 1952"),
    Term("broadcasting", ("broadcast",),
         "The rule that lets arrays of different shapes combine elementwise: shapes are right-aligned, and "
         "any axis of size 1 (or missing) is stretched to match. It replaces most Python loops.",
         "01", (), ""),
    Term("bucketing", ("pad_to_bucket", "bucket sizes"),
         "Padding variable-length inputs up to one of a few fixed sizes so jit compiles only a few "
         "programs.",
         "06", (), ""),
    Term("byte-level BPE", ("byte-level",),
         "BPE starting from the 256 byte values, so any text can be encoded with no unknown tokens. Used "
         "by GPT-2 and most modern LLMs.",
         "13", ("40",), "Radford et al., 2019"),
    Term("byte-pair encoding", ("BPE", "byte pair encoding"),
         "A tokenizer trained by repeatedly merging the most frequent adjacent pair of tokens into a new "
         "token, starting from bytes or characters. The merges are applied in order to tokenize.",
         "13", ("27", "30"), "Gage, 1994; Sennrich et al., 2016"),
    Term("calibration", ("calibrate", "calibrated", "uncalibrated"),
         "Correcting a model's built-in lean toward some options regardless of the input, e.g. by "
         "subtracting each option's score on a content-free question, so the choice reflects what this "
         "question adds.",
         "39", ("27",), "Zhao et al., 2021"),
    Term("capstone", (),
         "The course's final project: specialize, evaluate and serve a model of your own, end to end, with "
         "optional cloud scale-up.",
         "25", (), ""),
    Term("carry", ("loop carry",),
         "The state lax.scan passes from one step to the next (an RNN hidden state, a running sum, the "
         "train state).",
         "08", ("05", "09", "10", "14", "24"), ""),
    Term("causal mask", ("causal masking", "causal_mask"),
         "Setting attention scores for future positions to −∞ before softmax, so each token sees only "
         "itself and the past. Required for next-token training.",
         "15", ("32",), ""),
    Term("causal tracing", (),
         "Finding where a model computes something by corrupting the input and restoring one layer's activations "
         "at a time: the layers whose restoration brings the answer back are where it lives.",
         "46", (), "Meng et al., 2022"),
    Term("chain of thought", ("chain-of-thought", "CoT"),
         "Having a model write out intermediate reasoning steps before its final answer, which improves "
         "accuracy on multi-step problems.",
         "26", (), "Wei et al., 2022"),
    Term("chain rule", (),
         "The derivative of a composition is the product of the derivatives of its parts. Backpropagation "
         "is the chain rule applied systematically over a computation graph.",
         "03", (), ""),
    Term("channel", ("channels",),
         "One feature map in a convolutional layer, like the R, G, B of an image or one learned detector's "
         "output.",
         "12", ("01",), ""),
    Term("chat template", ("chat templates", "Jinja template"),
         "The rule (usually a Jinja template shipped with the model) that turns a list of chat messages "
         "into the exact token string, with role markers, that the model was fine-tuned on.",
         "27", (), ""),
    Term("ChatML", ("<|im_start|>", "im_start"),
         "A chat format that wraps each message as <|im_start|>role … <|im_end|>; used by SmolLM2 and "
         "Qwen.",
         "27", ("20", "26"), "OpenAI, 2023"),
    Term("checkpoint", ("checkpoints", "checkpointing", "Orbax"),
         "A saved snapshot of parameters and optimizer state so training can resume or the model be "
         "reused. The course uses Orbax, JAX's checkpointing library.",
         "11", ("10", "16", "26", "39"), ""),
    Term("Chinchilla scaling", ("Chinchilla", "compute-optimal"),
         "The finding that, for a fixed compute budget, parameters and training tokens should grow "
         "together, roughly 20 tokens per parameter. Many earlier large models were under-trained.",
         "16", ("42",), "Hoffmann et al., 2022"),
    Term("chunking", ("chunk size", "chunk_sentences"),
         "Splitting documents into passages before indexing. Chunk size trades context per hit against "
         "retrieval precision.",
         "38", (), ""),
    Term("circuit", ("circuits",),
         "A subnetwork of heads and neurons that together implement a specific behaviour.",
         "37", (), "Olah et al., 2020"),
    Term("classifier-free guidance", (),
         "Training a diffusion model with and without its conditioning, then extrapolating from the "
         "unconditional toward the conditional prediction to follow the prompt more strongly.",
         "33", (), "Ho & Salimans, 2022"),
    Term("CLIP", (),
         "Contrastive Language–Image Pretraining: an image encoder and a text encoder trained so matching "
         "image–caption pairs have high cosine similarity. Enables zero-shot classification and guides "
         "image generation.",
         "33", ("38", "43"), "Radford et al., 2021"),
    Term("clipped objective", ("clipped_pg_loss", "clip ratio"),
         "PPO's trick: cap how far the probability ratio between new and old policy can move in one "
         "update, so a step can't change the policy too much.",
         "26", (), "Schulman et al., 2017"),
    Term("closed book", ("closed-book",),
         "Answering questions from the model's parameters alone, with no retrieved sources.",
         "38", (), ""),
    Term("coding agent", ("coding assistant", "coding assistants"),
         "An LLM in a loop that reads code, proposes edits and runs tools. Module 19 builds a tiny one "
         "against a local server.",
         "19", (), ""),
    Term("ColBERT", ("late interaction",),
         "A retriever keeping one vector per token and scoring by summed max-similarity: between a "
         "bi-encoder and a cross-encoder in cost and accuracy.",
         "38", (), "Khattab & Zaharia, 2020"),
    Term("collective", ("collectives", "collective communication"),
         "A communication operation involving all devices along an axis, such as all-reduce or all-gather.",
         "24", ("25",), ""),
    Term("command buffer", ("command buffers",),
         "Recording a sequence of kernel launches once and replaying it, to cut per-launch CPU overhead "
         "(CUDA/HIP graphs).",
         "22", (), ""),
    Term("communication-to-compute ratio", ("dp_comm_ratio", "comm/compute ratio"),
         "In data parallelism, time spent all-reducing gradients divided by time spent computing. Model "
         "size cancels, so it depends on tokens per device per step and on the hardware's "
         "FLOP/s-to-bandwidth ratio.",
         "25", ("24",), ""),
    Term("compilation cache", ("persistent compilation cache",),
         "Storing compiled XLA programs on disk so a restarted process skips recompiling them.",
         "16", (), ""),
    Term("compile time", ("compilation time",),
         "The one-off cost of tracing and compiling a jitted function, separate from the per-call run "
         "time. Unrolled Python loops make it grow with the loop length.",
         "06", (), ""),
    Term("computation graph", ("computational graph",),
         "A record of a computation as nodes (operations) and edges (values). Autodiff walks it backwards "
         "to compute gradients; JAX's jaxpr is such a graph.",
         "03", (), ""),
    Term("compute unit", ("compute units",),
         "AMD's basic GPU building block (like an NVIDIA SM): it runs workgroups and has its own LDS. The "
         "RX 7600 XT has 32.",
         "23", (), ""),
    Term("compute-bound", ("compute bound",),
         "When the GPU's arithmetic units are the bottleneck: the work does many FLOPs per byte moved. "
         "Large matmuls are compute-bound.",
         "10", ("19",), ""),
    Term("condition number", ("ill-conditioned",),
         "The ratio of the largest to smallest curvature of a loss (or singular values of a matrix). A "
         "large one means a long narrow valley where plain gradient descent zig-zags slowly.",
         "05", (), ""),
    Term("confusion matrix", (),
         "A table counting, for each true class, how often the model predicted each class. Shows which "
         "classes get mixed up.",
         "09", (), ""),
    Term("constrained decoding", ("grammar-constrained", "logit masking"),
         "Masking the logits at each step so only tokens allowed by a grammar or JSON schema can be "
         "generated, guaranteeing well-formed output (e.g. tool calls) without retraining.",
         "39", (), ""),
    Term("container", ("containers/run.sh", "Containerfile"),
         "A packaged Linux userland (here a Podman image) that holds ROCm, JAX and every library, so "
         "nothing GPU-related is installed on the host. `containers/run.sh` starts it with the GPU device "
         "files passed through.",
         "00", (), ""),
    Term("contamination", ("data contamination", "test-set contamination"),
         "Test items leaking into training data, inflating scores. Detected by n-gram overlap and by "
         "comparing exact vs perturbed items.",
         "21", (), ""),
    Term("content-addressable memory", ("associative memory", "content addressable"),
         "Memory retrieved by giving part of its content rather than an address; a Hopfield network is "
         "one.",
         "29", (), ""),
    Term("context window", ("context length", "block size"),
         "How many previous tokens a model can see when predicting the next one.",
         "14", ("13", "27", "36", "39"), ""),
    Term("contextual embedding", ("contextual embeddings", "contextual"),
         "A vector for a word that depends on its sentence, as produced by a transformer, unlike "
         "word2vec's single vector per word.",
         "32", ("30", "38"), ""),
    Term("convolution", ("conv", "convolutions", "convolutional"),
         "Sliding a small kernel of weights across the input and taking a dot product at every position. "
         "The same weights look for the same pattern everywhere.",
         "12", ("00b", "33", "36"), ""),
    Term("convolutional neural network", ("CNN", "CNNs", "convnet"),
         "A network built from convolution layers (plus pooling and a final classifier). The standard "
         "image model before vision transformers.",
         "12", ("33",), "LeCun et al., 1989"),
    Term("corpus", ("corpora",),
         "The collection of documents searched or trained on.",
         "38", ("13", "21", "30", "32"), ""),
    Term("cosine decay", ("cosine schedule", "warmup_cosine_decay"),
         "Lowering the learning rate along a half cosine curve from its peak to near zero by the end of "
         "training.",
         "05", (), "Loshchilov & Hutter, 2016"),
    Term("cosine similarity", (),
         "The dot product of two vectors divided by both their lengths: the cosine of the angle between "
         "them, from -1 to 1. The standard way to compare embeddings.",
         "02", (), ""),
    Term("cross-encoder", ("cross-encoders",),
         "A model that reads query and document together and outputs a relevance score. More accurate than "
         "a bi-encoder but too slow for the whole corpus, so used to rerank.",
         "38", (), ""),
    Term("cross-entropy", ("cross entropy", "cross-entropy loss"),
         "The average surprise when outcomes come from the true distribution but you predicted with "
         "another: −Σ p log q. With one-hot targets it is −log of the probability given to the right "
         "answer, the standard classification and language-model loss.",
         "04", ("08", "09", "11", "14", "16"), "Shannon, 1948"),
    Term("CUDA", (),
         "NVIDIA's GPU programming platform (2007), which made GPUs general-purpose compute devices. "
         "ROCm/HIP is AMD's counterpart.",
         "00b", ("23", "25"), ""),
    Term("DALL·E", ("DALL-E",),
         "OpenAI's text-to-image models (2021–); with CLIP they launched the text-to-image era.",
         "33", (), "Ramesh et al., 2021"),
    Term("Dartmouth workshop", ("Dartmouth",),
         "The 1956 summer workshop, proposed by McCarthy, Minsky, Rochester and Shannon, that named the "
         'field "artificial intelligence".',
         "00b", (), "McCarthy et al., 1955 proposal"),
    Term("data parallelism", ("data parallel",),
         "Each device holds a full copy of the model and processes different examples; gradients are "
         "all-reduced.",
         "24", ("25",), ""),
    Term("DCGAN", (),
         "A convolutional GAN architecture with design rules that made GAN training stable enough for "
         "images.",
         "33", (), "Radford, Metz, Chintala, 2015"),
    Term("DDIM", (),
         "A deterministic diffusion sampler that skips steps, generating in tens instead of hundreds of "
         "steps from the same trained model.",
         "33", (), "Song, Meng, Ermon, 2020"),
    Term("DDPM", ("denoising diffusion",),
         "Denoising Diffusion Probabilistic Models: the 2020 formulation that trains a network to predict "
         "the added noise with a simple squared-error loss.",
         "33", (), "Ho, Jain, Abbeel, 2020"),
    Term("decode", ("decode phase", "generation phase"),
         "The second phase of generation: produce one token per step, each reading all the weights and the "
         "KV cache. Memory-bandwidth-bound.",
         "18", ("17", "19", "41"), ""),
    Term("decoder-only", (),
         "A transformer with only causally masked self-attention, trained to predict the next token. The "
         "architecture of GPT and nearly every modern LLM.",
         "16", (), ""),
    Term("deep learning", (),
         "Machine learning with neural networks of many layers, trained end to end by gradient descent on "
         "large data. It took off in 2012, when AlexNet won ImageNet on GPUs.",
         "00b", ("06", "08", "12", "22"), ""),
    Term("DeepSeek Sparse Attention", ("DSA", "lightning indexer", "indexer_scores"),
         "DeepSeek V3.2's attention: a small ReLU-scored indexer ranks every past token and each query reads "
         "only the top k (2,048); the indexer is trained to match the main attention's weights.",
         "44", (), "DeepSeek-AI, 2025"),
    Term("DeepSeek-V3", ("DeepSeek V3", "deepseek-v3-jax"),
         "DeepSeek's 2024 MoE LLM combining MLA, DeepSeekMoE, aux-loss-free balancing and MTP. The owner's "
         "deepseek-v3-jax replication is a recurring reference in this course.",
         "17", ("05", "08", "11", "13", "16"), "DeepSeek-AI, 2024"),
    Term("dense retrieval", ("dense retriever", "embedding search", "semantic search"),
         "Search by embedding query and documents as dense vectors and finding the nearest by cosine "
         "similarity, so matches need not share words.",
         "38", (), "Karpukhin et al., 2020 (DPR)"),
    Term("depthwise convolution", ("causal depthwise convolution", "depthwise"),
         "A convolution applied to each channel separately; Mamba blocks use a short causal one before the "
         "SSM.",
         "36", ("40",), ""),
    Term("derivative", ("derivatives",),
         "The slope of a function at a point: how much the output changes per unit change in the input.",
         "03", ("10",), ""),
    Term("determinant", (),
         "How much a matrix scales area (or volume); zero means it squashes space flat and cannot be "
         "undone, negative means it flips orientation.",
         "02", (), ""),
    Term("device", ("devices", "device_put"),
         "A piece of hardware JAX can place arrays on and run computation on, such as `rocm:0` or the CPU. "
         "`jax.device_put` moves data there.",
         "01", ("00", "08", "22", "24", "25"), ""),
    Term("diffusion model", ("diffusion", "diffusion models"),
         "Learn to reverse a process that gradually adds noise to data: a network predicts the noise at "
         "each step, and generation starts from pure noise and denoises step by step.",
         "33", ("00b", "45"), "Sohl-Dickstein et al., 2015; Ho et al., 2020"),
    Term("dilated convolution", ("dilated causal convolution",),
         "A convolution whose kernel taps are spaced apart, so the receptive field doubles per layer with "
         "dilations 1-2-4-8. WaveNet stacks them causally.",
         "12", ("40",), "van den Oord et al., 2016 (WaveNet)"),
    Term("discounted return", ("discount factor", "discounted sum"),
         "The total (usually discounted) reward from a point onward; what an RL agent maximizes.",
         "34", (), ""),
    Term("discretization", ("zero-order hold", "ZOH"),
         "Converting a continuous-time state-space model into a step-by-step update with a step size Δ; "
         "zero-order hold assumes the input is constant within each step.",
         "36", (), ""),
    Term("distillation", ("knowledge distillation", "distill", "distilled"),
         "Training a smaller student model on a larger teacher's outputs (or reasoning traces) so it "
         "inherits the teacher's ability.",
         "26", ("12", "31", "42"), "Hinton, Vinyals, Dean, 2015"),
    Term("distributional hypothesis", (),
         '"You shall know a word by the company it keeps": words in similar contexts have similar '
         "meanings. The idea behind all learned embeddings.",
         "30", (), "Harris, 1954; Firth, 1957"),
    Term("divergence", (),
         "Training going unstable: the loss shoots up or becomes NaN, usually from too large a learning "
         "rate.",
         "05", ("04",), ""),
    Term("domain shift", ("distribution shift",),
         "When test data differs in kind from training data, so in-distribution scores overstate real "
         "performance.",
         "21", (), ""),
    Term("donation", ("donate_argnums", "donate", "donated", "buffer donation"),
         "Telling jit that an input buffer won't be used again, so XLA can write the output into it. "
         "Halves peak memory for parameters and optimizer state in a train step.",
         "16", ("22", "25"), ""),
    Term("dot product", (),
         "Multiply two vectors elementwise and sum. It measures alignment: large and positive when they "
         "point the same way, zero when perpendicular. Attention scores are dot products.",
         "02", ("08", "12", "17"), ""),
    Term("Double DQN", (),
         "Pick the next action with the online network but evaluate it with the target network, reducing "
         "Q-learning's overestimation.",
         "34", (), "van Hasselt et al., 2015"),
    Term("DPO", ("Direct Preference Optimization", "dpo_loss"),
         "Preference learning without a reward model or RL: a classification-style loss raises the "
         "policy's log-probability ratio (vs a reference model) for chosen responses over rejected ones.",
         "35", (), "Rafailov et al., 2023"),
    Term("DPR", ("Dense Passage Retrieval",),
         "Dense Passage Retrieval: a BERT bi-encoder trained with in-batch and BM25 hard negatives that "
         "beat BM25 for open-domain question answering.",
         "38", (), "Karpukhin et al., 2020"),
    Term("DQN", ("deep Q-network", "Deep Q-Network"),
         "Q-learning with a neural network for Q-values, made stable by experience replay and a target "
         "network; learned Atari games from pixels.",
         "34", ("00b",), "Mnih et al., 2013, 2015"),
    Term("dropout", (),
         "During training, randomly zero a fraction of activations and scale up the rest, so the network "
         "can't rely on any one unit. Turned off at inference.",
         "11", (), "Srivastava et al., 2014"),
    Term("dtype", ("dtypes", "data type"),
         "The number format of an array's elements, such as float32, bfloat16 or int32. It sets precision, "
         "range and memory per element.",
         "01", ("06", "08", "10", "23", "32"), ""),
    Term("eigenvector", ("eigenvectors", "eigenvalue", "eigenvalues"),
         "A direction a matrix only stretches, never turns; the stretch factor is its eigenvalue. They "
         "explain repeated application, e.g. why some directions blow up or vanish in an RNN.",
         "02", (), ""),
    Term("einsum", ("jnp.einsum",),
         "A one-line notation for products and sums over named axes, e.g. `einsum('bij,bjk->bik', a, b)` "
         "for a batched matmul. Indices that appear in the inputs but not the output are summed.",
         "01", ("15", "17", "31", "36"), ""),
    Term("ELBO", ("evidence lower bound",),
         "A lower bound on the log-likelihood of the data: reconstruction quality minus the KL divergence "
         "of the encoder's latent distribution from the prior. VAEs maximize it.",
         "33", (), "Kingma & Welling, 2013"),
    Term("Elman network", ("Elman cell", "Elman"),
         "The simple RNN: h_t = tanh(W h_{t-1} + U x_t + b).",
         "31", (), "Elman, 1990"),
    Term("embedding", ("embeddings", "embedding table", "embedding layer"),
         "A learned vector for each token id, looked up from a table (V × d). Similar tokens end up with "
         "nearby vectors.",
         "14", ("15", "16", "27", "30", "38"), "Bengio et al., 2003"),
    Term("embedding model", ("embedding models", "Qwen3-Embedding", "text embeddings"),
         "A model whose output is a single vector per text, trained so related texts are close; the "
         "encoder side of dense retrieval.",
         "38", ("27",), ""),
    Term("encoder-decoder transformer", ("encoder-decoders", "T5"),
         "A transformer with an encoder reading the input and a decoder generating with cross-attention to "
         "it, as in the original Transformer and T5.",
         "32", (), "Vaswani et al., 2017; Raffel et al., 2019"),
    Term("encoder-only", ("encoders",),
         "A transformer without a causal mask, so every token sees the whole input; produces "
         "representations rather than generating text.",
         "32", (), ""),
    Term("energy function", (),
         "A scalar a Hopfield network's updates never increase, so the network rolls downhill into stored "
         "patterns (local minima).",
         "29", (), "Hopfield, 1982"),
    Term("entropy", (),
         "The average surprise of a distribution, −Σ p log p: how uncertain it is. Measured in bits (log "
         "base 2) or nats (natural log).",
         "04", ("33", "37"), "Shannon, 1948"),
    Term("epoch", ("epochs",),
         "One full pass over the training set.",
         "10", ("09", "11", "12", "29", "33"), ""),
    Term("evaluator-optimizer", ("evaluator–optimizer", "Reflexion"),
         "A workflow in which one call proposes an answer and another critiques it, repeating until the "
         "critic is satisfied.",
         "39", (), "Anthropic, 2024; Shinn et al., 2023"),
    Term("exact match", (),
         "Scoring a generated answer as right only if it equals the reference after normalization.",
         "21", ("20",), ""),
    Term("expectation", ("expected value",),
         "The probability-weighted average of a random variable: its long-run mean.",
         "04", (), ""),
    Term("experience replay", ("replay buffer",),
         "Storing past transitions and training on random samples of them, which breaks correlations "
         "between consecutive steps and reuses data.",
         "34", (), "Lin, 1992; Mnih et al., 2015"),
    Term("expert parallelism", ("expert parallel",),
         "Placing different MoE experts on different devices and using all-to-all to send each token to "
         "its experts' devices.",
         "24", ("40",), ""),
    Term("expert system", ("expert systems",),
         "AI built from hand-written if-then rules elicited from human experts. Commercially popular in "
         "the 1980s, they were brittle outside their rules, which helped bring on the second AI winter.",
         "00b", (), ""),
    Term("explicit sharding", ("sharding in types", "Auto axes"),
         "JAX mode in which an array's sharding is part of its type and visible at trace time, so "
         "ambiguous cases are errors rather than compiler guesses.",
         "24", (), ""),
    Term("exponent bits", (),
         "The bits of a float that set its scale. bfloat16 keeps float32's 8, so it has the same range.",
         "08", (), ""),
    Term("exponential moving average", ("EMA", "running average", "running averages"),
         "A running average where each new value gets weight (1 − β) and the old average β, so recent "
         "values count most. Used by momentum, Adam, BatchNorm and target networks.",
         "08", ("05",), ""),
    Term("family tree", ("lineage", "LINEAGE.md"),
         "The course's registry of about fifty models, the models each one builds on, and the building "
         "blocks each uses, filed by slot (position, normalization, attention, …).",
         "00b", ("16", "17"), ""),
    Term("fan-in", ("fan_in", "fan-out", "fan_out"),
         "The number of inputs (fan-in) or outputs (fan-out) of a layer, used to scale initial weights and "
         "some optimizer updates.",
         "11", ("05",), ""),
    Term("Fashion-MNIST", ("MNIST",),
         "70,000 28×28 grayscale images of clothing in 10 classes: the course's image benchmark for "
         "regression, MLPs and CNNs.",
         "09", ("07", "10", "11", "12", "33"), "Xiao, Rasul, Vollgraf, 2017"),
    Term("feature", ("features",),
         "One input variable to a model. In interpretability, a direction in activation space that "
         "represents a concept.",
         "09", ("11", "12", "15", "17", "37"), ""),
    Term("FFT convolution", ("FFT", "fast Fourier transform"),
         "Computing a long convolution in O(L log L) via the fast Fourier transform instead of O(L²).",
         "36", (), ""),
    Term("filter", ("filters", "convolution kernel"),
         "The small weight array a convolution slides over the input; each learns to detect one pattern, "
         "producing one output channel.",
         "12", (), ""),
    Term("fine-tuning", ("fine-tune", "fine-tuned", "finetuning", "full fine-tuning"),
         "Continuing to train a pretrained model on a smaller, task-specific dataset.",
         "20", ("26", "32", "35", "38"), ""),
    Term("finite differences", ("finite difference",),
         "Estimating a derivative by nudging an input and measuring the change in output. Simple but needs "
         "one evaluation per input and suffers truncation and round-off error, so it is used only to check "
         "gradients.",
         "03", (), ""),
    Term("FlashAttention", ("flash attention",),
         "Exact attention computed tile by tile with an online softmax, never writing the L × L score "
         "matrix to memory. Fast because it cuts memory traffic, not FLOPs.",
         "23", ("15", "19"), "Dao et al., 2022"),
    Term("Flax NNX", ("Flax", "nnx", "nnx.Module"),
         "JAX's neural-network library in which layers are Python objects holding their parameters, yet "
         "still convertible to pytrees so jit and grad apply.",
         "10", ("11", "12"), ""),
    Term("float16", ("fp16", "f16", "half precision"),
         "16-bit IEEE float with 5 exponent bits: more precision than bfloat16 but a tiny range, so "
         "training in it needs loss scaling. Common for storing inference weights.",
         "08", ("19", "23", "27"), ""),
    Term("float32", ("fp32", "f32", "single precision"),
         "32-bit float: about 7 significant digits. The safe default, and the format for master weights "
         "and accumulations.",
         "08", ("00", "01", "06", "23", "27"), ""),
    Term("floating point", (),
         "Numbers stored as sign × mantissa × 2^exponent. Exponent bits set the range, mantissa bits the "
         "precision.",
         "08", (), ""),
    Term("forward mode", ("forward-mode",),
         "Autodiff that pushes derivatives forward alongside the values, one input direction per pass. "
         "Cheap when there are few inputs and many outputs.",
         "03", (), ""),
    Term("forward process", ("noise schedule", "forward diffusion"),
         "In diffusion, the fixed procedure that adds a scheduled amount of Gaussian noise to the data at "
         "each step until only noise remains.",
         "33", (), ""),
    Term("freezing", ("freeze_mask",),
         "Keeping some parameters fixed during training, either by differentiating only the trainable part "
         "or masking the optimizer.",
         "20", (), ""),
    Term("FSDP", ("fully sharded data parallel",),
         "Data parallelism with parameters and optimizer state sharded across devices and all-gathered "
         "just before use, so memory per device shrinks.",
         "24", ("25",), "Rajbhandari et al., 2019 (ZeRO)"),
    Term("functional state", ("explicit state",),
         "Keeping parameters, optimizer state and RNG keys as values passed into and returned from pure "
         "functions, instead of mutating objects. JAX's way of training.",
         "07", (), ""),
    Term("GAN", ("GANs", "generative adversarial network", "generator", "discriminator"),
         "A generator makes fakes from noise while a discriminator learns to tell them from real data; "
         "each trains against the other.",
         "33", ("00b",), "Goodfellow et al., 2014"),
    Term("gates", ("forget gate", "input gate", "output gate", "update gate"),
         "A sigmoid-valued vector that multiplies another, controlling how much passes through: what to "
         "keep, forget or emit.",
         "31", ("36",), ""),
    Term("Gaussian", ("normal distribution", "Gaussians"),
         "The bell-curve distribution described by a mean and variance. Used for weight initialization, "
         "VAE latents and diffusion noise.",
         "33", (), ""),
    Term("GELU", (),
         "A smooth version of ReLU, x·Φ(x), used in GPT-2 and BERT.",
         "10", ("22", "23"), "Hendrycks & Gimpel, 2016"),
    Term("GEMM", ("GEMMs",),
         "GEneral Matrix Multiply: the C = A·B (+ C) operation that dominates deep-learning compute. "
         "Kernels and libraries are judged mostly by how fast their GEMMs run.",
         "22", (), ""),
    Term("generalization", ("generalize", "generalizes"),
         "How well a model performs on data it was not trained on. The real goal of training.",
         "10", ("11",), ""),
    Term("generative model", ("generative models",),
         "A model that learns a data distribution so it can produce new samples resembling the training "
         "data.",
         "33", (), ""),
    Term("generative perplexity", ("gen_nll", "generative NLL"),
         "A score for generated text: a separate autoregressive model's loss on it. Easy to game with dull, "
         "repetitive text, so it is read alongside diversity measures such as entropy and repeated n-grams.",
         "45", (), "Lou et al., 2024; Sahoo et al., 2024"),
    Term("gfx1102", ("gfx1100",),
         "The instruction-set name of the RX 7600 XT (Navi 33, RDNA3). JAX's ROCm plugin only accepts a "
         "fixed list of architectures, so the card is made to report gfx1100 (the RX 7900's) instead.",
         "00", ("22", "23"), ""),
    Term("GGUF", (),
         "The single-file model format of llama.cpp: weights (often quantized), tokenizer, chat template "
         "and metadata together.",
         "19", ("27",), ""),
    Term("glitch tokens", ("glitch token", "SolidGoldMagikarp"),
         "Tokens that exist in the vocabulary but were almost never seen in training, so their embeddings "
         "are untrained and the model behaves strangely on them.",
         "13", (), ""),
    Term("Goodhart's law", ("Goodhart",),
         '"When a measure becomes a target, it ceases to be a good measure." Optimizing a proxy reward too '
         "hard makes the true quality fall while the proxy keeps climbing.",
         "35", (), "Goodhart, 1975"),
    Term("GPT", ("GPT-2", "GPT-3", "generative pre-trained transformer"),
         "A decoder-only transformer trained on next-token prediction. GPT-2 (2019) is the template the "
         "course's model follows.",
         "16", ("13", "21", "25", "27", "32"), "Radford et al., 2018, 2019"),
    Term("gradient", ("gradients",),
         "The vector of all partial derivatives of a scalar loss with respect to the parameters. It points "
         "in the direction of steepest increase, so training steps the other way.",
         "03", ("05", "07", "11", "20", "31"), ""),
    Term("gradient accumulation", (),
         "Summing gradients over several micro-batches before one optimizer step: a larger effective batch "
         "without more memory, and fewer all-reduces per token.",
         "25", (), ""),
    Term("gradient clipping",
         ("clipping by global norm", "clip_by_global_norm", "global gradient norm", "gradient norm"),
         "If the gradient's overall norm exceeds a threshold, scale it down to the threshold. Prevents one "
         "bad batch from wrecking the weights.",
         "05", ("02", "07", "11", "16"), ""),
    Term("gradient descent", ("steepest descent",),
         "Repeatedly step the parameters a small amount against the gradient: θ ← θ − η·∇L.",
         "05", ("09", "14"), "Cauchy, 1847"),
    Term("greedy decoding", ("greedy", "argmax decoding"),
         "Always picking the most probable next token. Deterministic and often repetitive.",
         "18", ("20", "26", "27", "35"), ""),
    Term("group-wise quantization",
         ("per-channel", "per output channel", "group size", "grouped quantization"),
         "Quantizing with one scale per row/channel or per small group of weights, so one outlier can't "
         "wreck the precision of everything else.",
         "18", (), ""),
    Term("grouped-query attention", ("GQA", "grouped query attention"),
         "Attention in which several query heads share each key/value head, shrinking the KV cache by the "
         "group factor with little quality loss.",
         "17", ("18", "19"), "Ainslie et al., 2023"),
    Term("GRPO", ("Group Relative Policy Optimization",),
         "Policy-gradient RL for LLMs: sample a group of answers per prompt, score them, use each answer's "
         "reward minus the group mean (over the std) as its advantage, with PPO-style clipping and a KL "
         "penalty. No value network needed.",
         "26", ("27", "34", "35"), "Shao et al., 2024 (DeepSeekMath)"),
    Term("GRU", ("gated recurrent unit",),
         "A simpler gated RNN cell than the LSTM, with reset and update gates and no separate cell state.",
         "31", (), "Cho et al., 2014"),
    Term("guardrails", ("guardrail", "tool allowlist", "approval gate", "turn limit"),
         "Checks around an agent: input rules, tool allowlists, turn limits and human approval for "
         "high-risk actions.",
         "39", (), ""),
    Term("Gumbel-max trick", ("Gumbel", "Gumbel noise"),
         "Sample from a softmax by adding Gumbel noise to the logits and taking the argmax. Makes sampling "
         "a deterministic function of the key, and reappears in Gumbel MuZero.",
         "04", (), "Gumbel, 1954"),
    Term("handoff", ("handoffs", "hand off"),
         "A one-way transfer of a conversation from one agent to another, which then owns the reply.",
         "39", (), "OpenAI Agents SDK"),
    Term("hard negative", ("hard negatives",),
         "A wrong document that looks right (e.g. one BM25 ranks highly); training against them makes a "
         "retriever sharper.",
         "38", (), "Karpukhin et al., 2020"),
    Term("He initialization", ("He init", "Kaiming initialization", "he_init"),
         "Draw initial weights with variance 2/fan_in so activations keep their scale through ReLU layers.",
         "10", ("11",), "He et al., 2015"),
    Term("Hebbian learning", ("Hebbian", "Hebb's rule", "Hebb"),
         '"Cells that fire together wire together": strengthen a weight when both units are active. The '
         "Hopfield network stores patterns this way.",
         "29", (), "Hebb, 1949"),
    Term("Hessian", (),
         "The matrix of second derivatives of a scalar function: how the gradient itself changes. It "
         "describes curvature.",
         "03", (), ""),
    Term("hidden layer", ("hidden layers", "hidden units"),
         "A layer between input and output whose values (activations) are learned features rather than "
         "given data or predictions.",
         "10", ("29", "30"), ""),
    Term("hidden state", (),
         "The vector an RNN or state-space model carries from step to step, summarizing everything read so "
         "far.",
         "31", (), ""),
    Term("HIP", ("HIP runtime", "HIP kernel"),
         "AMD's C++ GPU programming interface and runtime, deliberately close to CUDA so code ports almost "
         "line for line. A hand-written HIP kernel is the low-level alternative to letting XLA or Pallas "
         "generate one.",
         "23", ("00b", "19", "22"), ""),
    Term("hipBLASLt", ("rocBLAS", "Tensile"),
         "AMD's tuned matrix-multiply libraries. With Triton GEMM disabled, XLA hands every large matmul "
         "(a GEMM) to hipBLASLt or rocBLAS, which is both safe and fast on this card.",
         "00", ("22", "23"), ""),
    Term("HiPPO", (),
         "A principled initialization of an SSM's state matrix so its state optimally summarizes the input "
         "history.",
         "36", (), "Gu et al., 2020"),
    Term("HLO", ("StableHLO", "compiled HLO"),
         "XLA's High-Level Operations IR: the program JAX hands the compiler (as StableHLO), and the "
         "optimized, fused program XLA produces. Reading it shows what actually runs.",
         "06", ("22", "23", "24"), ""),
    Term("HNSW", ("hierarchical navigable small world",),
         "A graph-based approximate nearest-neighbour index used by most vector databases.",
         "38", (), "Malkov & Yashunin, 2016"),
    Term("Hopfield capacity", ("patterns per neuron",),
         "A classic Hopfield network reliably stores only about 0.14 patterns per neuron before recall "
         "breaks down.",
         "29", (), "Amit, Gutfreund, Sompolinsky, 1985"),
    Term("Hopfield network", ("Hopfield networks", "Hopfield"),
         "A recurrent network of binary units with symmetric weights that stores patterns as minima of an "
         "energy; started from a corrupted pattern, it settles into the stored one.",
         "29", ("00b",), "Hopfield, 1982"),
    Term("HSA_OVERRIDE_GFX_VERSION", (),
         "An environment variable that makes the ROCm runtime report a different GPU architecture. Set to "
         "11.0.0 here so the RX 7600 XT (gfx1102) is accepted as gfx1100; the instruction sets are close "
         "enough that the code runs.",
         "00", (), ""),
    Term("Huber loss", ("Huber",),
         "Squared error for small errors, absolute error for large ones, so outliers don't produce huge "
         "gradients.",
         "34", (), "Huber, 1964"),
    Term("Hugging Face", (),
         "The main hub for open models and datasets, and the `transformers` library that loads them.",
         "27", ("13", "18", "26", "35"), ""),
    Term("hybrid model", ("hybrids", "Jamba", "hybrid models"),
         "An architecture interleaving attention layers with SSM (Mamba) layers to get most of attention's "
         "recall at a fraction of its cost; Jamba is one.",
         "36", ("40", "41"), "Lieber et al., 2024 (Jamba)"),
    Term("hybrid search", ("hybrid retrieval",),
         "Combining keyword (BM25) and dense retrieval results, which usually beats either alone.",
         "38", (), ""),
    Term("hyperparameter", ("hyperparameters",),
         "A setting chosen by you rather than learned: learning rate, batch size, depth, weight decay.",
         "09", (), ""),
    Term("im2col", (),
         "Unfolding every input patch into a column so a convolution becomes one big matrix multiply, "
         "which GPUs do fast.",
         "12", (), ""),
    Term("ImageNet", (),
         "A dataset of over a million labelled photos in 1,000 classes whose yearly competition became "
         "deep learning's proving ground.",
         "00b", (), "Deng et al., 2009"),
    Term("immutability", ("immutable", ".at[]"),
         "JAX arrays cannot be changed in place. To update an element you write `x.at[i].set(v)`, which "
         "returns a new array (and XLA usually makes it an in-place update under jit).",
         "01", (), ""),
    Term("implicit reward", ("implicit_reward",),
         "In DPO, β × the log-ratio of policy to reference probability for a response; the reward the "
         "policy implicitly optimizes.",
         "35", (), "Rafailov et al., 2023"),
    Term("in-context learning", ("ICL", "few-shot", "few-shot prompting", "zero-shot"),
         "A model doing a new task from examples given in the prompt, with no weight updates.",
         "20", ("33", "37"), "Brown et al., 2020 (GPT-3)"),
    Term("in_axes", ("out_axes",),
         "Arguments to vmap saying which axis of each input (or None for no axis) is the batch axis to map "
         "over.",
         "07", (), ""),
    Term("induction head", ("induction heads", "induction circuit", "induction_score"),
         "A two-layer attention circuit that completes repeated patterns ([A][B] … [A] → [B]): a "
         "previous-token head plus a head that looks for the token after the last copy. The mechanism "
         "behind much in-context learning.",
         "37", (), "Elhage et al., 2021; Olsson et al., 2022"),
    Term("inductive bias", ("inductive biases", "encodes assumptions"),
         "The assumptions an architecture builds in, such as locality for CNNs or order for RNNs. Good "
         "biases trade flexibility for data efficiency.",
         "12", (), ""),
    Term("inference-time scaling", ("test-time compute", "test-time scaling", "inference scaling"),
         "Getting better answers by spending more compute at generation time: more samples, longer "
         "reasoning, voting or search.",
         "26", (), "Snell et al., 2024"),
    Term("InfoNCE", ("in-batch negatives", "in-batch negative"),
         "A contrastive loss: treat each matching pair in a batch as the right answer of a softmax "
         "classification against all the other items in the batch (the in-batch negatives).",
         "33", ("38",), "van den Oord et al., 2018"),
    Term("information retrieval", ("retriever", "retrievers"),
         "Finding the documents most relevant to a query from a large collection.",
         "38", (), ""),
    Term("initialization", ("weight initialization",),
         "Choosing a network's starting weights. Scale them so signals neither explode nor vanish layer to "
         "layer.",
         "11", ("05", "07", "10", "16", "20"), ""),
    Term("instruction tuning", ("instruct model", "instruction-tuned", "Instruct"),
         'Fine-tuning a base model on instruction–response examples so it follows requests; "-Instruct" '
         "models have had it.",
         "20", ("21",), "Wei et al., 2021; Ouyang et al., 2022"),
    Term("interpret mode", ("interpret=True",),
         "Running a Pallas kernel as ordinary JAX on the CPU, for debugging before it touches the GPU.",
         "23", (), ""),
    Term("intrinsic dimension", ("intrinsic dimensionality",),
         "The finding that fine-tuning can succeed within a very low-dimensional subspace of the weights, "
         "which is why LoRA works.",
         "20", (), "Aghajanyan et al., 2020"),
    Term("IVF", ("inverted file", "nprobe"),
         "Inverted-file vector index: cluster the vectors (k-means) and search only the few clusters "
         "nearest the query.",
         "38", (), "Jégou, Douze, Schmid, 2011"),
    Term("Jacobian", ("jacfwd", "jacrev"),
         "The matrix of all partial derivatives of a vector-valued function: one row per output, one "
         "column per input.",
         "03", (), ""),
    Term("JAX", (),
         "A Python library with a NumPy-like API whose functions can be transformed: differentiated "
         "(`grad`), compiled (`jit`), vectorized (`vmap`) and parallelized. It works by tracing Python "
         "into a graph that XLA compiles.",
         "00", ("03", "06", "07", "23", "24"), "Frostig, Johnson, Leary, 2018"),
    Term("jax.grad", ("value_and_grad",),
         "JAX's transformation that turns a scalar-valued function into one that returns its gradient "
         "(reverse mode). `value_and_grad` returns the loss too.",
         "03", ("07", "08", "10", "11", "20"), ""),
    Term("jaxpr", ("jaxprs", "make_jaxpr"),
         "JAX's intermediate representation: the traced recording of a function as a list of primitive "
         "operations. `jax.make_jaxpr` prints it.",
         "06", ("07",), ""),
    Term("jit", ("jax.jit", "jitted", "just-in-time"),
         "Just-in-time compilation: the first call traces the function into a graph, XLA compiles it, and "
         "later calls with the same shapes and dtypes reuse the compiled program.",
         "06", ("07", "08", "10", "16", "23"), ""),
    Term("JVP", ("Jacobian-vector product",),
         "Jacobian-vector product: the change in a function's outputs for a given input direction, "
         "computed by forward mode without building the Jacobian.",
         "03", (), ""),
    Term("k-means", ("kmeans", "k-means clustering"),
         "Cluster points by alternating: assign each to its nearest centre, move each centre to its "
         "points' mean.",
         "38", (), "Lloyd, 1957/1982"),
    Term("K-quants", ("Q4_K_M", "Q8_0", "Q5_K_M", "Q6_K"),
         "llama.cpp's quantization types: Q8_0 is 8-bit (~8.5 bits/weight, nearly lossless), Q4_K_M a "
         "mixed ~4.8 bits/weight that costs a little perplexity.",
         "19", ("18", "27"), ""),
    Term("kernel", ("kernels", "GPU kernel"),
         "One function launched on the GPU over a grid of threads. Every jitted program runs as a series "
         "of kernels.",
         "23", ("00", "06", "12", "22", "36"), ""),
    Term("kernel fusion", ("fusion", "fusions", "fused"),
         "XLA combining several elementwise operations into one GPU kernel so intermediates stay in "
         "registers instead of round-tripping through memory. Often the biggest speed-up jit gives.",
         "06", ("15", "22", "23", "27", "38"), ""),
    Term("KL divergence", ("Kullback-Leibler", "Kullback–Leibler", "KL"),
         "How many extra bits you pay by using distribution q when the truth is p: cross-entropy minus "
         "entropy. Always ≥ 0 and not symmetric; used as a leash in RLHF and in the VAE loss.",
         "04", ("26", "33", "35"), "Kullback & Leibler, 1951"),
    Term("KL penalty", ("kl_penalty", "KL leash", "reference model"),
         "A term that penalizes the trained policy's divergence from a frozen reference model, keeping RL "
         "fine-tuning from drifting into gibberish or exploits.",
         "26", ("35",), ""),
    Term("KV cache", ("KV-cache", "k/v cache", "key-value cache", "kv_cache", "KV caches"),
         "During generation, store each layer's keys and values for past tokens so each new token computes "
         "only its own, instead of re-running the whole prefix. Its size often limits context length and "
         "batch.",
         "17", ("18", "19", "25", "36", "39", "41"), ""),
    Term("L2 regularization", ("L2 penalty", "ridge penalty"),
         "Adding λ·‖w‖² to the loss to penalize large weights. Equivalent to weight decay for plain SGD "
         "but not for Adam.",
         "05", (), ""),
    Term("LangGraph", ("LangChain",),
         "A framework for building agents and workflows as graphs of LLM calls and tools.",
         "39", (), ""),
    Term("language model", ("language models", "language modeling"),
         "A model that assigns a probability to the next token given the previous ones; chained, that "
         "gives a probability for any text and a way to generate it.",
         "14", ("04", "15", "31", "34", "37"), ""),
    Term("language-model head", ("LM head", "output head", "unembedding"),
         "The final linear layer that maps each token's hidden vector to logits over the vocabulary.",
         "16", (), ""),
    Term("latency", (),
         "Time from request to response (or to the first token). Batching improves throughput at some cost "
         "in latency.",
         "18", (), ""),
    Term("latent diffusion", ("Stable Diffusion",),
         "Diffusion run in an autoencoder's compact latent space instead of on pixels, which made "
         "high-resolution text-to-image generation cheap (Stable Diffusion).",
         "33", (), "Rombach et al., 2021"),
    Term("latent space", ("latent variable",),
         "A lower-dimensional space of hidden codes from which a generator or decoder produces data.",
         "33", (), ""),
    Term("lax.cond", ("lax.switch", "jnp.where"),
         "Traced control flow: `lax.cond` picks one of two branches based on a traced boolean and compiles "
         "both; `lax.switch` picks from many by index; `jnp.where` selects elementwise.",
         "08", ("06", "17", "29"), ""),
    Term("lax.fori_loop", ("fori_loop", "while_loop", "lax.while_loop"),
         "A compiled loop whose body is traced once, so compile time doesn't grow with the iteration "
         "count. `while_loop` stops on a condition but can't be reverse-differentiated.",
         "08", ("18", "23"), ""),
    Term("lax.scan", ("jax.lax.scan",),
         "A compiled loop that threads a carry through steps and stacks each step's output. It is how JAX "
         "writes RNNs, training loops and decoding without unrolling.",
         "08", ("05", "10", "14", "16", "36"), ""),
    Term("LayerNorm", ("layer normalization", "layer norm"),
         "Normalize each example's feature vector to mean 0 and variance 1, then apply a learned scale and "
         "shift. The transformer's standard normalization.",
         "11", ("16", "17"), "Ba, Kiros, Hinton, 2016"),
    Term("LDS", ("local data share", "shared memory"),
         "Local data share: fast on-chip memory shared by a workgroup (CUDA's shared memory), 64 KiB per "
         "workgroup here. Tiled kernels stage data in it.",
         "23", ("22",), ""),
    Term("leaf", ("tree leaves",),
         "An array (or other non-container value) at the bottom of a pytree.",
         "07", ("08",), ""),
    Term("learning rate", ("step size",),
         "The multiplier η on each gradient step. Too small is slow, too large oscillates or diverges; it "
         "is the most important hyperparameter.",
         "05", ("09", "11", "16", "20", "26"), ""),
    Term("learning-rate schedule", ("LR schedule", "learning rate schedule"),
         "A rule that changes the learning rate over training, typically a warmup followed by a decay.",
         "05", (), ""),
    Term("length extrapolation", ("extrapolate", "extrapolation"),
         "Running a model on sequences longer than any it trained on. Position schemes differ in how "
         "gracefully they degrade.",
         "17", (), ""),
    Term("length normalization", ("length-normalized", "per-token average"),
         "Dividing an answer's log-likelihood by its length (in tokens or bytes) so longer options aren't "
         "unfairly penalized.",
         "21", (), ""),
    Term("linear probe", ("linear probes", "probing"),
         "A linear classifier trained on a model's frozen activations to test whether some property (like "
         "a board square's state) is linearly readable from them.",
         "37", (), "Alain & Bengio, 2016"),
    Term("linear regression", (),
         "Fitting a straight line (or hyperplane) y ≈ Xw + b by minimizing squared error. Solvable in "
         "closed form or by gradient descent.",
         "09", (), ""),
    Term("linear time-invariant", ("LTI", "time-invariant"),
         "A system whose update matrices are the same at every step and independent of the input, so it "
         "can be computed as one long convolution.",
         "36", (), ""),
    Term("linearly separable", ("linear separability", "linearly inseparable"),
         "Two classes that a single straight line (or hyperplane) can separate. XOR is not.",
         "29", (), ""),
    Term("Llama", ("Llama-3", "Llama 3"),
         "Meta's family of open-weight LLMs; their architecture (RMSNorm, RoPE, SwiGLU, GQA) became the "
         "standard modern recipe.",
         "17", (), "Touvron et al., 2023"),
    Term("llama.cpp", ("llama-server", "llama.cpp server"),
         "A C/C++ LLM inference engine that runs GGUF models on CPUs and GPUs (via ROCm/HIP or Vulkan) and "
         "serves an OpenAI-compatible API.",
         "19", ("18", "21", "27", "39"), ""),
    Term("LLM-as-judge", ("LLM judge", "LLM-as-a-judge"),
         "Using a language model to grade other models' outputs. Cheap and scalable, but biased (e.g. "
         "position bias) and must be checked.",
         "21", ("38",), "Zheng et al., 2023"),
    Term("LM Studio", ("llmster",),
         "A desktop app and server for running local LLMs (via llama.cpp engines), with OpenAI-, "
         "Anthropic-compatible and native APIs.",
         "27", ("26", "38", "39"), ""),
    Term("load balancing", ("load-balancing", "auxiliary loss", "aux loss", "aux-loss-free"),
         "Keeping MoE experts evenly used. Classic MoE adds an auxiliary loss; DeepSeek-V3 instead nudges "
         "a per-expert bias in the router after each step (aux-loss-free).",
         "17", ("40",), "DeepSeek-AI, 2024 (V3)"),
    Term("local-global attention", ("local:global", "interleaved local and global layers"),
         "Mixing sliding-window layers with a few full-attention layers (Gemma 2 1 : 1, Gemma 3 5 : 1): most "
         "of the KV-cache saving of a window, with a path for long-range information.",
         "44", (), "Gemma Team, 2024"),
    Term("log-sum-exp", ("logsumexp", "log_softmax", "log-softmax"),
         "log(Σ exp xᵢ), computed safely by subtracting the max first so nothing overflows. It is the "
         "normalizer inside log-softmax and cross-entropy.",
         "04", ("08", "11", "39"), ""),
    Term("logistic regression", (),
         "A linear model followed by a sigmoid, giving the probability of the positive class; trained with "
         "binary cross-entropy.",
         "09", (), ""),
    Term("logit", ("logits",),
         "An unnormalized score, one per class or token, that softmax turns into probabilities. Logits are "
         "log-probabilities up to a shared constant.",
         "04", ("09", "11", "14", "18", "27"), ""),
    Term("long-term memory", ("episodic memory", "episodic log"),
         "Information an agent stores across sessions and retrieves when relevant, often with RAG over "
         "past episodes.",
         "39", (), ""),
    Term("LoRA", ("low-rank adaptation", "LoRA adapter", "LoRA adapters"),
         "Fine-tuning by freezing the base weights W and learning a low-rank update B·A (rank r) added to "
         "them, scaled by α/r. B starts at zero, so training starts from the base model, and only a tiny "
         "adapter is trained and shipped.",
         "20", ("02", "26", "27", "35"), "Hu et al., 2021"),
    Term("loss", ("loss function", "loss functions"),
         "A single number measuring how wrong the model is on data; training adjusts parameters to make it "
         "small.",
         "05", ("04", "10", "11", "14", "16"), ""),
    Term("loss at init", ("ln V",),
         "A sanity check: a freshly initialized classifier should predict near-uniformly, so its "
         "cross-entropy should be about ln V for V classes.",
         "16", (), ""),
    Term("loss landscape", ("landscape",),
         "The loss pictured as a surface over parameter space. Training walks downhill on it; narrow "
         "valleys, plateaus and saddle points make the walk hard.",
         "05", (), ""),
    Term("loss spike", ("loss spikes",),
         "A sudden jump in training loss, often from a bad batch, too high a learning rate or a numerical "
         "problem. Triage: check data, gradient norm, learning rate, precision.",
         "11", (), ""),
    Term("low-confidence remasking", ("confidence-based unmasking", "by_confidence"),
         "LLaDA's sampling rule for diffusion LMs: at each step keep the proposals the model is surest of and "
         "leave the rest masked for later.",
         "45", (), "Nie et al., 2025"),
    Term("low-rank approximation", ("low-rank", "low rank"),
         "Keep only the top few singular values of a matrix and drop the rest; the best approximation of "
         "that rank. LoRA and MLA both rely on weight changes or activations being nearly low-rank.",
         "02", ("20",), "Eckart & Young, 1936"),
    Term("LSTM", ("long short-term memory", "LSTMs"),
         "An RNN cell with a separate cell state and input, forget and output gates, so information (and "
         "gradients) can be kept unchanged over long spans.",
         "31", ("00b", "36"), "Hochreiter & Schmidhuber, 1997"),
    Term("Mamba", ("Mamba-2",),
         "A selective state-space model with a gated block structure and a hardware-aware scan; linear in "
         "sequence length with a constant-size state for generation.",
         "36", ("00b", "41"), "Gu & Dao, 2023"),
    Term("mantissa", ("significand",),
         "The bits of a float that hold its significant digits. bfloat16 keeps only 7 (about 2–3 decimal "
         "digits).",
         "08", (), ""),
    Term("masked language modeling", ("MLM", "masked language model", "mask_tokens", "[MASK]"),
         "BERT's training task: hide ~15% of tokens (80% replaced by [MASK], 10% random, 10% unchanged) "
         "and predict them from both sides.",
         "32", ("45",), "Devlin et al., 2018"),
    Term("masked diffusion language model", ("MDLM", "masked diffusion", "diffusion language model",
                                             "diffusion language models", "diffusion LM", "LLaDA"),
         "A language model that generates by filling in blanks: trained to predict tokens hidden at every "
         "masking rate (weighted by 1/t, a bound on the likelihood), it starts from an all-[MASK] passage "
         "and reveals several tokens per step, in any order.",
         "45", ("33",), "Sahoo et al., 2024; Nie et al., 2025"),
    Term("master weights", ("master weights in float32",),
         "The float32 copy of the parameters the optimizer updates. Tiny updates would be rounded away if "
         "added to bfloat16 weights.",
         "08", (), ""),
    Term("matrix", ("matrices",),
         "A 2-D array, best understood as a linear transformation of space: each column says where a basis "
         "vector lands. A neural-network layer is mostly a matrix.",
         "02", ("00", "05", "07", "12", "22"), ""),
    Term("matrix multiplication", ("matmul", "matmuls", "matrix multiply"),
         "Combining two matrices so the result applies one transformation after the other (right to left). "
         "Order matters. It is the operation GPUs are built to do fast.",
         "02", ("08", "18", "22", "23", "24"), ""),
    Term("MCP", ("Model Context Protocol",),
         "Model Context Protocol: an open standard for exposing tools and data to LLM applications through "
         "a common server interface.",
         "39", (), "Anthropic, 2024"),
    Term("mean squared error", ("MSE", "squared error"),
         "The average squared difference between prediction and target: the standard regression loss.",
         "09", ("10", "33", "34"), ""),
    Term("mechanistic interpretability", (),
         "Reverse-engineering what a trained network computes, finding the features and circuits inside "
         "it.",
         "37", (), ""),
    Term("memory bandwidth", (),
         "How fast the GPU can read and write its memory, 288 GB/s on the RX 7600 XT. Sets the speed limit "
         "for memory-bound work.",
         "18", ("06",), ""),
    Term("memory profile", ("memory_stats", "peak memory", "device memory profile"),
         "Measurements of how much GPU memory is live and at peak, used to see the effect of donation and "
         "precision.",
         "22", (), ""),
    Term("memory-bandwidth-bound", ("memory-bound", "memory bound", "bandwidth-bound"),
         "When time is set by bytes moved from GPU memory, not arithmetic. Single-sequence decoding is, so "
         "its speed ≈ bandwidth ÷ model size.",
         "18", ("19", "22"), ""),
    Term("merge", ("merges",),
         'One BPE rule, "replace pair (a, b) with new token c", learned in frequency order.',
         "13", (), ""),
    Term("merging", ("merge_lora", "merge the LoRA"),
         "Folding a LoRA update into the base weights (W + scale·B·A) so inference needs no extra "
         "computation.",
         "20", ("27",), ""),
    Term("mesh", ("device mesh", "jax.make_mesh"),
         "A grid of devices with named axes (e.g. 'data', 'model') that sharding specs refer to.",
         "24", ("25",), ""),
    Term("min-max fusion", ("min-max", "minmax_fuse"),
         "Rescale each retriever's scores to [0, 1] over the candidates and mix them with a weight.",
         "38", (), ""),
    Term("minibatch", ("minibatches", "mini-batch", "batch size"),
         "A small random subset of the training data used for one gradient step.",
         "05", ("11", "18"), ""),
    Term("minimax", ("alpha-beta", "alpha–beta"),
         "Classic game search: assume both players play optimally, maximize your worst case; alpha-beta "
         "pruning skips branches that can't matter. Deep Blue's method.",
         "34", (), ""),
    Term("mixed precision", ("mixed-precision",),
         "Doing the heavy matmuls in bfloat16 while keeping master weights, optimizer state and sensitive "
         "reductions in float32. Most of the speed, little of the risk.",
         "08", ("16",), "Micikevicius et al., 2017"),
    Term("mixture of experts", ("MoE", "mixture-of-experts"),
         "Replace one big MLP with many expert MLPs and a router that sends each token to a few of them. "
         "Parameters grow without per-token compute growing.",
         "17", ("00b", "16", "19", "24", "40"), "Shazeer et al., 2017"),
    Term("MoBA", ("mixture of block attention", "moba_mask"),
         "Block-sparse attention where each query reads its own block plus the earlier blocks whose mean key "
         "scores highest against it: mixture-of-experts routing over blocks of the context.",
         "44", ("17",), "Lu et al., 2025"),
    Term("mode collapse", ("mode-collapses", "mode collapses"),
         "A GAN failure in which the generator produces only a few kinds of output, ignoring the variety "
         "of the data.",
         "33", (), ""),
    Term("model FLOPs utilization", ("MFU",),
         "Achieved training FLOP/s (≈ 6 × parameters × tokens per second) divided by the GPU's peak. A "
         "single number for how well you use the hardware.",
         "16", ("22", "25", "42"), "Chowdhery et al., 2022 (PaLM)"),
    Term("model merging", ("TIES", "TIES-Merging", "DARE", "mergekit"),
         "Combining fine-tunes of one base model by arithmetic on their weights, with no further training. TIES "
         "trims small changes and resolves sign conflicts; DARE drops most of each change at random and rescales.",
         "46", ("20",), "Yadav et al., 2023; Yu et al., 2023"),
    Term("modern Hopfield network", ("modern Hopfield", "dense associative memory"),
         "Hopfield networks with sharper energy functions that store exponentially many patterns; their "
         "update rule is softmax attention.",
         "29", (), "Ramsauer et al., 2020"),
    Term("momentum", ("heavy ball", "Nesterov"),
         "Keep a running average of past gradients and step along it, like a ball rolling downhill. It "
         "damps zig-zagging and speeds travel along consistent directions; Nesterov evaluates the gradient "
         "at the look-ahead point.",
         "05", ("11",), "Polyak, 1964"),
    Term("Monte Carlo", ("Monte Carlo estimate", "Monte Carlo sampling"),
         "Estimating an expectation by averaging random samples. The error shrinks like 1/√n, so 100× more "
         "samples buys only 10× more accuracy.",
         "04", ("34",), ""),
    Term("Monte Carlo tree search", ("MCTS", "tree search"),
         "Plan by repeatedly simulating from the current position down a search tree, choosing moves that "
         "balance promise and uncertainty, and backing up the results.",
         "34", ("00b",), "Coulom, 2006; Kocsis & Szepesvári, 2006"),
    Term("MRR", ("mean reciprocal rank", "MRR@10", "mrr_at_k"),
         "Mean reciprocal rank: average of 1/(rank of the first relevant result) over queries.",
         "38", (), ""),
    Term("multi-agent system", ("multi-agent", "manager pattern", "decentralized pattern"),
         "Several agents cooperating: a manager calling specialists as tools, or agents handing off to one "
         "another. Worth it only when one agent's prompt or toolset gets unwieldy.",
         "39", (), "OpenAI, 2025 (A practical guide to building agents)"),
    Term("multi-head attention", ("MHA", "attention heads", "attention head"),
         "Running several attention operations in parallel on smaller slices of the vector, each free to "
         "look for a different relation, then concatenating the results.",
         "15", ("17", "18"), "Vaswani et al., 2017"),
    Term("multi-head latent attention", ("MLA",),
         "DeepSeek's attention: compress keys and values into a small shared latent vector and cache only "
         "that, with a separate small RoPE part. Much smaller KV cache than GQA at similar quality.",
         "17", ("15", "16", "18"), "DeepSeek-AI, 2024 (V2)"),
    Term("multi-host", ("multiple hosts", "jax.distributed", "multi-node"),
         "Running one JAX program across several machines, each with its own devices, coordinated by "
         "`jax.distributed`.",
         "24", ("25",), ""),
    Term("multi-query attention", ("MQA",),
         "The extreme of GQA: all query heads share a single key/value head.",
         "17", (), "Shazeer, 2019"),
    Term("multi-token prediction", ("MTP",),
         "An extra training head that predicts the token after next, densifying the training signal; its "
         "predictions can also drive speculative decoding.",
         "17", (), "Gloeckle et al., 2024; DeepSeek-AI, 2024"),
    Term("multilayer perceptron", ("MLP", "MLPs", "feed-forward network", "feedforward network", "FFN"),
         "Layers of linear maps with a nonlinearity between them. The basic neural network, and the "
         "per-token feed-forward block inside every transformer.",
         "10", ("11", "12", "14", "16", "31"), "Rumelhart, Hinton, Williams, 1986"),
    Term("multiple-choice evaluation",
         ("multiple-choice", "multiple choice", "ARC-Easy", "log-likelihood scoring"),
         "Scoring a model on a question by which answer option it gives the highest (often "
         "length-normalized) log-likelihood, without generating text.",
         "21", ("39",), ""),
    Term("Muon", (),
         "An optimizer for hidden 2-D weight matrices: take the momentum, orthogonalize it with "
         "Newton–Schulz iterations (set all its singular values to ≈1), and step along that. Embeddings, "
         "the output head and vector parameters stay on AdamW.",
         "05", ("25", "42"), "Keller Jordan, 2024"),
    Term("MuZero", ("Gumbel MuZero", "mctx"),
         "AlphaZero with a learned model of the game's dynamics, so it plans without knowing the rules. "
         "`mctx` is DeepMind's JAX MCTS library built around it.",
         "34", (), "Schrittwieser et al., 2019"),
    Term("n-gram", ("n-grams", "n-gram model"),
         "A run of n consecutive tokens; an n-gram model predicts from the previous n − 1. Also used to "
         "measure text overlap in evaluation and contamination checks.",
         "14", ("21", "40"), "Shannon, 1948"),
    Term("naive Bayes", ("log-odds readout",),
         "A classifier that assumes features are independent given the class; for binary features its "
         "log-probabilities are a linear layer whose weights are log-odds counted from data, not trained.",
         "46", (), ""),
    Term("NaN", ("NaNs",),
         "Not-a-Number: the result of invalid float math like 0/0 or inf − inf. One NaN poisons everything "
         "it touches, so catching its source is a key debugging skill.",
         "11", ("05", "09", "25", "35", "39"), ""),
    Term("Native Sparse Attention", ("NSA",),
         "DeepSeek's trainable sparse attention: compressed block summaries, a few selected blocks and a "
         "sliding window, combined by a learned gate and laid out for GPU kernels.",
         "44", (), "Yuan et al., 2025"),
    Term("nats", (),
         "Units of information: a nat uses the natural log, a bit uses log base 2 (1 nat ≈ 1.44 bits). "
         "Losses in JAX are usually in nats.",
         "04", ("14", "32", "33", "35", "37"), ""),
    Term("nDCG", ("nDCG@10", "ndcg_at_k", "normalized discounted cumulative gain"),
         "Normalized discounted cumulative gain: rewards relevant results more the higher they rank (with "
         "a log discount), scaled so a perfect ranking scores 1.",
         "38", (), "Järvelin & Kekäläinen, 2002"),
    Term("negative log-likelihood", ("NLL", "log-likelihood", "bigram_nll"),
         "−log of the probability a model assigns to the observed data. Minimizing it is "
         "maximum-likelihood training and equals minimizing cross-entropy.",
         "14", ("21",), ""),
    Term("negative sampling", ("SGNS",),
         "Instead of a softmax over the whole vocabulary, train the model to tell real (word, context) "
         "pairs from a few random fake ones, using a sigmoid each.",
         "30", (), "Mikolov et al., 2013"),
    Term("neural probabilistic language model", ("Bengio et al., 2003", "MLP language model"),
         "Bengio's 2003 model: embed the previous few tokens, concatenate, and predict the next with an "
         "MLP. It introduced learned word embeddings.",
         "14", (), "Bengio et al., 2003"),
    Term("Newton–Schulz iteration", ("Newton-Schulz", "Newton–Schulz", "newton_schulz", "polar factor"),
         "A few matrix-multiply-only iterations that push a matrix's singular values toward 1, "
         "approximating its polar factor (the nearest orthogonal matrix). It is the core of Muon and runs "
         "well on GPUs.",
         "05", (), ""),
    Term("next-token prediction", ("next token prediction", "next-token", "next token"),
         "The training task of language models: predict token t+1 from tokens up to t. Inputs and targets "
         "are the same sequence shifted by one.",
         "14", ("04", "18", "32", "35", "37"), ""),
    Term("NHWC", ("HWIO",),
         "Axis-order conventions for images (batch, height, width, channels) and conv kernels (height, "
         "width, in, out) used by JAX's convolution API.",
         "12", (), ""),
    Term("non-saturating loss", ("non-saturating",),
         "The practical GAN generator loss, maximizing log D(fake) instead of minimizing log(1 − D(fake)), "
         "so it gets strong gradients early on.",
         "33", (), "Goodfellow et al., 2014"),
    Term("NoPE", ("no positional encoding",),
         "No positional encoding at all. The causal mask still leaks order (a token at position p averages "
         "over p + 1 values), so decoders can learn positions without being given them.",
         "17", ("00b", "36"), "Haviv et al., 2022"),
    Term("norm", ("L2 norm", "Frobenius norm", "vector norm"),
         "A measure of a vector's or matrix's size, most often the L2 (Euclidean) length. Used for "
         "regularization, clipping and diagnosing training.",
         "02", ("05", "07", "08", "11", "12"), ""),
    Term("normal equations", ("closed form", "closed-form", "least squares"),
         "The direct least-squares solution w = (XᵀX)⁻¹Xᵀy, found by setting the gradient to zero instead "
         "of iterating.",
         "09", ("16",), ""),
    Term("normalization layer", ("normalization", "normalization layers"),
         "A layer that re-centres and re-scales activations to a stable range, so deep networks train "
         "reliably.",
         "11", ("33",), ""),
    Term("occupancy", ("waves per SIMD", "latency hiding"),
         "How many wavefronts are resident per SIMD. More lets the GPU hide memory latency by switching "
         "waves; if not even one workgroup fits, the launch fails.",
         "23", (), ""),
    Term("offloading", ("gpu_layers", "GPU layers", "offload", "n-gpu-layers"),
         "Putting some layers on the GPU and the rest on the CPU when a model doesn't fit in VRAM. Each "
         "CPU layer slows generation a lot.",
         "19", (), ""),
    Term("one-hot", ("one-hot encoding", "one_hot"),
         "A vector of zeros with a single 1 at the index of a category. Multiplying a one-hot by a matrix "
         "just selects a row, which is what an embedding lookup does.",
         "01", ("14", "15", "30"), ""),
    Term("online softmax", ("streaming softmax",),
         "Computing softmax in one pass over blocks by keeping a running max and sum and rescaling as a "
         "new max appears, so the full row never has to be in memory.",
         "23", (), "Milakov & Gimelshein, 2018"),
    Term("OpenAI-compatible API", ("OpenAI-compatible", "/v1/chat/completions", "chat completions"),
         "The de facto HTTP API for LLM servers (`/v1/chat/completions`), so any client written for OpenAI "
         "can point at a local server instead.",
         "19", ("27", "39"), ""),
    Term("optax", (),
         "JAX's optimizer library: optimizers are composable gradient transformations (`init` and "
         "`update`) chained together, e.g. clip then AdamW.",
         "05", ("09", "10", "14", "16", "20"), ""),
    Term("orchestrator-workers", ("orchestrator–workers",),
         "A workflow in which one model breaks the task into sub-tasks at run time, worker calls do them, "
         "and the results are combined.",
         "39", (), "Anthropic, 2024"),
    Term("orthogonal matrix",
         ("orthonormal", "Gram-Schmidt", "Gram–Schmidt", "gram_schmidt", "orthogonalize"),
         "A matrix whose columns are perpendicular unit vectors; it rotates or reflects without "
         "stretching. Gram–Schmidt builds one from any set of vectors.",
         "02", (), ""),
    Term("Othello-GPT", ("Othello",),
         "A GPT trained only on legal Othello move sequences, which nonetheless develops an internal board "
         "representation detectable with probes.",
         "37", (), "Li et al., 2022; Nanda et al., 2023"),
    Term("overfit one batch", ("overfit a single batch",),
         "A debugging check: a correct model and training loop should drive the loss to near zero on one "
         "small batch. If it can't, there's a bug.",
         "11", (), ""),
    Term("overfitting", ("overfit", "overfits"),
         "Fitting the training data so closely, noise included, that performance on new data gets worse. "
         "The gap between training and validation loss shows it.",
         "09", ("10", "11", "35"), ""),
    Term("overhead-bound", ("overhead bound",),
         "When a program's time goes to Python, dispatch and kernel-launch overhead rather than arithmetic "
         "or memory. Small models on a big GPU are usually overhead-bound; jit and scan fix it.",
         "10", (), ""),
    Term("padding", ("same padding", "valid padding"),
         "Adding zeros around the input so a convolution's output keeps a chosen size; also padding "
         "sequences to a common length.",
         "12", ("06", "16", "22"), ""),
    Term("Pallas", ("pallas_call",),
         "JAX's kernel language: write a kernel in Python on blocks of arrays (refs), and it is compiled "
         "via Triton (GPU) or Mosaic (TPU). Has an interpret mode for debugging on CPU.",
         "23", (), ""),
    Term("Pallas grid", ("BlockSpec", "block spec"),
         "In Pallas, the set of kernel instances to launch; each instance gets a block of the inputs "
         "chosen by its BlockSpec.",
         "23", (), ""),
    Term("Pallas ref", ("x_ref",),
         "In Pallas (and JAX state), a mutable reference to a block of memory that a kernel reads with "
         "`x_ref[...]` and writes in place.",
         "23", (), ""),
    Term("parallelization", ("sectioning",),
         "A workflow running several LLM calls at once and combining them, either by voting on repeated "
         "attempts or by splitting the task into sections.",
         "39", (), "Anthropic, 2024"),
    Term("parameter-efficient fine-tuning", ("PEFT", "parameter-efficient"),
         "Fine-tuning methods that train only a small number of extra or selected parameters, like LoRA.",
         "20", (), ""),
    Term("parameters", (),
         "The learned numbers of a model (weights and biases). Training is the search for good parameter "
         "values.",
         "10", ("03", "05", "12", "16", "20"), ""),
    Term("partial derivative", ("partial derivatives",),
         "The derivative with respect to one input while holding the others fixed.",
         "03", (), ""),
    Term("partial RoPE", ("partial rotary", "rope_partial", "partial_rotary_factor"),
         "RoPE applied to only some of each head's features; the rest stay unrotated and match on content "
         "regardless of distance.",
         "17", ("00b",), "Black et al., 2022 (GPT-NeoX-20B)"),
    Term("PartitionSpec", ("PartitionSpecs",),
         "A tuple naming which mesh axis, if any, each array dimension is split over, e.g. P('data', "
         "None).",
         "24", (), ""),
    Term("pass@k", ("pass at k",),
         "The probability that at least one of k sampled answers is correct; an upper bound for what "
         "voting or verification can reach.",
         "26", (), "Chen et al., 2021 (Codex)"),
    Term("patch embedding", ("patchify", "patches"),
         "A ViT's first layer: cut the image into p × p patches, flatten each, and multiply by one matrix (the "
         "same as a p × p convolution with stride p). Each patch becomes a token.",
         "43", (), "Dosovitskiy et al., 2020"),
    Term("per-example gradients", ("per-example gradient", "per_example_grads"),
         "A separate gradient for every example in a batch, obtained as `vmap(grad(loss))`. Used in "
         "differential privacy and data diagnostics.",
         "07", (), ""),
    Term("perceptron", ("perceptrons", "Rosenblatt"),
         "Rosenblatt's 1958 trainable linear threshold unit and its learning rule: when it misclassifies, "
         "add (or subtract) the input to the weights. Converges whenever the classes are linearly "
         "separable.",
         "29", ("00b", "10"), "Rosenblatt, 1958"),
    Term("perceptron convergence theorem", ("convergence theorem",),
         "If the data are linearly separable, the perceptron rule finds a separating line in a finite "
         "number of mistakes.",
         "29", (), "Novikoff, 1962"),
    Term("Perceptrons (Minsky & Papert)", ("Minsky and Papert", "Minsky & Papert", "Perceptrons (1969)"),
         "Minsky and Papert's 1969 book proving the limits of single-layer perceptrons (e.g. XOR), widely "
         "blamed for the first AI winter.",
         "29", ("00b",), "Minsky & Papert, 1969"),
    Term("permutation equivariance", ("permutation-equivariant",),
         "Shuffling the inputs just shuffles the outputs the same way. Plain attention has it, so it can't "
         "tell word order without positional information.",
         "15", (), ""),
    Term("perplexity", (),
         "exp of the average per-token cross-entropy (in nats): the effective number of choices the model "
         "is torn between at each step. Lower is better; only comparable across models with the same "
         "tokenizer.",
         "04", ("14", "18", "21", "27"), ""),
    Term("pipeline parallelism",
         ("pipeline parallel", "pipelining", "microbatches", "microbatch", "pipeline bubble"),
         "Putting different layers on different devices and streaming microbatches through, like an "
         "assembly line. Startup and drain leave idle time, the bubble.",
         "24", (), "Huang et al., 2018 (GPipe)"),
    Term("pixel shuffle", ("pixel_shuffle", "space-to-depth"),
         "Merging each r × r neighbourhood of image tokens into one wider token: r² fewer tokens for the "
         "language model to read. SmolVLM uses it to cut 1,024 patches to 64 tokens.",
         "43", (), "Shi et al., 2016; Marafioti et al., 2025"),
    Term("PJRT", ("PJRT plugin", "PJRT ROCm plugin"),
         "The plugin interface between JAX and a hardware backend. On this machine the `jax-rocm` PJRT "
         "plugin is the bridge from JAX to the AMD GPU.",
         "00", (), ""),
    Term("policy", ("policies",),
         "In RL, the rule (here the model's token distribution) that maps a state to actions. Fine-tuning "
         "an LLM with RL treats the model as the policy.",
         "34", ("20", "26", "35", "39"), ""),
    Term("policy gradient", ("policy gradients", "REINFORCE"),
         "Improve a policy by raising the log-probability of actions in proportion to how much better than "
         "expected they turned out (the advantage).",
         "34", ("26",), "Williams, 1992 (REINFORCE)"),
    Term("polysemantic", ("polysemanticity", "monosemantic"),
         "A neuron that responds to several unrelated features, a consequence of superposition.",
         "37", (), ""),
    Term("pooling", ("max pooling", "average pooling"),
         "Downsampling by taking the max or mean over small windows, which shrinks feature maps and adds "
         "tolerance to small shifts.",
         "12", (), ""),
    Term("position bias", (),
         "A judge preferring whichever answer comes first (or second) regardless of content. Countered by "
         "scoring both orders.",
         "21", (), ""),
    Term("positional encoding",
         ("positional embedding", "positional embeddings", "position embedding", "positional information"),
         "Information about each token's position added to (or rotated into) its representation so "
         "attention can use word order.",
         "15", ("16", "17"), "Vaswani et al., 2017"),
    Term("post-training", (),
         "Everything done to a pretrained base model to make it a useful assistant: SFT, preference "
         "learning, RL with verifiable rewards.",
         "26", ("39",), ""),
    Term("PPO", ("Proximal Policy Optimization",),
         "The policy-gradient algorithm classic RLHF uses: clip the policy-probability ratio so each "
         "update stays small, with a learned value function as the baseline.",
         "35", ("26", "34"), "Schulman et al., 2017"),
    Term("pre-norm", ("pre-LayerNorm", "pre-norm residual"),
         "Applying normalization at the start of each residual branch, x + f(norm(x)), rather than after "
         "the addition. More stable for deep transformers; used by GPT-2 onward.",
         "11", ("17",), ""),
    Term("pre-tokenization", ("pre-tokenizer",),
         "Splitting text into chunks (words, numbers, spaces) with a regex before BPE, so merges never "
         "cross those boundaries.",
         "13", ("27",), ""),
    Term("preallocation", ("XLA_PYTHON_CLIENT_PREALLOCATE", "preallocate"),
         "By default JAX grabs 75% of GPU memory at startup so later allocations are fast. The course "
         "turns it off (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) so several notebooks can share one card.",
         "00", (), ""),
    Term("preference data", ("preference pairs", "chosen and rejected", "chosen/rejected"),
         "Pairs of responses to the same prompt with a label saying which is better; the training data for "
         "reward models and DPO.",
         "35", (), ""),
    Term("prefill", ("prompt processing",),
         "The first phase of generation: run the whole prompt through the model at once, filling the KV "
         "cache. Compute-bound.",
         "18", ("17", "19", "39", "41"), ""),
    Term("prefix scan", ("parallel scan", "cumulative sum", "cumsum"),
         "Computing all running results (running sums, running products) of a sequence; with an "
         "associative operator it parallelizes in O(log L) steps.",
         "36", (), "Blelloch, 1990"),
    Term("previous-token head", ("previous token head", "previous_token_score"),
         "An attention head that mostly attends to the immediately preceding token; the first half of an "
         "induction circuit.",
         "37", (), ""),
    Term("primitive", ("primitives",),
         "One of JAX's built-in elementary operations (add, dot_general, exp, …) that every function is "
         "traced down to and that each transformation knows how to handle.",
         "06", (), ""),
    Term("PRNG key", ("random key", "random keys", "jax.random.key", "PRNGKey", "key splitting"),
         "JAX's explicit random state: a key you pass to every random function and `split` into fresh "
         "subkeys. Same key in, same numbers out, so randomness is reproducible and safe under jit and "
         "vmap.",
         "04", ("11", "16", "17", "31", "34"), ""),
    Term("probability distribution", (),
         "An assignment of probabilities to outcomes that sum (or integrate) to 1. A language model "
         "outputs one distribution over the vocabulary at each position.",
         "04", (), ""),
    Term("profiler trace", ("profiler", "jax.profiler", "Perfetto", "trace viewer"),
         "A timeline of every kernel and host event during a run, captured with `jax.profiler` and viewed "
         "in Perfetto. Shows which kernels dominate and where the GPU sits idle.",
         "22", ("25",), ""),
    Term("projection", ("project onto",),
         "Dropping a vector onto a line or subspace: the closest point there. The dot product with a unit "
         "vector gives the projection's length.",
         "02", ("15", "20"), ""),
    Term("prompt caching", ("prefix caching",),
         "Reusing the KV cache for a prompt prefix shared across requests (like a long system prompt), so "
         "it is prefilled once.",
         "39", (), ""),
    Term("prompt chaining", (),
         "A workflow of fixed LLM steps, each feeding the next, with programmatic checks (gates) between "
         "them.",
         "39", (), "Anthropic, 2024"),
    Term("prompt format", ("prompt sensitivity",),
         "The exact wording and layout of a prompt. Changing it can move benchmark scores by several "
         "points, so it is part of the result.",
         "21", (), ""),
    Term("PUCT", ("UCB", "UCT"),
         "The rule AlphaZero's MCTS uses to pick which move to explore: value estimate plus a bonus "
         "proportional to the network's prior for the move, shrinking as it is visited more.",
         "34", (), "Silver et al., 2017"),
    Term("pure function", ("pure functions", "side effect", "side effects"),
         "A function whose output depends only on its inputs and which changes nothing outside itself. jit "
         "and the other transformations assume purity; side effects run only once, at trace time.",
         "06", ("05", "07", "10"), ""),
    Term("PyTorch", (),
         "The most widely used deep-learning framework, imperative and object-oriented. The course "
         "compares it with JAX in side notes, using a separate ROCm PyTorch image.",
         "10", ("03", "15", "22", "23", "26"), ""),
    Term("pytree", ("pytrees", "tree_map", "jax.tree"),
         "Any nested structure of lists, tuples, dicts (or registered classes) with arrays at the leaves. "
         "JAX transformations and optax accept pytrees, so model parameters are just a dict of arrays.",
         "07", ("05", "08", "10", "16", "20"), ""),
    Term("Q-learning", ("tabular Q-learning",),
         "Learn Q-values by nudging each estimate toward reward + discounted best next Q, from experience, "
         "without a model of the environment.",
         "34", (), "Watkins, 1989"),
    Term("Q-value", ("action-value", "Q-values", "Q function"),
         "The expected return of taking a given action in a given state, then acting well.",
         "34", (), ""),
    Term("QK-Norm", ("qk norm", "query-key normalization"),
         "Normalizing queries and keys before their dot product, which keeps attention logits from blowing "
         "up and stabilizes training.",
         "00b", (), "Henry et al., 2020"),
    Term("quadratic cost", ("O(L²)", "quadratic"),
         "Attention compares every pair of positions, so compute and the score matrix grow with the square "
         "of the sequence length.",
         "15", (), ""),
    Term("quantization", ("quantize", "quantized", "int8", "int4"),
         "Storing weights in fewer bits (int8, int4) with a scale per group or channel. Smaller models "
         "load faster and decoding speeds up, at a small accuracy cost.",
         "18", ("01", "19", "27"), ""),
    Term("query likelihood", (),
         "Ranking documents by the probability a language model would generate the query from them.",
         "38", (), "Ponte & Croft, 1998"),
    Term("query, key, value", ("Q, K, V",),
         "The three projections in attention: a query asks what a position is looking for, keys advertise "
         "what each position has, values are what gets copied. Score = query · key.",
         "15", ("23",), "Vaswani et al., 2017"),
    Term("Qwen", ("Qwen2.5", "Qwen3", "Qwen2.5-Coder"),
         "Alibaba's family of open-weight LLMs, used in the course as capable local models in LM Studio "
         "and for embeddings.",
         "27", ("19", "26", "38"), ""),
    Term("random variable", ("random variables",),
         "A quantity whose value is drawn from a probability distribution, like a die roll or the next "
         "token.",
         "04", (), ""),
    Term("rank", ("matrix rank",),
         "The number of independent directions a matrix can output: the count of its nonzero singular "
         "values. An (m, n) matrix written as A·B with inner size r has rank at most r.",
         "02", ("05", "17", "20"), ""),
    Term("RASP", ("Tracr", "compiled transformer"),
         "A small programming language whose select and aggregate operations are what attention can do; Tracr "
         "compiles RASP programs into real transformer weights.",
         "46", ("37",), "Weiss et al., 2021; Lindner et al., 2023"),
    Term("RDNA3", ("RDNA",),
         "The AMD GPU architecture of the RX 7600 XT, with 32-thread wavefronts and compute units grouped "
         "into SIMDs.",
         "23", ("22",), ""),
    Term("ReAct", (),
         "Prompting a model to interleave reasoning (Thought), tool calls (Action) and their results "
         "(Observation).",
         "39", ("26",), "Yao et al., 2022"),
    Term("reasoning model", ("reasoning models", "DeepSeek-R1"),
         "An LLM trained, typically with RL on verifiable rewards, to produce long chains of thought "
         "before answering.",
         "26", ("27",), "DeepSeek-AI, 2025"),
    Term("recall@k", ("recall at k", "recall_at_k"),
         "The fraction of relevant documents that appear in the top k results.",
         "38", (), ""),
    Term("receptive field", ("receptive fields",),
         "The region of the input that can influence one output unit. Stacking and dilating convolutions "
         "grows it.",
         "12", (), ""),
    Term("reciprocal rank fusion", ("RRF",),
         "Merge ranked lists by summing 1/(k + rank) for each document across lists (k = 60 by "
         "convention); needs no score calibration.",
         "38", (), "Cormack, Clarke, Buettcher, 2009"),
    Term("recompilation", ("recompile", "recompiles", "retrace"),
         "jit compiles again whenever a call brings a new input shape, dtype or static value. Varying "
         "shapes cause a recompile storm; padding to a few bucket sizes avoids it.",
         "06", (), ""),
    Term("recurrent neural network", ("RNN", "RNNs"),
         "A network that reads a sequence one step at a time, updating a hidden state from the previous "
         "state and the current input with the same weights at every step.",
         "31", ("08", "36"), "Elman, 1990"),
    Term("reduce-scatter", ("reduce scatter",),
         "Sum across devices, but each device keeps only its own slice of the result; an all-reduce is a "
         "reduce-scatter plus an all-gather.",
         "24", ("25",), ""),
    Term("reduction", ("reductions",),
         "An operation that collapses an axis into one value per remaining position: sum, mean, max, "
         "argmax.",
         "01", ("08", "24"), ""),
    Term("regularization", ("regularizer",),
         "Anything that discourages fitting noise and so improves generalization: weight decay, dropout, "
         "data augmentation, early stopping.",
         "11", ("05", "09"), ""),
    Term("reinforcement learning", ("RL",),
         "Learning to act by trial and error: an agent observes the state of an environment, takes "
         "actions, and adjusts its policy to maximize total reward.",
         "34", ("25", "26", "35", "39"), ""),
    Term("relative position bias", ("T5 bias",),
         "A learned number per attention head and distance bucket, added to the attention scores: T5's "
         "position scheme.",
         "17", ("32", "00b"), "Raffel et al., 2019"),
    Term("ReLU", ("rectified linear unit",),
         "max(0, x): the simplest common nonlinearity. Networks of ReLUs compute piecewise-linear "
         "functions.",
         "10", ("11", "12", "17", "33", "37"), "Nair & Hinton, 2010"),
    Term("reparameterization trick", ("reparameterization",),
         "Write a sample as μ + σ·ε with ε random noise, so gradients can flow through the sampling step "
         "to μ and σ.",
         "33", (), "Kingma & Welling, 2013"),
    Term("repetition penalty", (),
         "Lowering the logits of tokens already generated to discourage loops.",
         "18", (), "Keskar et al., 2019"),
    Term("reranker", ("rerankers",),
         "A second stage that rescores the top candidates from a fast retriever with a slower, more "
         "accurate model.",
         "38", (), ""),
    Term("residual connection", ("residual connections", "skip connection", "skip connections"),
         "Adding a block's input to its output, x + f(x). Gradients get a direct path back through the "
         "network, which is what makes very deep networks trainable.",
         "11", ("31",), "He et al., 2015"),
    Term("residual stream", (),
         "The vector each token carries through the network; every attention and MLP block reads from it "
         "and adds its output back. A useful mental model for interpretability.",
         "16", ("37",), "Elhage et al., 2021"),
    Term("ResNet", (),
         "The 2015 deep convolutional network built from residual blocks, which made 100+ layer networks "
         "trainable.",
         "11", (), "He et al., 2015"),
    Term("retrieval-augmented generation", ("RAG", "retrieval-augmented"),
         "Retrieve relevant documents for a question and put them in the prompt, so the model answers from "
         "sources instead of memory alone.",
         "38", ("32", "39"), "Lewis et al., 2020"),
    Term("reverse mode", ("reverse-mode",),
         "Autodiff that runs forward, then sweeps backwards from the output, getting the gradient for all "
         "inputs in one pass. Ideal for one scalar loss and millions of parameters, which is why training "
         "uses it.",
         "03", ("08",), ""),
    Term("reverse process", (),
         "In diffusion, the learned procedure that removes noise step by step, turning random noise into a "
         "sample.",
         "33", (), ""),
    Term("reward", ("rewards",),
         "The scalar feedback an RL environment (or a reward model or verifier) gives for an action or "
         "outcome.",
         "34", ("26", "35"), ""),
    Term("reward hacking", ("specification gaming",),
         "A policy maximizing the reward in a way that defeats its purpose, such as gaming a format check, "
         "because the reward measures the goal imperfectly.",
         "26", (), ""),
    Term("reward model", ("reward models",),
         "A model trained to score a response so preferred responses score higher; it stands in for human "
         "judgment during RL.",
         "35", (), "Christiano et al., 2017"),
    Term("ridge point", (),
         "The arithmetic intensity at which a roofline's memory and compute roofs meet: peak FLOP/s ÷ "
         "bandwidth.",
         "22", (), ""),
    Term("ridge regression", ("ridge",),
         "Linear regression with an L2 penalty on the weights, which tames overfitting and makes the "
         "solution stable.",
         "09", ("22",), "Hoerl & Kennard, 1970"),
    Term("RLAIF", ("AI feedback", "Constitutional AI"),
         "Using an AI model's judgments instead of human labels to produce preference data.",
         "35", (), "Bai et al., 2022"),
    Term("RLHF", ("reinforcement learning from human feedback",),
         "Train a reward model on human preference comparisons, then fine-tune the LM with RL to maximize "
         "that reward while a KL penalty keeps it near the original. How ChatGPT-style assistants were "
         "aligned.",
         "35", ("34",), "Christiano et al., 2017; Ouyang et al., 2022"),
    Term("RLVR", ("RL with verifiable rewards", "reinforcement learning with verifiable rewards"),
         "Reinforcement learning whose reward comes from an automatic verifier (correct answer, passing "
         "tests) rather than a learned reward model. How DeepSeek-R1-style reasoning is trained.",
         "26", (), "Lambert et al., 2024; DeepSeek-AI, 2025"),
    Term("RMSNorm", (),
         "LayerNorm without the mean subtraction: divide by the root-mean-square, then scale. Cheaper and "
         "works as well; used by Llama and DeepSeek.",
         "11", ("16", "17"), "Zhang & Sennrich, 2019"),
    Term("RMSProp", (),
         "Divide each parameter's step by a running root-mean-square of its recent gradients, so steep "
         "directions take smaller steps and flat ones larger.",
         "05", (), "Hinton, 2012 (lecture)"),
    Term("ROCm", ("ROCm 10",),
         "AMD's open GPU software stack: driver interface, the HIP runtime and tuned math libraries. It "
         "plays the role CUDA plays on NVIDIA cards, and in this course it lives only inside the "
         "container.",
         "00", ("00b", "19", "22", "23", "27"), ""),
    Term("rocprof", ("rocprofv3", "rocprofv2", "omniperf"),
         "AMD's command-line GPU profiler; it reports every kernel launch with its resources, which is how "
         "the Triton-GEMM failure is diagnosed.",
         "22", (), ""),
    Term("ROME", ("rank-one model editing", "knowledge editing", "MEMIT"),
         "Rank-One Model Editing: rewrite one fact by changing one MLP output matrix so the subject's key produces "
         "a new value, sized by the covariance of the layer's usual inputs. MEMIT does thousands at once.",
         "46", (), "Meng et al., 2022"),
    Term("roofline", ("roofline model",),
         "A plot of achievable FLOP/s against arithmetic intensity: a sloped memory roof for low intensity "
         "and a flat compute roof for high. Tells you whether a kernel is memory- or compute-bound and how "
         "far from peak.",
         "22", ("18",), "Williams, Waterman, Patterson, 2009"),
    Term("RoPE", ("rotary position embedding", "rotary positional embedding", "rotary embeddings", "rotary"),
         "Rotary position embedding: rotate pairs of query and key dimensions by angles proportional to "
         "position, so their dot product depends only on relative position. The base sets the rotation "
         "frequencies.",
         "17", ("16",), "Su et al., 2021"),
    Term("round-off error", ("rounding error", "round-off"),
         "The error from storing a number with finite precision. It compounds in long sums and is worst in "
         "low-precision formats.",
         "08", ("03", "18", "27"), ""),
    Term("router", ("gating network", "top-k routing"),
         "In MoE, the small layer that scores the experts for each token and picks the top few.",
         "17", (), ""),
    Term("S4", ("structured state space",),
         "Structured State Space sequence model: an LTI SSM with a special (HiPPO-initialized) state "
         "matrix, computed by FFT convolution, that handled very long sequences.",
         "36", (), "Gu, Goel, Ré, 2021"),
    Term("safetensors", (),
         "A simple, safe file format for model weights (a header plus raw tensors), standard on Hugging "
         "Face.",
         "27", (), ""),
    Term("sampling", (),
         "Generating text by drawing each next token from the model's predicted distribution, rather than "
         "always taking the most likely.",
         "04", ("18", "21", "26", "33", "35"), ""),
    Term("scaled dot-product attention", ("scaled dot product", "1/√d", "sqrt(d)"),
         "Attention whose scores are divided by √d before softmax, because dot products of random "
         "d-dimensional vectors have variance d and would otherwise saturate the softmax.",
         "15", (), "Vaswani et al., 2017"),
    Term("scaling laws", ("scaling law",),
         "Empirical power laws relating a model's loss to its parameter count, data and compute. They let "
         "you predict how much bigger and better a model gets for a given budget.",
         "00b", ("42",), "Kaplan et al., 2020; Hoffmann et al., 2022"),
    Term("seatbelt", ("the seatbelt", "auto-stop"),
         "The course's AWS safety rails: budgets, quotas, automatic stop of idle instances, and never "
         "applying infrastructure changes without the owner's go-ahead.",
         "25", (), ""),
    Term("selective scan", ("selective SSM",),
         "Mamba's core: a selective, input-dependent SSM recurrence computed with a scan rather than a "
         "convolution.",
         "36", (), "Gu & Dao, 2023"),
    Term("selectivity", ("input-dependent",),
         "Making an SSM's step size and B, C matrices depend on the current input, so it can choose what "
         "to store or ignore. Breaks time-invariance (no more convolution) but enables content-based "
         "recall.",
         "36", (), "Gu & Dao, 2023"),
    Term("self-attention", ("self attention",),
         "Attention in which queries, keys and values all come from the same sequence, so tokens attend to "
         "each other.",
         "15", ("31",), "Vaswani et al., 2017"),
    Term("self-consistency", ("majority vote", "majority voting", "majority_vote"),
         "Sample several reasoning chains and take the most common final answer.",
         "26", (), "Wang et al., 2022"),
    Term("self-play", ("self play",),
         "Training a game agent on games it plays against itself, generating its own ever-harder data.",
         "34", (), "Samuel, 1959; Tesauro, 1992"),
    Term("self-refinement", ("self-refine", "self-correction"),
         "Asking a model to critique and revise its own answer.",
         "26", (), "Madaan et al., 2023"),
    Term("semi-autoregressive sampling", ("block diffusion", "BD3-LM", "block_order"),
         "Generate in blocks left to right, filling each block by diffusion: a middle ground between "
         "autoregressive and fully parallel decoding.",
         "45", (), "Arriola et al., 2025; Nie et al., 2025"),
    Term("seq2seq", ("sequence to sequence", "sequence-to-sequence", "encoder–decoder"),
         "An encoder RNN reads the input into one fixed-size vector and a decoder RNN generates the output "
         "from it; the 2014 recipe for translation. That single vector is a bottleneck, which attention "
         "removed.",
         "31", ("00b",), "Sutskever, Vinyals, Le, 2014"),
    Term("shape", ("ndim",),
         "The tuple of an array's sizes along each axis, e.g. (batch, seq, d). Most deep-learning bugs are "
         "shape bugs, so reading shapes is the first skill of the course.",
         "01", ("06", "07", "12", "17", "23"), ""),
    Term("shard_map", ("shmap",),
         "A JAX transformation in which you write the per-device program and its communication explicitly, "
         "instead of letting the compiler partition a global one.",
         "24", (), ""),
    Term("sharding", ("sharded", "shard", "shards", "NamedSharding"),
         "How an array is split (or replicated) across the devices of a mesh. With sharded inputs, jit "
         "produces a program that runs on all of them.",
         "24", ("25",), ""),
    Term("shared experts", ("shared expert", "routed experts", "routed expert"),
         "DeepSeekMoE experts that every token always uses, alongside the routed ones the router chooses; "
         "they hold common knowledge.",
         "17", (), "Dai et al., 2024 (DeepSeekMoE)"),
    Term("short-term memory", ("sliding window", "running summary", "conversation history"),
         "What an agent keeps of the current conversation: the full history, a sliding window of recent "
         "turns, a running summary or a structured ledger.",
         "39", ("21",), ""),
    Term("SigLIP", ("sigmoid loss", "siglip_loss", "SigLIP 2"),
         "A CLIP-style image-text model trained with a sigmoid loss: each image-text pair in the batch is its "
         "own yes/no question, with no softmax across the batch. The vision encoder of most open VLMs.",
         "43", (), "Zhai et al., 2023"),
    Term("sigmoid", ("logistic function",),
         "σ(x) = 1/(1 + e⁻ˣ): squashes any number into (0, 1). The two-class special case of softmax.",
         "09", ("10", "33"), ""),
    Term("SIMD", ("SIMDs",),
         "Single instruction, multiple data: a hardware unit applying one instruction to a whole wavefront "
         "at once.",
         "23", (), ""),
    Term("skip-gram", ("skipgram", "skip_gram"),
         "word2vec's objective: from a centre word, predict the words within a small window around it.",
         "30", (), "Mikolov et al., 2013"),
    Term("sliding-window attention", ("sliding window",),
         "Each token attends only to the previous W tokens, so compute and the KV cache grow with W "
         "instead of the context length.",
         "44", ("00b",), "Beltagy, Peters, Cohan, 2020 (Longformer)"),
    Term("SmolLM2", ("SmolLM2-135M", "SmolLM2-360M", "SmolLM"),
         "Hugging Face's small open LLMs (135M–1.7B); the real pretrained weights the course loads for "
         "inference, LoRA, evaluation, RAG and agents.",
         "18", ("20", "21", "27", "35", "38"), "Allal et al., 2025"),
    Term("smoothing", ("add-one smoothing", "Laplace smoothing"),
         "Adding a small count to every possibility so unseen pairs don't get probability zero (and "
         "infinite loss).",
         "14", (), ""),
    Term("softmax", (),
         "Exponentiate a vector of logits and divide by the total, turning arbitrary scores into a "
         "probability distribution. Large logits dominate.",
         "04", ("08", "15", "18", "22", "23"), ""),
    Term("softmax regression", ("multinomial logistic regression",),
         "Logistic regression for many classes: one linear score per class, then softmax and "
         "cross-entropy.",
         "09", ("10",), ""),
    Term("sparse activation", ("active parameters", "sparse compute", "dense vs sparse"),
         "Only part of the model runs per token (the chosen experts), so active parameters are far fewer "
         "than total parameters.",
         "17", (), ""),
    Term("sparse autoencoder", ("sparse autoencoders", "SAE", "SAEs"),
         "An autoencoder with a sparsity penalty trained on a model's activations, to pull superposed "
         "features apart into separate, more interpretable units.",
         "37", (), "Bricken et al., 2023"),
    Term("sparse retrieval", ("lexical search", "keyword search"),
         "Search by exact word overlap, using sparse vectors with one dimension per vocabulary word "
         "(TF-IDF, BM25).",
         "38", (), ""),
    Term("special tokens", ("special token", "BOS", "EOS", "end-of-text", "<|endoftext|>"),
         "Reserved tokens with structural meaning, like end-of-text or chat-role markers, that never come "
         "from ordinary text.",
         "13", (), ""),
    Term("speculative decoding", ("speculative sampling", "draft model", "drafter"),
         "A small draft model guesses several tokens, the big target model checks them all in one forward "
         "pass, and every accepted guess is a token gained. The output distribution is unchanged.",
         "18", ("17", "27"), "Leviathan et al., 2022; Chen et al., 2023"),
    Term("speed ladder", (),
         "Module 22's sequence of changes, each timed on the same model: eager → jit → mixed precision → "
         "donation → scanned steps.",
         "22", (), ""),
    Term("SPMD", ("SPMD partitioner", "single program, multiple data"),
         "Single program, multiple data: every device runs the same program on its own shard. XLA's SPMD "
         "partitioner rewrites one global program into that form, inserting communication.",
         "24", (), ""),
    Term("spurious memories", ("spurious memory", "spurious states"),
         "Stable states of a Hopfield network that weren't stored, such as mixtures or negatives of stored "
         "patterns.",
         "29", (), ""),
    Term("state-space model", ("SSM", "SSMs", "state space model", "state-space models"),
         "A sequence model that keeps a fixed-size state updated linearly at each step, h_t = A h_{t−1} + "
         "B x_t, y_t = C h_t. Runs as a recurrence, a convolution or a scan.",
         "36", (), "Kalman, 1960; Gu et al., 2021"),
    Term("static argument", ("static_argnums", "static_argnames"),
         "A jit argument treated as a compile-time constant rather than traced. Python control flow may "
         "depend on it, but each new value triggers a recompile.",
         "06", ("08",), ""),
    Term("step bubble", (),
         "Idle GPU time between training steps while the host does Python work or waits. Scanning many "
         "steps per call removes it.",
         "22", (), ""),
    Term("stochastic gradient descent", ("SGD", "stochastic gradient", "stochastic gradients"),
         "Gradient descent on a random minibatch instead of all the data each step: noisier but far "
         "cheaper, and the noise can even help.",
         "05", (), "Robbins & Monro, 1951"),
    Term("stop_gradient", ("stop gradient", "lax.stop_gradient"),
         "Marks a value as a constant for differentiation: it passes through forward unchanged, but no "
         "gradient flows back through it. Used for targets, frozen parts and bias updates that shouldn't "
         "be learned by backprop.",
         "17", (), ""),
    Term("streaming", ("server-sent events",),
         "Sending generated tokens to the client as they are produced rather than all at the end.",
         "19", ("23",), ""),
    Term("stride", ("strides",),
         "How far a convolution or pooling window moves between positions; stride 2 halves the output "
         "size.",
         "12", (), ""),
    Term("structured output", ("json_schema", "JSON schema", "JSON mode", "structured outputs"),
         "Constraining a model's output to match a JSON schema, typically enforced by constrained decoding "
         "on the server.",
         "27", (), ""),
    Term("subsampling", ("frequent-word subsampling",),
         "Randomly dropping very frequent words (the, of) from training pairs so rarer words get more "
         "updates.",
         "30", (), "Mikolov et al., 2013"),
    Term("subword", ("subwords", "subword tokenization"),
         "A token that is part of a word: common words stay whole while rare ones split into pieces, "
         "balancing vocabulary size and sequence length.",
         "13", (), ""),
    Term("superposition", (),
         "Networks representing more features than they have dimensions by overlapping them in nearly "
         "orthogonal directions, which works when features are sparse.",
         "37", (), "Elhage et al., 2022"),
    Term("supervised fine-tuning", ("SFT",),
         "The first post-training step: fine-tune on demonstration responses by ordinary next-token "
         "prediction.",
         "35", ("25", "26"), "Ouyang et al., 2022"),
    Term("surprise", ("surprisal", "self-information"),
         "−log p of an outcome: rare events are very surprising, certain ones not at all. Entropy is the "
         "expected surprise.",
         "04", ("21",), "Shannon, 1948"),
    Term("SVD", ("singular value decomposition", "singular values", "singular value"),
         "Any matrix factors as rotate, scale along axes, rotate: A = U Σ Vᵀ. The scale factors (singular "
         "values) rank the directions by importance.",
         "02", ("05", "30"), ""),
    Term("SwiGLU", ("GLU", "gated linear unit", "SiLU", "swish"),
         "A gated MLP: one projection passed through SiLU (swish) multiplies another elementwise before "
         "the down-projection. Used by Llama and DeepSeek in place of a GELU MLP.",
         "17", ("10",), "Shazeer, 2020"),
    Term("symbolic AI", ("rules and symbols", "GOFAI"),
         "The tradition of building intelligence from explicit rules, logic and symbols, as opposed to "
         "learning from examples. The course's history module contrasts these two traditions.",
         "00b", ("29",), ""),
    Term("system prompt", ("system message",),
         "The first message in a chat, setting the assistant's role and rules before the user speaks.",
         "27", ("39",), ""),
    Term("tanh", ("hyperbolic tangent",),
         "A sigmoid-shaped nonlinearity with outputs in (−1, 1). Its flat tails cause vanishing gradients "
         "in deep or recurrent networks.",
         "10", ("03", "06", "07", "11", "31"), ""),
    Term("target network", (),
         "A slowly updated copy of the Q-network used to compute training targets, so the targets don't "
         "chase every update.",
         "34", (), "Mnih et al., 2015"),
    Term("task arithmetic", ("task vector", "task vectors"),
         "Treating the weight change from fine-tuning as a vector you can add, subtract and scale to "
         "combine or remove skills.",
         "20", ("46",), "Ilharco et al., 2022"),
    Term("temperature", (),
         "Divide logits by T before softmax: T < 1 sharpens the distribution toward the top choice, T > 1 "
         "flattens it, T → 0 is greedy decoding.",
         "04", ("14", "16", "18", "19", "26", "42"), ""),
    Term("temporal-difference learning", ("TD learning", "TD error", "TD-Gammon", "temporal difference"),
         "Updating value estimates toward a target built from the next step's estimate, learning from "
         "incomplete episodes. TD-Gammon used it to play backgammon in 1992.",
         "34", (), "Sutton, 1988; Tesauro, 1992"),
    Term("tensor parallelism", ("tensor parallel", "column-parallel", "row-parallel", "model parallelism"),
         "Splitting each weight matrix across devices (column- or row-wise) so each computes part of every "
         "layer, with collectives in between.",
         "24", (), "Shoeybi et al., 2019 (Megatron-LM)"),
    Term("Terraform", ("terraform apply",),
         "An infrastructure-as-code tool that creates cloud resources from configuration files; `apply` is "
         "the step that actually provisions and costs money.",
         "25", (), ""),
    Term("TF-IDF", ("term frequency", "inverse document frequency", "IDF"),
         "Scoring a term in a document by how often it appears there (term frequency) times how rare it is "
         "across documents (inverse document frequency).",
         "38", (), "Spärck Jones, 1972"),
    Term("TFLOP/s", ("TFLOPS", "FLOP/s", "FLOPs"),
         "Floating-point operations per second, in trillions: the speed unit for GPU arithmetic. A matrix "
         "multiply of (m,k)·(k,n) costs about 2·m·k·n FLOPs, so time × TFLOP/s tells you whether the card "
         "is busy.",
         "00", ("06", "16", "18", "22", "25"), ""),
    Term("throughput", ("tokens per second", "tokens/s", "tok/s"),
         "Work done per unit time, e.g. tokens per second; distinct from latency, the time for one "
         "request.",
         "18", ("06", "16", "19", "22", "25"), ""),
    Term("tiling", ("tiled",),
         "Splitting a big computation into blocks that fit in registers or LDS, so each piece of data is "
         "loaded from memory once and reused many times.",
         "23", ("24",), ""),
    Term("token", ("tokens",),
         "One unit of text as a model sees it, an integer id for a character, byte, word piece or word.",
         "13", ("16", "17", "18", "21", "32"), ""),
    Term("tokenizer", ("tokenizers", "tokenization"),
         "The component that turns text into a sequence of token ids and back. It is fixed before training "
         "and shapes cost and behaviour.",
         "13", ("16", "18", "21", "27", "32"), ""),
    Term("tool calling", ("tool call", "tool calls", "function calling", "tool use"),
         "A model emitting a structured request (tool name + JSON arguments) that the host program "
         "executes, feeding the result back as a message.",
         "27", ("25", "39"), "Schick et al., 2023 (Toolformer)"),
    Term("top-k sampling", ("top-k",),
         "Sample only from the k most probable tokens, renormalized.",
         "18", (), "Fan et al., 2018"),
    Term("top-p sampling", ("top-p", "nucleus sampling", "nucleus"),
         "Sample from the smallest set of tokens whose probabilities add up to p, so the candidate set "
         "adapts to how confident the model is.",
         "18", (), "Holtzman et al., 2019"),
    Term("TPU", ("TPUs",),
         "Google's Tensor Processing Unit, a chip built for matrix multiplies. XLA and JAX grew up "
         "targeting it, which is why JAX's programming model is compiler-first.",
         "00b", ("25",), ""),
    Term("tracer", ("tracers",),
         "The placeholder value JAX passes through your function while tracing. It knows its shape and "
         "dtype but not its numbers, which is why Python `if` on it fails.",
         "06", (), ""),
    Term("tracing", ("traced",),
         "Running a Python function with abstract stand-in values (tracers) that record each array "
         "operation instead of computing it. The recording is what JAX transforms and compiles.",
         "06", ("08",), "Frostig, Johnson, Leary, 2018"),
    Term("train/validation/test split",
         ("validation set", "test set", "training set", "held-out", "train_val_test_split"),
         "Training data fits the parameters, validation data picks hyperparameters and detects "
         "overfitting, and the test set is touched once at the end for an honest estimate.",
         "09", ("10", "13", "21", "35", "37"), ""),
    Term("Transformer", ("transformer block",),
         "The architecture of stacked blocks of self-attention plus a per-token MLP, each wrapped in a "
         "residual connection and normalization. The basis of all modern LLMs.",
         "15", ("00b", "17", "31", "32", "37"), "Vaswani et al., 2017"),
    Term("translation equivariance", (),
         "Shifting the input shifts the output the same way. Convolutions have it built in; attention is "
         "instead permutation-equivariant.",
         "12", (), ""),
    Term("Tree of Thoughts", (),
         "Search over partial chains of thought, evaluating and expanding the most promising branches "
         "rather than following one chain.",
         "26", (), "Yao et al., 2023"),
    Term("Triton", ("Triton GEMM",),
         "A Python-embedded language and compiler for GPU kernels. XLA can use it to generate its own "
         "matmul kernels; on this card some of those request more registers or shared memory than exist "
         "and the launch fails, so the course disables them.",
         "22", ("00", "23"), "Tillet et al., 2019"),
    Term("type promotion", ("weakly typed", "weak type"),
         "The rule for the result dtype when you combine arrays of different dtypes. Python scalars are "
         "weakly typed in JAX, so `x * 2.0` keeps x's dtype instead of upcasting.",
         "01", (), ""),
    Term("U-Net", ("UNet",),
         "A convolutional encoder-decoder with skip connections between matching resolutions, the standard "
         "denoiser in image diffusion.",
         "33", (), "Ronneberger et al., 2015"),
    Term("underfitting", ("underfit",),
         "A model too simple, or trained too little, to capture the pattern even on the training data.",
         "09", ("11",), ""),
    Term("universal approximation", ("universal approximation theorem",),
         "One hidden layer with enough units can approximate any continuous function. It says nothing "
         "about how many units or how to find the weights.",
         "10", (), "Cybenko, 1989; Hornik, 1991"),
    Term("unrolling", ("unroll", "unrolled"),
         "Turning a Python loop into a long straight-line program during tracing. It makes compile time "
         "grow with the loop length; lax.scan avoids it.",
         "08", (), ""),
    Term("UTF-8", ("UTF-8 bytes",),
         "The variable-length encoding of Unicode text as bytes (1 to 4 per character). Byte-level "
         "tokenizers start from these 256 values.",
         "13", ("21",), ""),
    Term("VAE", ("variational autoencoder", "VAEs"),
         "An autoencoder whose encoder outputs a Gaussian over latents; trained on the ELBO, so the latent "
         "space is smooth enough to sample new data from.",
         "33", (), "Kingma & Welling, 2013"),
    Term("value function", ("state value", "value head", "value network", "value loss"),
         "The expected return from a state under a policy. AlphaZero's network predicts it to judge "
         "positions.",
         "34", ("17",), ""),
    Term("vanishing gradients", ("vanishing gradient", "exploding gradients", "exploding gradient"),
         "Gradients shrinking (or growing) exponentially as they are multiplied back through many layers "
         "or time steps, so early layers stop learning (or blow up).",
         "31", (), "Hochreiter, 1991; Bengio et al., 1994"),
    Term("variance", (),
         "The expected squared distance of a random variable from its mean: how spread out it is. The "
         "standard deviation is its square root.",
         "04", ("10", "11", "15", "33"), ""),
    Term("vector", ("vectors",),
         "An ordered list of numbers, best pictured as an arrow in space. Embeddings, gradients and hidden "
         "states are all vectors.",
         "02", ("15", "17", "30", "32", "38"), ""),
    Term("vector database", ("vector databases", "vector store", "pgvector", "OpenSearch"),
         "A database that stores embeddings and answers nearest-neighbour queries, usually with an ANN "
         "index; pgvector and OpenSearch are two.",
         "38", ("39",), ""),
    Term("verifier", ("verifiers", "verifiable reward", "verifiable rewards"),
         "A program that checks an answer automatically (e.g. exact arithmetic). Its pass/fail becomes the "
         "reward, so no learned reward model is needed.",
         "26", ("35", "39"), ""),
    Term("VGPR", ("VGPRs", "vector registers", "register file"),
         "Vector general-purpose registers: each thread's private fast storage. A SIMD's register file is "
         "shared by its resident waves, so more VGPRs per thread means fewer waves fit.",
         "23", (), ""),
    Term("vision transformer", ("ViT", "ViTs", "vision transformers"),
         "A Transformer encoder applied to an image cut into patches, one token per patch. Weaker than a CNN "
         "on small datasets, stronger once pretrained on enough data.",
         "43", ("12",), "Dosovitskiy et al., 2020"),
    Term("vision-language model", ("VLM", "VLMs", "multimodal LLM", "projector", "connector", "LLaVA"),
         "A language model that reads images: a vision encoder's patch tokens, mapped by a small projector "
         "into the LLM's embedding space, are placed in the prompt like words (LLaVA's recipe).",
         "43", ("33",), "Liu et al., 2023"),
    Term("VJP", ("vector-Jacobian product",),
         "Vector-Jacobian product: pull a weighting of the outputs back to the inputs, computed by reverse "
         "mode. `grad` is a VJP with the vector 1.",
         "03", (), ""),
    Term("vmap", ("jax.vmap",),
         "JAX's automatic vectorization: write a function for one example, and `vmap` turns it into one "
         "that runs on a whole batch, as if the loop were written by hand.",
         "07", ("03", "04", "14", "16", "34"), ""),
    Term("vocabulary", ("vocab", "vocabulary size", "vocab size"),
         "The set of tokens a tokenizer can produce. Its size V fixes the embedding table and output "
         "layer, which cost V × d parameters each.",
         "13", ("14", "16", "17", "22", "31"), ""),
    Term("VRAM", ("GPU memory", "video memory"),
         "The GPU's own memory, 16 GB on the RX 7600 XT. Weights + KV cache + overhead must fit for "
         "full-speed inference.",
         "19", ("00",), ""),
    Term("Vulkan", (),
         "A cross-vendor graphics and compute API; llama.cpp's Vulkan backend is an alternative to "
         "ROCm/HIP and is sometimes faster at generation on this card.",
         "19", ("27",), ""),
    Term("warmup", ("learning-rate warmup", "linear warmup"),
         "Ramping the learning rate up from near zero over the first steps, so the early, badly scaled "
         "updates (and Adam's noisy statistics) don't blow the model up.",
         "05", ("11", "16"), ""),
    Term("wavefront", ("wavefronts", "wave32", "warp"),
         "A group of threads (32 on RDNA3) that execute in lock-step on one SIMD; AMD's name for a warp.",
         "23", (), ""),
    Term("WaveNet", (),
         "DeepMind's 2016 audio model built from dilated causal convolutions; an early autoregressive "
         "sequence model without recurrence.",
         "12", (), "van den Oord et al., 2016"),
    Term("weight decay", (),
         "Shrinking every weight slightly toward zero each step, which discourages large weights and "
         "reduces overfitting. Decoupled from the gradient in AdamW.",
         "05", ("11",), ""),
    Term("weight sharing", ("shared weights",),
         "Using the same parameters at every position (convolution) or every time step (RNN), so the model "
         "generalizes across position with far fewer parameters.",
         "12", (), ""),
    Term("weight tying", ("tied embeddings", "tying"),
         "Using the same matrix for the input embedding and the output projection to logits, saving V × d "
         "parameters.",
         "16", (), "Press & Wolf, 2016"),
    Term("weight-space programming", ("WSP",),
         "An LLM reading and editing another network's raw weights by reasoning, with a few forward passes and no "
         "gradients (Takehi et al., 2026); later work had models edit their own weights.",
         "46", (), "Takehi et al., 2026"),
    Term("win rate", ("win_rate", "pairwise accuracy", "pairwise_accuracy"),
         "The fraction of comparisons in which one model's answer is preferred over another's (by people, "
         "a judge or a reward model).",
         "35", (), ""),
    Term("word analogy", ("analogy", "analogies"),
         'Solving "a is to b as c is to ?" by vector arithmetic b − a + c and finding the nearest word.',
         "30", (), "Mikolov et al., 2013"),
    Term("word2vec", (),
         "Mikolov's 2013 method for learning word embeddings by training a shallow model to predict "
         "context words; it showed vectors capture meaning (king − man + woman ≈ queen).",
         "30", ("00b", "32"), "Mikolov et al., 2013"),
    Term("workflow", ("workflows",),
         "LLMs and tools orchestrated through predefined code paths, as opposed to an agent that directs "
         "itself. Anthropic's five patterns are prompt chaining, routing, parallelization, "
         "orchestrator-workers and evaluator-optimizer.",
         "39", (), "Anthropic, 2024 (Building effective agents)"),
    Term("workgroup", ("workgroups", "thread block"),
         "Threads that run on one compute unit and can share LDS and synchronize; at most 1024 threads.",
         "23", ("22",), ""),
    Term("world model", ("world models", "latent structure"),
         "An internal representation of the state of the world behind the data (like the board behind the "
         "moves) that a model builds and uses.",
         "37", (), ""),
    Term("Xavier initialization", ("Glorot", "Xavier"),
         "Initial weight variance 2/(fan_in + fan_out), designed for tanh-like activations.",
         "11", (), "Glorot & Bengio, 2010"),
    Term("XLA", ("Accelerated Linear Algebra",),
         "The compiler underneath JAX: it takes the traced graph of array operations, fuses and optimizes "
         "it, and generates GPU kernels.",
         "06", ("00", "08", "16", "22", "23"), ""),
    Term("XLA dump", ("xla_dump_to", "dumping"),
         "Asking XLA to write its intermediate programs (before and after optimization) to files so you "
         "can see what it fused.",
         "22", (), ""),
    Term("XLA_FLAGS", ("xla_gpu_enable_triton_gemm",),
         "Environment variable that passes options to the XLA compiler. This course sets "
         "`--xla_gpu_enable_triton_gemm=false` so matrix multiplies go to AMD's tuned libraries instead of "
         "Triton-generated kernels that crash this card at full size.",
         "00", ("22",), ""),
    Term("XOR", (),
         "The exclusive-or function: true when exactly one input is true. No straight line separates its "
         "classes, the classic thing a single linear layer or perceptron cannot learn.",
         "10", ("29",), "Minsky & Papert, 1969"),
    Term("ε-greedy", ("epsilon-greedy", "ε-greedily"),
         "Take a random action with probability ε, otherwise the best-known one: the simplest balance of "
         "exploration and exploitation.",
         "34", (), ""),
    Term("capacity factor", ("capacity factors",),
         "How many token slots an expert (or a device's experts) accepts per batch, as a multiple of the even "
         "share. Slots past capacity are dropped; 1.25 leaves room for uneven routing.",
         "40", ("24",), "Lepikhin et al., 2020 (GShard)"),
    Term("conditional memory", ("Engram", "Engram module", "n-gram memory"),
         "A lookup table read at a few layers, indexed by the last few tokens (hashed) and gated by the hidden "
         "state: knowledge fetched in O(1) instead of rebuilt by computation. DeepSeek's Engram.",
         "40", (), "Cheng et al. (DeepSeek-AI), 2026"),
    Term("hashing trick", ("hash embeddings", "feature hashing", "hashed n-gram", "hash collision"),
         "Map a huge or open-ended set of keys (words, n-grams) to rows of a fixed-size table with a hash "
         "function, accepting that some keys share a row. Several independent hashes make a shared row harmless.",
         "40", (), "Weinberger et al., 2009"),
    Term("LatentMoE", ("latent experts", "ℓ-MoE", "latent MoE"),
         "A mixture-of-experts layer whose routed experts work in a narrower latent width ℓ = d/α: tokens are "
         "projected down before they're sent to experts and up after. The savings buy α× as many experts.",
         "40", (), "Elango et al. (NVIDIA), 2026"),
    Term("pinned memory", ("pinned_host", "page-locked memory", "pinned host memory"),
         "Host RAM the operating system promises not to move, so the GPU can copy to and from it directly "
         "(DMA) without a staging copy. JAX exposes it as the memory kind `pinned_host`.",
         "40", (), ""),
    Term("prefetch", ("prefetching", "prefetched"),
         "Start loading data you'll need next while the current work runs, so the wait hides behind it. "
         "Possible whenever the addresses are known in advance.",
         "40", (), ""),
    Term("sparsity", ("sparse model", "sparse parameters"),
         "Having many parameters but using only a few for each input: an MoE's unpicked experts, an Engram "
         "table's unread rows. Capacity grows without per-token compute growing.",
         "40", ("17",), ""),
    Term("tokenizer compression", ("canonical ids", "vocabulary compression", "compressed tokenizer"),
         "Map token ids that mean the same thing after normalization (case, accents, spaces) to one canonical "
         "id. Engram indexes its tables this way, so variants share rows.",
         "40", ("13",), "Cheng et al. (DeepSeek-AI), 2026"),
    Term("decay mask", ("segment sum", "segsum"),
         "The lower-triangular matrix of products a_{s+1}···a_t that turns decayed linear attention into masked "
         "attention, (QKᵀ ⊙ L)V. Built in log space from differences of a cumulative sum.",
         "41", ("36",), "Dao & Gu, 2024"),
    Term("delta rule", ("Widrow–Hoff rule", "delta-rule"),
         "Update a memory by writing only the error: S ← S + β k (v − Sᵀk)ᵀ, one gradient step on ‖Sᵀk − v‖². "
         "In a linear-attention layer it overwrites a key's old value instead of adding to it.",
         "41", (), "Widrow & Hoff, 1960; Schlag et al., 2021"),
    Term("fast weights", ("fast-weight", "fast weight programmer"),
         "A weight matrix written by the network itself while it reads its input, rather than by training. A "
         "linear-attention state is one: a key→value map built from the tokens seen so far.",
         "41", (), "Schmidhuber, 1992"),
    Term("gated attention", (),
         "Softmax attention whose output is multiplied by a sigmoid gate computed from the layer input; used "
         "in Qwen3-Next's attention layers.",
         "41", (), "Qiu et al., 2025"),
    Term("Gated DeltaNet", ("GDN", "gated delta rule", "Gated Delta Networks"),
         "A linear-attention layer combining Mamba-2's per-step decay with the delta rule: "
         "S_t = a_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ. Three of every four layers of Qwen3-Next and Qwen3.5.",
         "41", (), "Yang, Kautz & Hatamizadeh, 2025"),
    Term("linear attention", ("linear transformer", "linear-attention"),
         "Attention without the softmax: o_t = Σ_{s≤t} (q_t·k_s) v_s, which regroups into a recurrence on a fixed "
         "d_k × d_v state, S_t = S_{t−1} + k_t v_tᵀ. Constant memory and time per generated token.",
         "41", ("36",), "Katharopoulos et al., 2020"),
    Term("state space duality", ("SSD",),
         "Mamba-2's observation that a selective SSM with a scalar decay is decayed linear attention: the same "
         "function computed as a recurrence or as masked attention, and fastest as chunks of both.",
         "41", ("36",), "Dao & Gu, 2024"),
    Term("WY representation", ("UT transform",),
         "A compact form for a product of (I − β k kᵀ) factors. It lets the delta rule run chunk by chunk with "
         "one triangular solve and a few matmuls per chunk.",
         "41", (), "Yang et al., 2024 (DeltaNet)"),
    Term("block influence", ("BI",),
         "How much a layer changes its input: 1 − the mean cosine similarity of each token's hidden state before "
         "and after it. Layers with low block influence are the first to remove when pruning depth.",
         "42", (), "Men et al., 2024 (ShortGPT)"),
    Term("continued pretraining", ("continual pretraining", "domain-adaptive pretraining"),
         "Keep training an already-pretrained model with the same next-token objective on new data (a domain, "
         "a language), instead of starting from random weights.",
         "42", (), ""),
    Term("Newton–Schulz iteration", ("Newton-Schulz", "Newton–Schulz"),
         "A matmul-only iteration that pushes a matrix's singular values toward 1 while keeping its singular "
         "vectors: an approximate orthogonalization. Muon runs five steps of a tuned quintic version.",
         "42", ("05",), "Keller Jordan, 2024 (Muon's coefficients)"),
    Term("overtraining", ("overtrained", "inference-optimal"),
         "Training a model on many more tokens than the compute-optimal ~20 per parameter. It costs more to "
         "train but gives a better model at a size that's cheap to run; SmolLM2-135M saw ~15,000 tokens per "
         "parameter.",
         "42", (), ""),
    Term("perplexity filtering", ("perplexity filter",),
         "Choosing training data by how predictable a reference model finds it: keep the text it scores well, "
         "drop the rest. Cheap, but it favors easy, repetitive text as well as clean text.",
         "42", (), ""),
    Term("sparse upcycling", ("upcycling", "upcycled"),
         "Turn a trained dense model into a mixture of experts by copying its feed-forward block into every "
         "expert, adding a fresh router, and training on.",
         "42", ("17",), "Komatsuzaki et al., 2022"),
    Term("structured pruning", ("pruning", "width pruning", "depth pruning", "pruned"),
         "Make a trained model smaller by removing whole pieces — layers, attention heads, feed-forward neurons, "
         "embedding channels — chosen by an importance score, then retrain briefly (ideally by distillation).",
         "42", (), "Muralidharan et al., 2024 (Minitron)"),
    Term("µP", ("muP", "maximal update parametrization", "µTransfer"),
         "A way to scale initialization and learning rates with width so the best hyperparameters found on a "
         "narrow model still work on a wide one.",
         "42", (), "Yang et al., 2022"),
]


