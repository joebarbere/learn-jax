"""Module 42 exercises: what makes a small model good — budgets, distillation, pruning, and Muon.

Replace each `raise NotImplementedError` with your code, then run `make test N=42`. The lesson trains students
with your loss, prunes a real SmolLM2 checkpoint with your scores, and optimizes with your Newton–Schulz.

Model parameters follow learnjax/llama.py: layer weights are stacked along a leading layer axis, e.g. the
feed-forward weights w_gate and w_up are (L, D, F) and w_down is (L, F, D), used as
x + (silu(h @ w_gate) * (h @ w_up)) @ w_down.
"""
import jax  # noqa: F401  (for your code)
import jax.numpy as jnp  # noqa: F401


# ---------------------------------------------------------------- budgets
def training_cost(n_params: float, n_tokens: float, tflops: float, mfu: float = 0.4) -> dict:
    """What training a dense model costs, by the standard estimate: 6 FLOPs per parameter per token (2 for the
    forward pass, 4 for the backward).

    tflops: the GPU's peak (dense bf16) throughput in TFLOP/s; mfu: the fraction of it you actually get.
    Return a dict of floats:
      "flops"              6 · n_params · n_tokens
      "hours"              flops / (tflops · 1e12 · mfu), in hours
      "tokens_per_param"   n_tokens / n_params
      "chinchilla_tokens"  20 · n_params (the compute-optimal rule of thumb from the Chinchilla paper)
    """
    raise NotImplementedError


# ---------------------------------------------------------------- distillation
def kd_loss(student_logits, top_ids, top_logp, T: float = 1.0):
    """Knowledge distillation from a teacher's top-k predictions (Hinton et al. 2015, sparse version).

    student_logits: (..., V). top_ids: (..., k) int, the teacher's k most likely next tokens; top_logp:
    (..., k) the teacher's log-probabilities for them (normalized over the full vocabulary, at T = 1).

      p = softmax(top_logp / T) over the k entries          the teacher, softened and renormalized
      log q = log_softmax(student_logits / T) over all V     the student, softened
      loss = T² · mean over positions of Σ_k p_k (log p_k − log q[top_ids_k])

    The T² keeps gradients the same size as you change T (Hinton et al.). Return a scalar. Note the minimum
    isn't 0: the student is also pushed to drop the mass the teacher put outside its top k.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- pruning
def ffn_importance(acts):
    """Minitron-style importance of each feed-forward neuron: the mean absolute activation over a calibration
    set. acts: (..., F), the hidden activations silu(h @ w_gate) * (h @ w_up). Returns (F,).
    """
    raise NotImplementedError


def prune_ffn(layers, keep):
    """Keep only some feed-forward neurons in every layer.

    layers: a dict of stacked layer weights including "w_gate" (L, D, F), "w_up" (L, D, F) and "w_down"
    (L, F, D); keep: (L, k) int, the neurons to keep in each layer. Return a new dict with those three
    sliced to (L, D, k), (L, D, k), (L, k, D) and every other entry unchanged. Hint: jnp.take_along_axis.
    """
    raise NotImplementedError


def layer_importance(h_in, h_out):
    """How much a whole layer changes its input ("block influence", Men et al. 2024): 1 − the mean cosine
    similarity between each token's hidden state before (h_in) and after (h_out) the layer. Both (..., D).
    A layer whose output points where its input did does little, and is the first to drop. Returns a scalar.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- Muon
def newton_schulz(G, steps: int = 5):
    """Approximately orthogonalize a matrix the way Muon does (Keller Jordan, 2024): push all its singular
    values toward 1 while keeping its singular vectors, using only matmuls.

      X = G / ‖G‖_F                      (so every singular value is ≤ 1); work with the wide orientation:
                                         if G has more rows than columns, use G.T and transpose back at the end
      repeat `steps` times:  A = X Xᵀ;  X = a X + (b A + c A A) X      with (a, b, c) = (3.4445, −4.7750, 2.0315)

    Returns an array of G's shape, in float32.
    """
    raise NotImplementedError
