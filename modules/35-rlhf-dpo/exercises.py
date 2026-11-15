"""Module 35 exercises: learning from preferences.

Replace each `raise NotImplementedError` with your code, then run `make test N=35`. The lesson trains a
reward model and a DPO policy with *your* functions, so the cells come alive as you finish them.

Conventions: a "pair" is (chosen, rejected) — the answer the labeler preferred and the one it didn't.
Rewards and log-probs are float arrays of shape [B] (one number per sequence) unless stated otherwise.
"""
import jax  # noqa: F401  (you'll want jax.nn.log_sigmoid / log_softmax)
import jax.numpy as jnp  # noqa: F401
import numpy as np  # noqa: F401


def bradley_terry_loss(r_chosen, r_rejected):
    """Reward-model loss under the Bradley–Terry model (1952): P(chosen ≻ rejected) = σ(r_c − r_r).

    Return the mean negative log-likelihood over the batch, −mean(log σ(r_chosen − r_rejected)).
    Use `jax.nn.log_sigmoid` — writing log(sigmoid(x)) yourself underflows for large negative x.
    """
    raise NotImplementedError


def pairwise_accuracy(r_chosen, r_rejected):
    """Fraction of pairs where the reward model ranks the chosen answer strictly higher. Python float."""
    raise NotImplementedError


def sequence_logprob(logits, tokens, mask):
    """Summed log-probability of each completion.

    logits [B, N, V] predict tokens [B, N] (logits[:, i] scores tokens[:, i]); mask [B, N] is 1 for real
    completion tokens and 0 for padding after the end. Return [B] = Σ_i mask·log softmax(logits)[token].
    Compute the log-softmax in float32.
    """
    raise NotImplementedError


def implicit_reward(logp_policy, logp_ref, beta):
    """DPO's implicit reward: β · (log π(y|x) − log π_ref(y|x)). Arrays of shape [B]."""
    raise NotImplementedError


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    """Direct Preference Optimization (Rafailov et al., 2023).

    Each argument is a [B] array of summed completion log-probs under the policy (pi_*) or the frozen
    reference (ref_*). Loss = −mean log σ( implicit_reward(chosen) − implicit_reward(rejected) ).
    It's exactly the Bradley–Terry loss with the reward model replaced by β·log(π/π_ref).
    """
    raise NotImplementedError


def win_rate(scores_a, scores_b):
    """How often A beats B, judged item by item: mean of 1 (a > b), 0.5 (tie), 0 (a < b). Python float."""
    raise NotImplementedError


def best_of_n(scores, n):
    """Rejection sampling ("best-of-n"): scores is [items, K] reward-model scores for K samples per item.

    Using only the first `n` samples of each item, return the index (0..n-1) of the highest-scoring one,
    shape [items] (ties → the earliest, like np.argmax).
    """
    raise NotImplementedError
