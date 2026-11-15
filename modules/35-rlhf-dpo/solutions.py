"""Module 35 reference solutions. Try the exercises first — `make test N=35` grades yours."""
import jax
import jax.numpy as jnp
import numpy as np


def bradley_terry_loss(r_chosen, r_rejected):
    return jnp.mean(-jax.nn.log_sigmoid(r_chosen - r_rejected))


def pairwise_accuracy(r_chosen, r_rejected):
    return float(np.mean(np.asarray(r_chosen) > np.asarray(r_rejected)))


def sequence_logprob(logits, tokens, mask):
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    tok_lp = jnp.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]
    return jnp.sum(tok_lp * mask, axis=-1)


def implicit_reward(logp_policy, logp_ref, beta):
    return beta * (logp_policy - logp_ref)


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    margin = implicit_reward(pi_chosen, ref_chosen, beta) - implicit_reward(pi_rejected, ref_rejected, beta)
    return jnp.mean(-jax.nn.log_sigmoid(margin))


def win_rate(scores_a, scores_b):
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    return float(np.mean((a > b) + 0.5 * (a == b)))


def best_of_n(scores, n):
    s = np.asarray(scores)
    return np.argmax(s[:, :n], axis=1)
