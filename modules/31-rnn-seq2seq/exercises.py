"""Module 31 exercises: recurrent networks, gates, and the attention that replaced them.

Replace each `raise NotImplementedError` with your code, then run `make test N=31`.
The lesson calls these functions, so its "Your turn" cells come alive as you finish them.

Conventions used throughout:
  * A *cell* is a function cell(params, carry, x) -> new_carry. It processes ONE time step.
    The vanilla RNN and GRU carry a hidden state h; the LSTM carries a tuple (h, c).
  * Inputs may have any leading batch dimensions: x is (..., d_in), h is (..., H).
  * Sequences are TIME-MAJOR: xs has shape (T, ..., d_in), so lax.scan walks the first axis.
  * Weight matrices are used as x @ W (shape (d_in, out)), like everywhere else in the course.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def rnn_cell(params, h, x):
    """The Elman (1990) recurrent cell: h_new = tanh(x @ Wx + h @ Wh + b).

    params: {"Wx": (d_in, H), "Wh": (H, H), "b": (H,)}
    h: (..., H) previous hidden state     x: (..., d_in) this step's input
    Returns h_new, shape (..., H).
    """
    raise NotImplementedError


def lstm_cell(params, carry, x):
    """The LSTM cell (Hochreiter & Schmidhuber 1997, with the forget gate of Gers et al. 2000).

    params: {"Wx": (d_in, 4H), "Wh": (H, 4H), "b": (4H,)}
    carry: (h, c), each (..., H)          x: (..., d_in)

    z = x @ Wx + h @ Wh + b, split into four equal (..., H) chunks IN THIS ORDER: i, f, g, o.
        i = sigmoid(z_i)   input gate        f = sigmoid(z_f)   forget gate
        g = tanh(z_g)      candidate         o = sigmoid(z_o)   output gate
        c_new = f * c + i * g          <- the "constant error carousel": c is updated additively
        h_new = o * tanh(c_new)
    Returns the new carry (h_new, c_new).
    """
    raise NotImplementedError


def gru_cell(params, h, x):
    """The GRU cell (Cho et al. 2014), in the convention PyTorch and most libraries use.

    params: {"Wx": (d_in, 3H), "Wh": (H, 3H), "b": (3H,)}
    h: (..., H)     x: (..., d_in)

    Split gx = x @ Wx + b and gh = h @ Wh into three (..., H) chunks each, IN THIS ORDER: z, r, n.
        z = sigmoid(gx_z + gh_z)          update gate
        r = sigmoid(gx_r + gh_r)          reset gate
        n = tanh(gx_n + r * gh_n)         candidate (the reset gate scales the hidden-state part)
        h_new = (1 - z) * n + z * h
    Returns h_new, shape (..., H).
    """
    raise NotImplementedError


def run_rnn(cell, params, carry0, xs):
    """Run `cell` over a whole sequence with jax.lax.scan. Returns (final_carry, hs).

    xs: (T, ..., d_in), time-major.   carry0: the initial carry (an array, or (h, c) for the LSTM).
    hs: (T, ..., H) — the hidden state h after every step. When the carry is a tuple, h is its
    FIRST element (the LSTM's c is not output).
    No Python loop over time: write the step function and let lax.scan do the loop.
    """
    raise NotImplementedError


def grad_norm_by_distance(cell, params, carry0, xs, w_out):
    """How much does the final output still depend on each input? (the vanishing-gradient probe)

    Run the cell over xs (with run_rnn), read out a scalar from the FINAL hidden state,
        y = sum(h_T @ w_out)          (w_out: (H,); sum over any batch dimensions)
    and return the L2 norm of dy/dx_t for every time step t: a (T,) array.
    Index 0 is the FIRST input (furthest from the output), index T-1 the last.
    Use jax.grad with respect to xs.
    """
    raise NotImplementedError


def additive_attention(params, query, keys, values, mask=None):
    """Bahdanau (2014) additive attention. Returns (context, weights).

    params: {"Wq": (Hq, A), "Wk": (Hk, A), "v": (A,)}
    query: (B, Hq) — the decoder's state       keys: (B, S, Hk)   values: (B, S, Hv)
    mask: optional (B, S) boolean, True = this source position may be attended to.

        score[b, s] = v · tanh(query[b] @ Wq + keys[b, s] @ Wk)
        masked positions get -inf before the softmax
        weights = softmax(score) over s                      (B, S), rows sum to 1
        context[b] = sum_s weights[b, s] * values[b, s]      (B, Hv)
    You may assume every row of the mask allows at least one position.
    """
    raise NotImplementedError


def greedy_decode(step_fn, state, start_tokens, n_steps: int):
    """Generate n_steps tokens greedily from a decoder. Returns tokens, shape (B, n_steps), int32.

    step_fn(state, tokens) -> (logits, new_state): feed one token per sequence (tokens: (B,) int)
    and get the next-token logits (B, V) plus the updated decoder state (any pytree).
    Start by feeding start_tokens (B,). At each step pick argmax(logits) and feed it back in.
    Use jax.lax.scan (n_steps is a static Python int).
    """
    raise NotImplementedError
