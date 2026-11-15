"""Module 36 exercises: state-space models, three ways to run them, and the selective scan.

Replace each `raise NotImplementedError` with your code, then run `make test N=36`.
The lesson calls these functions, so its "Your turn" cells come alive as you finish them.

Conventions (the same as the S4/Mamba papers, specialised to keep shapes small):
  * The state matrix A is DIAGONAL, stored as a vector of its N entries (real, negative = stable).
  * A single-input, single-output (SISO) SSM has A, B, C each of shape (N,) and maps a scalar
    input sequence xs (L,) to a scalar output sequence ys (L,). The hidden state h has shape (N,).
  * Discrete-time recurrence:  h_t = A_bar * h_{t-1} + B_bar * x_t,   y_t = sum(C * h_t),   h_{-1} = 0.
"""
import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401


def discretize_zoh(A, B, dt):
    """Zero-order-hold discretization of the continuous system h' = A h + B x (diagonal A).

    A, B: (N,).  dt: scalar (or broadcastable) step size.
    Holding x constant over each step of length dt gives, exactly:
        A_bar = exp(dt * A)
        B_bar = (A_bar - 1) / A * B          (elementwise, because A is diagonal)
    Returns (A_bar, B_bar), each with A's shape.
    """
    raise NotImplementedError


def ssm_scan(A_bar, B_bar, C, xs):
    """Run the discrete SISO SSM recurrently with `jax.lax.scan`.

    A_bar, B_bar, C: (N,).   xs: (L,).   Start from h = 0.
    Each step: h = A_bar * h + B_bar * x_t ;  y_t = sum(C * h).
    Returns ys, shape (L,).
    """
    raise NotImplementedError


def ssm_kernel(A_bar, B_bar, C, L):
    """The SSM's impulse response, i.e. its convolution kernel, for L steps.

    Unrolling the recurrence from h = 0 gives y_t = sum_{k=0..t} K_k * x_{t-k} with
        K_k = sum_n C_n * A_bar_n**k * B_bar_n          (k = 0 .. L-1)
    Returns K, shape (L,). Hint: compute A_bar**k for all k at once with broadcasting.
    """
    raise NotImplementedError


def ssm_conv(kernel, xs):
    """Causal convolution y_t = sum_{k=0..t} kernel_k * x_{t-k}, computed with the FFT.

    kernel, xs: (L,). Zero-pad both to length 2L (so the circular FFT convolution doesn't wrap
    around), multiply in frequency space (jnp.fft.rfft / irfft), and keep the first L outputs.
    Returns ys, shape (L,).
    """
    raise NotImplementedError


def linear_recurrence_assoc(a, b):
    """Solve h_t = a_t * h_{t-1} + b_t (h_{-1} = 0) along axis 0 with `jax.lax.associative_scan`.

    a, b: (L, ...) — same shape. Returns h, shape (L, ...).

    Each step is an affine map h -> a*h + b. Composing "first (a1, b1), then (a2, b2)" gives another
    affine map:  (a1, b1) ∘ (a2, b2) = (a1 * a2,  a2 * b1 + b2).  That operator is associative, which is
    all associative_scan needs to compute every prefix in O(log L) parallel depth (Blelloch 1990).
    The second component of the prefix-combined pair at position t is exactly h_t.
    """
    raise NotImplementedError


def selective_scan(x, delta, A, B, C):
    """Reference Mamba selective scan for ONE sequence (write it with `jax.lax.scan`; clarity first).

    x:     (L, D)   input, D channels
    delta: (L, D)   per-step, per-channel step size Δ (already positive)
    A:     (D, N)   diagonal state matrix per channel (negative entries)
    B, C:  (L, N)   input-dependent: a different B_t and C_t at every step, shared across channels

    Per step t and channel d, with state h of shape (D, N):
        A_bar = exp(delta_t[:, None] * A)                     (D, N)
        Bx    = delta_t[:, None] * B_t[None, :] * x_t[:, None]  (D, N)   <- Mamba's simplified B_bar ≈ Δ·B
        h     = A_bar * h + Bx
        y_t   = h @ C_t                                        (D,)
    Returns ys, shape (L, D).
    """
    raise NotImplementedError
