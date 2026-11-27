"""Module 31b exercises: a recurrent network that sweeps an image in two dimensions, from all four corners.

Replace each `raise NotImplementedError` with your code, then run `make test N=51` (this module's registry
number; its label is 31b). Shapes: a batch of images as a grid of cells, `x` (B, R, C, F): B images, R rows,
C columns, F features per cell. A sweep's parameters are a dict
`p = {"wx": (F, H), "wu": (H, H), "wl": (H, H), "b": (H,)}`.
"""


def mdrnn_sweep_naive(p: dict, x):
    """One 2-D sweep from the top-left corner, cell by cell (Graves, Fernández & Schmidhuber 2007).

    h[i, j] = tanh(x[i, j] @ wx + h[i-1, j] @ wu + h[i, j-1] @ wl + b), where a neighbour outside the grid
    (row -1 or column -1) counts as zeros. x: (B, R, C, F) -> h: (B, R, C, H), same dtype as x.

    Visit the cells in an order where (i-1, j) and (i, j-1) are always done before (i, j): row by row,
    left to right. Plain Python loops over i and j are fine (it's the reference, not the fast version);
    collect the rows in a list and `jnp.stack` them, since JAX arrays can't be assigned in place.
    """
    raise NotImplementedError


def skew(x):
    """Shift row i of a grid right by i cells, padding with zeros. (B, R, C, H) -> (B, R, C + R - 1, H).

    After skewing, a cell's up neighbour (i-1, j) and left neighbour (i, j-1) both sit in the column just
    before it, so a whole skewed column can be computed at once from the previous one: the cells on one
    anti-diagonal of the original grid. Output[b, i, i + j] = x[b, i, j]; every other entry is 0.
    """
    raise NotImplementedError


def unskew(s, C: int):
    """Undo `skew`: (B, R, C + R - 1, H) -> (B, R, C, H), taking out[b, i, j] = s[b, i, i + j]."""
    raise NotImplementedError


def mdrnn_sweep(p: dict, x):
    """The same sweep as `mdrnn_sweep_naive`, as a diagonal wavefront: R + C - 1 sequential steps, not R * C.

    1. Project every cell's input at once: z = x @ wx + b, shape (B, R, C, H).
    2. `skew` it to (B, R, W, H), W = C + R - 1, and run `jax.lax.scan` over the W skewed columns. The carry
       is the previous skewed column h_prev (B, R, H). Row i's up neighbour is row i-1 of h_prev (zeros for
       row 0) and its left neighbour is row i of h_prev, so
           h = tanh(z_col + up @ wu + h_prev @ wl)
    3. Cells the skew padded with zeros aren't real cells: multiply h by a 0/1 mask so they stay exactly 0
       (otherwise tanh(b) would leak into the next column). The mask is `skew` applied to ones.
    4. Stack the scanned columns back to (B, R, W, H) and `unskew` to (B, R, C, H).
    Must match `mdrnn_sweep_naive` to float32 rounding, and work under jax.jit and jax.grad.
    """
    raise NotImplementedError


def four_direction_sweeps(params: list, x, sweep):
    """Sweep from all four corners and line the results up. Returns (B, R, C, 4H).

    params: a list of 1 or 4 sweep-parameter dicts. With 4, direction k uses params[k] (Graves' MDRNN: a
    separate hidden layer per corner). With 1, all four directions use params[0]: one set of weights, four
    directions. `sweep(p, x)` is a top-left sweep such as `mdrnn_sweep`.

    Each corner becomes the top-left by flipping the grid, in this order:
        k=0 top-left: x            k=1 bottom-left: flip rows    (axis 1)
        k=2 top-right: flip columns (axis 2)                     k=3 bottom-right: flip both
    Sweep each flipped view, flip each result back the same way so cell (i, j) means the same pixel in
    all four, and concatenate along the last axis in order k=0..3. Then every cell has seen the whole image.
    """
    raise NotImplementedError


def cross_scan(x):
    """VMamba's cross-scan (Liu et al. 2024): four 1-D orders of a grid. (B, R, C, F) -> (4, B, R*C, F).

    k=0 row-major (row 0 left to right, then row 1, ...), k=1 column-major (column 0 top to bottom, ...),
    k=2 the reverse of k=0, k=3 the reverse of k=1. A 1-D recurrence run along each order sees the grid from
    four sides, and each is an ordinary sequence, so it can use a parallel scan.
    """
    raise NotImplementedError


def cross_merge(ys, R: int, C: int):
    """Inverse of `cross_scan`, summing the four: (4, B, R*C, H) -> (B, R, C, H).

    Put each sequence back on the grid (undo the reversal for k=2 and k=3, the transpose for k=1 and k=3)
    and add the four grids. cross_merge(cross_scan(x), R, C) == 4 * x.
    """
    raise NotImplementedError
