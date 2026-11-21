"""Module 43 exercises: images as tokens, and teaching a language model to read them.

Replace each `raise NotImplementedError` with your code, then run `make test N=43`. models.py builds a
SigLIP vision tower on your `patchify`, scores images against captions with your `siglip_logits`, and
shrinks the image tokens for the language model with your `pixel_shuffle`.

Conventions: images are (B, H, W, C), channels last. A square grid of g × g patches is flattened row by
row into N = g² tokens.
"""
import jax  # noqa: F401  (for your code)
import jax.numpy as jnp  # noqa: F401


def patchify(images, p: int):
    """Cut images (B, H, W, C) into non-overlapping p × p patches and flatten each one: (B, N, p·p·C) with
    N = (H/p)·(W/p). Patches in row-major grid order (left to right, then top to bottom); inside a patch,
    pixels row-major with their C channels last, i.e. the same order as images[b, i:i+p, j:j+p].reshape(-1).

    (A ViT's "patch embedding" is then one matrix multiply: a p × p convolution with stride p.)
    """
    raise NotImplementedError


def resize_pos_embed(pos, new_grid: int):
    """A ViT trained at one resolution has one learned position embedding per patch, pos (g·g, d). To run
    it on a g' × g' grid, view pos as a (g, g, d) image, resize it bilinearly to (g', g', d)
    (jax.image.resize, method "bilinear") and flatten back: (g'·g', d).
    """
    raise NotImplementedError


def siglip_logits(img, txt, log_t, b):
    """SigLIP's pairwise scores: L2-normalize each row of img (N, d) and txt (M, d), then
    logits[i, j] = exp(log_t) · img_i · txt_j + b, shape (N, M). log_t and b are learned scalars.
    """
    raise NotImplementedError


def siglip_loss(logits):
    """SigLIP's sigmoid loss (Zhai et al. 2023) for a batch of n matching pairs, logits (n, n) where the
    diagonal pairs match: every (image, text) pair is its own yes/no question,

        loss = −(1/n) Σ_i Σ_j log σ(z_ij · logits_ij),   z_ij = +1 if i = j else −1.

    No softmax over the batch (CLIP's), so no all-to-all normalization. Use jax.nn.log_sigmoid.
    """
    raise NotImplementedError


def pixel_shuffle(x, r: int):
    """Trade image tokens for width (SmolVLM, Idefics3): x (B, N, d) is a g × g grid of tokens (N = g²,
    g divisible by r). Merge each r × r neighbourhood into one token by concatenating the r² tokens' features,
    in row-major order within the neighbourhood: (B, N / r², r²·d). Output tokens are row-major over the
    new (g/r) × (g/r) grid.
    """
    raise NotImplementedError
