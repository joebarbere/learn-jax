"""learnjax.interp on a tiny random Llama-shaped model (float32): capture, edits, lenses and SAE pieces."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from learnjax import interp, llama

CFG = llama.Config(vocab=50, d_model=16, n_layers=3, n_heads=4, n_kv_heads=2, d_ff=24, dtype="float32")


def _params(key=0):
    ks = iter(jax.random.split(jax.random.key(key), 20))
    L, d, f, hd = CFG.n_layers, CFG.d_model, CFG.d_ff, CFG.head_dim
    n = lambda *s: jax.random.normal(next(ks), s) * 0.3
    return {"embed": n(CFG.vocab, d), "final_norm": 1 + n(d),
            "layers": {"attn_norm": 1 + n(L, d), "wq": n(L, d, CFG.n_heads * hd), "wk": n(L, d, CFG.n_kv_heads * hd),
                       "wv": n(L, d, CFG.n_kv_heads * hd), "wo": n(L, CFG.n_heads * hd, d), "mlp_norm": 1 + n(L, d),
                       "w_gate": n(L, d, f), "w_up": n(L, d, f), "w_down": n(L, f, d)}}


P = _params()
TOK = jax.random.randint(jax.random.key(1), (2, 7), 0, CFG.vocab)


def test_forward_matches_llama():
    out = interp.forward(P, CFG, TOK, collect=("resid", "mid", "mlp_out"))
    np.testing.assert_allclose(out["logits"], llama.forward(P, TOK, CFG)[0], atol=1e-5)
    assert out["resid"].shape == (CFG.n_layers + 1, 2, 7, CFG.d_model)
    np.testing.assert_allclose(out["resid"][0], P["embed"][TOK])
    np.testing.assert_allclose(out["resid"][1:], out["mid"] + out["mlp_out"], atol=1e-6)
    np.testing.assert_allclose(interp.unembed(P, CFG, out["resid"][-1]), out["logits"], atol=1e-5)


def test_identity_edits_are_noops():
    a = interp.forward(P, CFG, TOK)["logits"]
    b = interp.forward(P, CFG, TOK, edit=lambda i, x: x, edit_mid=lambda i, x: x)["logits"]
    np.testing.assert_array_equal(a, b)


def test_edit_at_layer_only_changes_what_follows():
    v = jnp.ones(CFG.d_model)
    base = interp.forward(P, CFG, TOK, collect=True)["resid"]
    out = interp.forward(P, CFG, TOK, edit=interp.at_layer(2, lambda x: x + v), collect=True)["resid"]
    np.testing.assert_allclose(out[:2], base[:2])
    np.testing.assert_allclose(out[2], base[2] + v, atol=1e-5)
    assert not np.allclose(out[3], base[3])


def test_upto_stops_early():
    out = interp.forward(P, CFG, TOK, collect=True, upto=2)
    assert "logits" not in out and out["resid"].shape[0] == 3
    np.testing.assert_allclose(out["resid"], interp.forward(P, CFG, TOK, collect=True)["resid"][:3], atol=1e-6)


def test_tuned_lens_at_init_is_logit_lens():
    lens = interp.init_tuned_lens(CFG)
    h = interp.forward(P, CFG, TOK, collect=True)["resid"][1]
    np.testing.assert_allclose(interp.tuned_lens_logits(P, CFG, lens, 1, h), interp.unembed(P, CFG, h), atol=1e-6)


def test_kl_and_loss():
    lg = jax.random.normal(jax.random.key(2), (3, 5))
    np.testing.assert_allclose(interp.kl(lg, lg), 0.0, atol=1e-6)
    assert float(interp.kl(lg, lg[::-1]).min()) >= 0
    uniform = jnp.zeros((1, 4, CFG.vocab))
    assert float(interp.next_token_loss(uniform, TOK[:1, :4])) == pytest.approx(np.log(CFG.vocab), rel=1e-6)


def test_sae_pieces():
    sae = interp.init_sae(jax.random.key(3), 16, 64)
    np.testing.assert_allclose(jnp.linalg.norm(sae["W_dec"], axis=1), 1.0, rtol=1e-5)
    x = jax.random.normal(jax.random.key(4), (32, 16))
    z = interp.sae_encode(sae, x, "topk", 5)
    assert int((z > 0).sum(1).max()) <= 5 and float(z.min()) >= 0
    dead = jnp.zeros(64, bool).at[:10].set(True)
    (loss, m), g = jax.value_and_grad(interp.sae_loss, has_aux=True)(sae, x * interp.norm_scale(x), "topk", 5, 0.0,
                                                                      dead)
    assert np.isfinite(float(loss)) and float(m["aux"]) > 0 and float(m["l0"]) <= 5
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(g))
    _, m2 = interp.sae_loss(sae, x, "relu", 0, 1e-2, dead)
    assert float(m2["aux"]) == 0.0
    assert float(interp.fvu(x, x)) == 0.0
    assert float(jnp.mean(jnp.sum((x * interp.norm_scale(x)) ** 2, -1))) == pytest.approx(1.0, rel=1e-5)
