"""Tests for module 39: `make test N=39` (your exercises) or LEARNJAX_SOLUTIONS=1 (the reference)."""
from pathlib import Path

import numpy as np
import pytest

from learnjax.course import load_exercises

ex = load_exercises(Path(__file__).parent)


def test_parse_tool_call():
    good = 'Sure. <tool_call>{"name": "weekday", "arguments": {"date": "2024-01-01"}}</tool_call> done'
    assert ex.parse_tool_call(good) == {"name": "weekday", "arguments": {"date": "2024-01-01"}}
    two = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
           '<tool_call>{"name": "b", "arguments": {}}</tool_call>')
    assert ex.parse_tool_call(two)["name"] == "a"
    for bad in ['<tool_call>{"name": "weekday", "arguments": ', "no call here",
                '<tool_call>{"name": 3, "arguments": {}}</tool_call>',
                '<tool_call>{"name": "x", "arguments": [1]}</tool_call>',
                '<tool_call>["name", "x"]</tool_call>']:
        assert ex.parse_tool_call(bad) is None, bad


def test_dispatch_tool():
    reg = {"add": lambda a, b: a + b, "boom": lambda: 1 / 0}
    assert ex.dispatch_tool(reg, {"name": "add", "arguments": {"a": 2, "b": 3}}) == "5"
    assert ex.dispatch_tool(reg, {"name": "nope", "arguments": {}}).startswith("error:")
    err = ex.dispatch_tool(reg, {"name": "boom", "arguments": {}})
    assert err.startswith("error:") and "ZeroDivisionError" in err
    assert ex.dispatch_tool(reg, {"name": "add", "arguments": {"a": 1}}).startswith("error:")
    denied = ex.dispatch_tool(reg, {"name": "add", "arguments": {"a": 1, "b": 1}}, allowed={"boom"})
    assert denied.startswith("error:") and "allowed" in denied
    assert ex.dispatch_tool(reg, {"name": "add", "arguments": {"a": 1, "b": 1}}, allowed={"add"}) == "2"


def test_token_mask():
    m = ex.token_mask(["12", " 3", "a", '"}', "", "9-"], "0123456789-", ('"',))
    assert isinstance(m, np.ndarray) and m.dtype == bool
    assert m.tolist() == [True, False, False, True, False, True]
    assert ex.token_mask(["<", "</answer", "x"], "", ("<",)).tolist() == [True, True, False]


def test_select_by_logprob():
    assert ex.select_by_logprob([np.array([-1.0, -1.0]), np.array([-1.5])]) == 1     # total, not mean
    assert ex.select_by_logprob([np.array([-0.1]), np.array([-0.1])]) == 0            # tie -> first
    assert ex.select_by_logprob([[-3.0], [-0.5, -0.5, -0.5], [-2.0]]) == 1


def test_trim_window():
    msgs = [{"role": "system", "content": "aaaa"}] + [{"role": "user", "content": "b" * n} for n in (5, 3, 2, 4)]
    out = ex.trim_window(msgs, budget=4 + 4 + 2 + 1, count_tokens=len)
    assert [m["content"] for m in out] == ["aaaa", "bb", "bbbb"]
    out = ex.trim_window(msgs, budget=100, count_tokens=len)
    assert out == msgs
    no_sys = msgs[1:]
    assert [len(m["content"]) for m in ex.trim_window(no_sys, 6, len)] == [2, 4]
    assert [len(m["content"]) for m in ex.trim_window(no_sys, 5, len)] == [4]   # stop at first that doesn't fit


def test_recency_relevance_score():
    s = ex.recency_relevance_score([0, 10, 100], [0.2, 0.9, 0.5], decay=0.99)
    rec = 0.99 ** np.array([0, 10, 100.0])
    rec = (rec - rec.min()) / (rec.max() - rec.min())
    rel = (np.array([0.2, 0.9, 0.5]) - 0.2) / 0.7
    np.testing.assert_allclose(s, rec + rel, rtol=1e-6)
    s2 = ex.recency_relevance_score([5, 5], [1.0, 3.0], w_relevance=2.0)
    np.testing.assert_allclose(s2, [0.0, 2.0])


@pytest.mark.parametrize("answers,expected", [
    (["7", "8", "7", None], "7"), (["a", "b"], "a"), ([None, None], None), ([], None), ([None, "x", "y", "y"], "y"),
])
def test_majority_vote(answers, expected):
    assert ex.majority_vote(answers) == expected
