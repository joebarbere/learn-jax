from collections import namedtuple

import pytest

P = namedtuple("P", "title year")


def test_era_of(impl):
    assert impl.era_of(1958) == "Foundations"
    assert impl.era_of(1969) == "Foundations"
    assert impl.era_of(1970) == "Winters & expert systems"
    assert impl.era_of(2012) == "Deep learning"
    assert impl.era_of(2017) == "Transformers & scale"
    assert impl.era_of(2025) == "The LLM era"
    with pytest.raises(ValueError):
        impl.era_of(1900)


def test_events_between(impl):
    ev = [{"title": "a", "year": 1997}, {"title": "b", "year": 1958}, {"title": "c", "year": 2012}]
    assert [e["title"] for e in impl.events_between(ev, 1950, 1999)] == ["a", "b"]
    assert impl.events_between(ev, 1997, 1997) == [ev[0]]          # inclusive both ends
    assert impl.events_between(ev, 2020, 2030) == []


def test_sort_chronologically(impl):
    items = [P("t", 2017), {"title": "x", "year": 1958}, P("u", 2017), {"title": "y", "year": 1943}]
    out = impl.sort_chronologically(items)
    assert [impl.year_of(i) for i in out] == [1943, 1958, 2017, 2017]
    assert out[2].title == "t" and out[3].title == "u"                 # stable
    assert items[0].title == "t"                                       # input untouched


def test_papers_per_decade(impl):
    papers = [P("a", 1958), P("b", 1950), P("c", 1986), P("d", 2017), P("e", 2019)]
    got = impl.papers_per_decade(papers)
    assert got == {1950: 2, 1980: 1, 2010: 2}
    assert list(got) == sorted(got)
    assert impl.papers_per_decade([]) == {}


def test_module_for_concept(impl):
    idx = {"attention": "15-attention", "lora": "20-lora", "perceptron": "29-perceptron-hopfield"}
    assert impl.module_for_concept("Attention", idx) == "15-attention"
    assert impl.module_for_concept("multi-head attention", idx) == "15-attention"
    assert impl.module_for_concept("the Perceptron rule", idx) == "29-perceptron-hopfield"
    assert impl.module_for_concept("diffusion", idx) is None




# a toy family tree with a diamond: d builds on b and c, which both build on a
PARENTS = {"a": (), "b": ("a",), "c": ("a",), "d": ("b", "c"), "e": ("d",)}
USES = {"a": ("attn", "ln"), "b": ("attn", "rope"), "c": ("attn", "swiglu"), "d": ("rope", "swiglu", "moe"),
        "e": ("moe", "mla", "ln")}


def test_ancestors(impl):
    assert impl.ancestors("e", PARENTS) == {"a", "b", "c", "d"}
    assert impl.ancestors("d", PARENTS) == {"a", "b", "c"}          # the diamond: "a" once
    assert impl.ancestors("a", PARENTS) == set()                    # edge case: a root
    with pytest.raises(KeyError):
        impl.ancestors("zzz", PARENTS)


def test_new_blocks(impl):
    assert impl.new_blocks("d", PARENTS, USES) == {"moe"}           # rope from b, swiglu from c
    assert impl.new_blocks("e", PARENTS, USES) == {"mla"}           # "ln" comes from a, two levels up
    assert impl.new_blocks("a", PARENTS, USES) == {"attn", "ln"}    # edge case: a root added everything


def test_lineage_registry():
    """The course's real family tree obeys its own rules (see learnjax/lineage.py)."""
    from learnjax import lineage

    assert lineage.check() == []
