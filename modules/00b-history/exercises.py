"""Module 00b exercises: small tools for working with a timeline.

Replace each `raise NotImplementedError`, then run `make test N=00b`. Plain Python, no JAX.
Items (events or papers) are dicts or objects with a `year` (int) and a `title` (str).
"""

# (start_year, end_year_inclusive, name) — the six eras this module uses
ERAS = [
    (1940, 1969, "Foundations"),
    (1970, 1992, "Winters & expert systems"),
    (1993, 2011, "Statistical ML & the deep-learning revival"),
    (2012, 2016, "Deep learning"),
    (2017, 2021, "Transformers & scale"),
    (2022, 2100, "The LLM era"),
]


def year_of(item) -> int:
    """Helper (provided): the year of a dict or an object with a .year attribute."""
    return item["year"] if isinstance(item, dict) else item.year


def era_of(year: int) -> str:
    """Name of the era (from ERAS) containing `year`. Raise ValueError for years before 1940."""
    raise NotImplementedError


def events_between(events: list, start: int, end: int) -> list:
    """Items whose year is in [start, end] inclusive, keeping their original order."""
    raise NotImplementedError


def sort_chronologically(items: list) -> list:
    """A new list sorted by year, oldest first; ties keep their original order (a stable sort)."""
    raise NotImplementedError


def papers_per_decade(papers: list) -> dict[int, int]:
    """Count items per decade: {1950: 2, 1980: 1, ...}. Keys are the decade's first year, sorted ascending.
    Only decades that have at least one item appear."""
    raise NotImplementedError


def module_for_concept(concept: str, index: dict[str, str]) -> str | None:
    """Look up which module teaches a concept.

    `index` maps lower-case keywords to module slugs, e.g. {"attention": "15-attention"}.
    Match case-insensitively; if no key equals the concept, return the slug of the first key (in the
    index's order) that appears inside the concept ("multi-head attention" → "15-attention").
    Return None if nothing matches.
    """
    raise NotImplementedError
