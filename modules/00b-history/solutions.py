"""Module 00b reference solutions. Try the exercises first — `make test N=00b` grades yours."""

ERAS = [
    (1940, 1969, "Foundations"),
    (1970, 1992, "Winters & expert systems"),
    (1993, 2011, "Statistical ML & the deep-learning revival"),
    (2012, 2016, "Deep learning"),
    (2017, 2021, "Transformers & scale"),
    (2022, 2100, "The LLM era"),
]


def year_of(item) -> int:
    return item["year"] if isinstance(item, dict) else item.year


def era_of(year: int) -> str:
    for start, end, name in ERAS:
        if start <= year <= end:
            return name
    raise ValueError(f"no era for {year}")


def events_between(events: list, start: int, end: int) -> list:
    return [e for e in events if start <= year_of(e) <= end]


def sort_chronologically(items: list) -> list:
    return sorted(items, key=year_of)


def papers_per_decade(papers: list) -> dict[int, int]:
    counts: dict[int, int] = {}
    for p in papers:
        d = year_of(p) // 10 * 10
        counts[d] = counts.get(d, 0) + 1
    return dict(sorted(counts.items()))


def module_for_concept(concept: str, index: dict[str, str]) -> str | None:
    c = concept.lower().strip()
    if c in index:
        return index[c]
    for key, slug in index.items():
        if key in c:
            return slug
    return None
