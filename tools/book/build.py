"""Build the course book and site from rendered lessons (tools/book/render.sh writes site/build/<theme>/*.ipynb).

    podman run --rm -v $PWD:/course:z learn-jax-book python tools/book/build.py [--themes light dark] [--pdf]

Writes site/<theme>/index.html + one page per module (the browsable site), site/book/book-<theme>.html (one
document laid out by Paged.js) and, with --pdf, site/arrays-to-agents[-dark].pdf. Runs in the learn-jax-book image
(no GPU): Chromium via Playwright, Paged.js, KaTeX, Pygments and the fonts are pinned in containers/Containerfile.book.

This is the *reading edition*: lesson prose, figures and results. The code stays in the notebooks.
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SITE = ROOT / "site"
VENDOR = Path(os.environ.get("BOOK_VENDOR", "/opt/book/vendor"))
TITLE = "Arrays to Agents"
SUBTITLE = "Deep learning and language models in JAX, on a consumer AMD GPU"
PDF_NAME = {"light": "arrays-to-agents.pdf", "dark": "arrays-to-agents-dark.pdf"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


syl = _load("syllabus", ROOT / "learnjax" / "syllabus.py")
GLOSSARY = _load("glossary", ROOT / "learnjax" / "glossary.py").TERMS if (ROOT / "learnjax/glossary.py").exists() else []


# ---------------------------------------------------------------------------------------------- markdown + math

_CODE = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)
_MATH = re.compile(r"(\$\$.+?\$\$|(?<![\\$\w])\$(?=[^\s$])(?:[^$\n]|\n(?!\n))+?(?<=[^\s\\])\$(?![\w$]))", re.S)


def _math_html(src: str) -> str:
    """$...$ / $$...$$ -> KaTeX delimiters, HTML-escaped (KaTeX reads the text content)."""
    if src.startswith("$$"):
        return '<span class="math display">\\[' + html.escape(src[2:-2]) + "\\]</span>"
    return '<span class="math">\\(' + html.escape(src[1:-1]) + "\\)</span>"


def md_to_html(text: str, toc: bool = False) -> str:
    """Markdown with math protected from the markdown parser (underscores, backslashes)."""
    stash: list[str] = []

    def keep(m):
        stash.append(_math_html(m.group(0)))
        return f"MATHPH{len(stash) - 1}ZZ"

    pieces = _CODE.split(text)
    text = "".join(p if i % 2 else _MATH.sub(keep, p) for i, p in enumerate(pieces))
    text = _list_gap(text)
    exts = ["tables", "fenced_code", "codehilite", "sane_lists", "md_in_html"]
    cfg = {"codehilite": {"css_class": "hl", "guess_lang": False}}
    if toc:
        exts.append("toc")
        cfg["toc"] = {"slugify": _gh_slug}
    out = markdown.markdown(text, extensions=exts, extension_configs=cfg)
    return re.sub(r"MATHPH(\d+)ZZ", lambda m: stash[int(m.group(1))], out)


_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


def _list_gap(text: str) -> str:
    """A list right after a paragraph line needs a blank line for Python-Markdown (GitHub and marimo don't)."""
    out, fence = [], False
    for ln in text.split("\n"):
        if ln.lstrip().startswith(("```", "~~~")):
            fence = not fence
        prev = out[-1] if out else ""
        if (not fence and _ITEM.match(ln) and prev.strip() and not _ITEM.match(prev)
                and not prev.startswith((" ", "\t", "|", "#", ">"))):
            out.append("")
        out.append(ln)
    return "\n".join(out)


def _gh_slug(value: str, separator: str = "-") -> str:
    """GitHub-style heading anchors, so links written for GitHub (docs/ROCM_GOTCHAS.md) keep working."""
    value = re.sub(r"<[^>]+>", "", value).strip().lower()
    value = re.sub(r"[^\w\- ]", "", value)
    return value.replace(" ", separator)


# ---------------------------------------------------------------------------------------------- marimo output -> HTML

def _jattr(tag, name, default=None):
    v = tag.get(name)
    if v is None:
        return default
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return v


def _label_html(tag) -> str:
    lab = _jattr(tag, "data-label", "") or ""
    return convert(lab) if isinstance(lab, str) else html.escape(str(lab))


def _control(tag, kind: str) -> str:
    val = _jattr(tag, "data-initial-value")
    steps = _jattr(tag, "data-steps")
    if isinstance(steps, list) and isinstance(val, int) and 0 <= val < len(steps):
        val = steps[val]
    shown = "" if val in (None, "", []) else f' <span class="control-value">{html.escape(json.dumps(val) if isinstance(val, (list, dict)) else str(val))}</span>'
    return (f'<div class="control"><span class="control-kind">{kind}</span> '
            f'<span class="control-label">{_label_html(tag)}</span>{shown}</div>')


def _quiz(tag) -> str:
    opts = _jattr(tag, "data-options", []) or []
    items = "".join(f"<li>{html.escape(str(o))}</li>" for o in opts)
    return f'<div class="quiz"><div class="quiz-q">{_label_html(tag)}</div><ol type="a">{items}</ol></div>'


def _table(tag) -> str:
    data = _jattr(tag, "data-data")
    if isinstance(data, str) and data.startswith("["):
        try:
            data = json.loads(data)
        except ValueError:
            data = None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        cols = list(data[0])
        head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
        rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
                       for r in data[:60])
        return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
    return '<div class="control"><span class="control-kind">table</span> (interactive in the notebook)</div>'


def _mime(tag) -> str:
    return _render_mime(_jattr(tag, "data-mime", ""), _jattr(tag, "data-data", ""))


def _render_mime(mime: str, data) -> str:
    if mime == "application/vnd.marimo+mimebundle":   # {mime: data}: pick the best representation
        bundle = json.loads(data) if isinstance(data, str) else data
        for m in ("image/png", "image/svg+xml", "image/jpeg", "text/html", "text/markdown", "text/plain"):
            if isinstance(bundle, dict) and m in bundle:
                return _render_mime(m, bundle[m])
        return ""
    if not isinstance(data, str):
        data = json.dumps(data)
    if mime == "image/svg+xml" and not data.startswith("data:"):
        return f'<div class="plot-svg">{data}</div>'
    if mime.startswith("image/"):
        src = data if data.startswith("data:") else f"data:{mime};base64,{data}"
        return f'<img class="plot" src="{src}" alt="">'
    if mime == "text/html":
        return convert(data)
    if mime == "text/markdown":
        return md_to_html(data)
    if mime == "text/plain":
        return f"<pre class='plain'>{html.escape(data)}</pre>" if data.strip() else ""
    if "error" in mime:
        return f'<div class="callout danger">{html.escape(str(data))[:2000]}</div>'
    return ""


_CONTROLS = {"marimo-slider": "slider", "marimo-range-slider": "range", "marimo-number": "number",
             "marimo-dropdown": "dropdown", "marimo-multiselect": "select", "marimo-text": "text",
             "marimo-text-area": "text", "marimo-checkbox": "checkbox", "marimo-switch": "switch",
             "marimo-button": "button", "marimo-run-button": "button", "marimo-date": "date",
             "marimo-code-editor": "code"}


def convert(fragment: str) -> str:
    """Rewrite marimo's custom elements into static HTML (recursively: callouts carry HTML in attributes)."""
    if "marimo-" not in fragment:
        return fragment
    soup = BeautifulSoup(fragment, "html.parser")
    while True:
        tag = soup.find(lambda t: t.name and t.name.startswith("marimo-"))
        if tag is None:
            break
        name = tag.name
        if name == "marimo-tex":
            t = tag.get_text()
            t = t.replace("||(", "\\(").replace("||)", "\\)").replace("||[", "\\[").replace("||]", "\\]")
            cls = "math display" if t.lstrip().startswith("\\[") else "math"
            new = f'<span class="{cls}">{html.escape(t)}</span>'
        elif name == "marimo-callout-output":
            new = f'<div class="callout {html.escape(str(_jattr(tag, "data-kind", "neutral")))}">' \
                  f'{convert(_jattr(tag, "data-html", "") or "")}</div>'
        elif name == "marimo-mime-renderer":
            new = _mime(tag)
        elif name == "marimo-radio":
            new = _quiz(tag)
        elif name in _CONTROLS:
            new = _control(tag, _CONTROLS[name])
        elif name == "marimo-table":
            new = _table(tag)
        elif tag.get("data-label") is not None and not tag.contents:
            new = _control(tag, name.removeprefix("marimo-"))
        else:  # wrappers (marimo-ui-element, tabs, accordion...): keep what's inside
            tag.unwrap()
            continue
        tag.replace_with(BeautifulSoup(new, "html.parser"))
    return str(soup)


def _text(v) -> str:
    return "".join(v) if isinstance(v, list) else str(v)


def module_body(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text())
    parts = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            parts.append(md_to_html(_text(cell["source"])))
            continue
        for out in cell.get("outputs", []):
            d = out.get("data") or {}
            if "text/html" in d:
                parts.append(convert(_text(d["text/html"])))
            elif "text/markdown" in d:
                md = _text(d["text/markdown"])
                if md.lstrip().startswith("&lt;marimo-"):  # UI elements arrive as escaped HTML in text/markdown
                    parts.append(convert(html.unescape(md)))
                else:
                    parts.append(md_to_html(md))
            elif "image/png" in d:
                parts.append(f'<img class="plot" src="data:image/png;base64,{_text(d["image/png"]).strip()}" alt="">')
            elif out.get("output_type") == "error":
                parts.append(f'<div class="callout danger">{html.escape(out.get("ename", ""))}: '
                             f'{html.escape(out.get("evalue", ""))}</div>')
    return "\n".join(p for p in parts if p.strip())


# ---------------------------------------------------------------------------------------------- index anchors

_SKIP = {"pre", "code", "script", "style", "h1", "h2", "h3", "h4", "a", "svg", "math", "table"}


def _searchable(node) -> bool:
    for p in node.parents:
        if p.name in _SKIP or "math" in (p.get("class") or []) or "quiz" in (p.get("class") or []):
            return False
    return True


def term_patterns():
    pats = []
    for i, t in enumerate(GLOSSARY):
        names = sorted({t.term, *t.aliases}, key=len, reverse=True)
        alt = "|".join(re.escape(n).replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]?") for n in names if len(n) > 2)
        if alt:
            pats.append((i, re.compile(rf"(?<![\w-])(?:{alt})(?![\w-])", re.I)))
    return pats


def add_index_anchors(sections: dict[str, BeautifulSoup], pats) -> dict[int, list[tuple[str, str]]]:
    """First occurrence of each term in each module gets an anchor. Returns term -> [(module label, anchor id)]."""
    hits: dict[int, list[tuple[str, str]]] = {}
    for lab, soup in sections.items():
        nodes = [n for n in soup.find_all(string=True) if isinstance(n, NavigableString) and n.strip() and _searchable(n)]
        for i, pat in pats:
            for n in nodes:
                m = pat.search(str(n))
                if not m:
                    continue
                aid = f"ix-{i}-{lab}"
                s = str(n)
                anchor = soup.new_tag("span", attrs={"class": "ix", "id": aid})
                before, after = NavigableString(s[:m.start()]), NavigableString(s[m.start():])
                n.replace_with(before)
                before.insert_after(anchor)
                anchor.insert_after(after)
                nodes[nodes.index(n)] = after
                hits.setdefault(i, []).append((lab, aid))
                break
    return hits


# ---------------------------------------------------------------------------------------------- assembling

def _hero(dark: bool) -> str:
    svg = (ROOT / "docs/assets/hero.svg").read_text()
    svg = re.sub(r"^<\?xml[^>]*>", "", svg).replace('width="880" height="980"', 'class="hero"')
    # scope the SVG's own <style> rules to the hero, or they'd restyle the whole page once inlined
    return re.sub(r"(?m)^(\s*)\.(\w+)(\s*\{)", r"\1svg.hero .\2\3", svg)


def build_theme(theme: str):
    src = SITE / "build" / theme
    order = [(n, s, title, part) for n, s, title, part in syl.MODULES if (src / f"{s}.ipynb").exists()]
    missing = [s for _, s, _, _ in syl.MODULES if not (src / f"{s}.ipynb").exists()]
    if missing:
        print(f"[{theme}] not rendered yet, skipped: {', '.join(missing)}")
    sections, titles = {}, {}
    for _n, s, title, _part in order:
        lab = syl.label(s)
        sections[lab] = BeautifulSoup(module_body(src / f"{s}.ipynb"), "html.parser")
        titles[lab] = title
    hits = add_index_anchors(sections, term_patterns())

    # --- front matter
    toc, body, part = [], [], None
    for _n, s, title, p in order:
        lab = syl.label(s)
        if p != part:
            part = p
            mods = [(syl.label(s2), t2) for _, s2, t2, p2 in order if p2 == p]
            toc.append(f'<li class="toc-part"><a data-page="000" href="#part-{p}">Part {p} · {html.escape(syl.PARTS[p])}</a></li>')
            items = "".join(f'<li><span class="num">{lb}</span> {html.escape(t)}</li>' for lb, t in mods)
            first = " first" if not any('class="part' in b for b in body) else ""
            body.append(f'<section class="part{first}" id="part-{p}"><div class="part-num">Part {p}</div>'
                        f'<h1 class="part-title">{html.escape(syl.PARTS[p])}</h1><ol class="part-list">{items}</ol></section>')
        toc.append(f'<li class="toc-mod"><a data-page="000" href="#m-{lab}"><span class="num">{lab}</span> {html.escape(title)}</a></li>')
        body.append(f'<section class="module" id="m-{lab}" data-label="{lab}">'
                    f'<div class="running">Part {p} · {html.escape(syl.PARTS[p])}</div>'
                    f'<div class="running-chapter">{lab} · {html.escape(title)}</div>'
                    f'{sections[lab]}</section>')

    apps = [("rocm", "Appendix A · JAX on ROCm: gotchas", md_to_html((ROOT / "docs/ROCM_GOTCHAS.md").read_text(), toc=True)),
            ("papers", "Appendix B · Papers and books", md_to_html((ROOT / "docs/PAPERS.md").read_text(), toc=True)),
            ("glossary", "Appendix C · Glossary", glossary_html()),
            ("lineage", "Appendix D · The family tree", md_to_html((ROOT / "docs/LINEAGE.md").read_text(), toc=True)),
            ("index", "Index", index_html(hits))]
    for key, title, content in apps:
        toc.append(f'<li class="toc-app"><a data-page="000" href="#app-{key}">{html.escape(title)}</a></li>')
        body.append(f'<section class="appendix" id="app-{key}"><h1 class="app-title">{html.escape(title)}</h1>'
                    f'<div class="app-body">{_drop_h1(content)}</div></section>')

    cover = (f'<section class="cover"><div class="cover-title">{TITLE}</div><div class="cover-sub">{SUBTITLE}</div>'
             f'<div class="cover-art">{_hero(theme == "dark")}</div>'
             f'<div class="cover-foot">{len(order)} modules · JAX on ROCm · AMD Radeon RX 7600 XT</div></section>')
    colophon = (f'<section class="colophon"><p class="col-title">{TITLE}</p>'
                f'<p>{SUBTITLE}. {"Dark" if theme == "dark" else "Light"} edition, for '
                f'{"screens" if theme == "dark" else "screens and print"}.</p>'
                '<p>This is the <em>reading edition</em> of the course: each lesson\'s prose, figures and measured '
                'results, rendered with the reference solutions. The code, exercises and interactive controls live '
                'in the notebooks (<code>make lesson N=..</code>); sliders and quizzes appear here at their defaults.</p>'
                '<p>Every number was measured on one AMD Radeon RX 7600 XT (16 GB, gfx1102) with JAX on ROCm, in '
                'podman containers, in November 2026. Timings vary with load; the lessons say when.</p>'
                '<p>Seminal papers are credited in each module and collected in Appendix B; companion lectures and '
                'courses are credited where they are used.</p></section>')
    contents = f'<section class="toc"><h1 class="toc-title">Contents</h1><ol>{"".join(toc)}</ol></section>'
    doc = page(theme, TITLE, cover + colophon + contents + "\n".join(body), paged=True)
    out = SITE / "book"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"book-{theme}.html").write_text(doc)
    write_site(theme, order, sections, titles)
    print(f"[{theme}] {len(order)} modules, {sum(len(v) for v in hits.values())} index anchors for {len(hits)} terms")


def _drop_h1(h: str) -> str:
    return re.sub(r"<h1[^>]*>.*?</h1>", "", h, count=1, flags=re.S)


def glossary_html() -> str:
    if not GLOSSARY:
        return "<p>(glossary not generated yet)</p>"
    out, letter = [], None
    for i, t in enumerate(GLOSSARY):
        first = t.term[0].upper() if t.term[0].isalpha() else "#"
        if first != letter:
            letter = first
            out.append(f'<h2 class="gl-letter">{letter}</h2>')
        also = f"; also {', '.join(t.also)}" if t.also else ""
        origin = f' <span class="gl-origin">{html.escape(t.origin)}</span>' if t.origin else ""
        out.append(f'<p class="gl" id="g-{i}"><strong>{html.escape(t.term)}</strong> {md_inline(t.definition)} '
                   f'<span class="gl-mod">(<a href="#m-{t.module}">Module {t.module}</a>{also}.)</span>{origin}</p>')
    return "\n".join(out)


def md_inline(s: str) -> str:
    h = md_to_html(s)
    return re.sub(r"^<p>(.*)</p>$", r"\1", h.strip(), flags=re.S)


def index_html(hits) -> str:
    if not GLOSSARY:
        return "<p>(index needs the glossary)</p>"
    out, letter = [], None
    order = sorted(range(len(GLOSSARY)), key=lambda i: GLOSSARY[i].term.lower())
    for i in order:
        t = GLOSSARY[i]
        refs = hits.get(i, [])
        if not refs:
            continue
        pri = [r for r in refs if r[0] == t.module] + [r for r in refs if r[0] in t.also and r[0] != t.module]
        pri += [r for r in refs if r not in pri]
        pri = pri[:10]
        pri.sort(key=lambda r: refs.index(r))
        first = t.term[0].upper() if t.term[0].isalpha() else "#"
        if first != letter:
            letter = first
            out.append(f'<h2 class="ix-letter">{letter}</h2>')
        links = ", ".join(f'<a class="ixref{" def" if lab == t.module else ""}" href="#{aid}" data-mod="{lab}" '
                          f'data-page="000"></a>'
                          for lab, aid in pri)
        out.append(f'<p class="ix-entry"><a class="ix-term" href="#g-{i}">{html.escape(t.term)}</a> {links}</p>')
    return '<div class="ix-cols">' + "\n".join(out) + "</div>"


# After Paged.js lays out the pages: page numbers for the contents and index, and running headers.
# (Paged.js 0.4.3's target-counter and string-set proved unreliable across a long document.)
FINISH_JS = """
function finishPages() {
  const pages = [...document.querySelectorAll('.pagedjs_page')];
  const pageOf = {};
  pages.forEach((p, i) => p.querySelectorAll('[id]').forEach(e => { if (!(e.id in pageOf)) pageOf[e.id] = i + 1; }));
  document.querySelectorAll('a[data-page]').forEach(a => {
    const id = decodeURIComponent((a.getAttribute('href') || '').slice(1));
    a.setAttribute('data-page', id in pageOf ? pageOf[id] : '');
  });
  let part = '', mod = '';
  pages.forEach(p => {
    const r = p.querySelectorAll('.running'); if (r.length) part = r[r.length - 1].textContent;
    const c = p.querySelectorAll('.running-chapter'); if (c.length) mod = c[c.length - 1].textContent;
    const a = p.querySelectorAll('.app-title'); if (a.length) { part = 'Appendix'; mod = a[a.length - 1].textContent; }
    if (p.classList.contains('pagedjs_named_page') || p.classList.contains('pagedjs_blank_page')) return;
    const box = p.querySelector('.pagedjs_pagebox');
    if (!box || !mod) return;
    const h = document.createElement('div'); h.className = 'runhead';
    const l = document.createElement('span'); l.textContent = part;
    const rr = document.createElement('span'); rr.textContent = mod;
    h.append(l, rr); box.appendChild(h);
  });
}
"""


def page(theme: str, title: str, body: str, paged: bool, rel: str = "") -> str:
    v = f"{rel}vendor"
    head = (f'<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(title)}</title>'
            f'<link rel="stylesheet" href="{v}/katex/katex.min.css"><link rel="stylesheet" href="{rel}book.css">'
            f'<link rel="stylesheet" href="{rel}pygments-{theme}.css">'
            + (f'<link rel="stylesheet" href="{rel}print.css">' if paged else "")
            + f'<script src="{v}/katex/katex.min.js"></script><script src="{v}/katex/auto-render.min.js"></script>')
    run = ("window.PagedConfig = {auto: false};" if paged else "") + """
document.addEventListener('DOMContentLoaded', async () => {
  for (const el of document.querySelectorAll('span.math')) {
    try { renderMathInElement(el, {delimiters: [{left: '\\\\[', right: '\\\\]', display: true},
                                               {left: '\\\\(', right: '\\\\)', display: false}],
                                   throwOnError: false}); } catch (e) {}
  }
  """ + ("await window.PagedPolyfill.preview(); finishPages(); window.__bookReady = true;" if paged
         else "window.__bookReady = true;") + """
});"""
    paged_js = (f'<script>{FINISH_JS}</script><script src="{v}/paged.polyfill.js"></script>' if paged else "")
    return (f'<!doctype html><html lang="en" class="theme-{theme}"><head>{head}<script>{run}</script>{paged_js}</head>'
            f'<body>{body}</body></html>')


def write_site(theme: str, order, sections, titles):
    out = SITE / theme
    out.mkdir(parents=True, exist_ok=True)
    other = "dark" if theme == "light" else "light"
    nav_items = []
    part = None
    for _n, s, title, p in order:
        lab = syl.label(s)
        if p != part:
            part = p
            nav_items.append(f'<li class="toc-part">Part {p} · {html.escape(syl.PARTS[p])}</li>')
        nav_items.append(f'<li class="toc-mod"><a href="{lab}.html"><span class="num">{lab}</span> {html.escape(title)}</a></li>')
    nav = "".join(nav_items)
    for idx, (_n, s, title, _p) in enumerate(order):
        lab = syl.label(s)
        prev_ = f'<a href="{syl.label(order[idx - 1][1])}.html">← previous</a>' if idx else ""
        next_ = f'<a href="{syl.label(order[idx + 1][1])}.html">next →</a>' if idx + 1 < len(order) else ""
        bar = f'<nav class="bar"><a href="index.html">{TITLE}</a> {prev_} {next_} <a href="../{other}/{lab}.html">{other} mode</a></nav>'
        (out / f"{lab}.html").write_text(page(theme, f"{lab} · {title}", f'{bar}<main class="module">{sections[lab]}</main>{bar}',
                                              paged=False, rel="../"))
    home = (f'<main class="home"><div class="cover-title">{TITLE}</div><div class="cover-sub">{SUBTITLE}</div>'
            f'<div class="cover-art">{_hero(theme == "dark")}</div>'
            f'<p><a href="../{other}/index.html">{other} mode</a> · '
            f'<a href="../{PDF_NAME[theme]}">PDF</a></p><ol class="toc">{nav}</ol></main>')
    (out / "index.html").write_text(page(theme, TITLE, home, paged=False, rel="../"))


def assets():
    for f in ("book.css", "print.css"):
        shutil.copy(HERE / f, SITE / f)
    from pygments.formatters import HtmlFormatter
    for theme, style in (("light", "friendly"), ("dark", "github-dark")):
        (SITE / f"pygments-{theme}.css").write_text(HtmlFormatter(style=style).get_style_defs(".hl"))
    if (SITE / "vendor").exists():
        shutil.rmtree(SITE / "vendor")
    shutil.copytree(VENDOR, SITE / "vendor")
    shutil.copytree(VENDOR, SITE / "book" / "vendor", dirs_exist_ok=True)
    for f in ("book.css", "print.css", "pygments-light.css", "pygments-dark.css"):
        shutil.copy(SITE / f, SITE / "book" / f)


def print_pdf(theme: str):
    import functools
    import http.server
    import threading

    from playwright.sync_api import sync_playwright
    # Paged.js fetches the stylesheets with XHR, which Chromium blocks for file:// pages: serve site/ locally.
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/book/book-{theme}.html"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        pg.on("pageerror", lambda e: print(f"[{theme}] page error: {str(e)[:300]}"))
        pg.goto(url, wait_until="load", timeout=600_000)
        pg.wait_for_function("window.__bookReady === true", timeout=1_800_000, polling=1000)
        pg.pdf(path=str(SITE / PDF_NAME[theme]), prefer_css_page_size=True, print_background=True,
               outline=True, tagged=True)
        pages = pg.evaluate("document.querySelectorAll('.pagedjs_page').length")
        browser.close()
    srv.shutdown()
    print(f"[{theme}] {SITE / PDF_NAME[theme]}: {pages} pages")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--themes", nargs="+", default=["light", "dark"])
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    SITE.mkdir(exist_ok=True)
    assets()
    for t in a.themes:
        build_theme(t)
        if a.pdf:
            print_pdf(t)


if __name__ == "__main__":
    main()
