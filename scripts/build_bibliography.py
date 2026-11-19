"""Build APA-style references and a BibTeX export from verified source records."""

from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def author_name(author: dict) -> str:
    if "literal" in author:
        return author["literal"]
    initials = []
    for word in author.get("given", "").split():
        parts = [part for part in word.split("-") if part]
        initials.append("-".join(part[0] + "." for part in parts))
    return author["family"] + (", " + " ".join(initials) if initials else "")


def author_list(authors: list[dict]) -> str:
    names = [author_name(author) for author in authors]
    if len(names) > 20:
        return ", ".join(names[:19]) + ", … " + names[-1]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def date_label(source: dict) -> str:
    if source["year"] is None:
        return "n.d." + ("-" + source["year_suffix"] if source.get("year_suffix") else "")
    return str(source["year"]) + source.get("year_suffix", "")


def apa(source: dict) -> str:
    names = author_list(source["authors"])
    text = names + ("" if names.endswith(".") else ".") + f" ({date_label(source)}). "
    title, container = source["title"], source["container"]
    kind = source["type"]
    if kind == "journal_article":
        journal = f"*{container}, {source['volume']}*"
        if source.get("issue"):
            journal += f"({source['issue']})"
        detail = ("Article " + source["article_number"] if source.get("article_number") else
                  str(source.get("pages", "")).replace("-", "–"))
        text += f"{title}. {journal}" + (", " + detail if detail else "") + ". "
    elif kind == "conference_paper":
        text += title + ". In "
        if source.get("editors"):
            text += source["editors"] + " (Eds.), "
        text += f"*{container}*"
        if source.get("pages"):
            text += " (pp. " + source["pages"].replace("-", "–") + ")"
        text += ". "
        if source.get("publisher"):
            text += source["publisher"] + ". "
    elif kind == "dataset":
        text += f"*{title}* [Data set]. {container}. "
    elif kind == "documentation":
        text += f"*{title}* (Version {source['version']}). Retrieved November 18, 2026, from "
    else:
        raise ValueError(kind)
    return text + source["url"]


def bibtex(source: dict) -> str:
    kind = {"journal_article": "article", "conference_paper": "inproceedings"}.get(source["type"], "misc")
    authors = " and ".join("{" + a["literal"] + "}" if "literal" in a else
                           a["family"] + ", " + a.get("given", "") for a in source["authors"])
    fields = {"author": authors, "title": "{" + source["title"] + "}",
              "year": str(source["year"]) if source["year"] is not None else "n.d.", "url": source["url"]}
    if kind == "article":
        fields["journal"] = source["container"]
    elif kind == "inproceedings":
        fields["booktitle"] = source["container"]
    for key in ("volume", "issue", "pages", "doi", "publisher"):
        if source.get(key):
            fields["number" if key == "issue" else key] = str(source[key]).replace("–", "--")
    if source.get("article_number"):
        fields["eid"] = source["article_number"]
    if source["type"] == "documentation":
        fields["note"] = f"Version {source['version']}; accessed 2026-11-18; citation label {source['citation']}"
    lines = [f"@{kind}{{{source['id']},"] + [f"  {key} = {{{value}}}," for key, value in fields.items()]
    return "\n".join(lines + ["}"])


if __name__ == "__main__":
    registry = json.loads((ROOT / "docs/research/sources.json").read_text())
    def sort_key(source):
        authors = "|".join(a.get("family") or a["literal"] for a in source["authors"]).casefold()
        title = re.sub(r"^(the|an|a)\s+", "", source["title"], flags=re.IGNORECASE).casefold()
        return authors, source["year"] or 0, title
    sources = sorted(registry["sources"], key=sort_key)
    peer = sum(source["peer_reviewed"] for source in sources)
    header = ("# Bibliography\n\n"
              f"{len(sources)} sources; {peer} are classified as peer-reviewed journal or conference papers. "
              "The dataset and documentation have separate source types in the registry. "
              "References follow APA 7, with author-year citations in the literature review.\n\n")
    (ROOT / "docs/bibliography.md").write_text(header + "\n\n".join(apa(s) for s in sources) + "\n", encoding="utf-8")
    (ROOT / "docs/references.bib").write_text("\n\n".join(bibtex(s) for s in sources) + "\n", encoding="utf-8")
    print(f"Built bibliography and BibTeX: {len(sources)} sources, {peer} peer-reviewed.")
