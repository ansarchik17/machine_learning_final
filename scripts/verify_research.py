"""Check citation coverage and bibliography consistency, without evaluating models."""

import datetime as dt
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def normalise(url):
    return unquote(url).rstrip("/").casefold()


if __name__ == "__main__":
    registry = json.loads((ROOT / "docs/research/sources.json").read_text())
    sources = registry["sources"]
    review = (ROOT / "docs/literature_review.md").read_text()
    bibliography = (ROOT / "docs/bibliography.md").read_text()
    bibtex = (ROOT / "docs/references.bib").read_text()
    assert len(sources) >= 10
    assert len({source["id"] for source in sources}) == len(sources)
    assert len({normalise(source["url"]) for source in sources}) == len(sources)
    assert sum(source["peer_reviewed"] for source in sources) > len(sources) / 2
    citations = re.findall(r"\[\(([^\]]+)\)\]\((\S+)\)", review)
    lookup = {source["citation"]: source for source in sources}
    assert set(label for label, _ in citations) == set(lookup), "Uncited or unregistered references"
    for label, url in citations:
        assert normalise(url) == normalise(lookup[label]["url"]), (label, url)
    for source in sources:
        assert source["title"] in bibliography, source["id"]
        assert source["url"] in bibliography, source["id"]
        assert "{" + source["id"] + "," in bibtex, source["id"]
        assert source.get("evidence_url") and source.get("evidence_locator")
        assert source["year"] is None or source["year"] <= 2026
    for filename in ("docs/literature_review.md", "docs/research/related_work_matrix.md"):
        text = (ROOT / filename).read_text()
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not target.startswith(("https://", "http://", "#")):
                assert (ROOT / filename).parent.joinpath(target).resolve().exists(), target
    report = {"status": "passed", "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "source_count": len(sources), "peer_reviewed_count": sum(s["peer_reviewed"] for s in sources),
              "domain_studies_and_reviews": sum(s["category"] == "domain" for s in sources),
              "all_references_cited_in_review": True,
              "checks": ["unique records", "majority peer-reviewed", "author-year coverage", "citation URL identity",
                         "bibliography and BibTeX coverage", "evidence locations", "local document links"],
              "scope": "Checks internal consistency. Source facts and bibliographic metadata were reviewed separately against primary sources and publisher metadata."}
    (ROOT / "docs/research/validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Research checks passed: {len(sources)} sources, {report['peer_reviewed_count']} peer-reviewed; every reference cited.")
