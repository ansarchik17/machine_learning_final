"""Compare the written PDFs with saved evidence and record completed visual QA."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-visual-review", action="store_true",
                        help="Record a human/agent review of every rendered page of the current PDFs.")
    args = parser.parse_args()
    pdf_paths = [ROOT / "output/pdf/student_outcomes_report.pdf",
                 ROOT / "output/pdf/student_outcomes_research_proposal.pdf"]
    readers = [PdfReader(path) for path in pdf_paths]
    texts = [[page.extract_text() for page in reader.pages] for reader in readers]
    pages = [[compact(text) for text in document] for document in texts]
    report, proposal = ["".join(document) for document in pages]
    layout_path = ROOT / "artifacts/report_layout.json"
    layout = json.loads(layout_path.read_text())
    assert len(pages[0]) == layout["report_pages"]
    assert 2 <= len(pages[1]) == layout["proposal_pages"] <= 3
    assert layout["main_body_pages_before_conclusion"] >= 4
    for reader, document in zip(readers, texts, strict=True):
        for page, text in zip(reader.pages, document, strict=True):
            assert len(text.strip()) > 100, "Unexpected empty page"
            assert not any(char in text for char in ("\ufffd", "\u25a0")), "Possible missing glyph"
            embedded = False
            for font in page["/Resources"]["/Font"].values():
                descriptor = font.get_object().get("/FontDescriptor")
                if descriptor:
                    embedded |= "/FontFile2" in descriptor.get_object()
            assert embedded, "No embedded TrueType font on page"
    for member in ("Ансар Келес", "Мухамеджан Сагым", "Бекназар Сарсенбек"):
        assert compact(member) in report and compact(member) in proposal
    for section, number in layout["section_start_pages"].items():
        assert 1 <= number <= len(pages[0]), section

    table_root = ROOT / "artifacts/tables"
    comparison = pd.read_csv(table_root / "final_comparison.csv")
    model_rows = []
    for _, row in comparison.iterrows():
        expected = compact(f"{row.model} {row.validation_macro_f1:.4f} {row.test_macro_f1:.4f} {row.test_accuracy:.2%}")
        model_rows.append(expected)
    assert any(all(row in page for row in model_rows) for page in pages[0]), "Model table is split or differs from evidence"

    confusion = pd.read_csv(table_root / "selected_confusion_matrix.csv", index_col=0)
    matrix_rows = [compact(str(label) + " ".join(str(int(value)) for value in row)) for label, row in confusion.iterrows()]
    assert any(all(row in page for row in matrix_rows) for page in pages[0]), "Confusion matrix differs from saved predictions"
    ablation = pd.read_csv(table_root / "feature_ablation.csv")
    for _, row in ablation.iterrows():
        assert compact(f"{row.input_features} {row.validation_macro_f1:.4f} {row.test_macro_f1:.4f}") in report

    sources = json.loads((ROOT / "docs/research/sources.json").read_text())["sources"]
    bibliography = "".join(pages[0][layout["section_start_pages"]["bibliography"] - 1:])
    body = "".join(pages[0][2:layout["section_start_pages"]["conclusion"] - 1])
    for source in sources:
        assert compact(source["url"]) in bibliography, source["id"]
        assert compact(source["citation"]) in body, source["id"]
    for doi in ("10.3390/data7110146", "10.1145/3636555.3636906", "10.7160/eriesj.2026.190106", "10.1016/j.techsoc.2024.102474"):
        assert doi in proposal, doi
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in pdf_paths}
    validation_path = ROOT / "artifacts/report_validation.json"
    previous = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    visual = previous.get("visual_review") if previous.get("pdf_sha256") == hashes else None
    if args.record_visual_review:
        visual = {
            "status": "passed",
            "reviewed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "report_pages_reviewed": len(pages[0]),
            "proposal_pages_reviewed": len(pages[1]),
            "method": "Poppler page rendering and visual inspection of every final page",
            "checks": ["Cyrillic and chart labels readable", "no clipping or overlapping content",
                       "short tables remain on one page", "figure captions and page headers/footers clear"],
        }
    assert visual and visual["status"] == "passed", "Render and inspect every page, then use --record-visual-review"
    validation = {
        "status": "passed",
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "report_pages": len(pages[0]), "proposal_pages": len(pages[1]),
        "main_body_pages_before_conclusion": layout["main_body_pages_before_conclusion"],
        "pdf_sha256": hashes, "visual_review": visual,
        "checks": ["page counts", "embedded fonts and text extraction", "team names", "model table values and integrity",
                   "confusion matrix identity", "ablation metrics", "all 26 references cited and listed"],
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [
            ROOT / "docs/report.md", ROOT / "docs/research_proposal.md", ROOT / "docs/literature_review.md",
            ROOT / "docs/bibliography.md", ROOT / "scripts/build_report.py", ROOT / "artifacts/results_summary.json"]},
        "scope": "Document checks and visual review; not an independent statistical validation or a record of student contributions.",
    }
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    layout["status"] = "verified_and_visually_reviewed"
    layout_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n")
    print(f"Report checks passed: {len(pages[0])} report pages, {len(pages[1])} proposal pages; evidence and visual QA agree.")


if __name__ == "__main__":
    main()
