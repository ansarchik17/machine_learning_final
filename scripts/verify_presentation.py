"""Check final editable slide data and notes against the frozen source tables."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="output/presentation/student_outcomes_presentation.pptx")
    args = parser.parse_args()
    path = ROOT / args.path
    data = json.loads((ROOT / "artifacts/presentation_content.json").read_text())
    for filename, expected in data["source_sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
    assert len(data["slides"]) == 15 and sum(s["seconds"] for s in data["slides"]) == 720
    assert len(data["qa"]) == 10
    expected_series = {
        3: [[1421, 794, 2209]],
        6: [[r["mean_passed_sem1"] for r in data["training_profiles"]]],
        9: [[r["train_macro_f1"] for r in data["history"]], [r["validation_macro_f1"] for r in data["history"]]],
        10: [[r["validation_macro_f1"] for r in data["models"]], [r["test_macro_f1"] for r in data["models"]]],
        11: [[r["test_macro_f1"] for r in data["ablation"]]],
        12: [[r["cumulative_variance"] for r in data["pca"]]],
    }
    with zipfile.ZipFile(path) as package:
        slide_paths = [p for p in package.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", p)]
        assert len(slide_paths) == 15
        notes_paths = [p for p in package.namelist() if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", p)]
        assert len(notes_paths) == 15
        for i, meta in enumerate(data["slides"], 1):
            note = " ".join(ET.fromstring(package.read(f"ppt/notesSlides/notesSlide{i}.xml")).itertext())
            assert meta["speaker"] in note and meta["spoken_notes"] in note, i
            assert all(source in note for source in meta["sources"]), i
            slide = ET.fromstring(package.read(f"ppt/slides/slide{i}.xml"))
            assert not slide.findall(".//p:pic", NS), "Evidence should remain native and editable"
            if i in (7, 12, 13):
                assert len(slide.findall(".//a:tbl", NS)) == 1, i
            if i not in expected_series:
                continue
            refs = slide.findall(".//c:chart", NS)
            assert len(refs) == 1, i
            rels = ET.fromstring(package.read(f"ppt/slides/_rels/slide{i}.xml.rels"))
            ref = refs[0].get("{" + NS["r"] + "}id")
            target = next(rel.get("Target") for rel in rels if rel.get("Id") == ref)
            chart_path = posixpath.normpath(posixpath.join("ppt/slides", target)).lstrip("/")
            chart = ET.fromstring(package.read(chart_path))
            series = chart.findall(".//c:ser", NS)
            assert len(series) == len(expected_series[i]), i
            for observed, expected in zip(series, expected_series[i], strict=True):
                values = [float(v.text) for v in observed.findall("c:val/c:numRef/c:numCache/c:pt/c:v", NS)]
                assert len(values) == len(expected), i
                assert all(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12) for a, b in zip(values, expected, strict=True)), i
        matrix_slide = ET.fromstring(package.read("ppt/slides/slide13.xml"))
        cells = matrix_slide.findall(".//a:tbl/a:tr", NS)[1:]
        matrix = [[int("".join(t.text or "" for t in cell.findall(".//a:t", NS))) for cell in row.findall("a:tc", NS)[1:]] for row in cells]
        assert matrix == data["confusion"]
        assert sum(sum(row) for row in matrix) == 664
        assert sum(matrix[i][i] for i in range(3)) == 484
        final_note = " ".join(ET.fromstring(package.read("ppt/notesSlides/notesSlide15.xml")).itertext())
        assert all(q["question"] in final_note and q["answer"] in final_note for q in data["qa"])
        assert len([p for p in package.namelist() if p.startswith("ppt/embeddings/") and p.endswith(".xlsx")]) == 6
    print("Presentation checks passed: 15 slides, 15 notes, 6 native charts/workbooks, 3 native tables; source metrics and matrix agree.")


if __name__ == "__main__":
    main()
