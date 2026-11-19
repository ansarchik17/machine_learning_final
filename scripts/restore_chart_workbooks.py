"""Restore source chart workbooks omitted by PPTX import, translating labels only."""

import argparse
import copy
import io
import json
import math
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile

C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"c": C, "s": S}
ET.register_namespace("c", C)
ET.register_namespace("r", R)


def translate_workbook(payload, translations):
    source = zipfile.ZipFile(io.BytesIO(payload))
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            raw = source.read(info.filename)
            if info.filename.startswith("xl/") and info.filename.endswith(".xml"):
                root = ET.fromstring(raw)
                numbers_before = [(c.get("r"), c.findtext("s:v", namespaces=NS), c.findtext("s:f", namespaces=NS))
                                  for c in root.findall(".//s:c", NS) if c.get("t") not in ("inlineStr", "s")]
                changed = False
                for text in root.findall(".//s:t", NS):
                    value = translations.get(text.text, text.text)
                    if value != text.text:
                        text.text = value
                        changed = True
                numbers_after = [(c.get("r"), c.findtext("s:v", namespaces=NS), c.findtext("s:f", namespaces=NS))
                                 for c in root.findall(".//s:c", NS) if c.get("t") not in ("inlineStr", "s")]
                assert numbers_before == numbers_after, "Workbook numbers or formulas changed"
                if changed:
                    raw = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            target.writestr(info, raw)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("candidate")
    parser.add_argument("output")
    parser.add_argument("translations")
    args = parser.parse_args()
    translations = json.loads(Path(args.translations).read_text())
    with zipfile.ZipFile(args.source) as source, zipfile.ZipFile(args.candidate) as candidate:
        parts = {info.filename: candidate.read(info.filename) for info in candidate.infolist()}
        chart_paths = sorted(n for n in source.namelist() if "/charts/chart" in n and n.endswith(".xml"))
        assert len(chart_paths) == 6
        restored = []
        for chart_path in chart_paths:
            old = ET.fromstring(source.read(chart_path))
            new = ET.fromstring(parts[chart_path])
            old_series, new_series = old.findall(".//c:ser", NS), new.findall(".//c:ser", NS)
            assert len(old_series) == len(new_series)
            for a, b in zip(old_series, new_series, strict=True):
                old_values = [float(v.text) for v in a.findall("c:val//c:pt/c:v", NS)]
                new_values = [float(v.text) for v in b.findall("c:val//c:pt/c:v", NS)]
                assert len(old_values) == len(new_values)
                assert all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-13) for x, y in zip(old_values, new_values, strict=True))
            external = old.find("c:externalData", NS)
            assert external is not None, chart_path
            directory, basename = posixpath.split(chart_path)
            rel_path = directory + "/_rels/" + basename + ".rels"
            rels = ET.fromstring(source.read(rel_path))
            relation = next(rel for rel in rels if rel.get("Id") == external.get("{" + R + "}id"))
            assert relation.get("Type", "").endswith("/package")
            workbook = posixpath.normpath(posixpath.join(directory, relation.get("Target")))
            assert workbook.startswith("ppt/embeddings/") and workbook.endswith(".xlsx")
            # Restore the actual source workbook and its actual relationship.
            # The imported candidate has no surviving workbook to substitute.
            assert workbook not in parts
            parts[workbook] = translate_workbook(source.read(workbook), translations)
            parts[rel_path] = source.read(rel_path)
            existing = new.find("c:externalData", NS)
            if existing is not None:
                new.remove(existing)
            new.append(copy.deepcopy(external))
            parts[chart_path] = ET.tostring(new, encoding="utf-8", xml_declaration=True)
            restored.append(workbook)
        content_types = ET.fromstring(parts["[Content_Types].xml"])
        source_types = ET.fromstring(source.read("[Content_Types].xml"))
        for declaration in source_types:
            required = declaration.get("Extension") == "xlsx" or declaration.get("PartName", "").lstrip("/") in restored
            if required and not any(item.attrib == declaration.attrib for item in content_types):
                content_types.append(copy.deepcopy(declaration))
        ET.register_namespace("", CT)
        parts["[Content_Types].xml"] = ET.tostring(content_types, encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as output:
            for name, payload in parts.items():
                output.writestr(name, payload)
    print("Restored 6 original chart workbooks and relationships; translated labels with numeric cells and formulas unchanged.")


if __name__ == "__main__":
    main()
