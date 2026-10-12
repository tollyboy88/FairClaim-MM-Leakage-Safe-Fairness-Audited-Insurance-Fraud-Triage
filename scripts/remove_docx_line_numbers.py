"""Remove Word section line-numbering settings from one or more DOCX files."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def remove_line_numbers(path: Path) -> int:
    removed = 0
    with zipfile.ZipFile(path, "r") as source, NamedTemporaryFile(
        suffix=".docx", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        with zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "word/document.xml":
                    root = etree.fromstring(data)
                    nodes = root.xpath("//w:sectPr/w:lnNumType", namespaces={"w": W_NS})
                    for node in nodes:
                        node.getparent().remove(node)
                    removed += len(nodes)
                    data = etree.tostring(
                        root,
                        xml_declaration=True,
                        encoding="UTF-8",
                        standalone=True,
                    )
                target.writestr(item, data)
    temporary.replace(path)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        count = remove_line_numbers(path.resolve())
        print(f"{path}: removed {count} line-numbering element(s)")


if __name__ == "__main__":
    main()
