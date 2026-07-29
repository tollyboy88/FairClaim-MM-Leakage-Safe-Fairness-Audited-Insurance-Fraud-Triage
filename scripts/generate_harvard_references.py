"""Create a human-readable US Harvard reference list from a BibTeX file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def entries(text: str):
    starts = list(re.finditer(r"@(\w+)\{([^,]+),", text))
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        body = text[match.end() : end]
        fields = {}
        pattern = r"(\w+)\s*=\s*\{(.*?)\}(?=\s*,\s*\w+\s*=|\s*\}\s*$)"
        for field in re.finditer(pattern, body, flags=re.DOTALL):
            fields[field.group(1).lower()] = (
                " ".join(field.group(2).split()).replace("\\&", "&")
            )
        yield match.group(1).lower(), match.group(2), fields


def author_text(value: str) -> str:
    names = value.split(" and ")
    rendered = []
    for name in names:
        name = name.strip("{} ")
        if "," in name:
            family, given = [part.strip() for part in name.split(",", 1)]
            initials = "".join(
                f"{part[0]}." for part in re.findall(r"[A-Za-z]+", given) if part
            )
            rendered.append(f"{family}, {initials}" if initials else family)
        else:
            rendered.append(name)
    if len(rendered) == 1:
        return rendered[0]
    if len(rendered) == 2:
        return f"{rendered[0]} and {rendered[1]}"
    return ", ".join(rendered[:-1]) + f", and {rendered[-1]}"


def format_entry(kind: str, fields: dict[str, str]) -> str:
    author = author_text(fields.get("author", "Unknown author"))
    year = fields.get("year", "n.d.")
    title = fields.get("title", "Untitled").replace("{", "").replace("}", "")
    if kind == "article":
        tail = fields.get("journal", "")
        if fields.get("volume"):
            tail += f", {fields['volume']}"
        if fields.get("number"):
            tail += f"({fields['number']})"
        if fields.get("pages"):
            tail += f", pp. {fields['pages'].replace('--', '–')}"
    elif kind == "inproceedings":
        tail = f"in {fields.get('booktitle', '')}"
        if fields.get("pages"):
            tail += f", pp. {fields['pages'].replace('--', '–')}"
    elif kind == "book":
        tail = f"{fields.get('address', '')}: {fields.get('publisher', '')}".strip(
            ": "
        )
    else:
        tail = fields.get("publisher", fields.get("journal", "Online resource"))
    if fields.get("doi"):
        locator = f" https://doi.org/{fields['doi']}"
    elif fields.get("url"):
        locator = f" Available at: {fields['url']}"
    else:
        locator = ""
    return f"{author} ({year}) ‘{title}’, {tail}.{locator}".replace("..", ".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bib",
        type=Path,
        default=ROOT / "publication" / "aoa" / "manuscript" / "references.bib",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "publication"
        / "aoa"
        / "manuscript"
        / "references_harvard.md",
    )
    args = parser.parse_args()
    items = sorted(
        (
            format_entry(kind, fields)
            for kind, _, fields in entries(args.bib.read_text(encoding="utf-8"))
        ),
        key=str.casefold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "# References — US Harvard author–date style\n\n"
        + "\n\n".join(items)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(items)} references to {args.out}")


if __name__ == "__main__":
    main()
