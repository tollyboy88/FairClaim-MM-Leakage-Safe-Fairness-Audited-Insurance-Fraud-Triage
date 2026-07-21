"""Create a human-readable US Harvard reference list from references.bib."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "publication" / "manuscript" / "references.bib"
OUT = ROOT / "publication" / "manuscript" / "references_harvard.md"


def entries(text):
    starts = list(re.finditer(r"@(\w+)\{([^,]+),", text))
    for i, m in enumerate(starts):
        body = text[m.end(): starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        fields = {}
        pattern = r"(\w+)\s*=\s*\{(.*?)\}(?=\s*,\s*\w+\s*=|\s*\}\s*$)"
        for fm in re.finditer(pattern, body, flags=re.DOTALL):
            fields[fm.group(1).lower()] = " ".join(fm.group(2).split()).replace("\\&", "&")
        yield m.group(1).lower(), m.group(2), fields


def author_text(value):
    names = value.split(" and ")
    rendered = []
    for name in names:
        name = name.strip("{} ")
        if "," in name:
            family, given = [p.strip() for p in name.split(",", 1)]
            initials = "".join(f"{p[0]}." for p in re.findall(r"[A-Za-z]+", given) if p)
            rendered.append(f"{family}, {initials}" if initials else family)
        else:
            rendered.append(name)
    if len(rendered) == 1: return rendered[0]
    if len(rendered) == 2: return f"{rendered[0]} and {rendered[1]}"
    return ", ".join(rendered[:-1]) + f", and {rendered[-1]}"


def format_entry(kind, fields):
    a = author_text(fields.get("author", "Unknown author"))
    y = fields.get("year", "n.d.")
    t = fields.get("title", "Untitled").replace("{", "").replace("}", "")
    if kind == "article":
        venue = fields.get("journal", "")
        vol, no, pages = fields.get("volume"), fields.get("number"), fields.get("pages")
        tail = venue
        if vol: tail += f", {vol}"
        if no: tail += f"({no})"
        if pages: tail += f", pp. {pages.replace('--', '–')}"
    elif kind == "inproceedings":
        tail = f"in {fields.get('booktitle', '')}"
        if fields.get("pages"): tail += f", pp. {fields['pages'].replace('--', '–')}"
    elif kind == "book":
        tail = f"{fields.get('address', '')}: {fields.get('publisher', '')}".strip(": ")
    else:
        tail = fields.get("publisher", fields.get("journal", "Online resource"))
    locator = ""
    if fields.get("doi"): locator = f" https://doi.org/{fields['doi']}"
    elif fields.get("url"): locator = f" Available at: {fields['url']}"
    return f"{a} ({y}) ‘{t}’, {tail}.{locator}".replace("..", ".")


items = sorted((format_entry(kind, fields) for kind, _, fields in entries(BIB.read_text(encoding="utf-8"))), key=str.casefold)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("# References — US Harvard author–date style\n\n" + "\n\n".join(items) + "\n", encoding="utf-8")
print(f"Wrote {len(items)} references to {OUT}")
