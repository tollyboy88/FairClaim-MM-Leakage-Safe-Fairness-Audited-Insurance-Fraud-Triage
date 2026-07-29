"""Build the Applied Operations and Analytics Word submission files.

The anonymous manuscript is deliberately review-friendly: one file containing
text, tables, and figures, with double spacing, continuous line numbering, and
page numbers. Author-identifying information is confined to the title page and
cover letter.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
AOA = ROOT / "publication" / "aoa"
MANUSCRIPT = AOA / "manuscript" / "main.md"
REFERENCES = AOA / "manuscript" / "references_harvard.md"
TABLE_DIR = AOA / "tables"
FIGURE_DIR = AOA / "figures"
SUBMISSION_DIR = AOA / "submission"
FINAL_DIR = AOA / "delivery"

TITLE = "Calibrated Selective Prediction for Leakage-Safe and Fairness-Audited Motor Insurance Fraud Triage"
SHORT_TITLE = "Selective Motor Insurance Fraud Triage"
AUTHOR = "Adebayo Adetola Aliu"
JOURNAL = "Applied Operations and Analytics"

NAVY = RGBColor(31, 56, 78)
MID = RGBColor(69, 89, 105)
LIGHT = "E8EEF2"


def page_field(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, end])


def set_line_numbers(section) -> None:
    sect_pr = section._sectPr
    existing = sect_pr.find(qn("w:lnNumType"))
    if existing is not None:
        sect_pr.remove(existing)
    node = OxmlElement("w:lnNumType")
    node.set(qn("w:countBy"), "1")
    node.set(qn("w:restart"), "continuous")
    sect_pr.append(node)


def configure_document(doc: Document, *, line_numbers: bool = False) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)
    if line_numbers:
        set_line_numbers(section)

    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(20, 20, 20)
    normal.paragraph_format.line_spacing = 2 if line_numbers else 1.15
    normal.paragraph_format.space_after = Pt(0) if line_numbers else Pt(4)
    normal.paragraph_format.alignment = (
        WD_ALIGN_PARAGRAPH.JUSTIFY if line_numbers else WD_ALIGN_PARAGRAPH.LEFT
    )

    for name, size in (("Title", 18), ("Heading 1", 14), ("Heading 2", 12)):
        style = doc.styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = NAVY
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.15

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = header.add_run(SHORT_TITLE)
    run.font.name = "Arial"
    run.font.size = Pt(8)
    run.font.color.rgb = MID
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("Page ")
    run.font.name = "Arial"
    run.font.size = Pt(8)
    page_field(footer)


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def add_table(doc: Document, spec: str) -> None:
    filename, caption, columns = [part.strip() for part in spec.split("|", 2)]
    frame = pd.read_csv(TABLE_DIR / filename)
    selected = [column.strip() for column in columns.split(",")]
    frame = frame[selected]
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.line_spacing = 1
    run = paragraph.add_run(caption)
    run.bold = True
    run.font.size = Pt(9)

    table = doc.add_table(rows=1, cols=len(selected))
    table.style = "Table Grid"
    table.autofit = True
    header = table.rows[0]
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header._tr.get_or_add_trPr().append(repeat)
    for index, column in enumerate(selected):
        cell = header.cells[index]
        cell.text = column.replace("_", " ")
        shade_cell(cell, LIGHT)
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(7.5)
    for row_index, (_, row) in enumerate(frame.iterrows()):
        cells = table.add_row().cells
        for column_index, value in enumerate(row):
            if pd.isna(value):
                text = "—"
            elif isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            cells[column_index].text = text
            cells[column_index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if row_index % 2:
                shade_cell(cells[column_index], "F7F9FA")
            for paragraph in cells[column_index].paragraphs:
                paragraph.paragraph_format.line_spacing = 1
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(7.2)
    doc.add_paragraph()


def add_figure(doc: Document, spec: str) -> None:
    filename, caption = [part.strip() for part in spec.split("|", 1)]
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = 1
    paragraph.add_run().add_picture(str(FIGURE_DIR / filename), width=Inches(6.5))
    caption_paragraph = doc.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.line_spacing = 1
    run = caption_paragraph.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)


def add_inline_runs(paragraph, text: str) -> None:
    # Minimal Markdown emphasis for a stable journal review document.
    parts = re.split(r"(\*\*.*?\*\*|`.*?`)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
        else:
            paragraph.add_run(part)


def add_references(doc: Document) -> None:
    doc.add_heading("References", level=1)
    lines = REFERENCES.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        paragraph = doc.add_paragraph(stripped)
        paragraph.paragraph_format.left_indent = Inches(0.25)
        paragraph.paragraph_format.first_line_indent = Inches(-0.25)
        paragraph.paragraph_format.line_spacing = 1.15
        paragraph.paragraph_format.space_after = Pt(6)
        for run in paragraph.runs:
            run.font.size = Pt(9)


def build_anonymous_manuscript() -> Path:
    doc = Document()
    configure_document(doc, line_numbers=True)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run(TITLE)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.line_spacing = 1
    run = subtitle.add_run("Anonymous manuscript")
    run.italic = True
    run.font.color.rgb = MID

    lines = MANUSCRIPT.read_text(encoding="utf-8").splitlines()
    in_references = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line == "{{REFERENCES}}":
            in_references = True
            add_references(doc)
            continue
        if in_references:
            continue
        if line.startswith("{{TABLE:") and line.endswith("}}"):
            add_table(doc, line[len("{{TABLE:") : -2])
            continue
        if line.startswith("{{FIGURE:") and line.endswith("}}"):
            add_figure(doc, line[len("{{FIGURE:") : -2])
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:], level=1)
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:], level=2)
            continue
        if line.startswith("- "):
            paragraph = doc.add_paragraph(style="List Bullet")
            add_inline_runs(paragraph, line[2:])
            continue
        if re.match(r"^\d+\. ", line):
            paragraph = doc.add_paragraph(style="List Number")
            add_inline_runs(paragraph, re.sub(r"^\d+\. ", "", line))
            continue
        if line.startswith("$$") and line.endswith("$$"):
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.line_spacing = 1.15
            run = paragraph.add_run(line[2:-2].strip())
            run.font.name = "Cambria Math"
            run.font.size = Pt(10.5)
            continue
        paragraph = doc.add_paragraph()
        add_inline_runs(paragraph, line)

    output = FINAL_DIR / "Anonymous_Manuscript.docx"
    doc.save(output)
    return output


def build_title_page() -> Path:
    doc = Document()
    configure_document(doc)
    paragraph = doc.add_paragraph(style="Title")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run(TITLE)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = 1.15
    paragraph.add_run(AUTHOR).bold = True
    for label, value in [
        ("Affiliation", "[Insert institutional affiliation]"),
        ("ORCID", "[Insert ORCID]"),
        ("Email", "[Insert corresponding-author email]"),
        ("Corresponding author", AUTHOR),
        ("Target journal", JOURNAL),
        ("Article type", "Original research article"),
    ]:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.line_spacing = 1.15
        paragraph.add_run(f"{label}: ").bold = True
        paragraph.add_run(value)
    doc.add_heading("Author declarations", level=1)
    for label, value in [
        ("Funding", "No external funding was used, subject to author confirmation."),
        ("Competing interests", "The author declares no competing interests, subject to author confirmation."),
        ("Data availability", "The public benchmark and its persistent identifier are cited in the manuscript."),
        ("Code availability", "GitHub repository; the exact URL is provided in the manuscript."),
    ]:
        paragraph = doc.add_paragraph()
        paragraph.add_run(f"{label}: ").bold = True
        paragraph.add_run(value)
    output = FINAL_DIR / "Title_Page.docx"
    doc.save(output)
    return output


def build_cover_letter() -> Path:
    doc = Document()
    configure_document(doc)
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.line_spacing = 1.15
    today = date.today()
    paragraph.add_run(f"{today.day} {today.strftime('%B %Y')}")
    doc.add_paragraph("Editor-in-Chief\nApplied Operations and Analytics")
    paragraph = doc.add_paragraph()
    paragraph.add_run(f"Re: Submission of “{TITLE}”").bold = True
    for text in [
        "Dear Editor,",
        (
            "Please consider the enclosed original research article for Applied Operations and Analytics. "
            "The study frames motor-insurance fraud screening as a constrained operational allocation problem: "
            "calibrated risk is converted into approval, abstention, and investigation regions, then evaluated "
            "under fixed investigator capacity and transparent cost scenarios."
        ),
        (
            "The revision makes four contributions relevant to the journal. It separates model-family comparison "
            "from imbalance treatment; quantifies operational value at 5%, 10%, 20%, and 30% investigation "
            "capacity; evaluates calibration, abstention, robustness, and temporal shift; and audits fairness "
            "across all three operational actions, including a sensitivity analysis that removes the retained "
            "AgeOfPolicyHolder proxy."
        ),
        (
            "The image feasibility experiment from an earlier version has been removed because the image data "
            "are not linked to verified claim-level fraud labels. Financial results are explicitly labelled as "
            "illustrative scenarios rather than realised savings. Code, an executed notebook, machine-readable "
            "tables, and a reproducibility workflow accompany the submission."
        ),
        (
            "This manuscript is original, is not under consideration elsewhere, and all listed statements about "
            "funding, competing interests, authorship, and AI-assisted support must be confirmed by the author "
            "before submission."
        ),
        "Yours sincerely,",
        f"{AUTHOR}\n[Affiliation]\n[Email]\n[ORCID]",
    ]:
        paragraph = doc.add_paragraph(text)
        paragraph.paragraph_format.line_spacing = 1.15
        paragraph.paragraph_format.space_after = Pt(8)
    output = FINAL_DIR / "Cover_Letter.docx"
    doc.save(output)
    return output


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        build_anonymous_manuscript(),
        build_title_page(),
        build_cover_letter(),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
