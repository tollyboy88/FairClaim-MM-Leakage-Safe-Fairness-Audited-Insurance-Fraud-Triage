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
EQUATION_DIR = AOA / "equations"
SUBMISSION_DIR = AOA / "submission"
FINAL_DIR = AOA / "delivery"

TITLE = "Calibrated Three-Way Triage for Leakage-Safe and Fairness-Audited Motor Insurance Fraud Screening"
SHORT_TITLE = "Three-Way Motor Insurance Fraud Triage"
AUTHOR = "Adebayo Adetola Aliu"
JOURNAL = "Applied Operations and Analytics"

NAVY = RGBColor(31, 56, 78)
MID = RGBColor(69, 89, 105)
REVISION_BLUE = RGBColor(0, 92, 169)
LIGHT = "E8EEF2"
EQUATION_WIDTHS = [5.2, 5.8, 6.15, 6.15, 5.7]


def remove_paragraph_borders(target) -> None:
    """Remove inherited or direct paragraph borders (notably Word's Title rule)."""
    element = getattr(target, "_element", target)
    p_pr = element.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is not None:
        p_pr.remove(borders)


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
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.15
        remove_paragraph_borders(style)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = header.add_run(SHORT_TITLE)
    run.font.name = "Arial"
    run.font.size = Pt(8)
    run.font.color.rgb = MID
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
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


def add_table(doc: Document, spec: str, *, highlight_changes: bool = False) -> None:
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
    if highlight_changes:
        run.font.color.rgb = REVISION_BLUE

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
                    if highlight_changes:
                        run.font.color.rgb = REVISION_BLUE
    doc.add_paragraph()


def add_figure(doc: Document, spec: str, *, highlight_changes: bool = False) -> None:
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
    if highlight_changes:
        run.font.color.rgb = REVISION_BLUE


def add_equation(doc: Document, equation_index: int) -> None:
    """Insert a pre-rendered, publication-safe equation image."""
    path = EQUATION_DIR / f"equation_{equation_index:02d}.png"
    if not path.exists():
        raise FileNotFoundError(f"Missing equation render: {path}")
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = 1
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(3)
    width = EQUATION_WIDTHS[equation_index - 1]
    paragraph.add_run().add_picture(str(path), width=Inches(width))


def add_inline_runs(paragraph, text: str, *, highlight_changes: bool = False) -> None:
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
            run = paragraph.add_run(part)
        if highlight_changes:
            run.font.color.rgb = REVISION_BLUE


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


def build_anonymous_manuscript(
    filename: str = "Revised_Manuscript_Clean.docx",
    *,
    highlight_changes: bool = False,
) -> Path:
    doc = Document()
    configure_document(doc, line_numbers=True)
    title = doc.add_paragraph(style="Title")
    remove_paragraph_borders(title)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run(TITLE)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.line_spacing = 1
    run = subtitle.add_run("Anonymous manuscript")
    run.italic = True
    run.font.color.rgb = MID
    if highlight_changes:
        note = doc.add_paragraph()
        note.alignment = WD_ALIGN_PARAGRAPH.CENTER
        note.paragraph_format.line_spacing = 1
        note_run = note.add_run("Revised text is shown in blue")
        note_run.italic = True
        note_run.font.size = Pt(9)

    lines = MANUSCRIPT.read_text(encoding="utf-8").splitlines()
    in_references = False
    equation_index = 0
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
            add_table(
                doc,
                line[len("{{TABLE:") : -2],
                highlight_changes=highlight_changes,
            )
            continue
        if line.startswith("{{FIGURE:") and line.endswith("}}"):
            add_figure(
                doc,
                line[len("{{FIGURE:") : -2],
                highlight_changes=highlight_changes,
            )
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:], level=1)
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:], level=2)
            continue
        if line.startswith("- "):
            paragraph = doc.add_paragraph(style="List Bullet")
            add_inline_runs(paragraph, line[2:], highlight_changes=highlight_changes)
            continue
        if re.match(r"^\d+\. ", line):
            paragraph = doc.add_paragraph(style="List Number")
            add_inline_runs(
                paragraph,
                re.sub(r"^\d+\. ", "", line),
                highlight_changes=highlight_changes,
            )
            continue
        if line.startswith("$$") and line.endswith("$$"):
            equation_index += 1
            add_equation(doc, equation_index)
            continue
        paragraph = doc.add_paragraph()
        add_inline_runs(paragraph, line, highlight_changes=highlight_changes)

    output = FINAL_DIR / filename
    doc.save(output)
    return output


def build_title_page() -> Path:
    doc = Document()
    configure_document(doc)
    paragraph = doc.add_paragraph(style="Title")
    remove_paragraph_borders(paragraph)
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
        ("Article type", "Revised original research article"),
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
    paragraph.add_run(f"Re: Revised submission 264653829 “{TITLE}”").bold = True
    for text in [
        "Dear Editor,",
        (
            "Please consider the enclosed revised manuscript for Applied Operations and Analytics. "
            "We have addressed each reviewer and editor comment in the accompanying response. The revision "
            "frames motor-insurance fraud screening as an information-system work-allocation problem in which "
            "calibrated risk is converted into approval, defer, and investigation actions."
        ),
        (
            "The analysis now propagates validation-bootstrap threshold variation into approval NPV, fraud "
            "capture, investigation precision, and defer coverage. It also reports subgroup TPR and FNR with "
            "uncertainty intervals and compares binary and three-way policies at the same investigation threshold."
        ),
        (
            "The manuscript now uses two broader research questions, engages Information Systems design and "
            "human-AI delegation literature, and states four bounded design propositions. The Discussion contains "
            "separate contributions-to-literature and practical-implications subsections. Financial results remain "
            "explicitly illustrative rather than realised savings."
        ),
        (
            "Before submission, the author must confirm originality, exclusive consideration, funding, competing "
            "interests, authorship, and the accuracy of the AI-assistance disclosure. These matters have not been "
            "inferred from the analysis files."
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


def add_response_item(doc: Document, heading: str, response: str, location: str) -> None:
    doc.add_heading(heading, level=2)
    paragraph = doc.add_paragraph()
    paragraph.add_run("Response ").bold = True
    paragraph.add_run(response)
    paragraph = doc.add_paragraph()
    paragraph.add_run("Location in revision ").bold = True
    paragraph.add_run(location)


def build_response_to_reviewers() -> Path:
    doc = Document()
    configure_document(doc)
    title = doc.add_paragraph(style="Title")
    remove_paragraph_borders(title)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Response to Editors and Reviewers")
    paragraph = doc.add_paragraph()
    paragraph.add_run("Manuscript title ").bold = True
    paragraph.add_run(TITLE)
    paragraph = doc.add_paragraph()
    paragraph.add_run("Submission ID ").bold = True
    paragraph.add_run("264653829")
    doc.add_paragraph(
        "Thank you for the detailed comments. We revised the analysis, manuscript framing, terminology, "
        "discussion, references, and page layout. The responses below state only changes supported by the "
        "regenerated outputs. Revised manuscript text is also supplied in blue in a separate file."
    )

    doc.add_heading("Reviewer 1", level=1)
    add_response_item(
        doc,
        "Comment 1 Percentage changes",
        (
            "Corrected. Using unrounded results, removing AgeOfPolicyHolder changed test PR-AUC from "
            "0.255865 to 0.245526, an absolute reduction of 0.010339 and a relative reduction of 4.04%. "
            "The age-band investigation-rate gap changed from 0.162869 to 0.178197, an absolute increase "
            "of 0.015329 and a relative increase of 9.41%. The earlier 10% and 25.3% statements were removed."
        ),
        "Abstract and Section 4.3 Proxy sensitivity and fairness audit",
    )
    add_response_item(
        doc,
        "Comment 2 Temporal interpretation",
        (
            "Corrected. The chronological test PR-AUC of 0.157903 is now interpreted as evidence that the "
            "random split PR-AUC of 0.255865 is optimistic for temporal generalisation. The manuscript no "
            "longer describes this result as supporting random splits."
        ),
        "Abstract, Section 4.4 Threshold robustness and temporal shift, and Section 5.4",
    )
    add_response_item(
        doc,
        "Comment 3 Selective prediction terminology",
        (
            "We accepted the terminology concern and did not add an unsupported uncertainty model. The title "
            "and manuscript now describe calibrated three-way triage with an intermediate-risk defer region. "
            "The Methods state that the region is defined by approval NPV and F2 thresholds, not entropy, "
            "ensemble variance, or another direct uncertainty estimate."
        ),
        "Title, Abstract, Sections 2.1, 3.3, and 3.6",
    )
    add_response_item(
        doc,
        "Comment 4 Threshold stability propagation",
        (
            "Added. Each of 500 validation bootstraps reselects both thresholds and applies them to the fixed "
            "untouched test probabilities. The resulting 2.5th-97.5th percentile ranges are 98.55%-99.46% "
            "for approval NPV, 56.76%-79.49% for fraud capture, 14.67%-19.74% for investigation precision, "
            "and 8.15%-26.59% for defer coverage. The manuscript states that these ranges isolate threshold-"
            "selection instability and do not constitute a second test-set bootstrap."
        ),
        "Section 3.6, Section 4.4, and Table 10",
    )
    add_response_item(
        doc,
        "Comment 5 Fairness uncertainty and conditional errors",
        (
            "Added investigation TPR and FNR by group with Wilson intervals, together with 500-bootstrap "
            "intervals for approval, defer, investigation, TPR, and FNR gaps. The revision reports fraud-positive "
            "counts and warns that conditional and intersectional estimates can be imprecise. The text retains "
            "the distinction between descriptive disparity and a conclusion of discrimination."
        ),
        "Sections 3.5 and 4.3, Tables 7 and 8",
    )
    add_response_item(
        doc,
        "Comment 6 Binary and three way cost comparison",
        (
            "Clarified and rerun under an equivalent investigation constraint. Both policies use the same "
            "investigation threshold of 0.092956 and investigate the same 830 test claims. The binary policy "
            "approves all remaining claims; the three-way policy defers the 469 intermediate-risk claims. The "
            "reported difference therefore comes from the declared defer-delay and recovery assumptions, which "
            "remain explicitly illustrative."
        ),
        "Section 3.4 and Section 4.2",
    )

    doc.add_heading("Reviewer 2", level=1)
    add_response_item(
        doc,
        "Comment 1 NPV terminology",
        (
            "Corrected. NPV now means negative predictive value throughout. The erroneous phrase net present "
            "value was removed."
        ),
        "Sections 2.1, 3.3, 4.1, 4.3, 4.4, and 5.3",
    )
    add_response_item(
        doc,
        "Comment 2 Proofreading",
        (
            "Corrected the spellings Lindholm and Côté, removed the unclosed quotation and double full stop, "
            "and proofread the revised text. The Discussion was rewritten rather than retaining the affected sentences."
        ),
        "Section 2.2, Section 5, and References",
    )
    add_response_item(
        doc,
        "Comment 3 Page numbers and footer",
        (
            "Reformatted. Page numbers are right-aligned consistently in a dedicated footer, while the short "
            "title remains in the header. The footer contains no competing text."
        ),
        "All pages of both revised manuscript files",
    )

    doc.add_heading("Associate Editor", level=1)
    add_response_item(
        doc,
        "Comment 1 Information Systems literature",
        (
            "The revision now engages design science, managing AI, agentic-IS delegation, human-AI delegation, "
            "human-ML fairness augmentation, algorithmic social injustice, and the FAIR design theory. Added "
            "sources include Hevner et al. (2004), Gregor and Hevner (2013), Baird and Maruping (2021), "
            "Berente et al. (2021), Teodorescu et al. (2021), Fügener et al. (2022), Tanriverdi and Akinyemi "
            "(2025), and Rai, Tian, and Xue (2026)."
        ),
        "Introduction, Section 2.2, Section 5.1, and References",
    )
    add_response_item(
        doc,
        "Comment 2 Two broader research questions",
        (
            "The four technical questions were replaced with two questions about calibrated human-AI work "
            "allocation and the design and governance controls required for auditability under imbalance, "
            "proxy-sensitive fairness, threshold uncertainty, and temporal shift."
        ),
        "End of the Introduction",
    )
    add_response_item(
        doc,
        "Comment 3 Design theory and propositions",
        (
            "The paper now presents the prototype as an evaluated information-system artifact and derives four "
            "bounded design propositions: comparable learning choices, calibrated capacity-constrained allocation, "
            "explicit human deferral, and adaptive fairness governance. The manuscript states that these are "
            "situated implications requiring further field evaluation rather than universal causal laws."
        ),
        "Sections 5.1 and 5.2",
    )
    add_response_item(
        doc,
        "Comment 4 Discussion structure and implications",
        (
            "The Discussion has been reorganised into Contributions to literature, Design propositions, Practical "
            "implications, and Robustness fairness and governance implications. Practical implications are no "
            "longer presented as an early summary."
        ),
        "Section 5",
    )

    output = FINAL_DIR / "Response_to_Reviewers.docx"
    doc.save(output)
    return output


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        build_anonymous_manuscript(),
        build_anonymous_manuscript(
            "Revised_Manuscript_Changes_Highlighted.docx",
            highlight_changes=True,
        ),
        build_response_to_reviewers(),
        build_title_page(),
        build_cover_letter(),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
