"""Render the final plain-text report as a formatted ``.docx`` file."""
from datetime import datetime
from pathlib import Path

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("WARNING: Word export not available. Install: pip install python-docx")


def save_report_as_word(report_text: str, output_path: Path) -> None:
    """Write the report to a Word document with section/paragraph styling.

    ALL-CAPS lines under 120 chars become headings (main sections, paper
    titles, or subsections depending on content); everything else is
    collected into justified, line-spaced paragraphs.
    """
    if not DOCX_AVAILABLE:
        return

    doc = Document()

    title = doc.add_heading("Research Paper Synthesis Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.color.rgb = RGBColor(0, 51, 102)
        run.font.size = Pt(18)
        run.font.bold = True

    timestamp_para = doc.add_paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    timestamp_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in timestamp_para.runs:
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(128, 128, 128)
    doc.add_paragraph()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    main_section_keywords = (
        "OVERVIEW",
        "DETAILED PAPER-BY-PAPER ANALYSIS",
        "INTEGRATED SYNTHESIS",
        "COMPARISON",
        "CONCLUSIONS",
    )

    lines = report_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.isupper() and len(line) < 120 and not any(c.islower() for c in line):
            if any(keyword in line for keyword in main_section_keywords):
                h = doc.add_heading(line, level=1)
                color = RGBColor(0, 51, 102)
                size = Pt(16)
            elif line.startswith("PAPER ") and ":" in line:
                h = doc.add_heading(line, level=2)
                color = RGBColor(0, 76, 153)
                size = Pt(14)
            else:
                h = doc.add_heading(line, level=3)
                color = RGBColor(51, 102, 153)
                size = Pt(12)
            h.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in h.runs:
                run.font.size = size
                run.font.color.rgb = color
                run.font.bold = True
            i += 1
            continue

        para_lines = []
        while i < len(lines):
            current = lines[i].strip()
            if not current:
                if i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    if nxt and nxt.isupper() and len(nxt) < 120:
                        break
                i += 1
                continue
            if current.isupper() and len(current) < 120 and not any(c.islower() for c in current):
                break
            para_lines.append(current)
            i += 1

        if para_lines:
            para_text = " ".join(para_lines)
            if para_text.startswith(("•", "-", "*")):
                cleaned = para_text[1:].strip()
                p = doc.add_paragraph(cleaned, style="List Bullet")
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.space_after = Pt(4)
            else:
                p = doc.add_paragraph(para_text)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.line_spacing = 1.2
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.space_before = Pt(2)
                if len(para_text) > 80:
                    p.paragraph_format.first_line_indent = Inches(0.0)
                for run in p.runs:
                    run.font.size = Pt(11)
                    run.font.name = "Calibri"

    section = doc.sections[0]
    footer = section.footer
    footer_para = footer.paragraphs[0]
    footer_para.text = "Page "
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.save(str(output_path))
