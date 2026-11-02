"""Export the final Markdown report as PDF (ReportLab) or DOCX (python-docx)."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Characters that common PDF fonts lack are replaced with ASCII equivalents.
_TRANSLIT = {"→": "->", "←": "<-", "≥": ">=", "≤": "<=", "≈": "~", "×": "x", "−": "-", "✓": "yes", "✗": "no"}
_BODY_FONT_CANDIDATES = [
    ("/System/Library/Fonts/Supplemental/Times New Roman.ttf",
     "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
     "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
     "/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf"),
]


@dataclass(frozen=True)
class Block:
    kind: str  # title | h1 | h2 | h3 | para | bullet | reference
    text: str


def markdown_to_blocks(markdown: str) -> list[Block]:
    """A deliberately small Markdown reader: headings, paragraphs, bullets and reference lines."""
    blocks: list[Block] = []
    in_references = False
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            text = " ".join(s.strip() for s in paragraph).strip()
            if text:
                blocks.append(Block("reference" if in_references else "para", text))
            paragraph.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("# "):
            flush()
            blocks.append(Block("title", stripped[2:].strip()))
        elif stripped.startswith("## "):
            flush()
            heading = stripped[3:].strip()
            in_references = heading.lower() == "references"
            blocks.append(Block("h2", heading))
        elif stripped.startswith("### "):
            flush()
            blocks.append(Block("h3", stripped[4:].strip()))
        elif stripped.startswith(("- ", "* ")):
            flush()
            blocks.append(Block("bullet", stripped[2:].strip()))
        elif in_references and re.match(r"^\[\d+\]", stripped):
            flush()
            blocks.append(Block("reference", stripped))
        else:
            paragraph.append(stripped)
    flush()
    return blocks


def _transliterate(text: str) -> str:
    for src, dst in _TRANSLIT.items():
        text = text.replace(src, dst)
    return text


def _inline_segments(text: str) -> list[tuple[str, bool, bool]]:
    """Split **bold** and *italic* spans. Returns (segment, bold, italic) tuples."""
    segments: list[tuple[str, bool, bool]] = []
    for token in re.split(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)", text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) > 4:
            segments.append((token[2:-2], True, False))
        elif token.startswith("*") and token.endswith("*") and len(token) > 2:
            segments.append((token[1:-1], False, True))
        else:
            segments.append((token, False, False))
    return segments


# --------------------------------------------------------------------------- PDF

def _register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for regular, bold, italic, bold_italic in _BODY_FONT_CANDIDATES:
        if all(Path(p).exists() for p in (regular, bold, italic, bold_italic)):
            pdfmetrics.registerFont(TTFont("Body", regular))
            pdfmetrics.registerFont(TTFont("Body-Bold", bold))
            pdfmetrics.registerFont(TTFont("Body-Italic", italic))
            pdfmetrics.registerFont(TTFont("Body-BoldItalic", bold_italic))
            from reportlab.pdfbase.pdfmetrics import registerFontFamily

            registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                                boldItalic="Body-BoldItalic")
            return "Body"
    logger.warning("No serif TTF font found; falling back to Helvetica")
    return "Helvetica"


def build_pdf(markdown: str, title: str, generated_on: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    family = _register_fonts()
    bold = "Body-Bold" if family == "Body" else "Helvetica-Bold"
    body_font = family

    from xml.sax.saxutils import escape

    def rich(text: str) -> str:
        pieces = []
        for segment, is_bold, is_italic in _inline_segments(_transliterate(text)):
            safe = escape(segment)
            if is_bold:
                safe = f"<b>{safe}</b>"
            if is_italic:
                safe = f"<i>{safe}</i>"
            pieces.append(safe)
        return "".join(pieces)

    styles = {
        "title": ParagraphStyle("Title", fontName=bold, fontSize=20, leading=25, alignment=TA_CENTER,
                                spaceAfter=10, textColor=colors.HexColor("#111111")),
        "meta": ParagraphStyle("Meta", fontName=body_font, fontSize=10, leading=14, alignment=TA_CENTER,
                               textColor=colors.HexColor("#555555"), spaceAfter=18),
        "h2": ParagraphStyle("H2", fontName=bold, fontSize=14, leading=18, spaceBefore=16, spaceAfter=6,
                             textColor=colors.HexColor("#1B3A8A")),
        "h3": ParagraphStyle("H3", fontName=bold, fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=4),
        "para": ParagraphStyle("Para", fontName=body_font, fontSize=10.5, leading=15, alignment=TA_JUSTIFY,
                               spaceAfter=7),
        "bullet": ParagraphStyle("Bullet", fontName=body_font, fontSize=10.5, leading=15, leftIndent=14,
                                 bulletIndent=2, spaceAfter=4),
        "reference": ParagraphStyle("Reference", fontName=body_font, fontSize=9.5, leading=13, leftIndent=24,
                                    firstLineIndent=-24, spaceAfter=5),
    }

    flowables = []
    flowables.append(Paragraph(rich(title), styles["title"]))
    flowables.append(Paragraph(f"Generated on {escape(generated_on)} by the Multi-Agent Research Platform",
                               styles["meta"]))
    for block in markdown_to_blocks(markdown):
        if block.kind == "title":
            continue  # already printed as the cover title
        if block.kind == "h2":
            flowables.append(Paragraph(rich(block.text), styles["h2"]))
        elif block.kind == "h3":
            flowables.append(Paragraph(rich(block.text), styles["h3"]))
        elif block.kind == "bullet":
            flowables.append(Paragraph(rich(block.text), styles["bullet"], bulletText="•"))
        elif block.kind == "reference":
            flowables.append(Paragraph(rich(block.text), styles["reference"]))
        else:
            flowables.append(Paragraph(rich(block.text), styles["para"]))
    flowables.append(Spacer(1, 6))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm, topMargin=2.3 * cm,
        bottomMargin=2.2 * cm, title=_transliterate(title), author="Multi-Agent Research Platform",
    )

    def decorate(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(body_font, 8.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(2.3 * cm, A4[1] - 1.4 * cm, _transliterate(title)[:95])
        canvas.drawRightString(A4[0] - 2.3 * cm, 1.2 * cm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(flowables, onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()


# --------------------------------------------------------------------------- DOCX

def build_docx(markdown: str, title: str, generated_on: str) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    document = Document()
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)

    title_paragraph = document.add_heading(title, level=0)
    title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta = document.add_paragraph(f"Generated on {generated_on} by the Multi-Agent Research Platform")
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def add_runs(paragraph, text: str) -> None:
        for segment, is_bold, is_italic in _inline_segments(text):
            run = paragraph.add_run(segment)
            run.bold = is_bold
            run.italic = is_italic

    for block in markdown_to_blocks(markdown):
        if block.kind == "title":
            continue
        if block.kind == "h2":
            document.add_heading(block.text, level=1)
        elif block.kind == "h3":
            document.add_heading(block.text, level=2)
        elif block.kind == "bullet":
            add_runs(document.add_paragraph(style="List Bullet"), block.text)
        elif block.kind == "reference":
            add_runs(document.add_paragraph(), block.text)
        else:
            add_runs(document.add_paragraph(), block.text)

    # Page number field in the footer.
    footer = document.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Page ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
