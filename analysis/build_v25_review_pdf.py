from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
)


DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u207b": "-",
    }
)


def register_fonts() -> None:
    font_dir = Path(r"C:\Windows\Fonts")
    font_files = {
        "Arial": "arial.ttf",
        "Arial-Bold": "arialbd.ttf",
        "Arial-Italic": "ariali.ttf",
        "Arial-BoldItalic": "arialbi.ttf",
    }
    for name, filename in font_files.items():
        path = font_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "Arial",
        normal="Arial",
        bold="Arial-Bold",
        italic="Arial-Italic",
        boldItalic="Arial-BoldItalic",
    )


def inline_markup(text: str) -> str:
    text = text.translate(DASH_TRANSLATION)
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Arial">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#2f657e">\1</link>',
        escaped,
    )
    return escaped


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReviewTitle",
            parent=base["Title"],
            fontName="Arial-Bold",
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17233b"),
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "ReviewH1",
            parent=base["Heading1"],
            fontName="Arial-Bold",
            fontSize=13.5,
            leading=17,
            textColor=colors.HexColor("#17233b"),
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ReviewH2",
            parent=base["Heading2"],
            fontName="Arial-Bold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#71305f"),
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "ReviewH3",
            parent=base["Heading3"],
            fontName="Arial-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#17233b"),
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReviewBody",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=9.4,
            leading=13,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#17233b"),
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "reference": ParagraphStyle(
            "ReviewReference",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=8.8,
            leading=12,
            alignment=TA_LEFT,
            leftIndent=14,
            firstLineIndent=-14,
            textColor=colors.HexColor("#17233b"),
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "ReviewBullet",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=9.2,
            leading=12.6,
            leftIndent=14,
            firstLineIndent=-8,
            bulletIndent=3,
            textColor=colors.HexColor("#17233b"),
            spaceAfter=3,
            wordWrap="CJK",
        ),
    }


def markdown_story(markdown_path: Path, styles: dict[str, ParagraphStyle]):
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    story = []
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_buffer:
            return
        text = " ".join(piece.strip() for piece in paragraph_buffer).strip()
        paragraph_buffer.clear()
        style = styles["reference"] if re.match(r"^\d+\.\s", text) else styles["body"]
        story.append(Paragraph(inline_markup(text), style))

    for raw in lines:
        line = raw.rstrip()
        if not line:
            flush_paragraph()
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            style = styles["title"] if level == 1 and not story else styles[f"h{level}"]
            story.append(Paragraph(inline_markup(heading.group(2)), style))
            continue
        bullet = re.match(r"^[-*]\s+(.+)$", line)
        if bullet:
            flush_paragraph()
            story.append(
                Paragraph(inline_markup(bullet.group(1)), styles["bullet"], bulletText="•")
            )
            continue
        if re.match(r"^\d+\.\s", line):
            flush_paragraph()
            story.append(Paragraph(inline_markup(line), styles["reference"]))
            continue
        paragraph_buffer.append(line)
    flush_paragraph()
    return story


class ReviewDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, document_label: str):
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=0.85 * inch,
            rightMargin=0.85 * inch,
            topMargin=0.82 * inch,
            bottomMargin=0.72 * inch,
            title=document_label,
            author="PlumSPECTRA study team",
            subject="Frozen external-review draft",
        )
        self.document_label = document_label
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates(PageTemplate(id="review", frames=frame, onPage=self.draw_page))

    def draw_page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d8dde6"))
        canvas.setLineWidth(0.45)
        canvas.line(self.leftMargin, letter[1] - 0.58 * inch, letter[0] - self.rightMargin, letter[1] - 0.58 * inch)
        canvas.setFont("Arial", 7.6)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(self.leftMargin, letter[1] - 0.48 * inch, self.document_label)
        canvas.drawRightString(letter[0] - self.rightMargin, 0.42 * inch, f"Page {doc.page}")
        canvas.restoreState()


def build(markdown_path: Path, output_path: Path, label: str) -> None:
    register_fonts()
    styles = make_styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = ReviewDocTemplate(str(output_path), label)
    story = markdown_story(markdown_path, styles)
    story.append(Spacer(1, 0.1 * inch))
    document.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    build(args.markdown.resolve(), args.output.resolve(), args.label)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
