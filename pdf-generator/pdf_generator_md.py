#!/usr/bin/env python3
"""
Markdown-focused PDF Generator (CLI)

Generate customizable PDFs from Markdown content with configurable logo,
title, footer, and page layout settings.

Usage:
    python3 pdf_generator_md.py --help
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

DEJAVU_PATHS = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/dejavu",
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
]


def register_unicode_fonts() -> None:
    """Register DejaVu fonts when available."""
    fonts_to_register = [
        ("DejaVuSans", "DejaVuSans.ttf"),
        ("DejaVuSans-Bold", "DejaVuSans-Bold.ttf"),
        ("DejaVuSans-Oblique", "DejaVuSans-Oblique.ttf"),
        ("DejaVuSans-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
        ("DejaVuSansMono", "DejaVuSansMono.ttf"),
    ]

    for dejavu_path in DEJAVU_PATHS:
        if not os.path.exists(dejavu_path):
            continue
        try:
            for font_name, font_file in fonts_to_register:
                font_path = os.path.join(dejavu_path, font_file)
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
            return
        except Exception:
            continue


register_unicode_fonts()


@dataclass
class PDFConfig:
    output_file: str = "generated_document.pdf"
    logo_path: str = ""
    logo_width: float = 1.0
    logo_height: float = 1.0
    logo_position: str = "side-by-side"

    title: str = "Document Title"
    title_font_size: int = 14
    title_alignment: str = "left"

    body_markdown: str = "Document body content goes here."
    body_font_size: int = 10
    body_alignment: str = "left"

    footer_text: str = ""
    footer_font_size: int = 7
    footer_alignment: str = "center"

    include_header: bool = True
    include_footer: bool = True

    margin_top: float = 0.75
    margin_bottom: float = 1.0
    margin_left: float = 0.75
    margin_right: float = 0.75

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def from_dict(self, data: dict) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_file(self, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_file(self, filename: str) -> None:
        with open(filename, "r", encoding="utf-8") as f:
            self.from_dict(json.load(f))


def get_alignment(alignment: str) -> int:
    return {
        "left": TA_LEFT,
        "center": TA_CENTER,
        "right": TA_RIGHT,
        "justify": TA_JUSTIFY,
    }.get((alignment or "left").lower(), TA_LEFT)


def escape_markup(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def convert_inline_markdown(text: str) -> str:
    escaped = escape_markup(text)

    code_map = {}

    def stash_code(match: re.Match) -> str:
        token = f"@@CODE{len(code_map)}@@"
        code_map[token] = f'<font name="DejaVuSansMono">{escape_markup(match.group(1))}</font>'
        return token

    escaped = re.sub(r"`([^`]+)`", stash_code, escaped)

    escaped = re.sub(r"\[(.+?)\]\((https?://[^\s)]+)\)", r'<a href="\2"><u><font color="blue">\1</font></u></a>', escaped)
    escaped = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", escaped)
    escaped = re.sub(r"___(.+?)___", r"<b><i>\1</i></b>", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"__(.+?)__", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"~~(.+?)~~", r"<strike>\1</strike>", escaped)

    for token, value in code_map.items():
        escaped = escaped.replace(token, value)

    return escaped


def parse_list_block(lines: List[str], start_index: int) -> Tuple[ListFlowable, int]:
    def helper(idx: int, base_indent: int, ordered: bool) -> Tuple[List[ListItem], int]:
        items: List[ListItem] = []
        while idx < len(lines):
            raw = lines[idx]
            if not raw.strip():
                idx += 1
                break

            unordered_match = re.match(r"^(\s*)[-+*]\s+(.+)$", raw)
            ordered_match = re.match(r"^(\s*)(\d+)\.\s+(.+)$", raw)

            if not unordered_match and not ordered_match:
                break

            if ordered_match:
                indent = len(ordered_match.group(1).replace("\t", "    "))
                text = ordered_match.group(3)
                current_ordered = True
            else:
                indent = len(unordered_match.group(1).replace("\t", "    "))
                text = unordered_match.group(2)
                current_ordered = False

            if indent < base_indent:
                break

            if indent > base_indent and items:
                nested_items, idx = helper(idx, indent, current_ordered)
                nested_flow = ListFlowable(
                    nested_items,
                    bulletType="1" if current_ordered else "bullet",
                    start="1",
                    leftIndent=18,
                )
                last = items[-1]
                flowables = list(getattr(last, "_flowables", []))
                flowables.append(nested_flow)
                last._flowables = flowables
                continue

            para = Paragraph(convert_inline_markdown(text), getSampleStyleSheet()["BodyText"])
            items.append(ListItem(para, leftIndent=0))
            idx += 1

        return items, idx

    first_ordered = bool(re.match(r"^\s*\d+\.\s+", lines[start_index]))
    base_indent = len(re.match(r"^(\s*)", lines[start_index]).group(1).replace("\t", "    "))
    list_items, end_idx = helper(start_index, base_indent, first_ordered)

    flowable = ListFlowable(
        list_items,
        bulletType="1" if first_ordered else "bullet",
        start="1",
        leftIndent=18,
    )
    return flowable, end_idx


def parse_markdown_to_story(markdown_text: str, config: PDFConfig, body_style: ParagraphStyle) -> List:
    lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    story = []
    paragraph_buffer: List[str] = []
    code_style = ParagraphStyle(
        "CodeBlockStyle",
        parent=body_style,
        fontName="DejaVuSansMono",
        fontSize=max(8, config.body_font_size - 1),
        leading=max(10, config.body_font_size + 1),
        backColor=colors.HexColor("#f5f5f5"),
        borderColor=colors.HexColor("#dddddd"),
        borderWidth=0.6,
        borderPadding=6,
        leftIndent=8,
        rightIndent=8,
        spaceBefore=4,
        spaceAfter=6,
    )
    quote_style = ParagraphStyle(
        "QuoteStyle",
        parent=body_style,
        leftIndent=16,
        rightIndent=8,
        textColor=colors.HexColor("#444444"),
        fontName="DejaVuSans-Oblique",
        borderColor=colors.HexColor("#cccccc"),
        borderWidth=0.6,
        borderPadding=8,
        spaceBefore=3,
        spaceAfter=3,
    )

    def flush_paragraph() -> None:
        if paragraph_buffer:
            joined = " ".join(line.strip() for line in paragraph_buffer if line.strip())
            if joined:
                story.append(Paragraph(convert_inline_markdown(joined), body_style))
                story.append(Spacer(1, 0.08 * inch))
            paragraph_buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        fence_match = re.match(r"^```", stripped)
        if fence_match:
            flush_paragraph()
            i += 1
            code_lines = []
            while i < len(lines) and not re.match(r"^```", lines[i].strip()):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            story.append(Preformatted("\n".join(code_lines), style=code_style))
            story.append(Spacer(1, 0.08 * inch))
            continue

        if re.match(r"^\s{4,}\S", line):
            flush_paragraph()
            code_lines = [line[4:]]
            i += 1
            while i < len(lines) and (re.match(r"^\s{4,}\S", lines[i]) or not lines[i].strip()):
                code_lines.append(lines[i][4:] if lines[i].startswith("    ") else "")
                i += 1
            story.append(Preformatted("\n".join(code_lines).rstrip(), style=code_style))
            story.append(Spacer(1, 0.08 * inch))
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading_text = convert_inline_markdown(heading_match.group(2).strip())
            size_map = {1: 20, 2: 17, 3: 15, 4: 13, 5: 12, 6: 11}
            heading_style = ParagraphStyle(
                f"Heading{level}",
                parent=body_style,
                fontName="DejaVuSans-Bold",
                fontSize=max(config.body_font_size + 1, size_map[level]),
                leading=max(config.body_font_size + 2, size_map[level] + 2),
                spaceBefore=6,
                spaceAfter=6,
            )
            story.append(Paragraph(heading_text, heading_style))
            i += 1
            continue

        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            flush_paragraph()
            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#888888")))
            story.append(Spacer(1, 0.1 * inch))
            i += 1
            continue

        if re.match(r"^>\s?", stripped):
            flush_paragraph()
            quote_lines = []
            while i < len(lines) and re.match(r"^\s*>\s?", lines[i].strip()):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[i].strip()))
                i += 1
            quote_text = convert_inline_markdown("<br/>".join(quote_lines))
            story.append(Paragraph(quote_text, quote_style))
            story.append(Spacer(1, 0.06 * inch))
            continue

        image_match = re.match(r"^!\[(.*?)\]\(([^)\s]+)\)$", stripped)
        if image_match:
            flush_paragraph()
            image_path = image_match.group(2)
            if os.path.exists(image_path):
                try:
                    story.append(Image(image_path, width=3.5 * inch, preserveAspectRatio=True, hAlign="LEFT"))
                    story.append(Spacer(1, 0.08 * inch))
                except Exception:
                    alt_text = image_match.group(1) or image_path
                    story.append(Paragraph(f"[Image: {escape_markup(alt_text)}]", body_style))
            else:
                alt_text = image_match.group(1) or image_path
                story.append(Paragraph(f"[Image not found: {escape_markup(alt_text)}]", body_style))
            i += 1
            continue

        if re.match(r"^\s*([-+*]\s+|\d+\.\s+)", line):
            flush_paragraph()
            list_flowable, i = parse_list_block(lines, i)
            story.append(list_flowable)
            story.append(Spacer(1, 0.08 * inch))
            continue

        table_header_match = "|" in stripped and stripped.startswith("|") and stripped.endswith("|")
        if table_header_match and i + 1 < len(lines):
            separator = lines[i + 1].strip()
            if re.match(r"^\|(?:\s*:?-{2,}:?\s*\|)+$", separator):
                flush_paragraph()
                table_lines = [lines[i]]
                i += 1
                table_lines.append(lines[i])
                i += 1
                while i < len(lines):
                    row = lines[i].strip()
                    if not (row.startswith("|") and row.endswith("|")):
                        break
                    table_lines.append(lines[i])
                    i += 1

                rows = []
                for idx, row in enumerate(table_lines):
                    if idx == 1:
                        continue
                    cells = [convert_inline_markdown(c.strip()) for c in row.strip("|").split("|")]
                    rows.append([Paragraph(cell, body_style) for cell in cells])

                if rows:
                    col_count = max(len(r) for r in rows)
                    for row in rows:
                        while len(row) < col_count:
                            row.append(Paragraph("", body_style))
                    table = Table(rows, repeatRows=1)
                    table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efefef")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ]
                        )
                    )
                    story.append(table)
                    story.append(Spacer(1, 0.08 * inch))
                continue

        paragraph_buffer.append(line)
        i += 1

    flush_paragraph()
    return story


def generate_pdf(config: PDFConfig) -> None:
    if config.logo_path:
        if not os.path.exists(config.logo_path):
            raise FileNotFoundError(f"Logo file not found: {config.logo_path}")
        valid_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
        if not config.logo_path.lower().endswith(valid_extensions):
            raise ValueError(f"Logo file must be one of: {', '.join(valid_extensions)}")

    doc = SimpleDocTemplate(
        config.output_file,
        pagesize=A4,
        rightMargin=config.margin_right * inch,
        leftMargin=config.margin_left * inch,
        topMargin=config.margin_top * inch,
        bottomMargin=config.margin_bottom * inch,
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        fontName="DejaVuSans",
        fontSize=config.body_font_size,
        leading=config.body_font_size * 1.45,
        alignment=get_alignment(config.body_alignment),
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=4,
    )
    title_style = ParagraphStyle(
        "TitleStyle",
        fontName="DejaVuSans-Bold",
        fontSize=config.title_font_size,
        leading=config.title_font_size * 1.2,
        alignment=get_alignment(config.title_alignment),
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=12,
    )

    story = []

    def add_footer(canvas, _doc):
        if not (config.include_footer and config.footer_text):
            return
        canvas.saveState()
        canvas.setFont("DejaVuSans", config.footer_font_size)
        canvas.setFillColor(colors.HexColor("#666666"))
        y_pos = 0.5 * inch
        footer_alignment = get_alignment(config.footer_alignment)
        if footer_alignment == TA_CENTER:
            canvas.drawCentredString(A4[0] / 2, y_pos, config.footer_text)
        elif footer_alignment == TA_RIGHT:
            canvas.drawRightString(A4[0] - config.margin_right * inch, y_pos, config.footer_text)
        else:
            canvas.drawString(config.margin_left * inch, y_pos, config.footer_text)
        canvas.restoreState()

    if config.include_header:
        title_para = Paragraph(escape_markup(config.title), title_style)
        if config.logo_path and os.path.exists(config.logo_path):
            logo_img = Image(config.logo_path, width=config.logo_width * inch, height=config.logo_height * inch)
            if config.logo_position == "side-by-side":
                available_width = A4[0] / inch - config.margin_left - config.margin_right
                title_width = max(1.0, available_width - config.logo_width - 0.2)
                header_table = Table([[logo_img, title_para]], colWidths=[config.logo_width * inch, title_width * inch])
                header_table.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (0, 0), 0),
                            ("LEFTPADDING", (1, 0), (1, 0), 10),
                            ("RIGHTPADDING", (1, 0), (1, 0), 0),
                        ]
                    )
                )
                story.append(header_table)
            else:
                logo_table = Table([[logo_img]], colWidths=[config.logo_width * inch])
                logo_table.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "CENTER")]))
                story.append(logo_table)
                story.append(Spacer(1, 0.2 * inch))
                story.append(title_para)
        else:
            story.append(title_para)

        story.append(Spacer(1, 0.2 * inch))

    story.extend(parse_markdown_to_story(config.body_markdown or "", config, body_style))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(f"PDF generated successfully: {config.output_file}")


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PDF from Markdown content with configurable layout"
    )

    parser.add_argument("--output", "-o", default="generated_document.pdf", help="Output PDF filename")

    parser.add_argument("--logo", help="Path to logo image file")
    parser.add_argument("--logo-width", type=float, default=1.0, help="Logo width in inches (default: 1.0)")
    parser.add_argument("--logo-height", type=float, default=1.0, help="Logo height in inches (default: 1.0)")
    parser.add_argument(
        "--logo-position",
        choices=["side-by-side", "top-center"],
        default="side-by-side",
        help="Logo position (default: side-by-side)",
    )

    parser.add_argument("--title", default="Document Title", help="Document title text")
    parser.add_argument("--title-size", type=int, default=14, help="Title font size (default: 14)")
    parser.add_argument(
        "--title-align",
        choices=["left", "center", "right"],
        default="left",
        help="Title alignment (default: left)",
    )

    parser.add_argument(
        "--body",
        help="Markdown body text. Use \\n for line breaks and \\n\\n for paragraph separation.",
    )
    parser.add_argument("--body-file", help="Read Markdown body from file")
    parser.add_argument("--body-size", type=int, default=10, help="Body font size (default: 10)")
    parser.add_argument(
        "--body-align",
        choices=["left", "center", "right", "justify"],
        default="left",
        help="Body alignment (default: left)",
    )

    parser.add_argument("--footer", help="Footer text")
    parser.add_argument("--footer-size", type=int, default=7, help="Footer font size (default: 7)")
    parser.add_argument(
        "--footer-align",
        choices=["left", "center", "right"],
        default="center",
        help="Footer alignment (default: center)",
    )

    parser.add_argument("--margin-top", type=float, default=0.75, help="Top margin in inches (default: 0.75)")
    parser.add_argument("--margin-bottom", type=float, default=1.0, help="Bottom margin in inches (default: 1.0)")
    parser.add_argument("--margin-left", type=float, default=0.75, help="Left margin in inches (default: 0.75)")
    parser.add_argument("--margin-right", type=float, default=0.75, help="Right margin in inches (default: 0.75)")

    parser.add_argument("--config", help="Load settings from JSON config file")
    parser.add_argument("--save-config", help="Save current settings to JSON config file")

    args = parser.parse_args()

    config = PDFConfig()

    if args.config:
        config.load_from_file(args.config)

    config.output_file = args.output
    if args.logo:
        config.logo_path = args.logo
    config.logo_width = args.logo_width
    config.logo_height = args.logo_height
    config.logo_position = args.logo_position

    config.title = args.title
    config.title_font_size = args.title_size
    config.title_alignment = args.title_align

    if args.body:
        config.body_markdown = args.body.replace("\\n", "\n")
    elif args.body_file:
        with open(args.body_file, "r", encoding="utf-8") as f:
            config.body_markdown = f.read()

    config.body_font_size = args.body_size
    config.body_alignment = args.body_align

    if args.footer:
        config.footer_text = args.footer
    config.footer_font_size = args.footer_size
    config.footer_alignment = args.footer_align

    config.margin_top = args.margin_top
    config.margin_bottom = args.margin_bottom
    config.margin_left = args.margin_left
    config.margin_right = args.margin_right

    if args.save_config:
        config.save_to_file(args.save_config)
        print(f"Configuration saved to: {args.save_config}")

    generate_pdf(config)


if __name__ == "__main__":
    run_cli()
