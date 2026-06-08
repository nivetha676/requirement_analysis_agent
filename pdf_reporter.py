# pdf_reporter.py
# ─────────────────────────────────────────────────────────────
# Saves the analysis results as a formatted PDF report.
# Uses reportlab — fully offline, no external services.
#
# Usage (from main.py):
#   from pdf_reporter import save_pdf
#   save_pdf(results, Path("report.pdf"))
# ─────────────────────────────────────────────────────────────

from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

# ── Colour palette ────────────────────────────────────────────

C_BLACK       = colors.HexColor("#111111")
C_WHITE       = colors.HexColor("#FFFFFF")
C_LIGHT_GRAY  = colors.HexColor("#F5F5F5")
C_MID_GRAY    = colors.HexColor("#E0E0E0")
C_DARK_GRAY   = colors.HexColor("#666666")

C_HIGH_BG     = colors.HexColor("#FCEBEB")
C_HIGH_FG     = colors.HexColor("#A32D2D")
C_HIGH_BORDER = colors.HexColor("#E24B4A")

C_MEDIUM_BG   = colors.HexColor("#FAEEDA")
C_MEDIUM_FG   = colors.HexColor("#854F0B")
C_MEDIUM_BORDER = colors.HexColor("#EF9F27")

C_LOW_BG      = colors.HexColor("#EAF3DE")
C_LOW_FG      = colors.HexColor("#3B6D11")
C_LOW_BORDER  = colors.HexColor("#639922")

C_CLEAR_BG    = colors.HexColor("#E1F5EE")
C_CLEAR_FG    = colors.HexColor("#0F6E56")
C_CLEAR_BORDER = colors.HexColor("#1D9E75")

LEVEL_COLORS = {
    "High":   (C_HIGH_BG,   C_HIGH_FG,   C_HIGH_BORDER),
    "Medium": (C_MEDIUM_BG, C_MEDIUM_FG, C_MEDIUM_BORDER),
    "Low":    (C_LOW_BG,    C_LOW_FG,    C_LOW_BORDER),
    "Clear":  (C_CLEAR_BG,  C_CLEAR_FG,  C_CLEAR_BORDER),
}

LEVEL_ICONS = {
    "High":   "HIGH",
    "Medium": "MEDIUM",
    "Low":    "LOW",
    "Clear":  "CLEAR",
}


# ── Custom left-border flowable ───────────────────────────────

class LeftBorderBox(Flowable):
    """Draws a coloured left border beside a block of content."""

    def __init__(self, content_width, content_height, border_color,
                 bg_color, border_width=4):
        super().__init__()
        self.content_width  = content_width
        self.content_height = content_height
        self.border_color   = border_color
        self.bg_color       = bg_color
        self.border_width   = border_width
        self.width          = content_width
        self.height         = content_height

    def draw(self):
        canvas = self.canv
        # Background
        canvas.setFillColor(self.bg_color)
        canvas.rect(0, 0, self.content_width, self.content_height,
                    fill=1, stroke=0)
        # Left border stripe
        canvas.setFillColor(self.border_color)
        canvas.rect(0, 0, self.border_width, self.content_height,
                    fill=1, stroke=0)


# ── Styles ────────────────────────────────────────────────────

def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=C_BLACK,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            fontName="Helvetica",
            fontSize=10,
            textColor=C_DARK_GRAY,
            spaceAfter=2,
        ),
        "section_header": ParagraphStyle(
            "SectionHeader",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=C_BLACK,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "req_id": ParagraphStyle(
            "ReqId",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=C_DARK_GRAY,
            spaceAfter=2,
        ),
        "req_text": ParagraphStyle(
            "ReqText",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=C_BLACK,
            spaceAfter=6,
            leading=15,
        ),
        "field_label": ParagraphStyle(
            "FieldLabel",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=C_DARK_GRAY,
            spaceBefore=6,
            spaceAfter=2,
        ),
        "field_value": ParagraphStyle(
            "FieldValue",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_BLACK,
            spaceAfter=2,
            leading=13,
            leftIndent=10,
        ),
        "bullet_item": ParagraphStyle(
            "BulletItem",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_BLACK,
            spaceAfter=3,
            leading=13,
            leftIndent=14,
            bulletIndent=4,
        ),
        "rewrite": ParagraphStyle(
            "Rewrite",
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=colors.HexColor("#1a3a8a"),
            spaceAfter=2,
            leading=13,
            leftIndent=10,
        ),
        "sources": ParagraphStyle(
            "Sources",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=C_DARK_GRAY,
            spaceAfter=2,
            leftIndent=10,
        ),
        "footer": ParagraphStyle(
            "Footer",
            fontName="Helvetica",
            fontSize=8,
            textColor=C_DARK_GRAY,
            alignment=TA_CENTER,
        ),
        "toc_item": ParagraphStyle(
            "TocItem",
            fontName="Helvetica",
            fontSize=10,
            textColor=C_BLACK,
            spaceAfter=3,
            leading=14,
        ),
    }
    return styles


# ── Page template with header/footer ─────────────────────────

class _PageTemplate:
    def __init__(self, kb_name: str | None):
        self.kb_name = kb_name

    def on_page(self, canvas, doc):
        canvas.saveState()
        w, h = A4

        # Top rule
        canvas.setStrokeColor(C_MID_GRAY)
        canvas.setLineWidth(0.5)
        canvas.line(15 * mm, h - 12 * mm, w - 15 * mm, h - 12 * mm)

        # Footer rule
        canvas.line(15 * mm, 14 * mm, w - 15 * mm, 14 * mm)

        # Footer text
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(C_DARK_GRAY)
        canvas.drawString(15 * mm, 9 * mm, "Requirements Ambiguity Analysis Report")
        canvas.drawRightString(
            w - 15 * mm, 9 * mm,
            f"Page {doc.page}"
        )
        if self.kb_name:
            canvas.drawCentredString(
                w / 2, 9 * mm,
                f"Knowledge base: {self.kb_name}"
            )

        canvas.restoreState()


# ── Summary table ─────────────────────────────────────────────

def _build_summary_table(results: list[dict], styles: dict):
    counts = {"High": 0, "Medium": 0, "Low": 0, "Clear": 0}
    for r in results:
        lvl = r.get("ambiguityLevel", "Clear")
        counts[lvl] = counts.get(lvl, 0) + 1

    need_info = counts["High"] + counts["Medium"]

    data = [
        ["Metric", "Count"],
        ["Total requirements analysed", str(len(results))],
        ["Need more info  (High + Medium)", str(need_info)],
        ["  High ambiguity", str(counts["High"])],
        ["  Medium ambiguity", str(counts["Medium"])],
        ["  Low ambiguity", str(counts["Low"])],
        ["  Clear & testable", str(counts["Clear"])],
    ]

    col_widths = [110 * mm, 30 * mm]

    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",  (0, 0), (-1, 0), C_BLACK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 10),
        ("ALIGN",       (1, 0), (1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT_GRAY]),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("GRID",        (0, 0), (-1, -1), 0.3, C_MID_GRAY),
        # Colour the count cells
        ("TEXTCOLOR",   (1, 2), (1, 3), C_HIGH_FG),
        ("FONTNAME",    (1, 2), (1, 3), "Helvetica-Bold"),
        ("TEXTCOLOR",   (1, 4), (1, 4), C_MEDIUM_FG),
        ("FONTNAME",    (1, 4), (1, 4), "Helvetica-Bold"),
        ("TEXTCOLOR",   (1, 5), (1, 5), C_LOW_FG),
        ("TEXTCOLOR",   (1, 6), (1, 6), C_CLEAR_FG),
        ("FONTNAME",    (1, 6), (1, 6), "Helvetica-Bold"),
    ]))
    return table


# ── Badge paragraph ───────────────────────────────────────────

def _badge(level: str) -> Paragraph:
    _, fg, _ = LEVEL_COLORS.get(level, (C_LIGHT_GRAY, C_BLACK, C_MID_GRAY))
    bg, _, _ = LEVEL_COLORS.get(level, (C_LIGHT_GRAY, C_BLACK, C_MID_GRAY))
    label    = LEVEL_ICONS.get(level, level.upper())

    # Encode colors to hex strings for inline HTML
    fg_hex = fg.hexval() if hasattr(fg, 'hexval') else "#000000"
    bg_hex = bg.hexval() if hasattr(bg, 'hexval') else "#ffffff"

    style = ParagraphStyle(
        "Badge",
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=fg,
        backColor=bg,
        borderPadding=(3, 6, 3, 6),
        borderRadius=3,
    )
    return Paragraph(f" {label} ", style)


# ── Per-requirement card ──────────────────────────────────────

def _build_req_card(r: dict, styles: dict, page_width: float) -> list:
    """Return a list of flowables representing one requirement card."""
    lvl        = r.get("ambiguityLevel", "Clear")
    bg, fg, bd = LEVEL_COLORS.get(lvl, (C_LIGHT_GRAY, C_BLACK, C_MID_GRAY))
    story      = []

    inner_width = page_width - 30 * mm   # page margins = 15mm each side
    pad         = 10                      # inner padding points

    card_items = []

    # ── REQ ID + badge row ────────────────────────────────────
    badge_style = ParagraphStyle(
        "BadgeInline",
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=fg,
        backColor=bg,
    )
    id_style = ParagraphStyle(
        "ReqIdInline",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=C_DARK_GRAY,
    )

    header_data = [[
        Paragraph(r.get("id", "REQ"), id_style),
        Paragraph(f"[ {lvl.upper()} ]", badge_style),
    ]]
    header_table = Table(header_data, colWidths=[inner_width - 80, 70])
    header_table.setStyle(TableStyle([
        ("ALIGN",        (1, 0), (1, 0), "RIGHT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    card_items.append(header_table)

    # ── Requirement text ──────────────────────────────────────
    card_items.append(Paragraph(
        _escape(r.get("requirementText", "")),
        styles["req_text"],
    ))

    # ── Knowledge base sources ────────────────────────────────
    sources = r.get("_sources_used", [])
    if sources:
        card_items.append(Paragraph(
            "Knowledge base: " + ", ".join(sources),
            styles["sources"],
        ))

    # ── Section helper ────────────────────────────────────────
    def add_section(label: str, items: list, item_style="bullet_item"):
        if not items:
            return
        card_items.append(Paragraph(label, styles["field_label"]))
        for item in items:
            card_items.append(Paragraph(
                f"• {_escape(str(item))}",
                styles[item_style],
            ))

    # ── Missing information ───────────────────────────────────
    add_section("What is missing:", r.get("missingInformation", []))

    # ── Ambiguous terms ───────────────────────────────────────
    terms = r.get("ambiguousTerms", [])
    if terms:
        card_items.append(Paragraph("Vague terms:", styles["field_label"]))
        for t in terms:
            term    = _escape(str(t.get("term", "")))
            problem = _escape(str(t.get("problem", "")))
            example = _escape(str(t.get("example", "")))
            card_items.append(Paragraph(
                f'• <b>"{term}"</b> — {problem}',
                styles["bullet_item"],
            ))
            if example:
                card_items.append(Paragraph(
                    f'  &nbsp;&nbsp;&nbsp;&#8627; {example}',
                    styles["field_value"],
                ))

    # ── Missing acceptance criteria ───────────────────────────
    add_section(
        "Missing acceptance criteria:",
        r.get("missingAcceptanceCriteria", []),
    )

    # ── Edge cases ────────────────────────────────────────────
    add_section(
        "Edge cases not covered:",
        r.get("edgeCasesNotCovered", []),
    )

    # ── Clarifying questions ──────────────────────────────────
    add_section(
        "Ask the product owner:",
        r.get("clarifyingQuestions", []),
    )

    # ── Rewrite suggestion ────────────────────────────────────
    rewrite = r.get("rewriteSuggestion", "")
    if rewrite:
        card_items.append(Paragraph("Suggested rewrite:", styles["field_label"]))
        card_items.append(Paragraph(_escape(rewrite), styles["rewrite"]))

    # ── Wrap card items in a padded table ─────────────────────
    inner_table = Table(
        [[item] for item in card_items],
        colWidths=[inner_width - 2 * pad - 6],   # 6 = left border width
    )
    inner_table.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), pad),
        ("RIGHTPADDING", (0, 0), (-1, -1), pad),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))

    # ── Outer wrapper with coloured left border ───────────────
    outer_table = Table(
        [[inner_table]],
        colWidths=[inner_width],
    )
    outer_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), bg),
        ("LINEBEFORE",   (0, 0), (0, -1), 5, bd),
        ("LINEBELOW",    (0, -1), (-1, -1), 0.3, C_MID_GRAY),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))

    story.append(KeepTogether(outer_table))
    story.append(Spacer(1, 8))
    return story


# ── Main public function ──────────────────────────────────────

def save_pdf(
    results: list[dict],
    output_path: Path,
    kb_name: str | None = None,
) -> None:
    """
    Render *results* as a formatted PDF and write to *output_path*.

    Args:
        results:     list of requirement dicts from analyser.analyse_all()
        output_path: where to write the .pdf file
        kb_name:     optional knowledge base name shown in the footer
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles     = _build_styles()
    page_tmpl  = _PageTemplate(kb_name)
    page_w, _  = A4

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Requirements Ambiguity Analysis Report",
        author="Requirements Analysis Agent",
    )

    story = []

    # ── Cover / header ────────────────────────────────────────
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("Requirements Ambiguity Analysis", styles["title"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}  "
        + (f"  |  Knowledge base: <b>{kb_name}</b>" if kb_name else "  |  No knowledge base"),
        styles["subtitle"],
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_MID_GRAY))
    story.append(Spacer(1, 6 * mm))

    # ── Summary table ─────────────────────────────────────────
    story.append(Paragraph("Summary", styles["section_header"]))
    story.append(Spacer(1, 2 * mm))
    story.append(_build_summary_table(results, styles))
    story.append(Spacer(1, 8 * mm))

    # ── Per-requirement cards ─────────────────────────────────
    story.append(Paragraph("Detailed Analysis", styles["section_header"]))
    story.append(Spacer(1, 2 * mm))

    for r in results:
        story.extend(_build_req_card(r, styles, page_w))

    # ── Build ─────────────────────────────────────────────────
    doc.build(
        story,
        onFirstPage=page_tmpl.on_page,
        onLaterPages=page_tmpl.on_page,
    )


# ── Helpers ───────────────────────────────────────────────────

def _escape(text: str) -> str:
    """Escape characters that break ReportLab's XML parser in Paragraph."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
