from __future__ import annotations
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)

from shared.models import FirstCycleResult, SecondCycleResult

W, H = A4

BRAND_DARK = colors.HexColor("#1a1a2e")
BRAND_MID = colors.HexColor("#16213e")
BRAND_ACCENT = colors.HexColor("#0f3460")
BRAND_GOLD = colors.HexColor("#e94560")
LIGHT_GREY = colors.HexColor("#f5f5f5")


def _styles():
    base = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontSize=26, textColor=BRAND_DARK,
                                spaceAfter=6, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
                                   fontSize=12, textColor=BRAND_ACCENT,
                                   spaceAfter=20, alignment=TA_CENTER),
        "h1": ParagraphStyle("h1", parent=base["Heading1"],
                              fontSize=14, textColor=BRAND_DARK,
                              spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                              fontSize=11, textColor=BRAND_ACCENT,
                              spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontSize=9, leading=13, spaceAfter=4),
        "small": ParagraphStyle("small", parent=base["Normal"],
                                fontSize=7.5, textColor=colors.grey),
        "quote": ParagraphStyle("quote", parent=base["Normal"],
                                fontSize=8.5, leftIndent=18,
                                textColor=colors.HexColor("#444444"),
                                fontName="Helvetica-Oblique"),
    }
    return custom


TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BRAND_ACCENT),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 9),
    ("FONTSIZE", (0, 1), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
])


class PDFExporter:
    def export(self, summary: dict, fc: FirstCycleResult,
               sc: SecondCycleResult, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "analysis_report.pdf")

        doc = SimpleDocTemplate(out_path, pagesize=A4,
                                leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=2.5 * cm, bottomMargin=2 * cm)
        s = _styles()
        story = []

        # Title page
        story += self._title_page(s, summary)
        story.append(PageBreak())

        # Summary stats
        story += self._summary_section(s, summary, fc, sc)
        story.append(PageBreak())

        # First cycle
        story += self._first_cycle_section(s, fc)
        story.append(PageBreak())

        # Categories
        story += self._categories_section(s, sc)

        # Axial relationships
        if sc.axial_relationships:
            story.append(PageBreak())
            story += self._axial_section(s, sc)

        # Themes & core category
        if sc.themes or sc.core_category:
            story.append(PageBreak())
            story += self._theoretical_section(s, sc)

        doc.build(story)
        return out_path

    # ------------------------------------------------------------------ #

    def _title_page(self, s, summary):
        now = datetime.now().strftime("%d %B %Y, %H:%M")
        return [
            Spacer(1, 3 * cm),
            Paragraph("Qualitative Coding Analysis", s["title"]),
            Paragraph("NLP-Powered Qualitative Research Report", s["subtitle"]),
            HRFlowable(width="80%", thickness=2, color=BRAND_GOLD, hAlign="CENTER"),
            Spacer(1, 0.5 * cm),
            Paragraph(f"Generated: {now}", s["small"]),
            Spacer(1, 0.3 * cm),
            Paragraph(f"Files analysed: {summary.get('file_count', '—')}", s["body"]),
            Paragraph(f"Total segments: {summary.get('total_segments', '—')}", s["body"]),
            Paragraph(f"Total codes assigned: {summary.get('total_codes', '—')}", s["body"]),
        ]

    def _summary_section(self, s, summary, fc, sc):
        items = [Paragraph("Executive Summary", s["h1"])]
        stats = [
            ["Metric", "Value"],
            ["Unique code labels", str(len(fc.all_codes))],
            ["Coded segments", str(len(fc.coded_segments))],
            ["Categories identified", str(len(sc.categories))],
            ["Axial relationships", str(len(sc.axial_relationships))],
            ["Themes", str(len(sc.themes))],
            ["Core category",
             sc.core_category.name if sc.core_category else "—"],
        ]
        items.append(Table(stats, colWidths=[9 * cm, 8 * cm],
                           style=TABLE_STYLE))
        return items

    def _first_cycle_section(self, s, fc):
        items = [Paragraph("First Cycle Codes", s["h1"])]
        top_codes = sorted(fc.code_frequencies.items(),
                           key=lambda x: x[1], reverse=True)[:50]
        rows = [["Code Label", "Type", "Frequency", "Sample Excerpt"]]
        for label, freq in top_codes:
            codes = fc.all_codes.get(label, [])
            ctype = codes[0].code_type.value if codes else ""
            excerpt = (codes[0].excerpt[:60] + "…") if codes else ""
            rows.append([label[:35], ctype, str(freq), excerpt])
        col_w = [6 * cm, 3 * cm, 2.5 * cm, 6 * cm]
        items.append(Table(rows, colWidths=col_w, style=TABLE_STYLE))
        return items

    def _categories_section(self, s, sc):
        items = [Paragraph("Categories (Second Cycle — Focused Coding)", s["h1"])]
        for cat in sc.categories:
            items.append(Paragraph(cat.name, s["h2"]))
            items.append(Paragraph(cat.description, s["body"]))
            items.append(Paragraph(
                f"<b>Frequency:</b> {cat.frequency} &nbsp;|&nbsp; "
                f"<b>Member codes:</b> {', '.join(cat.codes[:8])}",
                s["small"]))
            if cat.properties:
                items.append(Paragraph(
                    f"<b>Properties:</b> {', '.join(cat.properties)}", s["small"]))
            items.append(Spacer(1, 0.2 * cm))
        return items

    def _axial_section(self, s, sc):
        items = [Paragraph("Axial Relationships", s["h1"])]
        rows = [["Source Category", "Relationship", "Target Category", "Description"]]
        for rel in sc.axial_relationships:
            rows.append([rel.source_category[:25], rel.relationship_type,
                         rel.target_category[:25], rel.description[:50]])
        col_w = [5 * cm, 3 * cm, 5 * cm, 5 * cm]
        items.append(Table(rows, colWidths=col_w, style=TABLE_STYLE))
        return items

    def _theoretical_section(self, s, sc):
        items = [Paragraph("Theoretical Findings", s["h1"])]

        if sc.core_category:
            cc = sc.core_category
            items.append(Paragraph("Core Category", s["h2"]))
            items.append(Paragraph(f"<b>{cc.name}</b>", s["body"]))
            items.append(Paragraph(cc.description, s["body"]))
            if cc.theoretical_statement:
                items.append(Paragraph(cc.theoretical_statement, s["quote"]))
            items.append(Spacer(1, 0.3 * cm))

        if sc.themes:
            items.append(Paragraph("Themes", s["h2"]))
            for i, theme in enumerate(sc.themes, 1):
                items.append(Paragraph(f"{i}. [{theme.level.upper()}] {theme.statement}",
                                       s["body"]))
                if theme.evidence:
                    items.append(Paragraph(f"Evidence: {theme.evidence[0][:80]}…",
                                           s["quote"]))
                items.append(Spacer(1, 0.15 * cm))

        return items
