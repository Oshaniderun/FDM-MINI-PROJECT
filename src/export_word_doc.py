#!/usr/bin/env python
"""
export_word_doc.py
------------------
Converts the Stage 11 Markdown Technical Report (reports/final_technical_report.md)
into a beautifully styled, publication-grade Microsoft Word (.docx) document:
- reports/Stage_11_Technical_Report.docx
- reports/final_technical_report.docx

Features:
- Academic cover/title page with university branding and student authorship table.
- Custom typography (Calibri, Calibri Light, Consolas) with industrial blue palette (#1E3A8A).
- Styled tables with dark navy headers, bold white text, zebra-striped rows, and thin borders.
- Callouts / blockquotes with left accent borders and light shading.
- Embedded high-resolution figures from reports/figures/ with formal academic captions.
- Architecture diagrams formatted in monospace callout frames.
- Running header and page numbering.
"""

import os
import re
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

BASE_DIR = Path(__file__).resolve().parent.parent
MD_PATH = BASE_DIR / "reports" / "final_technical_report.md"
DOCX_OUT1 = BASE_DIR / "reports" / "Stage_11_Technical_Report.docx"
DOCX_OUT2 = BASE_DIR / "reports" / "final_technical_report.docx"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FRONTEND_BG = BASE_DIR / "frontend" / "header_bg.jpg"

# Color Palette
COLOR_NAVY = RGBColor(0x1E, 0x3A, 0x8A)      # #1E3A8A - Primary Headers
COLOR_ROYAL = RGBColor(0x25, 0x63, 0xEB)     # #2563EB - Secondary Headers
COLOR_DARK = RGBColor(0x0F, 0x17, 0x2A)      # #0F172A - Subheadings & Strong Text
COLOR_BODY = RGBColor(0x33, 0x41, 0x55)      # #334155 - Body Text
COLOR_MUTED = RGBColor(0x64, 0x74, 0x8B)     # #64748B - Captions & Footers
COLOR_ACCENT = RGBColor(0x02, 0x84, 0xC7)    # #0284C7 - Accents

HEX_NAVY = "1E3A8A"
HEX_LIGHT_BG = "F8FAFC"
HEX_BORDER = "CBD5E1"
HEX_CALLOUT_BG = "EFF6FF"
HEX_CALLOUT_BORDER = "2563EB"

def set_cell_background(cell, hex_color):
    """Sets background shading of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding (in twips/dxa) for a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'  <w:top w:w="{top}" w:type="dxa"/>'
        f'  <w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'  <w:left w:w="{left}" w:type="dxa"/>'
        f'  <w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, hex_color="CBD5E1"):
    """Sets subtle borders for a table."""
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="{hex_color}"/>'
            f'  <w:left w:val="none"/>'
            f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="{HEX_NAVY}"/>'
            f'  <w:right w:val="none"/>'
            f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{hex_color}"/>'
            f'  <w:insideV w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)

def add_callout_box(doc, text_lines, title=None):
    """Adds a stylish shaded callout box with a thick colored left border."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    
    cell = tbl.cell(0, 0)
    set_cell_background(cell, HEX_CALLOUT_BG)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
    
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{HEX_CALLOUT_BORDER}"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    
    if title:
        run_title = p.add_run(f"{title}\n")
        run_title.font.name = "Calibri"
        run_title.font.size = Pt(10.5)
        run_title.font.bold = True
        run_title.font.color.rgb = COLOR_ROYAL
        
    for i, line in enumerate(text_lines):
        if i > 0 or title:
            p = cell.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
        add_formatted_runs(p, line, default_color=COLOR_BODY, default_size=Pt(10))

def add_code_box(doc, code_lines):
    """Adds a shaded box for code or ASCII diagrams."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=120, bottom=120, left=160, right=160)
    
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'  <w:left w:val="single" w:sz="12" w:space="0" w:color="{HEX_NAVY}"/>'
        f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    
    text = "\n".join(code_lines)
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.0)
    run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

def add_formatted_runs(paragraph, text, default_color=COLOR_BODY, default_size=Pt(10.5), is_bold=False):
    """Parses markdown inline syntax: **bold**, *italic*, `code`, and $latex$."""
    # Clean file links: [text](file:///...) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Clean simple math notation: $...$ -> ...
    text = re.sub(r'\$([^\$]+)\$', r'\1', text)
    
    # Tokenize by bold, code, italic
    token_pattern = re.compile(r'(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)')
    tokens = token_pattern.split(text)
    
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**'):
            inner = token[2:-2]
            run = paragraph.add_run(inner)
            run.font.name = "Calibri"
            run.font.size = default_size
            run.font.bold = True
            run.font.color.rgb = COLOR_DARK
        elif token.startswith('`') and token.endswith('`'):
            inner = token[1:-1]
            run = paragraph.add_run(inner)
            run.font.name = "Consolas"
            run.font.size = Pt(default_size.pt * 0.92)
            run.font.color.rgb = RGBColor(0x93, 0x33, 0xEA)
        elif token.startswith('*') and token.endswith('*'):
            inner = token[1:-1]
            run = paragraph.add_run(inner)
            run.font.name = "Calibri"
            run.font.size = default_size
            run.font.italic = True
            run.font.color.rgb = default_color
        else:
            run = paragraph.add_run(token)
            run.font.name = "Calibri"
            run.font.size = default_size
            run.font.bold = is_bold
            run.font.color.rgb = default_color

def insert_figure(doc, img_path, caption_text, width_inches=6.0):
    """Inserts a centered image with an academic figure caption."""
    if not os.path.exists(img_path):
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(2)
    run_img = p_img.add_run()
    run_img.add_picture(str(img_path), width=Inches(width_inches))
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(8)
    run_cap = p_cap.add_run(f"Figure: {caption_text}")
    run_cap.font.name = "Calibri"
    run_cap.font.size = Pt(9.5)
    run_cap.font.italic = True
    run_cap.font.color.rgb = COLOR_MUTED

def parse_markdown_table(lines):
    """Parses markdown table lines into header and rows."""
    header = []
    rows = []
    for line in lines:
        parts = [c.strip() for c in line.strip().strip('|').split('|')]
        if not parts or not any(parts):
            continue
        # Check if separator line
        if all(re.match(r'^:?-+:?$', p) for p in parts):
            continue
        if not header:
            header = parts
        else:
            rows.append(parts)
    return header, rows

def add_table_to_doc(doc, header, rows):
    """Builds a formatted Word table matching academic standards."""
    if not header:
        return
    num_cols = len(header)
    table = doc.add_table(rows=1 + len(rows), cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)
    
    # Format Header Row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(header):
        cell = hdr_cells[i]
        set_cell_background(cell, HEX_NAVY)
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        add_formatted_runs(p, title, default_color=RGBColor(0xFF, 0xFF, 0xFF), default_size=Pt(9.5), is_bold=True)
    
    # Format Data Rows with Zebra Striping
    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = HEX_LIGHT_BG if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx in range(num_cols):
            cell = row_cells[c_idx]
            val = row_data[c_idx] if c_idx < len(row_data) else ""
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=90, bottom=90, left=140, right=140)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            add_formatted_runs(p, val, default_color=COLOR_BODY, default_size=Pt(9))
            
    # Add subtle spacing after table
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

def setup_page_layout(doc):
    """Sets standard 1-inch margins and document properties."""
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)
        
        # Header
        header = section.header
        p_hdr = header.paragraphs[0]
        p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_hdr.paragraph_format.space_after = Pt(4)
        run_hdr = p_hdr.add_run("SLIIT IT3051 Fundamentals of Data Mining — Mini-Project Final Technical Report (2026)")
        run_hdr.font.name = "Calibri"
        run_hdr.font.size = Pt(8.5)
        run_hdr.font.color.rgb = COLOR_MUTED
        
        # Footer
        footer = section.footer
        p_ftr = footer.paragraphs[0]
        p_ftr.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_ftr.paragraph_format.space_before = Pt(4)
        run_ftr = p_ftr.add_run("Stage 11 Technical Report | Group 05 — Cognita | Predictive Maintenance Diagnostic System")
        run_ftr.font.name = "Calibri"
        run_ftr.font.size = Pt(8.5)
        run_ftr.font.color.rgb = COLOR_MUTED

def generate_report():
    print(f"Reading markdown source: {MD_PATH}")
    if not MD_PATH.exists():
        raise FileNotFoundError(f"Missing {MD_PATH}")
        
    with open(MD_PATH, 'r', encoding='utf-8') as f:
        md_text = f.read()

    doc = docx.Document()
    setup_page_layout(doc)
    
    # ----------------------------------------------------
    # COVER / TITLE PAGE
    # ----------------------------------------------------
    p_inst = doc.add_paragraph()
    p_inst.paragraph_format.space_before = Pt(24)
    p_inst.paragraph_format.space_after = Pt(4)
    r_inst = p_inst.add_run("SRI LANKA INSTITUTE OF INFORMATION TECHNOLOGY (SLIIT)")
    r_inst.font.name = "Calibri"
    r_inst.font.size = Pt(13)
    r_inst.font.bold = True
    r_inst.font.color.rgb = COLOR_NAVY
    
    p_fac = doc.add_paragraph()
    p_fac.paragraph_format.space_before = Pt(0)
    p_fac.paragraph_format.space_after = Pt(4)
    r_fac = p_fac.add_run("Faculty of Computing — Department of Computer Science & Software Engineering")
    r_fac.font.name = "Calibri"
    r_fac.font.size = Pt(11)
    r_fac.font.bold = True
    r_fac.font.color.rgb = COLOR_MUTED
    
    p_mod = doc.add_paragraph()
    p_mod.paragraph_format.space_before = Pt(0)
    p_mod.paragraph_format.space_after = Pt(24)
    r_mod = p_mod.add_run("IT3051 — Fundamentals of Data Mining (Year 3 Semester 2, 2026)")
    r_mod.font.name = "Calibri"
    r_mod.font.size = Pt(11)
    r_mod.font.color.rgb = COLOR_ROYAL
    
    # Main Report Title
    p_title_tag = doc.add_paragraph()
    p_title_tag.paragraph_format.space_before = Pt(12)
    p_title_tag.paragraph_format.space_after = Pt(6)
    r_tag = p_title_tag.add_run("STAGE 11 — FINAL TECHNICAL REPORT (20%)")
    r_tag.font.name = "Calibri"
    r_tag.font.size = Pt(14)
    r_tag.font.bold = True
    r_tag.font.color.rgb = COLOR_ROYAL
    
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(16)
    p_title.paragraph_format.line_spacing = 1.15
    r_title = p_title.add_run("A Physics-Informed Predictive Maintenance Diagnostic System for Industrial CNC Milling Equipment Under Telemetry Irregularity and Class Imbalance")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(21)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_NAVY
    
    # Metadata Overview Callout
    meta_box = [
        "**Group Identifier:** Group 05 — \"Cognita\"",
        "**Academic Module:** IT3051 Fundamentals of Data Mining (Year 3 Semester 2, 2026)",
        "**Target Benchmark:** AI4I Predictive Maintenance Dataset with Irregularities (AI4I-PMDI)",
        "**Target Variable:** Machine Health Diagnostic State (`Diagnostic` — 6 Multiclass Categories)",
        "**Selected Champion Model:** Tuned Random Forest Classifier (`n_estimators=200`, `max_depth=10`, `class_weight='balanced_subsample'`)",
        "**Evaluation Metric Rationale:** Macro-F1 ($0.7101$ Test Score) prioritized over trivial Accuracy ($98.50\%$) due to 508:1 Class Imbalance",
        "**Deployment Stack:** FastAPI REST Microservice + Vanilla HTML5/CSS3/ES6 Industrial Single-Page Dashboard"
    ]
    add_callout_box(doc, meta_box, title="PROJECT SPECIFICATION & VERIFICATION SUMMARY")
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Authorship Table
    p_auth_head = doc.add_paragraph()
    p_auth_head.paragraph_format.space_before = Pt(12)
    p_auth_head.paragraph_format.space_after = Pt(6)
    r_auth = p_auth_head.add_run("Student Authorship & Contribution Roster")
    r_auth.font.name = "Calibri"
    r_auth.font.size = Pt(12)
    r_auth.font.bold = True
    r_auth.font.color.rgb = COLOR_DARK
    
    auth_header = ["Student Full Name", "Registration No.", "Degree Specialization", "Primary Project Ownership"]
    auth_rows = [
        ["Oshani De Run (Lead Author)", "[EVIDENCE REQUIRED: Student ID]", "Software Engineering / Data Science", "Pipeline Architecture, Preprocessing, Modeling & Tuning, REST API"],
        ["[EVIDENCE REQUIRED: Member 2]", "[EVIDENCE REQUIRED: Student ID]", "Computer Science / Software Engineering", "Exploratory Data Analysis, Missingness Statistical Proof, Visualizations"],
        ["[EVIDENCE REQUIRED: Member 3]", "[EVIDENCE REQUIRED: Student ID]", "Information Technology", "Hyperparameter Optimization, Feature Ablation, Permutation Importance"],
        ["[EVIDENCE REQUIRED: Member 4]", "[EVIDENCE REQUIRED: Student ID]", "Information Systems", "Frontend Interface Development, System Verification, Quality Audit"]
    ]
    add_table_to_doc(doc, auth_header, auth_rows)
    
    # Cover Page Break
    doc.add_page_break()
    
    # ----------------------------------------------------
    # PARSE SECTIONS AND CONTENT
    # ----------------------------------------------------
    lines = md_text.splitlines()
    i = 0
    in_cover = True
    in_table = False
    table_lines = []
    in_code = False
    code_lines = []
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check if past cover metadata
        if "## Executive Summary" in line:
            in_cover = False
            
        if in_cover:
            i += 1
            continue
            
        # Code block handling
        if stripped.startswith("```"):
            if in_code:
                in_code = False
                add_code_box(doc, code_lines)
                code_lines = []
            else:
                in_code = True
                code_lines = []
            i += 1
            continue
            
        if in_code:
            code_lines.append(line)
            i += 1
            continue
            
        # Table handling
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                in_table = True
                table_lines = [stripped]
            else:
                table_lines.append(stripped)
            i += 1
            continue
        else:
            if in_table:
                in_table = False
                hdr, rws = parse_markdown_table(table_lines)
                add_table_to_doc(doc, hdr, rws)
                table_lines = []
                
        # Empty line
        if not stripped:
            i += 1
            continue
            
        # Horizontal Rule
        if stripped in ["---", "___", "***"]:
            i += 1
            continue
            
        # Blockquote / Alert
        if stripped.startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q_text = lines[i].strip().lstrip('>').strip()
                quote_lines.append(q_text)
                i += 1
            add_callout_box(doc, quote_lines)
            continue
            
        # Heading 1 (# ...)
        if stripped.startswith("# "):
            h_text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(h_text)
            r.font.name = "Calibri"
            r.font.size = Pt(16)
            r.font.bold = True
            r.font.color.rgb = COLOR_NAVY
            i += 1
            continue
            
        # Heading 2 (## ...)
        if stripped.startswith("## "):
            h_text = stripped[3:].strip()
            
            # Add page break before major numbered sections for clean academic structure
            if re.match(r'^(## )?([1-9]|1[0-8])\.', stripped):
                doc.add_page_break()
                
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            r = p.add_run(h_text)
            r.font.name = "Calibri"
            r.font.size = Pt(13.5)
            r.font.bold = True
            r.font.color.rgb = COLOR_ROYAL
            i += 1
            continue
            
        # Heading 3 (### ...)
        if stripped.startswith("### "):
            h_text = stripped[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            r = p.add_run(h_text)
            r.font.name = "Calibri"
            r.font.size = Pt(11)
            r.font.bold = True
            r.font.color.rgb = COLOR_DARK
            i += 1
            
            # Automatic Figure Injection corresponding to subheadings
            if "3.5 Target Class Distribution" in h_text:
                insert_figure(doc, FIGURES_DIR / "fig01_target_imbalance.png", "Target Class Distribution and Severe Class Imbalance (508:1 Ratio)")
            elif "4.2 Descriptive Statistics" in h_text:
                insert_figure(doc, FIGURES_DIR / "fig04_sensor_distributions_kde.png", "Kernel Density Estimation (KDE) Distributions of Continuous Sensor Telemetry")
            elif "4.4 Missingness Mechanism" in h_text:
                insert_figure(doc, FIGURES_DIR / "fig02_missingness_overview.png", "Sensor Missingness Rates and Pattern Incomplete Coverage")
                insert_figure(doc, FIGURES_DIR / "fig03_comissingness_by_control.png", "Deterministic Co-Missingness Pattern Strictly Conditional on Multiplexer Mode (Control A, B, C)")
            elif "4.5 Sensor Outlier Concentration" in h_text:
                insert_figure(doc, FIGURES_DIR / "fig05_sensor_boxplots_by_diagnostic.png", "Sensor Outlier Distributions Demonstrating Failure Mode Clustering")
            elif "4.7 Multi-Sensor Physical Correlations" in h_text:
                insert_figure(doc, FIGURES_DIR / "fig06_correlation_heatmaps.png", "Pearson and Spearman Correlation Matrices Showing Non-Linear Physical Couplings")
            elif "6.1 Feature Engineering Rationale" in h_text:
                insert_figure(doc, FIGURES_DIR / "fig08_domain_physics_failure_boundaries.png", "Domain Physics Boundaries Separating Mechanical Power and Thermal Failures")
            elif "6.3 Controlled Feature Engineering Ablation" in h_text:
                insert_figure(doc, FIGURES_DIR / "03_feature_ablation_comparison.png", "Controlled Feature Ablation Experiment Demonstrating +14.89% Macro-F1 Lift")
            elif "6.4 Permutation Feature Importance" in h_text:
                insert_figure(doc, FIGURES_DIR / "07_feature_importance_champion.png", "Permutation Feature Importance for Champion Tuned Random Forest")
            elif "8.1 Systematic Baseline Leaderboard" in h_text:
                insert_figure(doc, FIGURES_DIR / "01_model_comparison_metrics.png", "Systematic Benchmark of Five Diverse Machine Learning Algorithms under Stratified 5-Fold CV")
            elif "8.2 In-Depth Performance Analysis" in h_text:
                insert_figure(doc, FIGURES_DIR / "02_per_class_f1_comparison.png", "Per-Class F1-Score Comparison Across Evaluated Algorithms")
            elif "9.1 Imbalance Strategy Benchmark" in h_text:
                insert_figure(doc, FIGURES_DIR / "04_imbalance_strategy_comparison.png", "Class Imbalance Strategy Benchmark: Cost-Weighting vs SMOTE vs Unweighted")
            elif "9.3 Baseline vs. Tuned Comparison" in h_text:
                insert_figure(doc, FIGURES_DIR / "05_baseline_vs_tuned_comparison.png", "Performance Comparison of Baseline vs. Hyperparameter-Tuned Models")
            elif "10.3 Per-Class Performance" in h_text:
                insert_figure(doc, FIGURES_DIR / "06_final_confusion_matrices.png", "Final Confusion Matrices of Champion Model on Untouched Test Partition (N=2,000)")
            elif "13.2 User Interface Components" in h_text:
                if FRONTEND_BG.exists():
                    insert_figure(doc, FRONTEND_BG, "Industrial Single-Page Diagnostic Dashboard Visual Architecture and Telemetry Banner")
            continue
            
        # Heading 4 (#### ...)
        if stripped.startswith("#### "):
            h_text = stripped[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(h_text)
            r.font.name = "Calibri"
            r.font.size = Pt(10.5)
            r.font.bold = True
            r.font.color.rgb = COLOR_ROYAL
            i += 1
            continue
            
        # Bullet list (* or -)
        if stripped.startswith("* ") or stripped.startswith("- "):
            item_text = stripped[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_runs(p, item_text)
            i += 1
            continue
            
        # Numbered list (1. ...)
        m_num = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if m_num:
            num_str, item_text = m_num.groups()
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_runs(p, item_text)
            i += 1
            continue
            
        # Standard Body Paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        add_formatted_runs(p, stripped)
        i += 1
        
    # Flush remaining table if any
    if in_table and table_lines:
        hdr, rws = parse_markdown_table(table_lines)
        add_table_to_doc(doc, hdr, rws)
        
    # Flush remaining code if any
    if in_code and code_lines:
        add_code_box(doc, code_lines)

    # Save documents
    print(f"Saving Word document: {DOCX_OUT1}")
    doc.save(str(DOCX_OUT1))
    
    print(f"Saving copy to: {DOCX_OUT2}")
    doc.save(str(DOCX_OUT2))
    
    size1 = os.path.getsize(DOCX_OUT1)
    size2 = os.path.getsize(DOCX_OUT2)
    print(f"Successfully generated docx files! Size: {size1:,} bytes ({size1 / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    generate_report()
