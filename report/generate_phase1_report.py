#!/usr/bin/env python3
"""Generate the six-page ERC 2026 Phase 1 submission-draft report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output' / 'pdf' / 'ERC_Phase_1_Report_Draft.pdf'
WIDTH, HEIGHT = A4
MARGIN = 15 * mm
CONTENT_WIDTH = WIDTH - 2 * MARGIN

NAVY = colors.HexColor('#102A43')
BLUE = colors.HexColor('#1473E6')
CYAN = colors.HexColor('#00A6A6')
ORANGE = colors.HexColor('#F28C28')
GREEN = colors.HexColor('#1F8A70')
RED = colors.HexColor('#C73E3A')
INK = colors.HexColor('#243B53')
MUTED = colors.HexColor('#627D98')
PALE = colors.HexColor('#F0F4F8')
LINE = colors.HexColor('#BCCCDC')
YELLOW = colors.HexColor('#FFF4CC')


def wrap_lines(text: str, font: str, size: float, width: float) -> list[str]:
    words = text.split()
    if not words:
        return ['']
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f'{current} {word}'
        if stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def paragraph(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font: str = 'Helvetica',
    size: float = 8.3,
    leading: float = 10.4,
    color=INK,
) -> float:
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in wrap_lines(text, font, size, width):
        pdf.drawString(x, y, line)
        y -= leading
    return y


def bullets(
    pdf: canvas.Canvas,
    items: Iterable[str],
    x: float,
    y: float,
    width: float,
    *,
    size: float = 8.1,
    leading: float = 10.0,
    gap: float = 2.5,
) -> float:
    for item in items:
        pdf.setFillColor(BLUE)
        pdf.circle(x + 2.4, y + 2.5, 1.5, fill=1, stroke=0)
        y = paragraph(
            pdf,
            item,
            x + 9,
            y,
            width - 9,
            size=size,
            leading=leading,
        )
        y -= gap
    return y


def section_label(pdf: canvas.Canvas, text: str, x: float, y: float) -> float:
    pdf.setFillColor(NAVY)
    pdf.setFont('Helvetica-Bold', 10.2)
    pdf.drawString(x, y, text)
    pdf.setStrokeColor(CYAN)
    pdf.setLineWidth(1.5)
    pdf.line(x, y - 4, x + CONTENT_WIDTH, y - 4)
    return y - 16


def page_header(pdf: canvas.Canvas, title: str, body_page: int) -> float:
    pdf.setFillColor(NAVY)
    pdf.rect(0, HEIGHT - 25 * mm, WIDTH, 25 * mm, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont('Helvetica-Bold', 15)
    pdf.drawString(MARGIN, HEIGHT - 14 * mm, title)
    pdf.setFont('Helvetica', 7.5)
    pdf.drawRightString(
        WIDTH - MARGIN,
        HEIGHT - 14 * mm,
        f'ERC 2026 PHASE 1  |  BODY {body_page} OF 5',
    )
    return HEIGHT - 33 * mm


def footer(pdf: canvas.Canvas, body_page: int | None = None) -> None:
    pdf.setStrokeColor(LINE)
    pdf.line(MARGIN, 12 * mm, WIDTH - MARGIN, 12 * mm)
    pdf.setFillColor(RED)
    pdf.setFont('Helvetica-Bold', 6.8)
    pdf.drawString(MARGIN, 8 * mm, 'SUBMISSION DRAFT - FINAL TEAM DATA AND FIVE RANDOMIZED TRIALS REQUIRED')
    pdf.setFillColor(MUTED)
    pdf.setFont('Helvetica', 6.8)
    suffix = f'Body page {body_page}/5' if body_page else 'Title page (excluded from five-page body limit)'
    pdf.drawRightString(WIDTH - MARGIN, 8 * mm, suffix)


def draw_table(
    pdf: canvas.Canvas,
    data: Sequence[Sequence[object]],
    x: float,
    y_top: float,
    widths: Sequence[float],
    *,
    font_size: float = 7.0,
    row_backgrounds: bool = True,
) -> float:
    paragraph_style = ParagraphStyle(
        'cell',
        fontName='Helvetica',
        fontSize=font_size,
        leading=font_size + 1.5,
        textColor=INK,
        alignment=TA_LEFT,
    )
    header_style = ParagraphStyle(
        'header',
        parent=paragraph_style,
        fontName='Helvetica-Bold',
        textColor=colors.white,
    )
    converted = []
    for row_index, row in enumerate(data):
        converted.append(
            [
                Paragraph(str(value), header_style if row_index == 0 else paragraph_style)
                for value in row
            ]
        )
    table = Table(converted, colWidths=list(widths), repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('GRID', (0, 0), (-1, -1), 0.35, LINE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]
    if row_backgrounds:
        for row in range(1, len(converted)):
            if row % 2 == 0:
                style.append(('BACKGROUND', (0, row), (-1, row), PALE))
    table.setStyle(TableStyle(style))
    _, height = table.wrapOn(pdf, sum(widths), HEIGHT)
    table.drawOn(pdf, x, y_top - height)
    return y_top - height


def draw_image_box(
    pdf: canvas.Canvas,
    path: Path,
    x: float,
    y_top: float,
    width: float,
    height: float,
    caption: str,
) -> None:
    pdf.setFillColor(PALE)
    pdf.roundRect(x, y_top - height, width, height, 5, fill=1, stroke=0)
    image_height = height - 17
    if path.exists():
        image = ImageReader(str(path))
        source_width, source_height = image.getSize()
        scale = min(width / source_width, image_height / source_height)
        rendered_width = source_width * scale
        rendered_height = source_height * scale
        pdf.drawImage(
            image,
            x + (width - rendered_width) / 2,
            y_top - 2 - rendered_height,
            rendered_width,
            rendered_height,
            preserveAspectRatio=True,
            mask='auto',
        )
    else:
        pdf.setFillColor(MUTED)
        pdf.setFont('Helvetica-Oblique', 8)
        pdf.drawCentredString(x + width / 2, y_top - height / 2, 'Evidence image unavailable')
    pdf.setFillColor(NAVY)
    pdf.rect(x, y_top - height, width, 15, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont('Helvetica', 6.4)
    caption_lines = wrap_lines(caption, 'Helvetica', 6.4, width - 8)
    pdf.drawString(x + 4, y_top - height + 5, caption_lines[0])


def draw_architecture(pdf: canvas.Canvas, x: float, y_top: float) -> float:
    boxes = [
        ('Perception', 'live RGB-D -> markers, book row, red bin', BLUE),
        ('Mission manager', 'bounded state machine + evidence gates', NAVY),
        ('Navigation', 'odom/LiDAR holonomic feedback', CYAN),
        ('Manipulation', 'fixed-torso left-arm IK + payload checks', ORANGE),
    ]
    box_w = 118
    box_h = 46
    gap = 18
    positions = [
        (x, y_top),
        (x + box_w + gap, y_top),
        (x + 2 * (box_w + gap), y_top),
        (x + 3 * (box_w + gap), y_top),
    ]
    for (title, detail, color), (box_x, box_y) in zip(boxes, positions):
        pdf.setFillColor(color)
        pdf.roundRect(box_x, box_y - box_h, box_w, box_h, 6, fill=1, stroke=0)
        pdf.setFillColor(colors.white)
        pdf.setFont('Helvetica-Bold', 8.5)
        pdf.drawCentredString(box_x + box_w / 2, box_y - 15, title)
        pdf.setFont('Helvetica', 5.9)
        for index, line in enumerate(wrap_lines(detail, 'Helvetica', 5.9, box_w - 10)[:2]):
            pdf.drawCentredString(box_x + box_w / 2, box_y - 27 - index * 7, line)
    pdf.setStrokeColor(MUTED)
    pdf.setFillColor(MUTED)
    pdf.setLineWidth(1.1)
    for index in range(3):
        start_x = positions[index][0] + box_w
        end_x = positions[index + 1][0]
        arrow_y = y_top - box_h / 2
        pdf.line(start_x + 2, arrow_y, end_x - 4, arrow_y)
        pdf.line(end_x - 8, arrow_y + 3, end_x - 4, arrow_y)
        pdf.line(end_x - 8, arrow_y - 3, end_x - 4, arrow_y)
    return y_top - box_h - 10


def load_navigation(path: Path) -> tuple[list[dict], list[dict]]:
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding='utf-8'))
    return (
        payload.get('planned_path', {}).get('samples', []),
        payload.get('executed_path', {}).get('samples', []),
    )


def draw_path_plot(
    pdf: canvas.Canvas,
    path: Path,
    x: float,
    y_top: float,
    width: float,
    height: float,
) -> None:
    planned, executed = load_navigation(path)
    pdf.setFillColor(colors.white)
    pdf.setStrokeColor(LINE)
    pdf.roundRect(x, y_top - height, width, height, 5, fill=1, stroke=1)
    all_samples = planned + executed
    if not all_samples:
        pdf.setFillColor(MUTED)
        pdf.drawCentredString(x + width / 2, y_top - height / 2, 'Path data unavailable')
        return
    xs = [float(item['x']) for item in all_samples]
    ys = [float(item['y']) for item in all_samples]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(x_min, x_max):
        x_max += 0.01
    if math.isclose(y_min, y_max):
        y_max += 0.01
    pad = 16

    def project(item: dict) -> tuple[float, float]:
        px = x + pad + (float(item['x']) - x_min) / (x_max - x_min) * (width - 2 * pad)
        py = y_top - height + pad + (float(item['y']) - y_min) / (y_max - y_min) * (height - 2 * pad)
        return px, py

    for samples, color, line_width in ((planned, BLUE, 2.2), (executed, ORANGE, 1.5)):
        if len(samples) < 2:
            continue
        pdf.setStrokeColor(color)
        pdf.setLineWidth(line_width)
        last = project(samples[0])
        for sample in samples[1:]:
            current = project(sample)
            pdf.line(last[0], last[1], current[0], current[1])
            last = current
    pdf.setFillColor(GREEN)
    for sample in (planned[-1:] if planned else []):
        px, py = project(sample)
        pdf.circle(px, py, 3, fill=1, stroke=0)
    pdf.setFillColor(BLUE)
    pdf.rect(x + 10, y_top - 12, 10, 2, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont('Helvetica', 6.5)
    pdf.drawString(x + 24, y_top - 14, 'planned')
    pdf.setFillColor(ORANGE)
    pdf.rect(x + 72, y_top - 12, 10, 2, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.drawString(x + 86, y_top - 14, 'executed')


def title_page(pdf: canvas.Canvas) -> None:
    pdf.setFillColor(NAVY)
    pdf.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    pdf.setFillColor(BLUE)
    pdf.circle(WIDTH - 48, HEIGHT - 54, 92, fill=1, stroke=0)
    pdf.setFillColor(CYAN)
    pdf.circle(WIDTH - 12, HEIGHT - 100, 55, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont('Helvetica-Bold', 12)
    pdf.drawString(MARGIN, HEIGHT - 58 * mm, 'EMIRATES ROBOTICS COMPETITION 2026')
    pdf.setFont('Helvetica-Bold', 29)
    pdf.drawString(MARGIN, HEIGHT - 76 * mm, 'Phase 1 Technical Report')
    pdf.setFont('Helvetica', 13)
    pdf.drawString(MARGIN, HEIGHT - 87 * mm, 'Autonomous visual book retrieval and bin placement')

    pdf.setFillColor(YELLOW)
    pdf.roundRect(MARGIN, HEIGHT - 115 * mm, CONTENT_WIDTH, 17 * mm, 6, fill=1, stroke=0)
    pdf.setFillColor(RED)
    pdf.setFont('Helvetica-Bold', 10)
    pdf.drawString(MARGIN + 8, HEIGHT - 105 * mm, 'SUBMISSION DRAFT')
    pdf.setFillColor(INK)
    pdf.setFont('Helvetica', 8.5)
    pdf.drawString(
        MARGIN + 124,
        HEIGHT - 105 * mm,
        'Five randomized scored trials, team identity, video link and unedited run evidence remain required.',
    )

    fields = [
        ('Team', '[TEAM NAME REQUIRED]'),
        ('University', '[UNIVERSITY REQUIRED]'),
        ('Members', '[MEMBER NAMES REQUIRED]'),
        ('Contact', '[EMAIL REQUIRED]'),
        ('Repository', 'github.com/HishaamA/erc-2026-phase1-solution'),
        ('Video', '[UNEDITED YOUTUBE URL REQUIRED]'),
        ('Simulator', 'Official repository v1.0.3 development baseline; evaluator tag unconfirmed'),
        ('Report date', '4 September 2026'),
        ('Deadline', '15 September 2026'),
    ]
    y = HEIGHT - 139 * mm
    for label, value in fields:
        pdf.setFillColor(CYAN)
        pdf.setFont('Helvetica-Bold', 8)
        pdf.drawString(MARGIN, y, label.upper())
        pdf.setFillColor(colors.white)
        pdf.setFont('Helvetica', 9)
        pdf.drawString(MARGIN + 82, y, value)
        y -= 9 * mm

    pdf.setFillColor(colors.HexColor('#0B2035'))
    pdf.roundRect(MARGIN, 30 * mm, CONTENT_WIDTH, 27 * mm, 6, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont('Helvetica-Bold', 8)
    pdf.drawString(MARGIN + 8, 47 * mm, 'EVALUATOR COMMAND')
    pdf.setFont('Courier', 7.6)
    pdf.drawString(
        MARGIN + 8,
        38 * mm,
        'ros2 launch erc_phase1_solution solution.launch.py shelf_column_number:=2 book_colour:=red',
    )
    footer(pdf)
    pdf.showPage()


def architecture_page(pdf: canvas.Canvas) -> None:
    y = page_header(pdf, '1. System architecture', 1)
    y = paragraph(
        pdf,
        'The solution is a deterministic ROS 2 package layered on the official TIAGo Pro interfaces. '
        'A mission manager coordinates three bounded workers; every task perception result comes from the live onboard RGB-D stream, never from scene ground truth.',
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=8.8,
        leading=11,
    ) - 8
    y = draw_architecture(pdf, MARGIN, y)
    y = section_label(pdf, 'Node and interface responsibilities', MARGIN, y)
    data = [
        ['Node', 'Inputs', 'Outputs / role'],
        ['erc_perception', 'RGB, registered depth, camera info', 'Marker cloud; target book/bin points; timestamped PNG evidence + digest'],
        ['erc_navigation', '/odom, front/rear scans', '/cmd_vel; 41-sample planned path; executed path; reached/blocked/failed status'],
        ['erc_manipulation', 'Target PointStamped, joint states, TF', 'Head/torso trajectories; fixed-torso left-arm IK; clamped left-gripper commands; payload-safe route'],
        ['erc_mission_manager', 'Worker status, contact pairs, odometry', 'State sequencing; required row/column Int32 topics; retries; JSONL/summary; completion gated by classified contact'],
    ]
    y = draw_table(pdf, data, MARGIN, y, [82, 143, CONTENT_WIDTH - 225], font_size=6.6) - 10
    y = section_label(pdf, 'Mission sequence and safety gates', MARGIN, y)
    sequence = [
        'Ready/stow -> marker search -> fit a shelf-facing line and normal from at least three labels -> navigate to the requested column.',
        'Publish /erc/shelf_column_identification; detect requested colour and top-to-bottom row; publish /erc/shelf_row_identification; move to the 0.65 m grasp standoff and reacquire.',
        'Preplan fixed-torso clearance, pregrasp, grasp, reverse extraction, 0.18 m fixed-orientation lowering and a collision-checked compact transport/HOME candidate before arm motion.',
        'Back the lowered payload 0.70 m straight off the shelf -> perform the radius/tilt-gated compact fold -> turn and return to the recorded start -> locate and reacquire the red bin -> execute a payload-filtered approach -> correlate book/bin contact.',
    ]
    y = bullets(pdf, sequence, MARGIN, y, CONTENT_WIDTH)
    y -= 2
    pdf.setFillColor(PALE)
    pdf.roundRect(MARGIN, y - 40, CONTENT_WIDTH, 40, 5, fill=1, stroke=0)
    paragraph(
        pdf,
        "One-arm compliance: the right arm is moved once to PAL Robotics' official inactive home posture because the simulator initializes it fully extended. It is never used for sensing, grasping, carrying or placement; all book handling is left-arm-only. Confirm with the organizer if 'use' is interpreted as prohibiting even this safety parking.",
        MARGIN + 8,
        y - 12,
        CONTENT_WIDTH - 16,
        size=7.2,
        leading=8.7,
    )
    footer(pdf, 1)
    pdf.showPage()


def perception_page(pdf: canvas.Canvas) -> None:
    y = page_header(pdf, '2. Perception', 2)
    y = paragraph(
        pdf,
        'Number recognition combines bright-card contours with a dark-glyph fallback for cards that merge into the shelf. '
        'Digits 1-5 are rectified and compared to local templates. HSV connected components identify all coloured books; vertical clustering assigns rows 1-4 from top to bottom. Three stable frames are required before publication.',
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=8.3,
        leading=10.3,
    ) - 6
    image_gap = 8
    image_width = (CONTENT_WIDTH - image_gap) / 2
    image_height = 148
    draw_image_box(
        pdf,
        ROOT / 'erc_images' / 'column_20260827T191913_120044Z_ros40460000000.png',
        MARGIN,
        y,
        image_width,
        image_height,
        'Retry 10 live RGB: requested column 2 selected (confidence 0.88).',
    )
    draw_image_box(
        pdf,
        ROOT / 'erc_images' / 'book_20260827T192056_180543Z_ros53956000000.png',
        MARGIN + image_width + image_gap,
        y,
        image_width,
        image_height,
        'Retry 10 live RGB: requested red book selected in row 1 (confidence 0.89).',
    )
    y -= image_height + 14
    y = section_label(pdf, 'Measured development observations (same seed and request)', MARGIN, y)
    data = [
        ['Run set', 'Column result', 'Book result', 'Marker to evidence', 'Book to evidence'],
        ['Retry 10: 9133c7a3da61', '2 correct; 0.88', 'red row 1; 0.89', '16.560 sim s', '1.256 sim s'],
        ['Five same-seed passes', '5/5 correct; mean 0.874', '5/5 correct; mean 0.860', 'mean 16.691 sim s', 'mean 1.255 sim s'],
    ]
    y = draw_table(pdf, data, MARGIN, y, [85, 106, 106, 94, CONTENT_WIDTH - 391], font_size=6.5) - 10
    y = section_label(pdf, 'Why this approach', MARGIN, y)
    y = bullets(
        pdf,
        [
            'No network, external model, extra sensor or prerecorded scene data is required; the classifier is deterministic and ships with the package.',
            'RGB/depth timestamps must be within 0.20 s and no older than 1.0 s. Deprojection uses the live camera matrix and median valid depth around the detection centre.',
            'Evidence is written only after a stable live detection. Bounding boxes, confidence, ROS/UTC time, file path and SHA-256 are computed and published on /erc/perception/status; the digest is not claimed as a mission-JSONL field.',
            'Aggregate run IDs: cc2d0ab42139, a493a14e2cdf, 606455d9c2b0, 704e2a55e21a and 9133c7a3da61.',
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=7.7,
        leading=9.4,
    )
    pdf.setFillColor(YELLOW)
    pdf.roundRect(MARGIN, 19 * mm, CONTENT_WIDTH, 17 * mm, 5, fill=1, stroke=0)
    paragraph(
        pdf,
        'Accuracy limitation: these five observations repeat seed 101, column 2 and red; they are development checks, not randomized accuracy. Confidence is an internal similarity score, not a calibrated probability. Five reset, randomized final trials remain mandatory.',
        MARGIN + 8,
        30 * mm,
        CONTENT_WIDTH - 16,
        size=7.2,
        leading=8.6,
    )
    footer(pdf, 2)
    pdf.showPage()


def navigation_page(pdf: canvas.Canvas) -> None:
    y = page_header(pdf, '3. Navigation', 3)
    y = paragraph(
        pdf,
        'The controller generates a 41-sample straight-line odometry plan and follows it with holonomic x/y/yaw feedback, acceleration limits and a 0.28 m laser emergency stop. Position and yaw tolerances are 0.045 m and 0.045 rad. Planned paths are cyan/blue; executed paths are orange.',
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=8.4,
        leading=10.5,
    ) - 7
    plot_height = 152
    draw_path_plot(
        pdf,
        ROOT / 'results' / 'paths' / 'trial_25e950d0a9f8_navigation_goal_03_shelf_observation_pose.json',
        MARGIN,
        y,
        CONTENT_WIDTH * 0.42,
        plot_height,
    )
    screenshot_x = MARGIN + CONTENT_WIDTH * 0.44
    screenshot_w = CONTENT_WIDTH * 0.56
    draw_image_box(
        pdf,
        ROOT / 'report' / 'evidence' / 'rviz_planned_executed_25e950d0a9f8_zoom.png',
        screenshot_x,
        y,
        screenshot_w,
        plot_height,
        'Live retry 14 RViz: planned cyan and executed orange topics, status OK.',
    )
    y -= plot_height + 14
    y = section_label(pdf, 'Measured development navigation', MARGIN, y)
    data = [
        ['Run / purpose', 'Planned', 'Executed', 'Ratio', 'Sim time', 'End error', 'Outcome'],
        ['retry 19 shelf', '1.444066 m', '1.407512 m', '0.974687', '10.750 s', '44.561 mm', 'reached'],
        ['retry 19 grasp', '0.770540 m', '0.730692 m', '0.948286', '9.150 s', '44.507 mm', 'reached'],
    ]
    y = draw_table(pdf, data, MARGIN, y, [83, 72, 72, 61, 56, 66, CONTENT_WIDTH - 410], font_size=6.3) - 10
    y = section_label(pdf, 'Interpretation', MARGIN, y)
    bullets(
        pdf,
        [
            'Both retry 19 translational goals reached within the configured 45 mm tolerance and generated trial-scoped JSON artifacts with 41 planned samples plus 73 shelf and 62 grasp-standoff executed odometry samples.',
            'The screenshot is a genuine RViz capture from live transient-local /erc/navigation/planned_path and /erc/navigation/executed_path publishers. The adjacent plot independently renders the exact shelf artifact saved for trial 25e950d0a9f8.',
            'The table metrics are from retry 19 trial 2172a3aa1379; the retained RViz screenshot and adjacent plot are retry 14 evidence. Both are same-seed development runs, not randomized final trials. Slow host real-time factor is excluded.',
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=7.6,
        leading=9.3,
    )
    footer(pdf, 3)
    pdf.showPage()


def manipulation_page(pdf: canvas.Canvas) -> None:
    y = page_header(pdf, '4. Manipulation', 4)
    y = paragraph(
        pdf,
        'A URDF-derived damped-least-squares solver controls the torso plus seven left-arm joints. Shelf motion is solved at a fixed 0.35 m torso height and bin placement at 0.30 m. The book is reacquired after final base alignment so planning uses a fresh live 3D point.',
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=8.5,
        leading=10.7,
    ) - 9

    stages = [
        ('1', 'Reacquire', 'fresh RGB-D point'),
        ('2', 'Clearance', 'x = 0.45 m'),
        ('3', 'Pregrasp', '14 cm behind grasp'),
        ('4', 'Grasp/verify', 'row depth; contact + jaw'),
        ('5', 'Extract/lower', 'reverse; 18 cm down'),
        ('6', 'Carry/place', 'retreat -> compact -> bin'),
    ]
    stage_w = (CONTENT_WIDTH - 5 * 8) / 6
    for index, (number, title, detail) in enumerate(stages):
        x = MARGIN + index * (stage_w + 8)
        pdf.setFillColor(BLUE if index < 4 else GREEN)
        pdf.roundRect(x, y - 58, stage_w, 58, 5, fill=1, stroke=0)
        pdf.setFillColor(colors.white)
        pdf.setFont('Helvetica-Bold', 13)
        pdf.drawString(x + 7, y - 17, number)
        pdf.setFont('Helvetica-Bold', 7.2)
        pdf.drawString(x + 7, y - 33, title)
        pdf.setFont('Helvetica', 5.7)
        for line_index, line in enumerate(wrap_lines(detail, 'Helvetica', 5.7, stage_w - 14)[:2]):
            pdf.drawString(x + 7, y - 43 - line_index * 6, line)
    y -= 73
    y = section_label(pdf, 'Safety, single-arm use and placement', MARGIN, y)
    y = bullets(
        pdf,
        [
            "Only the left gripper contacts and transports the target book. The right arm is idempotently parked once at PAL's official home_right joint values to retract its 0.983 m zero-pose reach, then never commanded again.",
            'Before any pick motion, the complete fixed-torso Cartesian approach, reverse extraction, fixed-orientation lowering and carried route are solved. Payload checks use 61 samples per bounded leg; exact nonadjacent robot-mesh and closed-mesh containment checks cover endpoints plus adaptive interior samples and use measured parked-arm joints. The fixed fold passed an earlier 601-sample triangle-surface audit plus a 100-sample containment-enabled spot audit.',
            'The carried-book model uses the official 0.16 x 0.03 x 0.25 m dimensions, 15 mm inflation and a 20 mm shelf margin against the modeled base, torso, head/camera and arm links 1-7. An unsafe HOME fold remains lowered only for the straight shelf retreat, then follows a preflighted compact route with a 0.45 m modeled-geometry radius gate and 60-degree book-tilt ceiling. The audited route peaks at 42.43 degrees; free-joint IK reserves 0.01 rad from hard stops and bin placement reserves 0.03 rad. Tool-link/gripper meshes remain a live-validation limitation.',
            'The left pinch gate ignores right-gripper contacts. A 0.018 m hold and 0.016 m floor prevent further closing on undersized pinches. Pick recovery opens and completes a checked unloaded route or aborts; placement motion failure retains the payload and stops. Placement uses a 0.72 m bin standoff and a fresh bin point.',
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=7.2,
        leading=8.8,
    )
    y -= 4
    y = section_label(pdf, 'Observed development outcome and correction', MARGIN, y)
    data = [
        ['Evidence', 'Observation', 'Resulting control'],
        ['retry 10', 'Old +0.065 m depth closed to 0.000016 m despite target contact; verifier rejected the empty close.', 'Jaw-width and requested-colour contact must persist; depth required geometry calibration.'],
        ['retry 12', 'All three attempts were rejected before arm motion because the direct carried route exceeded the payload shelf margin by 3.63 mm.', 'Offline exact-pose audit selects delayed arm_left_4: 2001 samples/leg, payload max x 0.653437 vs 0.672600 m limit (19.163 mm clearance).'],
        ['retry 19', 'Hold passed at 0.018000433 m; all 5 extraction + 3 lowering legs retained. First home-transition leg contacted arm_left_5 at ROS 93.552 s; grasp_lost at 95.714 s.', 'Recovery opened and succeeded. This run predates the current robot-mesh guard and compact upright transport posture; live retention remains unresolved.'],
    ]
    y = draw_table(pdf, data, MARGIN, y, [72, 205, CONTENT_WIDTH - 277], font_size=6.3) - 12
    pdf.setFillColor(YELLOW)
    pdf.roundRect(MARGIN, y - 47, CONTENT_WIDTH, 47, 5, fill=1, stroke=0)
    paragraph(
        pdf,
        'Retry 19 selected loaded-clearance candidate 1 (5 extraction / 3 lowering / 2 HOME). Hold passed, recovery succeeded, then reacquisition failed. Terminal: success=false, book_reacquisition_failed, 551.510 wall s, 1 pick and 1 collision. No return, placement or mission success.',
        MARGIN + 8,
        y - 14,
        CONTENT_WIDTH - 16,
        font='Helvetica-Bold',
        size=7.5,
        leading=9.3,
        color=RED,
    )
    footer(pdf, 4)
    pdf.showPage()


def results_page(pdf: canvas.Canvas) -> None:
    y = page_header(pdf, '5. Results', 5)
    y = paragraph(
        pdf,
        "The rows below are same-seed engineering diagnostics, not the competition's mandated five randomized scored trials. They are retained to make limitations and changes auditable.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=8.2,
        leading=10.2,
    ) - 6
    development = [
        ['Development run', 'Wall s', 'Terminal reason / evidence', 'Col./row', 'Picks', 'Nav', 'Coll.', 'Bin'],
        ['retry 15 / f2e020010e12', '1044.258', 'book_reacquisition_failed after 2 failed picks', 'yes / 1', '2', '4', '3 shelf', 'no'],
        ['retry 16 / f59b1e43ab8f', '463.726', 'ID/nav reached; intentional shutdown after near-zero close', 'yes / 1', '1', '4', '0', 'no'],
        ['retry 17 / 85b3813da159', '1200.575', 'timeout entering retry after carried-route failure', 'yes / 1', '1', '4', '2', 'no'],
        ['retry 18 / e2ba29e074ef', '674.928', 'book_reacquisition_failed after safe pick recovery', 'yes / 1', '1', '4', '1', 'no'],
        ['retry 19 / 2172a3aa1379', '551.510', 'book_reacquisition_failed after home-transition loss', 'yes / 1', '1', '4', '1', 'no'],
    ]
    y = draw_table(pdf, development, MARGIN, y, [105, 43, 139, 55, 34, 31, 35, CONTENT_WIDTH - 442], font_size=5.9) - 9
    y = section_label(pdf, 'Required five randomized final trials - complete before submission', MARGIN, y)
    final_trials = [
        ['Trial', 'Random request / seed', 'Column', 'Row', 'Grasp', 'Place/contact', 'Time', 'Score'],
        ['FINAL_T1', '[pending]', 'TBD', 'TBD', 'TBD', 'TBD', 'TBD', 'TBD'],
        ['FINAL_T2', '[pending]', 'TBD', 'TBD', 'TBD', 'TBD', 'TBD', 'TBD'],
        ['FINAL_T3', '[pending]', 'TBD', 'TBD', 'TBD', 'TBD', 'TBD', 'TBD'],
        ['FINAL_T4', '[pending]', 'TBD', 'TBD', 'TBD', 'TBD', 'TBD', 'TBD'],
        ['FINAL_T5', '[pending]', 'TBD', 'TBD', 'TBD', 'TBD', 'TBD', 'TBD'],
        ['Aggregate', '5 resets required', 'TBD', 'TBD', 'TBD', 'TBD', 'mean TBD', 'mean TBD'],
    ]
    y = draw_table(pdf, final_trials, MARGIN, y, [50, 102, 52, 42, 52, 85, 50, CONTENT_WIDTH - 433], font_size=6.1) - 10
    y = section_label(pdf, '6. Limitations and Future Work', MARGIN, y)
    y = bullets(
        pdf,
        [
            'Validation coverage: current live observations use one seed/request. Complete five reset randomized trials, ground-truth each perception result, review the unedited video and record official scores.',
            'Manipulation: retry 19 retained through all extraction/lowering legs, then lost the book on first home transition. Live-test the new carried-book-vs-modeled-robot guard, including omitted tool/gripper geometry, and prove retention through return/place.',
            'Navigation: straight-line odometry control has no global replanning. The required RViz planned/executed evidence is captured for retry 14; test every shelf column plus return/bin paths.',
            'Perception: template/HSV thresholds assume official lighting and RGB-depth registration. Expand randomized evidence and report true accuracy rather than internal confidence.',
            'Baseline/submission: the supplied Phase 1 PDF names v1.0.0, while development uses official repository v1.0.3 with gripper/book fixes. Obtain organizer approval for the evaluator tag; then replace identity/link placeholders, publish GitHub and record one unedited maximum 5-minute YouTube run.',
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=7.2,
        leading=8.7,
        gap=1.5,
    )
    pdf.setFillColor(PALE)
    pdf.roundRect(MARGIN, 17 * mm, CONTENT_WIDTH, 18 * mm, 5, fill=1, stroke=0)
    paragraph(
        pdf,
        'Not competition-ready: retry 19 failed, five randomized official trials are not done, and no score is claimed. Historical ROS checks: 139 passed and colcon built; latest host-only geometry/helper checks: 118 passed, 2 ROS modules skipped. Live validation of the current guard remains required.',
        MARGIN + 8,
        29 * mm,
        CONTENT_WIDTH - 16,
        size=7.0,
        leading=8.5,
    )
    footer(pdf, 5)
    pdf.showPage()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=A4, pageCompression=1)
    pdf.setTitle('ERC 2026 Phase 1 Technical Report - Submission Draft')
    pdf.setAuthor('[TEAM NAME REQUIRED]')
    pdf.setSubject('Autonomous visual book retrieval and bin placement')
    title_page(pdf)
    architecture_page(pdf)
    perception_page(pdf)
    navigation_page(pdf)
    manipulation_page(pdf)
    results_page(pdf)
    pdf.save()
    print(OUTPUT)


if __name__ == '__main__':
    main()
