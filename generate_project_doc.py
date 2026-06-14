# -*- coding: utf-8 -*-
import os
import sys
import subprocess

# ✅ ตรวจสอบและติดตั้ง reportlab อัตโนมัติหากยังไม่มี
try:
    import reportlab
except ImportError:
    print("Installing reportlab library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def build_pdf(filename="Project_Details_Surveillance.pdf"):
    # ✅ ลงทะเบียนฟอนต์ภาษาไทยจากระบบ Windows (Tahoma) เพื่อป้องกันสระลอยและแสดงภาษาไทยถูกต้อง
    font_path = "C:/Windows/Fonts/tahoma.ttf"
    font_bold_path = "C:/Windows/Fonts/tahomabd.ttf"
    
    if not os.path.exists(font_path) or not os.path.exists(font_bold_path):
        print("Error: Windows Tahoma fonts not found in C:/Windows/Fonts. Falling back to system fonts.")
        # ลองหาฟอนต์อื่นหากไม่มี Tahoma (กรณีรันบน OS อื่น)
        fallback_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf"
        ]
        font_path = next((p for p in fallback_paths if os.path.exists(p)), None)
        if not font_path:
            print("No fonts available. Cannot guarantee Thai text display.")
            return

    pdfmetrics.registerFont(TTFont('Tahoma', font_path))
    pdfmetrics.registerFont(TTFont('Tahoma-Bold', font_bold_path))

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # ✅ กำหนดสไตล์ตัวอักษรภาษาไทย
    title_style = ParagraphStyle(
        'ThaiTitle',
        parent=styles['Normal'],
        fontName='Tahoma-Bold',
        fontSize=20,
        textColor=colors.HexColor('#FFFFFF'),
        alignment=1, # Center
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'ThaiSubtitle',
        parent=styles['Normal'],
        fontName='Tahoma',
        fontSize=12,
        textColor=colors.HexColor('#E2E8F0'),
        alignment=1, # Center
    )

    h1_style = ParagraphStyle(
        'ThaiH1',
        parent=styles['Normal'],
        fontName='Tahoma-Bold',
        fontSize=14,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=14,
        spaceAfter=6,
        borderPadding=(0, 0, 2, 0),
        borderColor=colors.HexColor('#1A365D')
    )

    h2_style = ParagraphStyle(
        'ThaiH2',
        parent=styles['Normal'],
        fontName='Tahoma-Bold',
        fontSize=11,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'ThaiBody',
        parent=styles['Normal'],
        fontName='Tahoma',
        fontSize=9.5,
        textColor=colors.HexColor('#2D3748'),
        leading=14,
        spaceAfter=6
    )

    body_bold_style = ParagraphStyle(
        'ThaiBodyBold',
        parent=body_style,
        fontName='Tahoma-Bold'
    )

    bullet_style = ParagraphStyle(
        'ThaiBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Tahoma-Bold',
        fontSize=9,
        textColor=colors.white,
        alignment=1
    )

    story = []

    # ── 1. HEADER BANNER ──────────────────────────────────────────
    banner_data = [
        [Paragraph("เอกสารรายละเอียดโครงการ Mini Project", subtitle_style)],
        [Paragraph("ระบบเฝ้าระวังพื้นที่อัจฉริยะโดยใช้ปัญญาประดิษฐ์", title_style)],
        [Paragraph("(AI-Based Smart Surveillance System)", subtitle_style)]
    ]
    banner_table = Table(banner_data, colWidths=[520])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1A365D')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 16),
        ('TOPPADDING', (0,0), (-1,-1), 16),
        ('LEFTPADDING', (0,0), (-1,-1), 20),
        ('RIGHTPADDING', (0,0), (-1,-1), 20),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 15))

    # ── 2. BACKGROUND & IMPORTANCE ────────────────────────────────
    story.append(Paragraph("1. ความเป็นมาและความสำคัญ", h1_style))
    story.append(Paragraph(
        "ในปัจจุบันระบบกล้องวงจรปิด (CCTV) ถูกติดตั้งอย่างแพร่หลายเพื่อความปลอดภัย "
        "อย่างไรก็ตาม การเฝ้าระวังแบบดั้งเดิมยังคงต้องอาศัยเจ้าหน้าที่จับตามองหน้าจอตลอดเวลา "
        "ซึ่งมีข้อจำกัดด้านสายตา สมาธิ และอาจเกิดความผิดพลาดจากมนุษย์ (Human Error) ได้ง่าย "
        "โครงการนี้จึงมุ่งพัฒนาระบบเฝ้าระวังพื้นที่อัตโนมัติโดยนำเทคโนโลยี Computer Vision และ "
        "Machine Learning มาช่วยวิเคราะห์ภาพจากกล้องแบบ Real-time เพื่อตรวจจับบุคคล ประเมินพฤติกรรม "
        "และแจ้งเตือนทันทีเมื่อพบความผิดปกติโดยไม่ต้องพึ่งพาเจ้าหน้าที่เฝ้าดูตลอดเวลา",
        body_style
    ))

    # ── 3. OBJECTIVES ─────────────────────────────────────────────
    story.append(Paragraph("2. วัตถุประสงค์ของโครงการ", h1_style))
    objectives = [
        "เพื่อพัฒนาระบบตรวจจับและติดตามบุคคลในพื้นที่แบบ Real-time ผ่านกล้องอย่างน้อย 2 ตัวพร้อมกัน",
        "เพื่อประเมินระดับความเสี่ยงจากพฤติกรรมและลักษณะภายนอกของบุคคลโดยใช้ปัญญาประดิษฐ์ (AI)",
        "เพื่อตรวจจับวัตถุอันตราย เช่น อาวุธปืนและมีด ด้วยโมเดลปัญญาประดิษฐ์เฉพาะทาง",
        "เพื่อแจ้งเตือนเมื่อพบเหตุการณ์ผิดปกติและบันทึกข้อมูลเหตุการณ์ (Event Log) พร้อมหลักฐานภาพถ่ายสำหรับย้อนดูย้อนหลัง"
    ]
    for idx, obj in enumerate(objectives, 1):
        story.append(Paragraph(f"<b>2.{idx}</b> {obj}", bullet_style))

    # ── 4. SCOPE ──────────────────────────────────────────────────
    story.append(Paragraph("3. ขอบเขตของโครงการ", h1_style))
    scopes = [
        "<b>การเชื่อมต่อกล้อง:</b> รองรับกล้องอย่างน้อย 2 ตัวทำงานพร้อมกันผ่าน Multi-threading แบบขนาน",
        "<b>การประมวลผลวิดีโอ:</b> ประมวลผลภาพนิ่งและวิดีโอแบบ Real-time บนอุปกรณ์ Edge (Local GPU)",
        "<b>ระดับความเสี่ยง:</b> จำแนกและประเมินระดับความเสี่ยงเป็น 4 ระดับ ได้แก่ LOW / MEDIUM / HIGH / CRITICAL",
        "<b>ความเป็นส่วนตัว (Privacy):</b> ไม่ครอบคลุมการระบุตัวตนบุคคลเชิงลึก (Face Recognition) เพื่อรักษาความเป็นส่วนตัวตามกฎหมาย PDPA"
    ]
    for scope in scopes:
        story.append(Paragraph(f"• {scope}", bullet_style))

    # ── 5. SYSTEM ARCHITECTURE & TECHNOLOGY ────────────────────────
    story.append(Paragraph("4. สถาปัตยกรรมระบบและเทคโนโลยีที่ใช้", h1_style))
    
    # ตารางเปรียบเทียบเทคโนโลยี
    tech_data = [
        [Paragraph("งานประมวลผล", table_header_style), Paragraph("เทคโนโลยี / โมเดลที่ใช้", table_header_style), Paragraph("บทบาทและหน้าที่", table_header_style)],
        
        [Paragraph("<b>Object Detection</b>", body_style), 
         Paragraph("YOLOv8 (Ultralytics)", body_style), 
         Paragraph("ตรวจจับบุคคล, วัตถุ และอาวุธแบบกรอบ Bounding Box", body_style)],
         
        [Paragraph("<b>Multi-Object Tracking</b>", body_style), 
         Paragraph("BoT-SORT + Kalman Filter", body_style), 
         Paragraph("ติดตามตัวบุคคลข้ามเฟรม ขจัดสัญญาณสั่นไหวของพิกัด และทำนายทิศทางล่วงหน้า", body_style)],
         
        [Paragraph("<b>Appearance Analysis</b>", body_style), 
         Paragraph("EfficientNet-B0 (Custom)", body_style), 
         Paragraph("วิเคราะห์เครื่องแต่งกายที่เป็นภัยเงียบ หรือการปิดบังใบหน้าด้วยหน้ากาก/หมวก", body_style)],
         
        [Paragraph("<b>Behavior Recognition</b>", body_style), 
         Paragraph("YOLOv8-Pose + LSTM", body_style), 
         Paragraph("วิเคราะห์คีย์พอยท์ข้อต่อร่างกายเพื่อจำแนกพฤติกรรมเสี่ยง เช่น การวิ่งหรือการสอดส่อง", body_style)],
         
        [Paragraph("<b>Weapon Detection</b>", body_style), 
         Paragraph("YOLOv8n (Custom Train)", body_style), 
         Paragraph("โมเดลตรวจจับปืนและมีดผ่านการฝึกฝนจากชุดข้อมูลจริง", body_style)],
         
        [Paragraph("<b>Image Processing</b>", body_style), 
         Paragraph("OpenCV", body_style), 
         Paragraph("จัดการฟีดภาพสดจากกล้อง, ย่อภาพ, วาด Graphic Overlay แจ้งเตือน", body_style)]
    ]
    
    tech_table = Table(tech_data, colWidths=[110, 160, 250])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 10))

    # ── 6. ADVANCED RISK ANALYSIS FEATURES ─────────────────────────
    story.append(Paragraph("5. ฟีเจอร์วิเคราะห์ความเสี่ยงขั้นสูง (เพิ่มเติม)", h1_style))
    story.append(Paragraph(
        "เพื่อให้ระบบเฝ้าระวังฉลาดและแม่นยำยิ่งขึ้น ระบบมีการนำลอจิกวิเคราะห์พฤติกรรมเชิงลึกเข้ามาผสาน:",
        body_style
    ))
    
    adv_features = [
        "<b>Time-Based Geofencing (การจำกัดพื้นที่ตามเวลา):</b> กำหนดเขตพื้นที่เสี่ยงที่แปรผันตามเวลา เช่น ประตูหลังอาคารในเวลาทำการความเสี่ยงปกติ (LOW) แต่หากล้ำเส้นหลังเวลา 22:00 น. จะปรับระดับเป็นวิกฤต (CRITICAL) ทันที",
        "<b>Sequential Risk State Machine (สถานะความเสี่ยงสะสม):</b> ป้องกันการแจ้งเตือนผิดพลาดโดยคิดคะแนนสะสมเชิงพฤติกรรมต่อเนื่อง เช่น ยืนนิ่งนานผิดปกติ ➡️ เริ่มเคลื่อนที่เร็วขึ้น ➡️ หยิบวัตถุต้องสงสัย",
        "<b>Heatmap & Path Outliers:</b> วาดแผนที่สะสมความหนาแน่นการใช้งานพื้นที่ของมนุษย์ เพื่อชี้วัดและแจ้งเตือนทันทีเมื่อมีคนเดินผ่านมุมอับหรือจุดต้องห้ามที่ปกติไม่มีคนผ่านเลย"
    ]
    for feat in adv_features:
        story.append(Paragraph(f"• {feat}", bullet_style))

    # ── 7. PROJECT CHECKLIST ──────────────────────────────────────
    story.append(Paragraph("6. สถานะความคืบหน้าของโครงการ (Project Status Checklist)", h1_style))
    
    checklist_data = [
        [Paragraph("หัวข้อฟังก์ชัน", table_header_style), Paragraph("คำอธิบายงาน", table_header_style), Paragraph("สถานะ", table_header_style)],
        
        [Paragraph("Threaded Camera Reader", body_style), Paragraph("อ่านเฟรมกล้องพร้อมกันหลายตัวผ่าน Background Thread เพื่อไม่ให้หน่วง", body_style), Paragraph("<font color='#38A169'><b>เสร็จสิ้น (Done)</b></font>", body_style)],
        
        [Paragraph("Kalman Trajectory & Prediction", body_style), Paragraph("ติดตามพิกัด ลดสัญญาณรบกวนของเส้นวิถี และทำนายทิศทางล่วงหน้า", body_style), Paragraph("<font color='#38A169'><b>เสร็จสิ้น (Done)</b></font>", body_style)],
         
        [Paragraph("Advanced Risk Assessment Engine", body_style), Paragraph("ประเมินความเสี่ยงด้วย Rule-based + Geofencing + Time Rule + Heuristic Behavior", body_style), Paragraph("<font color='#DD6B20'><b>กำลังดำเนินการ (In Progress)</b></font>", body_style)],
         
        [Paragraph("Event Logging & Screen Capture", body_style), Paragraph("บันทึกตารางเหตุการณ์ลง CSV พร้อมจับภาพสกรีนช็อคช่วงเกิดเหตุลงโฟลเดอร์", body_style), Paragraph("<font color='#DD6B20'><b>กำลังดำเนินการ (In Progress)</b></font>", body_style)],
         
        [Paragraph("Cross-Camera Re-ID / Heatmap Overlay", body_style), Paragraph("วาด Heatmap บนฟีดวิดีโอ และเก็บประวัติผู้ใช้ข้ามกล้อง", body_style), Paragraph("<font color='#E53E3E'><b>รอดำเนินการ (To Do)</b></font>", body_style)]
    ]
    
    checklist_table = Table(checklist_data, colWidths=[150, 250, 120])
    checklist_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4A5568')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (2,1), (2,-1), 'CENTER')
    ]))
    story.append(checklist_table)

    # ✅ เริ่ม Build PDF
    doc.build(story)
    print(f"Successfully generated PDF: {filename}")

if __name__ == "__main__":
    build_pdf()
