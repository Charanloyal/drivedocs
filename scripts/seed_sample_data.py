"""Regenerate the synthetic sample documents in data/sample/ (development only; requires reportlab).

The generated files are committed, so this script is not needed to run or deploy DriveDocs. The evaluation set
(data/sample/evaluation_set.json) is maintained by hand and is not touched by this script. Regenerating the PDF
changes its bytes and therefore its document ID.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def create_standalone_diagram(out_path: Path) -> Path:
    """Generate a clean synthetic technical architecture diagram using PIL."""
    img = Image.new("RGB", (800, 500), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Draw title
    draw.text((40, 30), "DRIVEDOCS MULTIMODAL SYSTEM ARCHITECTURE", fill=(248, 250, 252))
    draw.text((40, 55), "Dual-Track Vector Retrieval Pipeline & Grounded Synthesis", fill=(148, 163, 184))

    # Draw boxes
    boxes = [
        ((40, 120, 220, 260), (30, 41, 59), (99, 102, 241), "1. Content Parser\n- Multi-page PDF\n- Image Extractor\n- Text Chunker"),
        ((300, 120, 480, 260), (30, 41, 59), (6, 182, 212), "2. Vector Store\n- MiniLM (384-d)\n- CLIP Vision (512-d)\n- FAISS InnerProduct"),
        ((560, 120, 740, 260), (30, 41, 59), (16, 185, 129), "3. Grounded QA\n- Citation Verifier\n- Fact Alignment\n- Extractive Fallback"),
    ]

    for (x1, y1, x2, y2), bg_col, border_col, text in boxes:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill=bg_col, outline=border_col, width=2)
        lines = text.split("\n")
        y_text = y1 + 18
        draw.text((x1 + 14, y_text), lines[0], fill=(255, 255, 255))
        y_text += 24
        for l in lines[1:]:
            draw.text((x1 + 14, y_text), l, fill=(203, 213, 225))
            y_text += 20

    # Draw arrows
    draw.line([(220, 190), (300, 190)], fill=(99, 102, 241), width=3)
    draw.polygon([(290, 185), (300, 190), (290, 195)], fill=(99, 102, 241))

    draw.line([(480, 190), (560, 190)], fill=(6, 182, 212), width=3)
    draw.polygon([(550, 185), (560, 190), (550, 195)], fill=(6, 182, 212))

    # Bottom summary box
    draw.rounded_rectangle([40, 320, 740, 440], radius=8, fill=(24, 32, 47), outline=(71, 85, 105), width=1)
    draw.text((60, 340), "Performance Specifications & Bounds:", fill=(129, 140, 248))
    draw.text((60, 370), "• Latency Target: < 450 ms end-to-end CPU inference", fill=(203, 213, 225))
    draw.text((60, 395), "• Embedding Normalization: L2 unit vector spherical projection for exact cosine similarity", fill=(203, 213, 225))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path


def create_photonics_chart(out_path: Path) -> Path:
    """Generate a synthetic efficiency curve chart to embed into PDF."""
    img = Image.new("RGB", (500, 260), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)

    # Axis
    draw.line([(60, 20), (60, 220)], fill=(100, 116, 139), width=2)
    draw.line([(60, 220), (460, 220)], fill=(100, 116, 139), width=2)

    draw.text((70, 25), "SNSPD Quantum Efficiency vs Wavelength (nm)", fill=(30, 41, 59))

    # Draw curve
    points = [
        (60, 180), (120, 160), (180, 130), (240, 90),
        (300, 60), (360, 50), (420, 55), (460, 75)
    ]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=(79, 70, 229), width=3)

    # Highlight peak at 1550nm
    draw.ellipse([(355, 45), (365, 55)], fill=(239, 68, 68), outline=(185, 28, 28))
    draw.text((370, 40), "Peak: 98.4% @ 1550 nm", fill=(185, 28, 28))

    draw.text((60, 230), "1200nm", fill=(100, 116, 139))
    draw.text((220, 230), "1400nm", fill=(100, 116, 139))
    draw.text((340, 230), "1550nm", fill=(100, 116, 139))
    draw.text((430, 230), "1700nm", fill=(100, 116, 139))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path


def create_quantum_pdf(pdf_path: Path, chart_path: Path) -> Path:
    """Generate a realistic two-page technical report with embedded diagram."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=12
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10
    )

    story = []

    # PAGE 1
    story.append(Paragraph("Quantum Photonics Integrated Circuit Specification", title_style))
    story.append(Paragraph("Document Reference: QPIC-TR-2026-V4 | Page 1 Architecture Overview", styles['Italic']))
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Photonic quantum processors harness discrete photons as information-carrying qubits, leveraging integrated "
        "silicon nitride (SiN) waveguides for phase-stable spatial routing. The optical transmission loss of the primary "
        "waveguide bus is precisely calibrated to 0.12 dB/cm across the telecommunication C-band spectrum. "
        "A reconfigurable triangular lattice of Mach-Zehnder interferometers provides universal linear optical unitaries.",
        body_style
    ))
    story.append(Paragraph(
        "Phase modulation is executed through thermo-optic phase shifters with a switching rise time of 1.4 microseconds "
        "and low cross-talk below -34 dB. Cryogenic coupling is achieved through inverted taper edge couplers boasting "
        "an edge-coupling insertion loss of 0.8 dB per facet.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # PAGE BREAK
    story.append(PageBreak())

    # PAGE 2
    story.append(Paragraph("Detection Subsystem & Cryostat Calibration", title_style))
    story.append(Paragraph("Document Reference: QPIC-TR-2026-V4 | Page 2 Detector Performance", styles['Italic']))
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Detection occurs using Superconducting Nanowire Single-Photon Detectors (SNSPDs) micro-fabricated from "
        "amorphous molybdenum silicide (MoSi) thin films. The detection system achieves a peak quantum efficiency of 98.4% "
        "at 1550 nm telecommunication wavelength, operating in a closed-cycle pulse-tube cryostat maintaining a stable "
        "base temperature of 0.85 Kelvin. Timing jitter is suppressed below 14.8 picoseconds FWHM.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Embed chart on Page 2
    story.append(RLImage(str(chart_path), width=420, height=218))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Figure 2.1: Calibrated Single-Photon Detector Quantum Efficiency across the 1200-1700 nm optical band. "
        "Notice the high-fidelity resonant plateau centered at 1550 nm.",
        styles['Italic']
    ))

    doc.build(story)
    return pdf_path


def create_markdown_doc(md_path: Path) -> Path:
    """Generate Markdown document on autonomous agent safety protocols."""
    content = """# Autonomous Agent Safety Protocols & Governance

## System Boundary Enforcements
Autonomous AI agents operating in continuous execution environments must adhere to strict runtime containment rules.

### Containment Boundaries
1. **Tool Invocation Whitelist**: Agents are restricted to safe, sandboxed tools. Direct arbitrary shell execution without validation is prohibited.
2. **API Rate Limiting**: The agent orchestrator enforces a maximum throughput ceiling of 120 calls per minute per agent worker.
3. **Audit Trail Immutability**: All tool payloads, state transitions, and file mutations are committed to an append-only cryptographic ledger with SHA-256 block signatures.

## Human-in-the-Loop Review Thresholds
Any action involving financial fund disbursement exceeding $500 USD, infrastructure deletion, or schema drop operations triggers an asynchronous approval escalation modal that halts execution until an authorized human supervisor signs the approval token.
"""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    return md_path


def create_txt_doc(txt_path: Path) -> Path:
    """Generate plain text document on distributed consensus parameters."""
    content = """DISTRIBUTED CONSENSUS SPECIFICATION: RAFT ENGINE v2

1. LEADER ELECTION PROTOCOL
Nodes initialize in the Follower state. If a follower node receives no communication from a cluster leader within the randomized election timeout period, it transitions to the Candidate state and increments its current term number.
The randomized election timeout is configured between 300 milliseconds and 600 milliseconds to avoid split-vote oscillations.

2. HEARTBEAT INTERVALS
The cluster leader periodically dispatches empty AppendEntries RPC heartbeats at a fixed interval of 150 milliseconds to maintain cluster dominance and reset follower election countdown timers.

3. QUORUM THRESHOLDS
A cluster consisting of N nodes requires a minimum quorum of floor(N/2) + 1 active nodes to commit a state machine log entry. For a 5-node cluster, 3 nodes constitute an authoritative quorum.
"""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    return txt_path



def main():
    print("[*] Generating synthetic sample files...")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    arch_img = SAMPLE_DIR / "architecture_diagram.png"
    create_standalone_diagram(arch_img)
    print(f" [+] Created: {arch_img.name}")

    chart_img = SAMPLE_DIR / "snspd_efficiency_chart.png"
    create_photonics_chart(chart_img)
    print(f" [+] Created: {chart_img.name}")

    pdf_doc = SAMPLE_DIR / "quantum_photonics_spec.pdf"
    create_quantum_pdf(pdf_doc, chart_img)
    print(f" [+] Created: {pdf_doc.name}")

    md_doc = SAMPLE_DIR / "agent_safety_protocol.md"
    create_markdown_doc(md_doc)
    print(f" [+] Created: {md_doc.name}")

    txt_doc = SAMPLE_DIR / "distributed_systems_notes.txt"
    create_txt_doc(txt_doc)
    print(f" [+] Created: {txt_doc.name}")

    print("[*] Sample documents regenerated. data/sample/evaluation_set.json was not modified.")


if __name__ == "__main__":
    main()
