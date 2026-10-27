"""Generate the synthetic vehicle-manual demo corpus in data/sample/ (development only; requires reportlab).

All content describes a fictional vehicle, the "Northwind Aurora" electric sedan, and was written for this project.
It is not copied from any real owner's manual. The service record contains clearly fake personal data
(example.com e-mail, 555 phone number, a test card number) that exists only to demonstrate PII redaction.

The generated files are committed, so this script is not needed to run or deploy DriveDocs. The evaluation sets
(data/sample/evaluation_set.json, data/sample/security_eval_set.json) are maintained by hand. Regenerating a file
changes its bytes and therefore its document ID.

    python scripts/seed_sample_data.py
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

Color = Tuple[int, int, int]
INK: Color = (30, 41, 59)
MUTED: Color = (100, 116, 139)


def font(size: int, bold: bool = False):
    for name in (("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def centered(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, fnt, fill: Color) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


# --------------------------------------------------------------------------------------------------------------------
# Diagrams
# --------------------------------------------------------------------------------------------------------------------


def warning_lights_chart(out: Path) -> Path:
    """Dashboard warning-light chart: a grid of round coloured indicator symbols on a dark instrument panel."""
    img = Image.new("RGB", (900, 560), (17, 24, 39))
    d = ImageDraw.Draw(img)
    d.text((30, 22), "Aurora instrument cluster: warning and indicator lights", font=font(26, True), fill=(241, 245, 249))
    lights = [
        ((220, 38, 38), "!", "Brake system"),
        ((220, 38, 38), "+-", "12V battery"),
        ((220, 38, 38), "°C", "Battery temp"),
        ((245, 158, 11), "(!)", "Tire pressure"),
        ((245, 158, 11), "EV", "Powertrain"),
        ((245, 158, 11), "ABS", "ABS fault"),
        ((34, 197, 94), "P", "Ready to drive"),
        ((59, 130, 246), "HI", "High beam"),
    ]
    for i, (col, sym, label) in enumerate(lights):
        cx, cy = 130 + (i % 4) * 215, 170 + (i // 4) * 210
        d.ellipse((cx - 62, cy - 62, cx + 62, cy + 62), fill=(30, 41, 59), outline=col, width=8)
        centered(d, (cx, cy - 4), sym, font(40, True), col)
        centered(d, (cx, cy + 92), label, font(22), (203, 213, 225))
    d.text((30, 520), "Red = stop safely   Amber = service soon   Green/Blue = information", font=font(20), fill=(148, 163, 184))
    img.save(out, "PNG")
    return out


def tire_pressure_placard(out: Path) -> Path:
    """Top view of the car with the recommended cold tire pressure at each wheel."""
    img = Image.new("RGB", (900, 600), (248, 250, 252))
    d = ImageDraw.Draw(img)
    d.text((30, 20), "Tire and loading information placard (cold tires)", font=font(26, True), fill=INK)
    d.rounded_rectangle((330, 90, 570, 560), radius=90, outline=INK, width=6, fill=(226, 232, 240))
    d.rounded_rectangle((365, 170, 535, 300), radius=24, outline=MUTED, width=4)
    d.rounded_rectangle((365, 380, 535, 480), radius=24, outline=MUTED, width=4)
    for x, y, label in [(290, 150, "FRONT 2.5 bar / 36 psi"), (560, 150, "FRONT 2.5 bar / 36 psi"),
                        (290, 430, "REAR 2.7 bar / 39 psi"), (560, 430, "REAR 2.7 bar / 39 psi")]:
        d.rounded_rectangle((x, y, x + 50, y + 110), radius=12, fill=(15, 23, 42))
        tx = 40 if x < 400 else 640
        d.text((tx, y + 40), label, font=font(21, True), fill=(185, 28, 28))
    d.text((30, 568), "Tire size 235/45 R19 99W  -  Max. vehicle load 450 kg", font=font(18), fill=MUTED)
    img.save(out, "PNG")
    return out


def charging_port_diagram(out: Path) -> Path:
    """Charging inlet: Type 2 AC connector above the two DC pins of the CCS combo inlet, with status light."""
    img = Image.new("RGB", (900, 600), (241, 245, 249))
    d = ImageDraw.Draw(img)
    d.text((30, 20), "Charging inlet (CCS Combo 2) and status light", font=font(26, True), fill=INK)
    d.ellipse((300, 90, 600, 390), fill=(30, 41, 59))
    for i in range(7):
        ang = 2 * math.pi * i / 7
        px, py = 450 + (85 * math.cos(ang) if i else 0), 240 + (85 * math.sin(ang) if i else 0)
        d.ellipse((px - 24, py - 24, px + 24, py + 24), fill=(226, 232, 240))
    d.rounded_rectangle((320, 410, 580, 540), radius=60, fill=(30, 41, 59))
    for px in (395, 505):
        d.ellipse((px - 40, 435, px + 40, 515), fill=(250, 204, 21))
    d.ellipse((650, 220, 710, 280), fill=(34, 197, 94))
    d.text((725, 230), "status light", font=font(20), fill=INK)
    d.text((40, 200), "AC Type 2\nup to 11 kW", font=font(22, True), fill=INK)
    d.text((40, 450), "DC pins\nup to 150 kW", font=font(22, True), fill=INK)
    img.save(out, "PNG")
    return out


def fuse_box_layout(out: Path) -> Path:
    """Fuse box under the glovebox: grid of colour-coded blade fuses with positions and amperage."""
    img = Image.new("RGB", (900, 620), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((30, 20), "Interior fuse box (behind passenger footwell panel)", font=font(26, True), fill=INK)
    d.rectangle((60, 80, 840, 590), outline=INK, width=5, fill=(229, 231, 235))
    amp_colors = {5: (234, 179, 8), 7: (139, 69, 19), 10: (220, 38, 38), 15: (37, 99, 235), 20: (250, 204, 21),
                  25: (230, 230, 230), 30: (34, 197, 94)}
    fuses = [(1, 10), (2, 15), (3, 20), (4, 10), (5, 5), (6, 30), (7, 10), (8, 7), (9, 15), (10, 25),
             (11, 20), (12, 15), (13, 10), (14, 5), (15, 30), (16, 10)]
    for i, (num, amps) in enumerate(fuses):
        x, y = 110 + (i % 8) * 92, 140 + (i // 8) * 220
        d.rounded_rectangle((x, y, x + 60, y + 120), radius=8, fill=amp_colors[amps], outline=INK, width=3)
        centered(d, (x + 30, y + 60), f"{amps}A", font(20, True), INK)
        centered(d, (x + 30, y + 145), f"F{num}", font(22, True), INK)
    img.save(out, "PNG")
    return out


def front_compartment_diagram(out: Path) -> Path:
    """Standalone image: top view of the front compartment with service points labelled."""
    img = Image.new("RGB", (1000, 640), (15, 23, 42))
    d = ImageDraw.Draw(img)
    d.text((30, 20), "Front compartment (under the hood): service points", font=font(28, True), fill=(241, 245, 249))
    d.polygon([(170, 610), (220, 110), (780, 110), (830, 610)], outline=(148, 163, 184), width=6)
    parts = [
        ((260, 160, 470, 300), (59, 130, 246), "Washer fluid"),
        ((530, 160, 740, 300), (234, 179, 8), "Brake fluid"),
        ((260, 350, 470, 520), (239, 68, 68), "12V battery"),
        ((530, 350, 740, 520), (45, 212, 191), "Coolant"),
    ]
    for box, col, label in parts:
        d.rounded_rectangle(box, radius=18, outline=col, width=6, fill=(30, 41, 59))
        centered(d, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, font(26, True), (241, 245, 249))
    img.save(out, "PNG")
    return out


# --------------------------------------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------------------------------------


def owners_manual(pdf_path: Path, diagrams: dict) -> Path:
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=19, leading=23, textColor=colors.HexColor("#0F172A"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=17, textColor=colors.HexColor("#1E3A8A"))
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15.5, spaceAfter=8,
                          textColor=colors.HexColor("#1F2937"))
    cap = ParagraphStyle("Cap", parent=styles["Italic"], fontSize=9, leading=12, textColor=colors.HexColor("#475569"))

    def para(text: str):
        return Paragraph(text, body)

    def figure(key: str, caption: str, width: float = 400):
        w, h = Image.open(diagrams[key]).size
        return [Spacer(1, 6), RLImage(str(diagrams[key]), width=width, height=width * h / w), Paragraph(caption, cap)]

    s = []
    # Page 1 - introduction and identification
    s += [Paragraph("Northwind Aurora Owner's Manual", h1),
          Paragraph("Model year 2026 · Electric sedan · Chapter 1: Introduction and vehicle identification", cap),
          Spacer(1, 10),
          para("This manual describes the operation and care of the Northwind Aurora, a fictional battery-electric "
               "sedan used as sample content for DriveDocs. Read the safety chapters before driving."),
          Paragraph("Vehicle identification number (VIN)", h2),
          para("The 17-character vehicle identification number is stamped on a plate at the lower left corner of the "
               "windshield, visible from outside the vehicle, and printed on the certification label on the "
               "driver-side door pillar (B-pillar). Quote the VIN when ordering parts or booking service."),
          Paragraph("Key specifications", h2),
          para("The high-voltage traction battery has a usable capacity of 82 kWh. The estimated driving range is "
               "520 km on the WLTP combined cycle. The rear-mounted motor produces 250 kW, and the Aurora accelerates "
               "from 0 to 100 km/h in 5.9 seconds. Kerb weight is 2,050 kg."),
          PageBreak()]
    # Page 2 - warning lights
    s += [Paragraph("Chapter 2: Instrument cluster and warning lights", h1),
          para("Warning lights use three colours. A red light means stop safely as soon as possible. An amber light "
               "means the vehicle needs service soon. Green and blue lights are for information only."),
          para("If the red brake system warning light stays on after you release the parking brake, the brake fluid "
               "level may be low. Stop the vehicle safely and contact Northwind Assistance; do not continue driving."),
          para("The amber tire pressure warning light comes on when the pressure in one or more tires is 25% or more "
               "below the recommended cold pressure. Check and adjust the pressures as soon as possible."),
          para("The red 12V battery light indicates a fault in the low-voltage charging system. Accessories may stop "
               "working; drive to a safe place and contact service."),
          *figure("warning", "Figure 2.1: Dashboard warning and indicator lights of the Aurora instrument cluster."),
          PageBreak()]
    # Page 3 - tires
    s += [Paragraph("Chapter 3: Tires and wheels", h1),
          para("The recommended cold tire pressure is 2.5 bar (36 psi) for the front tires and 2.7 bar (39 psi) for the "
               "rear tires. Check pressures monthly, when the tires are cold, using the values on the tire and loading "
               "information placard on the driver-side door pillar."),
          para("Rotate the tires every 10,000 km to even out wear. After rotating the tires or adjusting pressures, reset "
               "the tire pressure monitoring system (see the troubleshooting guide)."),
          para("Tighten the wheel lug nuts to 140 N·m in a star pattern. The Aurora has no spare wheel; a tire repair "
               "kit with sealant and a 12V compressor is stored under the rear cargo floor."),
          *figure("tires", "Figure 3.1: Tire and loading information placard showing front and rear cold pressures."),
          PageBreak()]
    # Page 4 - charging
    s += [Paragraph("Chapter 4: Charging", h1),
          para("The charging inlet behind the left rear flap accepts Type 2 AC charging at up to 11 kW and CCS Combo 2 "
               "DC fast charging at up to 150 kW. On a DC fast charger, charging from 10% to 80% takes about 28 minutes "
               "when the battery is at operating temperature."),
          para("The status light next to the inlet shows the charging state: pulsing green means charging, solid green "
               "means charging is complete, blue means the vehicle is waiting for a scheduled charge, and red means a "
               "charging fault. If the light is red, unplug the cable, wait 10 seconds and reconnect."),
          para("For daily use, set the charge limit to 80% to reduce battery wear, and charge to 100% only before long "
               "trips."),
          *figure("charging", "Figure 4.1: CCS Combo 2 charging inlet with the AC connector, DC pins and status light."),
          PageBreak()]
    # Page 5 - fuses
    s += [Paragraph("Chapter 5: Fuses and 12V electrical system", h1),
          para("The interior fuse box is located behind the trim panel in the front passenger footwell, below the "
               "glovebox. Switch the vehicle off before replacing a fuse, and replace a fuse only with one of the same "
               "rating."),
          para("Fuse F12 (15 A) protects the rear window defroster. Fuse F7 (10 A) protects the infotainment display, "
               "and fuse F6 (30 A) protects the electric seat heaters. Fuse F15 (30 A) supplies the 12V power outlet."),
          para("The 12V battery is located in the front compartment under the hood, and powers the computers, lights "
               "and door locks."),
          *figure("fuses", "Figure 5.1: Interior fuse box layout with fuse positions F1 to F16 and ratings."),
          PageBreak()]
    # Page 6 - emergencies
    s += [Paragraph("Chapter 6: Emergencies and towing", h1),
          para("If the vehicle will not power on, the 12V battery may be discharged. Connect a booster or another "
               "vehicle's 12V battery to the jump-start terminals in the front compartment: red to positive first, then "
               "black to the marked ground point. Never connect jump leads to the high-voltage battery."),
          para("The Aurora must be transported on a flatbed truck. If a flatbed is not available, it may be towed with "
               "the wheels on the ground in Tow Mode for a maximum of 10 km at no more than 40 km/h."),
          para("In a collision that deploys the airbags, the high-voltage system is disconnected automatically. Have "
               "the vehicle inspected by an authorized Northwind service center before driving it again.")]
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=56, rightMargin=56, topMargin=50, bottomMargin=50,
                            title="Northwind Aurora Owner's Manual", author="DriveDocs sample content")
    doc.build(s)
    return pdf_path


MAINTENANCE_MD = """# Northwind Aurora Maintenance Schedule

Synthetic sample content for DriveDocs. Follow whichever interval comes first: time or distance.

## Service intervals

| Item | Interval |
| --- | --- |
| Cabin air filter | Replace every 12 months or 20,000 km |
| Tire rotation | Every 10,000 km |
| Brake fluid | Test for water content every 2 years; replace if above 3% |
| Battery coolant | Replace every 5 years or 150,000 km |
| Wiper blades | Replace every 12 months |
| 12V battery | Test at every annual inspection |

## Brakes

The Aurora uses regenerative braking, so the brake pads wear slowly. Inspect the pads and discs at every annual
service, and clean and lubricate the brake calipers every 12 months in regions where roads are salted in winter.

## Fluids and capacities

- Brake fluid: DOT 4, approximately 0.9 litres.
- Battery and motor coolant: Northwind G-48 blue coolant, approximately 9.5 litres. Only a service center may refill it.
- Washer fluid: 4.5 litres; use a winter mixture below 0 °C.

## Warranty

The high-voltage battery is covered for 8 years or 160,000 km, whichever comes first, against capacity falling below
70%. The rest of the vehicle is covered by a 4-year or 100,000 km limited warranty.
"""

TROUBLESHOOTING_TXT = """NORTHWIND AURORA TROUBLESHOOTING GUIDE (synthetic sample content)

SYMPTOM: The vehicle does not power on and the touchscreen stays dark.
CAUSE: The 12V battery is discharged.
ACTION: Jump-start the 12V system using the terminals in the front compartment, then drive for at least 30 minutes so the 12V battery recharges.

SYMPTOM: Charging always stops at 80%.
CAUSE: The charge limit is set to 80%, which is the recommended setting for daily driving.
ACTION: Open Charging > Charge Limit on the touchscreen and raise the limit before a long trip.

SYMPTOM: The tire pressure warning light stays on after the tires were rotated or inflated.
CAUSE: The tire pressure monitoring system still uses the old reference values.
ACTION: With the vehicle stopped, press and hold the TPMS reset button below the steering column for 3 seconds until the light flashes twice, then drive above 25 km/h for 10 minutes.

SYMPTOM: The key fob is not detected ("Key not found").
CAUSE: The fob battery is weak or there is radio interference.
ACTION: Place the fob on the marked backup reader in the centre console cup holder and press the brake pedal to start. Replace the CR2032 fob battery.

SYMPTOM: The infotainment screen is frozen.
CAUSE: The infotainment computer stopped responding.
ACTION: Press and hold both steering-wheel scroll buttons for 10 seconds until the screen restarts. Driving functions are not affected.

SYMPTOM: Squealing noise when braking after the car has been parked for several days.
CAUSE: Light surface rust on the brake discs, common because regenerative braking uses the friction brakes less.
ACTION: Brake firmly a few times at low speed in a safe place. If the noise continues, book a brake inspection.
"""

SERVICE_RECORD_TXT = """NORTHWIND SERVICE CENTER - SERVICE RECORD (synthetic sample; all personal data is fictitious)

Record number: SR-2026-0457
Date: 2026-03-14
Customer: Jane Example
Email: jane.example@example.com
Phone: (555) 010-2345
Vehicle: Northwind Aurora 2026, VIN 1HGBH41JXMN109186
Odometer: 40,250 km
Payment: card 4111 1111 1111 1111

Work performed: Brake fluid replaced (DOT 4) after the 2-year test showed 3.4% water content. Cabin air filter replaced. Tires rotated and TPMS reset. 12V battery tested: good.
Technician notes: Rear brake pads at 6 mm, front at 7 mm. Next service due at 60,000 km or March 2027.
"""


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR = Path(tempfile.mkdtemp(prefix="drivedocs_diagrams_"))
    diagrams = {
        "warning": warning_lights_chart(ASSET_DIR / "warning_lights.png"),
        "tires": tire_pressure_placard(ASSET_DIR / "tire_placard.png"),
        "charging": charging_port_diagram(ASSET_DIR / "charging_inlet.png"),
        "fuses": fuse_box_layout(ASSET_DIR / "fuse_box.png"),
    }
    owners_manual(SAMPLE_DIR / "aurora_owners_manual.pdf", diagrams)
    front_compartment_diagram(SAMPLE_DIR / "front_compartment_diagram.png")
    (SAMPLE_DIR / "maintenance_schedule.md").write_text(MAINTENANCE_MD.strip() + "\n", encoding="utf-8")
    (SAMPLE_DIR / "troubleshooting_guide.txt").write_text(TROUBLESHOOTING_TXT.strip() + "\n", encoding="utf-8")
    (SAMPLE_DIR / "service_record_2026-03.txt").write_text(SERVICE_RECORD_TXT.strip() + "\n", encoding="utf-8")
    for p in sorted(SAMPLE_DIR.iterdir()):
        if p.is_file():
            print(f" [+] {p.name} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
