"""Generate synthetic demo data that mirrors the real corpus schema.

The real training corpus (data/v5a_eos_vs_noneos.csv) contains proprietary
internal FA ticket descriptions and is excluded from version control.
This script produces a small synthetic sample (200 rows) with the same
columns and class distribution so notebooks and the Streamlit app run
end-to-end on a public machine.

Usage:
    python data/generate_synthetic_demo.py
"""
import random, csv, pathlib, re

random.seed(42)

EOS_TEMPLATES = [
    "short circuit at {pin} pin, {damage} observed under {tool}",
    "high current event at {pin}, {damage} with {symptom}",
    "{damage} at {pin} junction, EOS-like failure signature",
    "component failure: {damage} on {pin}, overstress indication",
    "pin {pin} burnt due to {symptom}, dielectric breakdown",
    "catastrophic {damage} at {pin}, surge event suspected",
    "latch-up condition at {pin}, {damage} confirmed by {tool}",
    "{symptom} at {pin} caused severe {damage}",
    "overcurrent damage: {damage} at {pin}",
    "EOS event: {pin} shows {damage} after {symptom}",
]
NON_EOS_TEMPLATES = [
    "bond wire open on {pin}, mechanical stress fracture",
    "corrosion at {pin} contact, humidity ingress",
    "intermittent open at {pin}, contamination",
    "pin {pin} missing solder, cold joint observed",
    "crack in passivation layer at {pin}, {symptom}",
    "delamination at {pin}, thermal cycling damage",
    "{symptom} at {pin}, not related to electrical overstress",
    "physical damage: {pin} lead bent, handling issue",
    "open circuit at {pin}, wire bond lift-off",
    "package crack near {pin}, thermomechanical fatigue",
]
PINS     = ["VCC", "GND", "VDDC", "IO1", "IO2", "CLK", "RST", "OUT", "IN", "EN"]
DAMAGE   = ["melting", "cratering", "metallization damage", "oxide breakdown", "junction damage"]
SYMPTOM  = ["voltage spike", "current surge", "reverse bias", "transient event", "ESD pulse"]
TOOLS    = ["SEM", "TEM", "cross-section", "optical microscope", "EMMI", "nanoprobe"]

def render(t):
    return t.format(
        pin=random.choice(PINS),
        damage=random.choice(DAMAGE),
        symptom=random.choice(SYMPTOM),
        tool=random.choice(TOOLS),
    )

rows = []
for i in range(140):  # ~70% non-EOS mirrors real 74.5%
    rows.append((f"DEMO{i+1:04d}", render(random.choice(NON_EOS_TEMPLATES)), "Non-EOS"))
for i in range(60):   # ~30% EOS mirrors real 25.5%
    rows.append((f"DEMO{i+141:04d}", render(random.choice(EOS_TEMPLATES)), "EOS"))

random.shuffle(rows)

out = pathlib.Path(__file__).with_name("v5a_demo_synthetic.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Nn Notif Nr", "PSI Failure Desc", "label"])
    w.writerows(rows)

print(f"Written {len(rows)} synthetic rows → {out}")
print("EOS:", sum(1 for r in rows if r[2]=="EOS"),
      "| Non-EOS:", sum(1 for r in rows if r[2]=="Non-EOS"))
