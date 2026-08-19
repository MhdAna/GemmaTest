import sys
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT      = Path(__file__).parent.parent
OPENI_DIR = ROOT / "openi"

parser = argparse.ArgumentParser()
parser.add_argument("--reports", type=Path, default=OPENI_DIR / "ecgen-radiology",
                    help="Folder containing OpenI XML report files")
parser.add_argument("--images", type=Path, default=OPENI_DIR / "NLMCXR_png",
                    help="Folder containing OpenI PNG images")
parser.add_argument("--holdout-n", type=int, default=20,
                    help="Number of cases to keep as final test holdout")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for sampling/splitting")
args = parser.parse_args()

REPORT_DIR = args.reports
IMAGE_DIR  = args.images
SUBSET_CSV = ROOT / "smoke_subset.csv"
HOLDOUT_CSV = ROOT / "smoke_test_holdout.csv"
TARGET_N   = 800  # set to 50 for a quick 5-min sanity check

KEYWORDS = [
    "broncho-vascular", "bronchovascular",
    "bronchitis", "peribronchial",
    "bronchial wall thickening", "bilateral hilar",
    "increased markings", "interstitial markings",
    "airway disease",
]

if not REPORT_DIR.exists():
    raise FileNotFoundError(
        f"XML reports folder not found: {REPORT_DIR}\n"
        "Pass a custom path with:  --reports /path/to/xml/folder\n"
        "Or download the default dataset:\n"
        "  curl -O https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz\n"
        "  curl -O https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz\n"
        "  tar -xzf NLMCXR_png.tgz -C openi/\n"
        "  tar -xzf NLMCXR_reports.tgz -C openi/"
    )

xml_files = sorted(REPORT_DIR.glob("*.xml"))
print(f"Found {len(xml_files)} XML files in {REPORT_DIR}")

rows = []
skipped_no_image = 0
for xml_file in xml_files:
    try:
        root = ET.parse(xml_file).getroot()

        findings, impression = "", ""
        for node in root.iter("AbstractText"):
            label = node.get("Label", "").upper()
            text  = (node.text or "").strip()
            if label == "FINDINGS":
                findings = text
            elif label == "IMPRESSION":
                impression = text

        # image ids come directly from <parentImage id="..."> attributes
        image_ids = [el.get("id") for el in root.findall("parentImage") if el.get("id")]
        if not image_ids:
            continue
        frontal_ids = [i for i in image_ids if i.endswith("1001")]
        lateral_ids = [i for i in image_ids if i.endswith("2001")]
        chosen_frontal = frontal_ids[0] if frontal_ids else image_ids[0]
        png_frontal = IMAGE_DIR / f"{chosen_frontal}.png"
        if not png_frontal.exists():
            skipped_no_image += 1
            continue

        row = {
            "path":               str(png_frontal.relative_to(IMAGE_DIR)),
            "section_findings":   findings,
            "section_impression": impression,
        }
        # add lateral path when available
        if lateral_ids:
            png_lateral = IMAGE_DIR / f"{lateral_ids[0]}.png"
            if png_lateral.exists():
                row["path_lateral"] = str(png_lateral.relative_to(IMAGE_DIR))
        rows.append(row)
    except Exception as e:
        print(f"  WARNING {xml_file.name}: {e}")
        continue

if skipped_no_image:
    print(f"Skipped {skipped_no_image} cases — PNG not found under {IMAGE_DIR}")
    if skipped_no_image == len(xml_files):
        raise FileNotFoundError(
            f"No PNGs matched any XML id under {IMAGE_DIR}\n"
            "Check that --images points to the folder containing CXR*.png files."
        )

df = pd.DataFrame(rows)
print(f"Parsed {len(df):,} reports")

text = df["section_findings"].fillna("") + " " + df["section_impression"].fillna("")
mask = text.str.contains("|".join(KEYWORDS), case=False, na=False)
matched = df[mask].copy()
print(f"Keyword matches : {len(matched):,} / {len(df):,}")

if len(matched) == 0:
    print("No keyword matches — using full dataset sample for smoke test")
    matched = df.sample(min(TARGET_N, len(df)), random_state=args.seed)
elif len(matched) > TARGET_N:
    matched = matched.sample(n=TARGET_N, random_state=args.seed)

holdout_n = max(0, args.holdout_n)
if holdout_n > 0 and len(matched) > holdout_n:
    holdout = matched.sample(n=holdout_n, random_state=args.seed)
    train_df = matched.drop(index=holdout.index).reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)
    holdout.to_csv(HOLDOUT_CSV, index=False)
    print(f"Saved holdout set: {len(holdout)} rows -> {HOLDOUT_CSV}")
else:
    train_df = matched.reset_index(drop=True)
    if holdout_n > 0:
        print(
            "WARNING: Not enough rows to create holdout split "
            f"(requested {holdout_n}, available {len(matched)})."
        )
        print("Holdout CSV not written; training CSV contains all rows.")

train_df.to_csv(SUBSET_CSV, index=False)
print(f"Saved training set: {len(train_df)} rows -> {SUBSET_CSV}")
