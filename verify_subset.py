import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dicom_utils import load_dicom, dicom_to_pil

ROOT         = Path(__file__).parent.parent
CHEXPERT_DIR = ROOT / "chexpert_plus"
SUBSET_CSV   = ROOT / "smoke_subset.csv"

df = pd.read_csv(SUBSET_CSV)
ok, fail = 0, 0

# CheXpert Plus 'path' is typically relative: files/p00000/s00000/xxx.dcm
for _, row in df.head(20).iterrows():
    dcm_path = CHEXPERT_DIR / row["path"]
    try:
        img = dicom_to_pil(load_dicom(dcm_path))
        print(f"  ✓  {row['path']}  {img.size}")
        ok += 1
    except Exception as exc:
        print(f"  ✗  {row['path']}  {exc}")
        fail += 1

print(f"\nAccessible: {ok}/20  |  Failed: {fail}/20")
if fail > 0:
    print("Adjust CHEXPERT_DIR / row['path'] to match your download layout.")
