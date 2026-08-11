"""
anonymize_burned_text.py
─────────────────────────────────────────────────────────────────────────────
Detect and remove burned-in PHI (patient name, DOB, MRN, dates) from the
pixel data of DICOM chest X-ray files.

Usage
─────
  # Single file:
  python anonymize_burned_text.py --input ./dataset/patient_001/image.dcm

  # Single file with custom output path:
  python anonymize_burned_text.py --input ./raw/image.dcm --output ./safe/image.dcm

  # Entire dataset directory (each subdirectory must contain image.dcm):
  python anonymize_burned_text.py --input ./raw_dataset --output ./safe_dataset

  # Preview only — show what would be redacted WITHOUT saving:
  python anonymize_burned_text.py --input ./raw_dataset --preview-only

  # Save side-by-side preview images for manual verification:
  python anonymize_burned_text.py --input ./raw_dataset --output ./safe_dataset --save-previews

  # Custom redaction margins (percent of image dimensions):
  python anonymize_burned_text.py --input ./raw_dataset --output ./safe_dataset --margin-h 10 --margin-w 45

  # Overwrite original files in-place (no separate output directory):
  python anonymize_burned_text.py --input ./dataset --inplace

Options
───────
  --input          Path to a single .dcm file OR a directory of patient subdirs
  --output         Where to save fixed files (default: <input>_anonymized)
  --inplace        Overwrite each source file directly (mutually exclusive with --output)
  --preview-only   Show matplotlib previews without saving any files
  --save-previews  Save before/after PNG previews to <output>/previews/
  --margin-h       Top/bottom redaction height as % of image height (default: 7)
  --margin-w       Left/right redaction width  as % of image width  (default: 40)
  --no-confirm     Skip interactive confirmation prompt
"""

import argparse
import sys
import shutil
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image


# ── ANSI colours for terminal output ──────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# DICOM helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_burned_in_tag(ds) -> str:
    """Return value of (0028,0301) BurnedInAnnotation tag, or 'UNKNOWN'."""
    tag = ds.get((0x0028, 0x0301), None)
    if tag is None:
        return "UNKNOWN"
    return str(tag.value).strip()


def pixels_to_display(pixels: np.ndarray, ds) -> np.ndarray:
    """Convert raw DICOM pixel array to a uint8 displayable image."""
    arr = pixels.astype(np.float32)

    # Apply rescale slope/intercept
    slope     = float(getattr(ds, "RescaleSlope",     1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    # Apply VOI LUT or window
    try:
        arr = apply_voi_lut(arr.astype(np.int16), ds)
    except Exception:
        wc = float(getattr(ds, "WindowCenter", arr.mean()))
        ww = float(getattr(ds, "WindowWidth",  arr.max() - arr.min() + 1))
        if isinstance(wc, list):
            wc = wc[0]
        if isinstance(ww, list):
            ww = ww[0]
        lo = wc - ww / 2
        hi = wc + ww / 2
        arr = np.clip(arr, lo, hi)

    # Normalise to 0-255
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    else:
        arr = np.zeros_like(arr)

    # MONOCHROME1 → invert
    pi = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip()
    if pi == "MONOCHROME1":
        arr = 255.0 - arr

    return arr.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Detection & redaction
# ─────────────────────────────────────────────────────────────────────────────

def detect_text_brightness(
    pixels: np.ndarray,
    margin_h: int,
    margin_w: int,
    threshold: float = 40.0,
) -> tuple[bool, dict]:
    """
    Check whether any of the four corner regions contain bright pixels that
    could be burned-in text annotations.

    Returns (suspicious: bool, corner_means: dict).
    """
    h, w = pixels.shape[:2]
    mh = min(margin_h, h // 2)
    mw = min(margin_w, w // 2)

    corners = {
        "top_left":     pixels[:mh,    :mw],
        "top_right":    pixels[:mh,    w - mw:],
        "bottom_left":  pixels[h - mh:, :mw],
        "bottom_right": pixels[h - mh:, w - mw:],
    }

    means = {}
    for name, region in corners.items():
        display = pixels_to_display_region(region)
        means[name] = float(display.mean()) if display.size > 0 else 0.0

    suspicious = any(v > threshold for v in means.values())
    return suspicious, means


def pixels_to_display_region(region: np.ndarray) -> np.ndarray:
    """Normalise a raw pixel region to 0-255 for brightness comparison."""
    arr = region.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    return arr


def redact_corners(pixels: np.ndarray, margin_h: int, margin_w: int) -> np.ndarray:
    """Black out the four corner regions of the pixel array."""
    result = pixels.copy()
    h, w   = result.shape[:2]
    mh     = min(margin_h, h // 2)
    mw     = min(margin_w, w // 2)

    result[:mh,     :mw]       = 0   # top-left
    result[:mh,     w - mw:]   = 0   # top-right
    result[h - mh:, :mw]       = 0   # bottom-left
    result[h - mh:, w - mw:]   = 0   # bottom-right
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────────────────────────────────────

def process_dicom(
    input_path: Path,
    output_path: Path,
    margin_h_pct: float = 7.0,
    margin_w_pct: float = 40.0,
    preview_only: bool  = False,
    save_previews: bool = False,
    preview_dir: Path   = None,
) -> dict:
    """
    Load one DICOM file, redact its corner regions, and save the result.
    Returns a result dict with keys: file, status, tag, suspicious, corners, message.
    """
    result = {
        "file":       str(input_path),
        "status":     "error",
        "tag":        "UNKNOWN",
        "suspicious": False,
        "corners":    {},
        "message":    "",
    }

    try:
        ds = pydicom.dcmread(str(input_path))
    except Exception as exc:
        result["message"] = f"Cannot read DICOM: {exc}"
        return result

    result["tag"] = check_burned_in_tag(ds)

    try:
        pixels_orig = ds.pixel_array
    except Exception as exc:
        result["message"] = f"Cannot decode pixel data: {exc}"
        return result

    h, w = pixels_orig.shape[:2]
    margin_h = max(1, int(round(h * margin_h_pct / 100)))
    margin_w = max(1, int(round(w * margin_w_pct / 100)))

    suspicious, corner_means = detect_text_brightness(pixels_orig, margin_h, margin_w)
    result["suspicious"] = suspicious
    result["corners"]    = corner_means

    pixels_fixed = redact_corners(pixels_orig, margin_h, margin_w)

    # ── Previews ──────────────────────────────────────────────────────────
    if preview_only:
        _show_preview(ds, pixels_orig, pixels_fixed, input_path, margin_h, margin_w)
        result["status"]  = "preview_only"
        result["message"] = "Preview shown — file not saved."
        return result

    if save_previews and preview_dir:
        _save_preview(ds, pixels_orig, pixels_fixed, input_path, preview_dir, margin_h, margin_w)

    # ── Save modified DICOM ───────────────────────────────────────────────
    try:
        ds.PixelData = pixels_fixed.tobytes()
        # Update tag to indicate annotations have been removed
        ds[(0x0028, 0x0301)].value = "NO"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(output_path))
    except Exception as exc:
        result["message"] = f"Failed to save: {exc}"
        return result

    result["status"]  = "fixed"
    result["message"] = (
        f"Redacted corners ({margin_h}px × {margin_w}px). "
        f"Tag was '{result['tag']}'. "
        f"Suspicious={'Yes' if suspicious else 'No'}"
    )
    return result


def _normalise_for_display(ds, pixels):
    """Return a uint8 array ready for PIL."""
    arr = pixels_to_display(pixels, ds)
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)   # grayscale → RGB
    return arr


def _save_preview(ds, pixels_orig, pixels_fixed, input_path, preview_dir, margin_h, margin_w):
    """Save a side-by-side PNG comparison to preview_dir."""
    try:
        from PIL import ImageDraw
        orig_arr  = _normalise_for_display(ds, pixels_orig)
        fixed_arr = _normalise_for_display(ds, pixels_fixed)

        orig_img  = Image.fromarray(orig_arr).resize((512, 512), Image.LANCZOS)
        fixed_img = Image.fromarray(fixed_arr).resize((512, 512), Image.LANCZOS)

        combined = Image.new("RGB", (1024 + 10, 512 + 30), (30, 30, 30))
        combined.paste(orig_img,  (0,   30))
        combined.paste(fixed_img, (514, 30))

        draw = ImageDraw.Draw(combined)
        draw.text((180, 5), "BEFORE — PHI may be visible", fill=(255, 80, 80))
        draw.text((680, 5), "AFTER  — corners redacted",   fill=(80, 255, 80))

        preview_dir.mkdir(parents=True, exist_ok=True)
        out = preview_dir / f"{input_path.parent.name}_preview.png"
        combined.save(str(out))
    except Exception as exc:
        print(f"  {YELLOW}⚠  Could not save preview: {exc}{RESET}")


def _show_preview(ds, pixels_orig, pixels_fixed, input_path, margin_h, margin_w):
    """Display an interactive matplotlib preview."""
    try:
        import matplotlib.pyplot as plt
        orig_arr  = _normalise_for_display(ds, pixels_orig)
        fixed_arr = _normalise_for_display(ds, pixels_fixed)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(str(input_path), fontsize=9, color="gray")
        axes[0].imshow(orig_arr,  cmap="gray"); axes[0].set_title("BEFORE — PHI may be visible", color="red")
        axes[1].imshow(fixed_arr, cmap="gray"); axes[1].set_title("AFTER  — corners redacted",   color="green")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"  {YELLOW}⚠  Cannot show preview: {exc}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Directory discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_dicom_files(input_path: Path) -> list[Path]:
    """
    Return a list of DICOM files to process.

    - If input_path is a .dcm file → return [input_path]
    - If input_path is a directory → find all *.dcm / *.DCM files recursively
    """
    if input_path.is_file():
        if input_path.suffix.lower() not in (".dcm", ""):
            print(f"{YELLOW}Warning: file does not have .dcm extension — processing anyway.{RESET}")
        return [input_path]

    if input_path.is_dir():
        dcm_files = sorted(
            list(input_path.rglob("*.dcm")) +
            list(input_path.rglob("*.DCM"))
        )
        if not dcm_files:
            print(f"{RED}No .dcm files found under {input_path}{RESET}")
            sys.exit(1)
        return dcm_files

    print(f"{RED}Input path does not exist: {input_path}{RESET}")
    sys.exit(1)


def resolve_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    """
    Mirror the input directory structure under output_root.

    Example:
      input_root  = ./raw_dataset
      input_file  = ./raw_dataset/patient_001/image.dcm
      output_root = ./safe_dataset
      → output    = ./safe_dataset/patient_001/image.dcm
    """
    rel = input_file.relative_to(input_root) if input_root.is_dir() else input_file.name
    return output_root / rel


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    total      = len(results)
    fixed      = sum(1 for r in results if r["status"] == "fixed")
    previews   = sum(1 for r in results if r["status"] == "preview_only")
    errors     = sum(1 for r in results if r["status"] == "error")
    suspicious = sum(1 for r in results if r.get("suspicious", False))

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}Summary{RESET}")
    print(f"{'─'*60}")
    print(f"  Total files processed : {total}")
    print(f"  Fixed & saved         : {GREEN}{fixed}{RESET}")
    print(f"  Previewed only        : {CYAN}{previews}{RESET}")
    print(f"  Errors                : {RED}{errors}{RESET}")
    print(f"  Suspicious corners    : {YELLOW}{suspicious}{RESET}  (high brightness detected in corners)")

    if errors:
        print(f"\n{RED}Files with errors:{RESET}")
        for r in results:
            if r["status"] == "error":
                print(f"  ✗  {r['file']}")
                print(f"     {r['message']}")

    print(f"{'─'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Remove burned-in PHI from DICOM pixel data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",         required=True,         help="Single .dcm file or dataset directory")
    parser.add_argument("--output",        default=None,          help="Output file or directory (default: <input>_anonymized)")
    parser.add_argument("--inplace",       action="store_true",   help="Overwrite source files directly")
    parser.add_argument("--preview-only",  action="store_true",   help="Show previews without saving")
    parser.add_argument("--save-previews", action="store_true",   help="Save before/after PNG previews")
    parser.add_argument("--margin-h",      type=float, default=7, help="Top/bottom redaction height %% (default: 7)")
    parser.add_argument("--margin-w",      type=float, default=40,help="Left/right redaction width  %% (default: 40)")
    parser.add_argument("--no-confirm",    action="store_true",   help="Skip confirmation prompt")
    args = parser.parse_args()

    if args.inplace and args.output:
        print(f"{RED}Error: --inplace and --output are mutually exclusive.{RESET}")
        sys.exit(1)

    input_path = Path(args.input).resolve()

    # ── Resolve output root ───────────────────────────────────────────────
    if args.inplace:
        output_root = input_path          # signals in-place mode
    elif args.output:
        output_root = Path(args.output).resolve()
    else:
        output_root = input_path.parent / (input_path.name + "_anonymized")

    preview_dir = (output_root / "previews") if args.save_previews and not args.inplace else None

    # ── Discover files ────────────────────────────────────────────────────
    dcm_files = discover_dicom_files(input_path)

    print(f"\n{BOLD}Burned-In PHI Redaction Tool{RESET}")
    print(f"{'─'*60}")
    print(f"  Input          : {input_path}")
    print(f"  Output         : {'[in-place]' if args.inplace else output_root}")
    print(f"  Files found    : {len(dcm_files)}")
    print(f"  Corner margins : {args.margin_h}% height × {args.margin_w}% width")
    print(f"  Preview only   : {args.preview_only}")
    print(f"  Save previews  : {args.save_previews}")
    print(f"{'─'*60}")

    if not args.preview_only and not args.no_confirm:
        dest_label = "overwritten in-place" if args.inplace else f"written to {output_root}"
        answer = input(f"\n{YELLOW}Proceed? Files will be {dest_label}  [y/N]: {RESET}").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    # ── Process each file ─────────────────────────────────────────────────
    results = []
    for i, dcm_path in enumerate(dcm_files, 1):
        out_path = dcm_path if args.inplace else resolve_output_path(dcm_path, input_path, output_root)

        print(f"[{i:>4}/{len(dcm_files)}] {dcm_path.parent.name}/{dcm_path.name} ", end="", flush=True)

        result = process_dicom(
            input_path    = dcm_path,
            output_path   = out_path,
            margin_h_pct  = args.margin_h,
            margin_w_pct  = args.margin_w,
            preview_only  = args.preview_only,
            save_previews = args.save_previews,
            preview_dir   = preview_dir,
        )
        results.append(result)

        # ── Per-file status line ──────────────────────────────────────────
        tag_display = f"tag={result['tag']}"
        if result["status"] == "fixed":
            suspicious_flag = f" {YELLOW}[SUSPICIOUS CORNERS]{RESET}" if result.get("suspicious") else ""
            print(f"{GREEN}✓  Fixed{RESET}  {tag_display}{suspicious_flag}")
        elif result["status"] == "preview_only":
            print(f"{CYAN}👁  Preview shown{RESET}  {tag_display}")
        elif result["status"] == "error":
            print(f"{RED}✗  Error: {result['message']}{RESET}")
        else:
            print(f"  {result['message']}")

        # Print per-corner brightness for suspicious files
        if result.get("suspicious") and result["corners"]:
            c = result["corners"]
            print(f"       Brightness — TL:{c['top_left']:.0f}  TR:{c['top_right']:.0f}  "
                  f"BL:{c['bottom_left']:.0f}  BR:{c['bottom_right']:.0f}  (0–255 scale)")

    print_summary(results)

    if not args.preview_only:
        if args.inplace:
            print(f"{GREEN}✅  Files updated in-place.{RESET}")
        else:
            print(f"{GREEN}✅  Safe dataset written to:{RESET}  {output_root}")
        if args.save_previews and not args.inplace:
            print(f"{GREEN}✅  Previews saved to:{RESET}        {preview_dir}")
        print(f"\n{YELLOW}⚠  Always visually inspect a sample of output files{RESET}")
        print(f"   before uploading to any cloud service.\n")


if __name__ == "__main__":
    main()
