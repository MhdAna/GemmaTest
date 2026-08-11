"""
dicom_utils.py – Utilities for loading and converting DICOM files
to PIL Images suitable for Gemma 4 vision input.
"""

from pathlib import Path
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except ImportError:
    raise ImportError("Install pydicom:  pip install pydicom")


def load_dicom(path: str | Path) -> pydicom.Dataset:
    """Load a DICOM file and return the dataset."""
    return pydicom.dcmread(str(path))


def dicom_to_pil(
    ds: pydicom.Dataset,
    window_center: float | None = None,
    window_width: float | None = None,
) -> Image.Image:
    """
    Convert a DICOM dataset to an 8-bit RGB PIL Image.

    Window centre / width are applied when provided; otherwise the
    VOI LUT embedded in the dataset is used (if present).  Falls back
    to a simple min-max normalisation when neither is available.

    Parameters
    ----------
    ds:             pydicom Dataset returned by load_dicom()
    window_center:  override WC (Hounsfield units or raw value)
    window_width:   override WW
    """
    pixel_array = ds.pixel_array.astype(np.float64)

    # Apply rescale slope / intercept if present (common in CT)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept

    # --- Windowing ---
    if window_center is not None and window_width is not None:
        lo = window_center - window_width / 2
        hi = window_center + window_width / 2
        pixel_array = np.clip(pixel_array, lo, hi)
    elif hasattr(ds, "WindowCenter"):
        # Use dataset VOI LUT
        pixel_array = apply_voi_lut(pixel_array, ds)
    else:
        # Min-max fallback
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255
        else:
            pixel_array = np.zeros_like(pixel_array)

    # Normalise to 0-255 uint8
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > pmin:
        pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
    img_8bit = pixel_array.clip(0, 255).astype(np.uint8)

    # Handle PhotometricInterpretation (invert MONOCHROME1)
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        img_8bit = 255 - img_8bit

    # Convert to RGB (Gemma 4 vision expects 3-channel input)
    if img_8bit.ndim == 2:
        pil_img = Image.fromarray(img_8bit, mode="L").convert("RGB")
    elif img_8bit.ndim == 3:
        # Some DICOMs already store RGB
        pil_img = Image.fromarray(img_8bit).convert("RGB")
    else:
        raise ValueError(f"Unexpected pixel array shape: {img_8bit.shape}")

    return pil_img


def get_dicom_metadata(ds: pydicom.Dataset) -> dict:
    """Return a small dict of clinically relevant DICOM tags."""
    tags = {
        "PatientID":         getattr(ds, "PatientID", "N/A"),
        "Modality":          getattr(ds, "Modality", "N/A"),
        "StudyDescription":  getattr(ds, "StudyDescription", "N/A"),
        "SeriesDescription": getattr(ds, "SeriesDescription", "N/A"),
        "BodyPartExamined":  getattr(ds, "BodyPartExamined", "N/A"),
        "SliceThickness":    getattr(ds, "SliceThickness", "N/A"),
        "PixelSpacing":      getattr(ds, "PixelSpacing", "N/A"),
        "Rows":              getattr(ds, "Rows", "N/A"),
        "Columns":           getattr(ds, "Columns", "N/A"),
    }
    return tags
