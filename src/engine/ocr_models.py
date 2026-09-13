"""Resolve the OCR model assets shipped with frozen application builds."""

import sys
from pathlib import Path

DETECTION_MODEL_NAME = "PP-OCRv6_medium_det"
RECOGNITION_MODEL_NAME = "PP-OCRv6_medium_rec"
OCR_MODEL_NAMES = (DETECTION_MODEL_NAME, RECOGNITION_MODEL_NAME)
REQUIRED_MODEL_FILES = (
    "inference.yml",
    "inference.json",
    "inference.pdiparams",
)


class BundledOCRModelError(RuntimeError):
    """Raised when a frozen release contains incomplete OCR model assets."""


def get_bundled_ocr_model_options(
    bundle_root: Path | str | None = None,
) -> dict[str, str]:
    """Return PaddleOCR options for complete model directories in the bundle."""
    if bundle_root is None:
        bundle_root = getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2])

    model_root = Path(bundle_root) / "ocr_models"
    model_dirs = {
        model_name: model_root / model_name
        for model_name in OCR_MODEL_NAMES
    }
    missing = [
        str(model_dir / filename)
        for model_dir in model_dirs.values()
        for filename in REQUIRED_MODEL_FILES
        if not (model_dir / filename).is_file()
    ]
    if missing:
        raise BundledOCRModelError(
            "The installed OCR model bundle is incomplete. Reinstall "
            "TeyvatTranslator. Missing: " + ", ".join(missing)
        )

    return {
        "text_detection_model_name": DETECTION_MODEL_NAME,
        "text_detection_model_dir": str(model_dirs[DETECTION_MODEL_NAME]),
        "text_recognition_model_name": RECOGNITION_MODEL_NAME,
        "text_recognition_model_dir": str(model_dirs[RECOGNITION_MODEL_NAME]),
    }
