import tempfile
import unittest
from pathlib import Path

from src.engine.ocr_models import (
    OCR_MODEL_NAMES,
    REQUIRED_MODEL_FILES,
    BundledOCRModelError,
    get_bundled_ocr_model_options,
)


class OCRModelBundleTests(unittest.TestCase):
    def test_incomplete_bundle_is_rejected_before_ocr_initialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_root = Path(temp_dir)
            for model_name in OCR_MODEL_NAMES:
                model_dir = bundle_root / "ocr_models" / model_name
                model_dir.mkdir(parents=True)
                for filename in REQUIRED_MODEL_FILES:
                    if not (
                        model_name == "PP-OCRv6_medium_rec"
                        and filename == "inference.yml"
                    ):
                        (model_dir / filename).write_text("fixture", encoding="utf-8")

            with self.assertRaisesRegex(
                BundledOCRModelError,
                r"PP-OCRv6_medium_rec[/\\]inference\.yml",
            ):
                get_bundled_ocr_model_options(bundle_root)


if __name__ == "__main__":
    unittest.main()
