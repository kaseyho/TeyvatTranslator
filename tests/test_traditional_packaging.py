import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import build

ROOT = Path(__file__).resolve().parents[1]


class TraditionalPackagingTests(unittest.TestCase):
    def test_pyinstaller_analysis_includes_the_staged_ocr_model_directories(self):
        captured = {}

        def fake_analysis(*args, **kwargs):
            captured["datas"] = kwargs["datas"]
            return types.SimpleNamespace(
                pure=[],
                zipped_data=[],
                scripts=[],
                binaries=[],
                zipfiles=[],
                datas=kwargs["datas"],
            )

        pyinstaller = types.ModuleType("PyInstaller")
        pyinstaller_utils = types.ModuleType("PyInstaller.utils")
        pyinstaller_hooks = types.ModuleType("PyInstaller.utils.hooks")
        pyinstaller_hooks.collect_data_files = lambda package: []
        pyinstaller_hooks.collect_submodules = lambda package: []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            for model_name in (
                "PP-OCRv6_medium_det",
                "PP-OCRv6_medium_rec",
            ):
                model_dir = temp_root / "build" / "ocr_models" / model_name
                model_dir.mkdir(parents=True)
                for filename in (
                    "inference.yml",
                    "inference.json",
                    "inference.pdiparams",
                ):
                    (model_dir / filename).write_text("fixture", encoding="utf-8")

            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_root)
                with patch.dict(
                    sys.modules,
                    {
                        "PyInstaller": pyinstaller,
                        "PyInstaller.utils": pyinstaller_utils,
                        "PyInstaller.utils.hooks": pyinstaller_hooks,
                    },
                ):
                    runpy.run_path(
                        str(ROOT / "TeyvatTranslator.spec"),
                        init_globals={
                            "Analysis": fake_analysis,
                            "PYZ": lambda *args, **kwargs: object(),
                            "EXE": lambda *args, **kwargs: object(),
                            "COLLECT": lambda *args, **kwargs: object(),
                        },
                    )
            finally:
                os.chdir(previous_cwd)

            bundled_destinations = {
                destination
                for source, destination in captured["datas"]
                if Path(source).parent.name == "ocr_models"
            }
            self.assertEqual(
                bundled_destinations,
                {
                    "ocr_models/PP-OCRv6_medium_det",
                    "ocr_models/PP-OCRv6_medium_rec",
                },
            )

    def test_build_stages_complete_ocr_models_before_running_pyinstaller(self):
        model_names = (
            "PP-OCRv6_medium_det",
            "PP-OCRv6_medium_rec",
        )
        required_files = (
            "inference.yml",
            "inference.json",
            "inference.pdiparams",
        )
        paddle_ocr_calls = []

        def fake_paddle_ocr(**options):
            paddle_ocr_calls.append(options)
            cache_root = Path(os.environ["PADDLE_PDX_CACHE_HOME"])
            for model_name in model_names:
                model_dir = cache_root / "official_models" / model_name
                if model_dir.exists():
                    continue
                model_dir.mkdir(parents=True)
                for filename in required_files:
                    (model_dir / filename).write_text("fixture", encoding="utf-8")
            return object()

        fake_paddleocr = types.ModuleType("paddleocr")
        fake_paddleocr.PaddleOCR = fake_paddle_ocr

        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "build"
            incomplete_model_dir = (
                build_dir
                / "source-model-cache"
                / "official_models"
                / "PP-OCRv6_medium_rec"
            )
            incomplete_model_dir.mkdir(parents=True)
            (incomplete_model_dir / "inference.json").write_text(
                "partial download",
                encoding="utf-8",
            )
            with (
                patch.object(build, "BUILD_DIR", build_dir),
                patch.object(build, "run_command", return_value=True) as run_command,
                patch.dict(sys.modules, {"paddleocr": fake_paddleocr}),
                patch.dict(
                    os.environ,
                    {
                        "PADDLE_PDX_CACHE_HOME": str(
                            build_dir / "source-model-cache"
                        )
                    },
                ),
            ):
                result = build.build_executable()

            self.assertTrue(result)
            self.assertTrue(
                paddle_ocr_calls,
                "The release build did not fetch the OCR models",
            )
            for model_name in model_names:
                for filename in required_files:
                    with self.subTest(model=model_name, filename=filename):
                        self.assertTrue(
                            (
                                build_dir
                                / "ocr_models"
                                / model_name
                                / filename
                            ).is_file()
                        )
            run_command.assert_called_once()

    def test_build_stops_before_pyinstaller_when_downloaded_model_is_incomplete(self):
        fake_paddleocr = types.ModuleType("paddleocr")

        def fake_paddle_ocr(**_options):
            cache_root = Path(os.environ["PADDLE_PDX_CACHE_HOME"])
            for model_name in (
                "PP-OCRv6_medium_det",
                "PP-OCRv6_medium_rec",
            ):
                model_dir = cache_root / "official_models" / model_name
                model_dir.mkdir(parents=True)
                (model_dir / "inference.json").write_text(
                    "incomplete fixture",
                    encoding="utf-8",
                )
            return object()

        fake_paddleocr.PaddleOCR = fake_paddle_ocr

        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "build"
            with (
                patch.object(build, "BUILD_DIR", build_dir),
                patch.object(build, "run_command", return_value=True) as run_command,
                patch.dict(sys.modules, {"paddleocr": fake_paddleocr}),
                patch.dict(
                    os.environ,
                    {
                        "PADDLE_PDX_CACHE_HOME": str(
                            build_dir / "source-model-cache"
                        )
                    },
                ),
            ):
                result = build.build_executable()

            self.assertFalse(result)
            run_command.assert_not_called()

    def test_frozen_build_collects_opencc_conversion_data(self):
        spec_text = (ROOT / "TeyvatTranslator.spec").read_text(encoding="utf-8")

        self.assertIn("collect_data_files('opencc')", spec_text)
        self.assertIn("'opencc'", spec_text)

    def test_declared_ocr_versions_match_the_verified_shared_profile(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("paddlepaddle==3.3.1", requirements)
        self.assertIn("paddleocr==3.7.0", requirements)
        self.assertIn("paddlex==3.7.2", requirements)

    def test_windows_only_capture_dependency_has_a_platform_marker(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn('pywin32>=306; sys_platform == "win32"', requirements)


if __name__ == "__main__":
    unittest.main()
