"""
Build script for TeyvatTranslator
Creates a standalone Windows executable using PyInstaller

Usage:
    python build.py           # Build the executable
    python build.py --clean   # Clean build artifacts first
    python build.py --zip     # Also create a distributable ZIP
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.engine.ocr_models import (
    DETECTION_MODEL_NAME,
    OCR_MODEL_NAMES,
    RECOGNITION_MODEL_NAME,
    REQUIRED_MODEL_FILES,
)

# Build configuration
APP_NAME = "TeyvatTranslator"
VERSION = "1.5.1"
BUILD_DIR = Path("build")
DIST_DIR = Path("dist")
SPEC_FILE = Path("TeyvatTranslator.spec")


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, shell=False)
    
    if result.returncode != 0:
        print(f"\nFailed: {description}")
        return False
    
    print(f"\nCompleted: {description}")
    return True


def clean_build():
    """Remove previous build artifacts."""
    print("\n🧹 Cleaning previous build artifacts...")
    
    for path in [BUILD_DIR, DIST_DIR]:
        if path.exists():
            shutil.rmtree(path)
            print(f"  Removed: {path}")
    
    print("Clean complete")


def install_build_dependencies():
    """Install PyInstaller if not present."""
    try:
        import PyInstaller
        print("PyInstaller is installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)


def build_executable():
    """Build the executable using PyInstaller."""
    if not SPEC_FILE.exists():
        print(f"Spec file not found: {SPEC_FILE}")
        return False

    if not stage_ocr_models():
        return False
    
    cmd = [sys.executable, "-m", "PyInstaller", str(SPEC_FILE), "--noconfirm"]
    return run_command(cmd, "Building executable with PyInstaller")


def stage_ocr_models() -> bool:
    """Download, validate, and stage the OCR models for the frozen bundle."""
    configured_cache_root = os.environ.get("PADDLE_PDX_CACHE_HOME")
    cache_root = Path(
        configured_cache_root or BUILD_DIR / "paddlex-cache"
    ).expanduser().resolve()
    official_models_root = cache_root / "official_models"
    staging_root = BUILD_DIR / "ocr_models"

    for model_name in OCR_MODEL_NAMES:
        cached_model_dir = official_models_root / model_name
        if cached_model_dir.exists() and any(
            not (cached_model_dir / filename).is_file()
            for filename in REQUIRED_MODEL_FILES
        ):
            print(f"Removing incomplete cached OCR model: {cached_model_dir}")
            shutil.rmtree(cached_model_dir)

    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_root)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
    os.environ["FLAGS_use_mkldnn"] = "0"

    print("Preparing bundled PP-OCRv6 models...")
    try:
        from paddleocr import PaddleOCR

        PaddleOCR(
            text_detection_model_name=DETECTION_MODEL_NAME,
            text_recognition_model_name=RECOGNITION_MODEL_NAME,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as exc:
        print(f"Failed to prepare bundled OCR models: {type(exc).__name__}: {exc}")
        return False

    if staging_root.exists():
        shutil.rmtree(staging_root)

    for model_name in OCR_MODEL_NAMES:
        source_dir = official_models_root / model_name
        missing = [
            source_dir / filename
            for filename in REQUIRED_MODEL_FILES
            if not (source_dir / filename).is_file()
        ]
        if missing:
            print(
                "Downloaded OCR model is incomplete; missing: "
                + ", ".join(str(path) for path in missing)
            )
            return False

        destination_dir = staging_root / model_name
        shutil.copytree(source_dir, destination_dir)
        print(f"Staged OCR model: {model_name}")

    return True


def create_zip():
    """Create a distributable ZIP file."""
    dist_folder = DIST_DIR / APP_NAME
    if not dist_folder.exists():
        print(f"Distribution folder not found: {dist_folder}")
        return False
    
    zip_name = f"{APP_NAME}-v{VERSION}-windows"
    zip_path = DIST_DIR / zip_name
    
    print(f"\nCreating ZIP: {zip_name}.zip")
    shutil.make_archive(str(zip_path), 'zip', DIST_DIR, APP_NAME)
    
    final_zip = Path(f"{zip_path}.zip")
    size_mb = final_zip.stat().st_size / (1024 * 1024)
    print(f"Created: {final_zip} ({size_mb:.1f} MB)")
    
    return True


def create_installer():
    """Create Windows installer using Inno Setup."""
    iss_file = Path("TeyvatTranslator.iss")
    if not iss_file.exists():
        print(f"Inno Setup script not found: {iss_file}")
        return False
    
    # Check if Inno Setup is installed
    iscc_paths = [
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    ]
    
    iscc_exe = None
    for path in iscc_paths:
        if path.exists():
            iscc_exe = path
            break
    
    if not iscc_exe:
        print("\n⚠️  Inno Setup not found!")
        print("   Download from: https://jrsoftware.org/isdl.php")
        print("   After installing, run this command again.")
        return False
    
    # Convert icon from PNG to ICO if needed
    icon_png = Path("assets/icon.png")
    icon_ico = Path("assets/icon.ico")
    if icon_png.exists() and not icon_ico.exists():
        print("\n📷 Converting icon.png to icon.ico...")
        try:
            from PIL import Image
            img = Image.open(icon_png)
            # Create multiple sizes for better quality
            img.save(icon_ico, format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
            print(f"   Created: {icon_ico}")
        except Exception as e:
            print(f"   Warning: Could not convert icon: {e}")
            print("   Installer will use default icon.")
    
    cmd = [str(iscc_exe), str(iss_file)]
    return run_command(cmd, "Creating Windows installer with Inno Setup")


def main():
    parser = argparse.ArgumentParser(description="Build TeyvatTranslator executable")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts first")
    parser.add_argument("--zip", action="store_true", help="Create distributable ZIP")
    parser.add_argument("--installer", action="store_true", help="Create Windows installer (requires Inno Setup)")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║           TeyvatTranslator Build Script                  ║
║                   Version {VERSION}                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Clean if requested
    if args.clean:
        clean_build()
    
    # Install dependencies
    install_build_dependencies()
    
    # Build
    if not build_executable():
        print("\nBuild failed!")
        sys.exit(1)
    
    # Create ZIP if requested
    if args.zip:
        if not create_zip():
            print("\nZIP creation failed, but executable was built")
    
    # Create installer if requested
    installer_created = False
    if args.installer:
        if create_installer():
            installer_created = True
        else:
            print("\nInstaller creation failed, but executable was built")
    
    # Print completion message
    if installer_created:
        print(f"""
╔══════════════════════════════════════════════════════════╗
║                    BUILD COMPLETE!                       ║
╠══════════════════════════════════════════════════════════╣
║  Installer: dist/{APP_NAME}-v{VERSION}-Setup.exe         ║
║                                                          ║
║  To distribute:                                          ║
║  Upload the Setup.exe to GitHub Releases                 ║
╚══════════════════════════════════════════════════════════╝
    """)
    else:
        print(f"""
╔══════════════════════════════════════════════════════════╗
║                    BUILD COMPLETE!                       ║
╠══════════════════════════════════════════════════════════╣
║  Executable: dist/{APP_NAME}/{APP_NAME}.exe       ║
║                                                          ║
║  To distribute:                                          ║
║  1. Run: python build.py --installer                     ║
║  2. Upload the Setup.exe to GitHub Releases              ║
╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
