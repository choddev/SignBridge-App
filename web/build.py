"""
Build script for SignBridge Web Application.
Bundles web/src/* and copies YOLO metadata into web/dist/ for production deployment.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DIST = ROOT / "dist"
MODELS = ROOT.parent / "models"


def build():
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True, exist_ok=True)

    for item in SRC.iterdir():
        if item.is_file():
            shutil.copy2(item, DIST / item.name)

    # Copy YOLO dataset config into dist/models for inspection
    dist_models = DIST / "models"
    dist_models.mkdir(exist_ok=True)
    if (MODELS / "data.yaml").exists():
        shutil.copy2(MODELS / "data.yaml", dist_models / "data.yaml")

    print(f"[Web Build] Successfully built web application to: {DIST}")
    for f in sorted(DIST.rglob("*")):
        if f.is_file():
            print(f"  - {f.relative_to(DIST)} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
