from __future__ import annotations

import ast
from pathlib import Path

TARGET = Path("src/model.py")
if not TARGET.exists():
    raise FileNotFoundError("Run this from the repository root; src/model.py was not found.")

text = TARGET.read_text(encoding="utf-8")
original = text

IMPORT_LINE = "from src.preseason_prior import blend_preseason_role_prior\n"
if IMPORT_LINE not in text:
    anchor = "from src.ensemble_models import BlendedClassifier, BlendedRegressor\n"
    if anchor not in text:
        raise RuntimeError("Could not locate the model import anchor.")
    text = text.replace(anchor, anchor + IMPORT_LINE, 1)

call = (
    "    xp_base_model = blend_preseason_role_prior(\n"
    "        base_view,\n"
    "        xp_base_model,\n"
    "        base_start,\n"
    "        base_app,\n"
    "    )\n"
)
if call not in text:
    anchor = (
        "    xp_base_model = _point_projection_for_source(\n"
        "        bundle, base_view, base_start, base_app, base_minutes\n"
        "    )\n"
    )
    if anchor not in text:
        raise RuntimeError("Could not locate the Base xP projection block.")
    text = text.replace(anchor, anchor + "\n" + call, 1)

ast.parse(text)

if text == original:
    print("✅ v8.2 preseason-prior patch was already applied.")
else:
    backup = Path("src/model.py.before_v8_2_preseason_prior")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print("✅ Patched src/model.py")
    print("Backup:", backup)

print("✅ Python syntax check passed.")
