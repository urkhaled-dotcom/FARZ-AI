#!/usr/bin/env python3
"""Apply the checked-in FARZ Kotlin bridge to a generated Flutter Android scaffold."""
from __future__ import annotations
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
ANDROID = APP / "android"
TEMPLATE = APP / "android_template" / "app" / "src" / "main" / "kotlin" / "com" / "farz" / "farz_app"
KOTLIN = ANDROID / "app" / "src" / "main" / "kotlin" / "com" / "farz" / "farz_app"

if not (ANDROID / "app" / "build.gradle.kts").is_file():
    raise SystemExit("Flutter Android scaffold not found. Run flutter create --platforms=android first.")

KOTLIN.mkdir(parents=True, exist_ok=True)
for filename in ("MainActivity.kt", "FarzClassifier.kt"):
    source = TEMPLATE / filename
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, KOTLIN / filename)

build_file = ANDROID / "app" / "build.gradle.kts"
text = build_file.read_text(encoding="utf-8")
if re.search(r"minSdk\s*=\s*flutter\.minSdkVersion", text):
    text = re.sub(r"minSdk\s*=\s*flutter\.minSdkVersion", "minSdk = 24", text, count=1)
elif not re.search(r"minSdk\s*=", text):
    text, count = re.subn(r"defaultConfig\s*\{", "defaultConfig {\n        minSdk = 24", text, count=1)
    if count != 1:
        raise RuntimeError("Could not set Android minSdk in app/build.gradle.kts")

if "abiFilters" not in text:
    text, count = re.subn(
        r"defaultConfig\s*\{",
        'defaultConfig {\n        ndk { abiFilters += setOf("arm64-v8a", "x86_64") }',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not set supported Android ABIs")

if "org.pytorch:executorch-android:1.5.1" not in text or "androidx.exifinterface:exifinterface:1.4.2" not in text:
    match = re.search(r"dependencies\s*\{", text)
    if match:
        additions = '\n    implementation("org.pytorch:executorch-android:1.5.1")\n    implementation("androidx.exifinterface:exifinterface:1.4.2")'
        text = text[:match.end()] + additions + text[match.end():]
    else:
        text += '\n\ndependencies {\n    implementation("org.pytorch:executorch-android:1.5.1")\n    implementation("androidx.exifinterface:exifinterface:1.4.2")\n}\n'
build_file.write_text(text, encoding="utf-8")

# FARZ CI workaround: disable crashing third-party lintVital task.
root_build_file = ANDROID / "build.gradle.kts"
if root_build_file.is_file():
    root_text = root_build_file.read_text(encoding="utf-8")
    if marker not in root_text:
        root_text += (
            "\n\n" + marker + "\n"
            + "subprojects {\n"
            + "    tasks.configureEach {\n"
            + '        if (name.startsWith("lintVital")) {\n'
            + "            enabled = false\n"
            + "        }\n"
            + "    }\n"
            + "}\n"
        )
    root_build_file.write_text(root_text, encoding="utf-8")


# ExecuTorch Module.load() reads native Android assets. Copy the .pte from
# the tracked Flutter app asset folder into Android's native asset directory.
native_assets = ANDROID / "app" / "src" / "main" / "assets" / "models"
native_assets.mkdir(parents=True, exist_ok=True)
pte_source = APP / "assets" / "models" / "farz_model.pte"
pte_target = native_assets / "farz_model.pte"
if pte_source.is_file():
    shutil.copy2(pte_source, pte_target)
    print(f"Bundled FARZ PTE model ({pte_target.stat().st_size:,} bytes).")
else:
    pte_target.unlink(missing_ok=True)
    print("NOTICE: PTE model not present; Flutter shell builds but classification stays disabled.")

print("FARZ Android native bridge configured.")
