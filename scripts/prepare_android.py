#!/usr/bin/env python3
"""Apply FARZ Android inference bridge, branding and ONNX Runtime packaging."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
ANDROID = APP / "android"
TEMPLATE_MAIN = APP / "android_template" / "app" / "src" / "main"
TEMPLATE_KOTLIN = TEMPLATE_MAIN / "kotlin" / "com" / "farz" / "farz_app"
TEMPLATE_RES = APP / "android_template" / "res"
KOTLIN = ANDROID / "app" / "src" / "main" / "kotlin" / "com" / "farz" / "farz_app"
RES = ANDROID / "app" / "src" / "main" / "res"

if not (ANDROID / "app" / "build.gradle.kts").is_file():
    raise SystemExit("Flutter Android scaffold not found. Run flutter create --platforms=android first.")

# Install Kotlin inference bridge.
KOTLIN.mkdir(parents=True, exist_ok=True)
for filename in ("MainActivity.kt", "FarzClassifier.kt"):
    source = TEMPLATE_KOTLIN / filename
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, KOTLIN / filename)

# Install FARZ launcher resources.
if not TEMPLATE_RES.is_dir():
    raise FileNotFoundError(f"FARZ Android resource templates missing: {TEMPLATE_RES}")
shutil.copytree(TEMPLATE_RES, RES, dirs_exist_ok=True)

# Configure Android 7+ and universal native library ABIs.
build_file = ANDROID / "app" / "build.gradle.kts"
text = build_file.read_text(encoding="utf-8")
text, min_count = re.subn(
    r"minSdk\s*=\s*flutter\.minSdkVersion", "minSdk = 24", text, count=1
)
if min_count == 0 and not re.search(r"minSdk\s*=", text):
    text, count = re.subn(
        r"defaultConfig\s*\{", "defaultConfig {\n        minSdk = 24", text, count=1
    )
    if count != 1:
        raise RuntimeError("Could not set Android minSdk.")

text = re.sub(r"(?ms)^\s*ndk\s*\{[^{}]*\}\s*", "", text)
text = re.sub(r"(?m)^\s*abiFilters\s*.*(?:\n|$)", "", text)
text, count = re.subn(
    r"defaultConfig\s*\{",
    'defaultConfig {\n        ndk { abiFilters += setOf("armeabi-v7a", "arm64-v8a", "x86_64") }',
    text,
    count=1,
)
if count != 1:
    raise RuntimeError("Could not set FARZ Android ABI filters.")

# Use the maintained ONNX Runtime Android package, not the 64-bit-only PTE runtime.
for artifact in (
    r"org\.pytorch:executorch-android",
    r"org\.pytorch:pytorch_android_lite",
    r"com\.microsoft\.onnxruntime:onnxruntime-android",
):
    text = re.sub(
        rf'^\s*implementation\("{artifact}:[^"]+"\)\s*\n',
        "",
        text,
        flags=re.MULTILINE,
    )

required = [
    'implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")',
    'implementation("androidx.exifinterface:exifinterface:1.4.2")',
]
missing = [line for line in required if line not in text]
if missing:
    match = re.search(r"dependencies\s*\{", text)
    if match:
        text = text[:match.end()] + "\n    " + "\n    ".join(missing) + "\n" + text[match.end():]
    else:
        text += "\n\ndependencies {\n    " + "\n    ".join(missing) + "\n}\n"

# Remove the previous ARMv7 exclusion if it exists.
text = re.sub(
    r'(?ms)^\s*packaging\s*\{\s*jniLibs\s*\{\s*excludes\s*\+=\s*setOf\("\*\*/armeabi-v7a/\*\*"\)\s*\}\s*\}\s*',
    "\n",
    text,
)
build_file.write_text(text, encoding="utf-8")

# Apply user-facing app name and launcher icons.
manifest = ANDROID / "app" / "src" / "main" / "AndroidManifest.xml"
if not manifest.is_file():
    raise FileNotFoundError(manifest)
manifest_text = manifest.read_text(encoding="utf-8")
# The ONNX Runtime AAR can merge telemetry network permissions into the app.
# Remove these permissions because FARZ performs inference entirely offline.
manifest_text = re.sub(
    r"<manifest\b",
    '<manifest xmlns:tools="http://schemas.android.com/tools"',
    manifest_text,
    count=1,
) if "xmlns:tools=" not in manifest_text else manifest_text
for permission in ("android.permission.INTERNET", "android.permission.ACCESS_NETWORK_STATE"):
    tag = f'<uses-permission android:name="{permission}" tools:node="remove" />'
    if tag not in manifest_text:
        manifest_text = re.sub(
            r"(<application\b)",
            tag + r"\n    \1",
            manifest_text,
            count=1,
        )
app_match = re.search(r"<application\b[^>]*>", manifest_text, flags=re.DOTALL)
if not app_match:
    raise RuntimeError("No application element in AndroidManifest.xml")
tag = app_match.group(0)
for attr, value in (
    ("label", "@string/app_name"),
    ("icon", "@mipmap/ic_launcher"),
    ("roundIcon", "@mipmap/ic_launcher_round"),
):
    pattern = rf'android:{attr}="[^"]*"'
    replacement = f'android:{attr}="{value}"'
    if re.search(pattern, tag):
        tag = re.sub(pattern, replacement, tag, count=1)
    else:
        tag = tag[:-1] + f" {replacement}>"
manifest.write_text(
    manifest_text[:app_match.start()] + tag + manifest_text[app_match.end():],
    encoding="utf-8",
)

# Preserve the existing CI workaround for a third-party lint task.
root_build = ANDROID / "build.gradle.kts"
marker = "// FARZ CI workaround: disable crashing third-party lintVital task."
if root_build.is_file():
    root_text = root_build.read_text(encoding="utf-8")
    if marker not in root_text:
        root_text += (
            "\n\n" + marker +
            "\nsubprojects {\n    tasks.configureEach {\n"
            "        if (name.startsWith(\"lintVital\")) {\n            enabled = false\n"
            "        }\n    }\n}\n"
        )
    root_build.write_text(root_text, encoding="utf-8")

# ONNX Runtime reads Android native assets, not Flutter's asset bundle.
native_assets = ANDROID / "app" / "src" / "main" / "assets" / "models"
native_assets.mkdir(parents=True, exist_ok=True)
model_source = APP / "assets" / "models" / "farz_model.onnx"
model_target = native_assets / "farz_model.onnx"
if not model_source.is_file() or model_source.stat().st_size == 0:
    model_target.unlink(missing_ok=True)
    raise SystemExit(
        "ERROR: FARZ ONNX model missing or empty. "
        "Export the verified farz_model.onnx before building."
    )
shutil.copy2(model_source, model_target)
print(f"Bundled FARZ ONNX model ({model_target.stat().st_size:,} bytes).")
print("Android ABIs enabled: armeabi-v7a, arm64-v8a and x86_64.")
print("FARZ app label and launcher icons installed.")
