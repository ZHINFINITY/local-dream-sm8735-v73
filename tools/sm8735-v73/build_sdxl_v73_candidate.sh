#!/usr/bin/env bash
set -Eeuo pipefail

# Experimental only: the default is a known Qualcomm v73 reference profile
# (SM8550, soc_model 43). It is not a device-specific SM8735 compile/validation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-illustrious_xl_v2}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"
SOC_MODEL="${SOC_MODEL:-43}"
BUILD_ROOT="${BUILD_ROOT:-${TMPDIR:-/tmp}/local-dream-sdxl-v73}"
ALLOW_RESOURCE_WARNINGS="${ALLOW_RESOURCE_WARNINGS:-0}"
CONVERTER_ZIP_URL="https://chino.icu/local-dream/convertsdxl.zip"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

[[ -n "$MODEL_PATH" ]] || fail 'Set MODEL_PATH to the original SDXL .safetensors checkpoint (not an MNN package).'
[[ -f "$MODEL_PATH" ]] || fail "Checkpoint not found: $MODEL_PATH"
[[ "$MODEL_PATH" == *.safetensors ]] || fail 'MODEL_PATH must point to the original .safetensors SDXL checkpoint.'
[[ "$MODEL_PATH" == /* ]] || fail 'Use an absolute MODEL_PATH.'
[[ "$MODEL_PATH" != *[[:space:]]* ]] || fail 'The upstream conversion scripts do not safely handle spaces in MODEL_PATH; use a no-space path.'
[[ "$MODEL_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail 'MODEL_NAME must use only letters, digits, dot, underscore, or hyphen.'
[[ -n "$QNN_SDK_ROOT" ]] || fail 'Set QNN_SDK_ROOT to an installed Qualcomm AI Engine Direct SDK 2.28 root.'
[[ -d "$QNN_SDK_ROOT" ]] || fail "QNN_SDK_ROOT not found: $QNN_SDK_ROOT"
[[ -f "$QNN_SDK_ROOT/bin/envsetup.sh" ]] || fail 'Expected bin/envsetup.sh under QNN_SDK_ROOT.'
[[ -f "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" ]] || fail 'Expected the Linux x86_64 QNN HTP library under QNN_SDK_ROOT.'
[[ -f "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnModelDlc.so" ]] || fail 'Expected libQnnModelDlc.so under QNN_SDK_ROOT.'
[[ -f "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so" ]] || fail 'Expected libQnnHtpNetRunExtensions.so under QNN_SDK_ROOT.'
[[ "$QNN_SDK_ROOT" != *[[:space:]]* ]] || fail 'Use a no-space QNN_SDK_ROOT path; upstream scripts do not consistently quote it.'
[[ "$BUILD_ROOT" != *[[:space:]]* ]] || fail 'Use a no-space BUILD_ROOT path; upstream scripts do not consistently quote it.'
command -v python3 >/dev/null || fail 'python3 is required.'
command -v uv >/dev/null || fail 'uv is required by the official converter workflow.'
[[ "$SOC_MODEL" =~ ^[0-9]+$ ]] || fail 'SOC_MODEL must be a Qualcomm numeric soc_model identifier.'

if [[ "$SOC_MODEL" == "43" ]]; then
  printf '%s\n' 'Using v73 / 8 MB VTCM / soc_model 43 (Qualcomm SM8550 reference profile).'
  printf '%s\n' 'This is NOT an SM8735-specific model ID and does NOT prove POCO F7 compatibility.'
else
  printf 'Using caller-supplied soc_model %s with HTP v73; verify this identifier against Qualcomm documentation for the intended device.\n' "$SOC_MODEL"
fi

mkdir -p "$BUILD_ROOT"
# Local Dream's official experimental SDXL guide recommends about 64 GB RAM+swap
# and 100 GB+ free disk for high-resolution conversion. Warn-and-stop by default.
read -r mem_kb swap_kb < <(awk '/^MemTotal:/ {m=$2} /^SwapTotal:/ {s=$2} END {print m+0, s+0}' /proc/meminfo)
mem_gib=$(( (mem_kb + swap_kb) / 1048576 ))
disk_kb=$(df -Pk "$BUILD_ROOT" | awk 'NR==2 {print $4}')
disk_gib=$(( disk_kb / 1048576 ))
resource_problem=0
if (( mem_gib < 64 )); then
  warn "Detected about ${mem_gib} GiB RAM+swap; the guide recommends about 64 GiB for high-resolution SDXL conversion."
  resource_problem=1
fi
if (( disk_gib < 100 )); then
  warn "Detected about ${disk_gib} GiB free under BUILD_ROOT; the guide warns high-resolution workflows can need 100 GB+."
  resource_problem=1
fi
if (( resource_problem )) && [[ "$ALLOW_RESOURCE_WARNINGS" != "1" ]]; then
  fail 'Refusing a likely out-of-memory/out-of-disk conversion. Use a suitable build host or set ALLOW_RESOURCE_WARNINGS=1 only if you accept the risk.'
fi

CONVERTER_PARENT="$BUILD_ROOT/convertsdxl"
CONVERTER_ROOT="$CONVERTER_PARENT/convertsdxl"
ARCHIVE="$BUILD_ROOT/convertsdxl.zip"
mkdir -p "$CONVERTER_PARENT"
if [[ ! -f "$CONVERTER_ROOT/export_sdxl.sh" ]]; then
  printf 'Downloading official Local Dream SDXL conversion scripts from %s\n' "$CONVERTER_ZIP_URL"
  python3 - "$CONVERTER_ZIP_URL" "$ARCHIVE" "$CONVERTER_PARENT" <<'PY'
import io, pathlib, sys, urllib.request, zipfile
url, archive, destination = sys.argv[1:]
data = urllib.request.urlopen(url, timeout=60).read()
pathlib.Path(archive).write_bytes(data)
root = pathlib.Path(destination).resolve()
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for item in z.infolist():
        target = (root / item.filename).resolve()
        if not target.is_relative_to(root):
            raise SystemExit(f"Unsafe archive path: {item.filename}")
    z.extractall(root)
print(f"Downloaded and extracted {len(data)} bytes")
PY
fi
[[ -f "$CONVERTER_ROOT/export_sdxl.sh" ]] || fail 'Official conversion archive did not contain export_sdxl.sh.'

# This selects the same-architecture SM8550 reference profile, not an SM8735
# target. Do not add SM8735 to Local Dream's built-in SDXL support list until a
# correct target profile and an actual POCO F7 validation exist.
python3 "$SCRIPT_DIR/build_v73_package.py" prepare-profile \
  --converter-root "$CONVERTER_ROOT" --soc-model "$SOC_MODEL"

export MODEL_PATH MODEL_NAME QNN_SDK_ROOT
printf '\nStarting upstream SDXL conversion with the experimental v73 / 8gen2 profile.\n'
(cd "$CONVERTER_ROOT" && bash ./export_sdxl.sh)

python3 "$SCRIPT_DIR/build_v73_package.py" package \
  --package-dir "$CONVERTER_ROOT/output/qnn_models_sdxl_8gen2" \
  --output "$BUILD_ROOT/${MODEL_NAME}_experimental_v73_soc${SOC_MODEL}_candidate.zip" \
  --soc-model "$SOC_MODEL"
