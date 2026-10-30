#!/usr/bin/env bash
set -euo pipefail

# Build the private, side-by-side-installable Local Dream SM8735 candidate.
# Requires a locally installed Android SDK/NDK whose licenses have already
# been accepted by the SDK owner, plus the user-authorized QAIRT 2.51 subset.
# This script never downloads SDKs, accepts licenses, or pushes artifacts.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_CPP="$ROOT/app/src/main/cpp"
SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
NDK_ROOT="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
QNN_ROOT="${QNN_SDK_ROOT:-}"

if [[ -z "$SDK_ROOT" || ! -d "$SDK_ROOT" ]]; then
    echo "ANDROID_SDK_ROOT (or ANDROID_HOME) must point to an installed Android SDK." >&2
    exit 2
fi
if [[ -z "$NDK_ROOT" && -d "$SDK_ROOT/ndk" ]]; then
    NDK_ROOT="$(find "$SDK_ROOT/ndk" -mindepth 1 -maxdepth 1 -type d -print | sort -V | tail -n 1)"
fi
if [[ -z "$NDK_ROOT" || ! -f "$NDK_ROOT/build/cmake/android.toolchain.cmake" ]]; then
    echo "Set ANDROID_NDK_HOME to an installed Android NDK." >&2
    exit 2
fi
if [[ -z "$QNN_ROOT" || ! -d "$QNN_ROOT/include/QNN" ]]; then
    echo "Set QNN_SDK_ROOT to the extracted QAIRT 2.51 SDK root." >&2
    exit 2
fi

required_qnn_files=(
    "$QNN_ROOT/lib/aarch64-android/libQnnHtp.so"
    "$QNN_ROOT/lib/aarch64-android/libQnnSystem.so"
    "$QNN_ROOT/lib/aarch64-android/libQnnHtpV73Stub.so"
    "$QNN_ROOT/lib/hexagon-v73/unsigned/libQnnHtpV73.so"
    "$QNN_ROOT/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so"
    "$QNN_ROOT/LICENSE.pdf"
    "$QNN_ROOT/NOTICE.txt"
    "$QNN_ROOT/QNN_NOTICE.txt"
)
for file in "${required_qnn_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Required QAIRT v73 subset file is missing: $file" >&2
        exit 2
    fi
done

CXX_RUNTIME="$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [[ ! -f "$CXX_RUNTIME" ]]; then
    echo "NDK libc++_shared.so is missing: $CXX_RUNTIME" >&2
    exit 2
fi

if ! command -v cmake >/dev/null || ! command -v java >/dev/null || ! command -v javac >/dev/null; then
    echo "CMake and a JDK with both java and javac are required." >&2
    exit 2
fi
if git -C "$ROOT" submodule status | grep -q '^-' ; then
    echo "One or more native source submodules are uninitialized." >&2
    exit 2
fi

BUILD_DIR="$APP_CPP/build/sm8735-v73-android"
JNI_DIR="$ROOT/app/src/main/jniLibs/arm64-v8a"
QNN_ASSET_DIR="$ROOT/app/src/main/assets/qnnlibs"
LICENSE_ASSET_DIR="$ROOT/app/src/main/assets/licenses"
JNI_BINARY="$JNI_DIR/libstable_diffusion_core.so"
LICENSE_FILES=(qnn-license.pdf qnn-notice.txt qnn-notice-qnn.txt)

# Never overwrite user-provided staged files or delete pre-existing SDK assets.
if [[ -e "$JNI_BINARY" || -e "$QNN_ASSET_DIR" ]]; then
    echo "Generated APK staging paths already exist; move them aside before building." >&2
    exit 2
fi
for file in "${LICENSE_FILES[@]}"; do
    if [[ -e "$LICENSE_ASSET_DIR/$file" ]]; then
        echo "Generated license staging path already exists: $LICENSE_ASSET_DIR/$file" >&2
        exit 2
    fi
done

cleanup() {
    rm -f "$JNI_BINARY"
    rmdir "$JNI_DIR" 2>/dev/null || true
    rm -rf "$QNN_ASSET_DIR"
    for file in "${LICENSE_FILES[@]}"; do
        rm -f "$LICENSE_ASSET_DIR/$file"
    done
    rmdir "$LICENSE_ASSET_DIR" 2>/dev/null || true
}
trap cleanup EXIT

export ANDROID_SDK_ROOT="$SDK_ROOT"
export ANDROID_HOME="$SDK_ROOT"
export ANDROID_NDK_HOME="$NDK_ROOT"
export QNN_SDK_ROOT="$QNN_ROOT"

cmake -S "$APP_CPP" -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DQNN_SDK_ROOT="$QNN_ROOT" \
    -DQNN_HTP_ARCHES=73
cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-2}"

BUILT_BINARY="$BUILD_DIR/bin/arm64-v8a/libstable_diffusion_core.so"
if [[ ! -s "$BUILT_BINARY" ]]; then
    echo "CMake did not produce the expected native executable: $BUILT_BINARY" >&2
    exit 1
fi
qnn_runtime_names=(
    libQnnHtp.so
    libQnnSystem.so
    libQnnHtpV73Stub.so
    libQnnHtpV73.so
    libQnnHtpV73Skel.so
)
for file in "${qnn_runtime_names[@]}"; do
    if [[ ! -s "$BUILD_DIR/qnnlibs/$file" ]]; then
        echo "CMake did not stage required QNN runtime library: $BUILD_DIR/qnnlibs/$file" >&2
        exit 1
    fi
done
mkdir -p "$JNI_DIR" "$QNN_ASSET_DIR" "$LICENSE_ASSET_DIR"
cp "$BUILT_BINARY" "$JNI_BINARY"
cp "$BUILD_DIR/qnnlibs/"* "$QNN_ASSET_DIR/"
cp "$CXX_RUNTIME" "$QNN_ASSET_DIR/libc++_shared.so"
cp "$QNN_ROOT/LICENSE.pdf" "$LICENSE_ASSET_DIR/qnn-license.pdf"
cp "$QNN_ROOT/NOTICE.txt" "$LICENSE_ASSET_DIR/qnn-notice.txt"
cp "$QNN_ROOT/QNN_NOTICE.txt" "$LICENSE_ASSET_DIR/qnn-notice-qnn.txt"

(
    cd "$ROOT"
    ./gradlew --no-daemon :app:assembleBasicDebug
)

APK_DIR="$ROOT/app/build/outputs/apk/basic/debug"
mapfile -t APKS < <(find "$APK_DIR" -maxdepth 1 -type f -name '*.apk' -print | sort)
if [[ "${#APKS[@]}" -ne 1 ]]; then
    echo "Expected one basicDebug APK under $APK_DIR; found ${#APKS[@]}." >&2
    exit 1
fi
APK_OUTPUT="${APK_OUTPUT:-$ROOT/../LocalDream_SM8735_v73_debug.apk}"
mkdir -p "$(dirname "$APK_OUTPUT")"
cp "${APKS[0]}" "$APK_OUTPUT"
echo "APK ready: $APK_OUTPUT"
echo "The APK staging copies were removed from the source tree after packaging."
