#!/usr/bin/env python3
"""Prepare and package an experimental Local Dream SDXL HTP v73 candidate."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import zipfile
from typing import Any

REQUIRED_FILES = {
    "SDXL",
    "tokenizer.json",
    "clip.mnn",
    "clip_2.mnn",
    "pos_emb.bin",
    "token_emb.bin",
    "pos_emb_2.bin",
    "token_emb_2.bin",
    "unet.bin",
    "vae_encoder.bin",
    "vae_decoder.bin",
}
PACKAGE_PREFIX = pathlib.PurePosixPath("output/qnn_models_sdxl_8gen2")


def _replace_single_line(
    text: str,
    pattern: str,
    replacement: str,
    *,
    already_patched: str | None = None,
    flags: int = re.MULTILINE,
) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count == 1:
        return updated
    if already_patched is not None and re.search(already_patched, text, flags):
        return text
    raise ValueError(f"Expected one upstream script line matching {pattern!r}")


def prepare_v73_profile(converter_root: pathlib.Path, soc_model: int) -> None:
    """Write QNN's v73/8 MB profile and adapt the official export shell scripts."""
    root = converter_root.resolve()
    export_path = root / "export_sdxl.sh"
    convert_path = root / "scripts" / "convert_all_sdxl.sh"
    if not export_path.is_file() or not convert_path.is_file():
        raise FileNotFoundError("Expected official export_sdxl.sh and scripts/convert_all_sdxl.sh")

    backend = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": "./htp_config_8gen2.json",
        }
    }
    htp = {
        "graphs": [{"vtcm_mb": 8, "graph_names": ["model"], "O": 3.0}],
        "devices": [
            {
                "dsp_arch": "v73",
                "soc_model": int(soc_model),
                "cores": [{"perf_profile": "burst", "rpc_control_latency": 100}],
            }
        ],
    }
    (root / "htp_backend_8gen2.json").write_text(json.dumps(backend, indent=2) + "\n")
    (root / "htp_config_8gen2.json").write_text(json.dumps(htp, indent=2) + "\n")

    text = export_path.read_text()
    text = _replace_single_line(
        text,
        r"^model_path=.*$",
        'model_path="${MODEL_PATH:?Set MODEL_PATH}"',
        already_patched=r'^model_path="\$\{MODEL_PATH:\?Set MODEL_PATH\}"$',
    )
    text = _replace_single_line(
        text,
        r"^model_name=.*$",
        'model_name="${MODEL_NAME:?Set MODEL_NAME}"',
        already_patched=r'^model_name="\$\{MODEL_NAME:\?Set MODEL_NAME\}"$',
    )
    text = _replace_single_line(
        text,
        r'^soc_versions=\("8gen3"\)$',
        'soc_versions=("8gen2")',
        already_patched=r'^soc_versions=\("8gen2"\)$',
    )
    text = text.replace(
        "python prepare_data_sdxl.py --model_path $model_path",
        'python prepare_data_sdxl.py --model_path "$model_path"',
    )
    text = text.replace(
        "python export_onnx_sdxl.py --model_path $model_path",
        'python export_onnx_sdxl.py --model_path "$model_path"',
    )
    export_path.write_text(text)

    text = convert_path.read_text()
    text = _replace_single_line(
        text,
        r"^QNN_SDK_ROOT=.*$",
        'QNN_SDK_ROOT="${QNN_SDK_ROOT:-/data/qairt/2.28.0.241029}"',
        already_patched=r'^QNN_SDK_ROOT="\$\{QNN_SDK_ROOT:-/data/qairt/2\.28\.0\.241029\}"$',
    )
    if not re.search(r"^export QNN_SDK_ROOT$", text, re.MULTILINE):
        text = re.sub(
            r'^(QNN_SDK_ROOT="\$\{QNN_SDK_ROOT:-/data/qairt/2\.28\.0\.241029\}")$',
            r"\1\nexport QNN_SDK_ROOT",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    convert_path.write_text(text)


def package_candidate(
    package_dir: pathlib.Path,
    output_path: pathlib.Path,
    soc_model: int,
) -> pathlib.Path:
    """Validate expected app files and package them with an explicit test-only note."""
    package_dir = package_dir.resolve()
    output_path = output_path.resolve()
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Package directory not found: {package_dir}")
    found = {item.name for item in package_dir.iterdir() if item.is_file()}
    missing = sorted(REQUIRED_FILES - found)
    if missing:
        raise ValueError("Incomplete Local Dream SDXL package; missing: " + ", ".join(missing))

    profile: dict[str, Any] = {
        "status": "EXPERIMENTAL_UNVALIDATED",
        "model_type": "SDXL",
        "htp_arch": "v73",
        "vtcm_mb": 8,
        "qnn_soc_model": int(soc_model),
        "target_note": (
            "Qualcomm SM8550 reference profile; not an SM8735-specific target."
            if int(soc_model) == 43
            else "Caller-supplied SoC model value; verify against Qualcomm documentation."
        ),
        "validation": "Not validated on POCO F7 / SM8735. Do not infer support from the v73 architecture alone.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for item in sorted(package_dir.rglob("*")):
            if item.is_file() and item.name != "EXPERIMENTAL_V73_UNVALIDATED.json":
                archive.write(item, str(PACKAGE_PREFIX / item.relative_to(package_dir)))
        archive.writestr(
            str(PACKAGE_PREFIX / "EXPERIMENTAL_V73_UNVALIDATED.json"),
            json.dumps(profile, indent=2) + "\n",
        )

    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())
        missing_from_zip = sorted(
            name for name in REQUIRED_FILES if str(PACKAGE_PREFIX / name) not in names
        )
        if missing_from_zip:
            raise ValueError("Final archive is missing: " + ", ".join(missing_from_zip))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("prepare-profile")
    profile_parser.add_argument("--converter-root", type=pathlib.Path, required=True)
    profile_parser.add_argument("--soc-model", type=int, required=True)

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--package-dir", type=pathlib.Path, required=True)
    package_parser.add_argument("--output", type=pathlib.Path, required=True)
    package_parser.add_argument("--soc-model", type=int, required=True)

    args = parser.parse_args()
    try:
        if args.command == "prepare-profile":
            prepare_v73_profile(args.converter_root, args.soc_model)
            print(f"Prepared HTP v73 / 8 MB profile under {args.converter_root}")
        else:
            output = package_candidate(args.package_dir, args.output, args.soc_model)
            print(f"Candidate archive written: {output}")
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
