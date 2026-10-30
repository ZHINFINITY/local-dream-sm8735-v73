#!/usr/bin/env python3
"""Prepare and package an experimental Local Dream SDXL HTP v73 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import zipfile
from typing import Any

REQUIRED_FILES = {
    "SDXL",
    "tokenizer.json",
    "clip.mnn",
    "clip.mnn.weight",
    "clip_2.mnn",
    "clip_2.mnn.weight",
    "pos_emb.bin",
    "token_emb.bin",
    "pos_emb_2.bin",
    "token_emb_2.bin",
    "unet.bin",
    "vae_encoder.bin",
    "vae_decoder.bin",
}
PACKAGE_PREFIX = pathlib.PurePosixPath("output/qnn_models_sdxl_8gen2")
SOURCE_REPO = "NyuromonX/Illustrious-XL-v2.0-Diffusers-fp16"
SOURCE_REVISION = "c3cd0edb4c4ebfc6c86011039fb2e81b865b5d85"


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
    *,
    unet_htp_profiled_sm8550: bool = False,
    vae_encoder_htp_profiled_sm8550: bool = False,
    vae_decoder_htp_profiled_sm8550: bool = False,
    unet_context_bytes: bytes | None = None,
    unet_link_job_id: str | None = None,
    vae_encoder_link_job_id: str | None = None,
    vae_decoder_link_job_id: str | None = None,
    consume_source_files: bool = False,
) -> pathlib.Path:
    """Validate expected app files and package them with an explicit test-only note."""
    package_dir = package_dir.resolve()
    output_path = output_path.resolve()
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Package directory not found: {package_dir}")
    found = {item.name for item in package_dir.iterdir() if item.is_file()}
    if unet_context_bytes is not None:
        if not unet_context_bytes:
            raise ValueError("AI Hub returned an empty U-Net context")
        if "unet.bin" in found:
            raise ValueError("Provide the U-Net context either from AI Hub memory or from package_dir, not both")
        found.add("unet.bin")
    missing = sorted(REQUIRED_FILES - found)
    if missing:
        raise ValueError("Incomplete Local Dream SDXL package; missing: " + ", ".join(missing))

    components_profiled = {
        "unet_htp_profiled_on_sm8550": bool(unet_htp_profiled_sm8550),
        "vae_encoder_htp_profiled_on_sm8550": bool(vae_encoder_htp_profiled_sm8550),
        "vae_decoder_htp_profiled_on_sm8550": bool(vae_decoder_htp_profiled_sm8550),
    }
    all_components_profiled = all(components_profiled.values()) and int(soc_model) == 43
    profile: dict[str, Any] = {
        "status": "COMPONENTS_PROFILED_SM8550_ONLY" if all_components_profiled else "EXPERIMENTAL_UNVALIDATED",
        "model_type": "SDXL",
        "htp_arch": "v73",
        "vtcm_mb": 8,
        "qnn_soc_model": int(soc_model),
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_identity_note": "The mirror is name-matched to Illustrious XL v2.0, but exact weight identity with the upstream OnomaAIResearch checkpoint is not independently verified.",
        "qairt_version": "2.51",
        "component_precision": {"unet": "W8A16", "vae_encoder": "HTP FP16", "vae_decoder": "HTP FP16", "clip_encoders": "MNN assets from the published Onoma-based conversion"},
        "target_note": (
            "Qualcomm SM8550 reference profile; not an SM8735-specific target."
            if int(soc_model) == 43
            else "Caller-supplied SoC model value; verify against Qualcomm documentation."
        ),
        "sm8550_reference_component_profiles": components_profiled,
        "sm8550_reference_job_ids": {
            "unet_link_job_id": unet_link_job_id,
            "vae_encoder_link_job_id": vae_encoder_link_job_id,
            "vae_decoder_link_job_id": vae_decoder_link_job_id,
        },
        "unet_context_sha256": hashlib.sha256(unet_context_bytes).hexdigest() if unet_context_bytes is not None else None,
        "sm8735_poco_f7_validated": False,
        "model_capabilities": {
            "text_to_image": True,
            "image_to_image": "vae_encoder.bin" in found,
        },
        "validation": (
            "The U-Net and both VAE contexts have separate HTP profiles on the SM8550 reference; the integrated app package and POCO F7 / SM8735 are not device-validated."
            if all_components_profiled
            else "Not validated on POCO F7 / SM8735. Do not infer support from an SM8550 v73 reference profile or the v73 architecture alone."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for item in sorted(package_dir.rglob("*")):
            if not item.is_file() or item.name == "EXPERIMENTAL_V73_UNVALIDATED.json":
                continue
            if unet_context_bytes is not None and item.relative_to(package_dir).as_posix() == "unet.bin":
                continue
            archive_info = zipfile.ZipInfo(str(PACKAGE_PREFIX / item.relative_to(package_dir)))
            archive_info.compress_type = zipfile.ZIP_DEFLATED
            with item.open("rb") as source, archive.open(archive_info, "w", force_zip64=True) as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            if consume_source_files:
                item.unlink()
        if unet_context_bytes is not None:
            archive_info = zipfile.ZipInfo(str(PACKAGE_PREFIX / "unet.bin"))
            archive_info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(archive_info, "w", force_zip64=True) as destination:
                view = memoryview(unet_context_bytes)
                chunk_size = 64 * 1024 * 1024
                for offset in range(0, len(view), chunk_size):
                    destination.write(view[offset : offset + chunk_size])
            del view
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
    package_parser.add_argument("--unet-htp-profiled-sm8550", action="store_true")
    package_parser.add_argument("--vae-encoder-htp-profiled-sm8550", action="store_true")
    package_parser.add_argument("--vae-decoder-htp-profiled-sm8550", action="store_true")
    package_parser.add_argument("--unet-link-job-id", help="Download the completed LinkJob context directly into the ZIP in memory")
    package_parser.add_argument("--vae-encoder-link-job-id")
    package_parser.add_argument("--vae-decoder-link-job-id")
    package_parser.add_argument("--project")
    package_parser.add_argument("--consume-source-files", action="store_true", help="Delete local package files after each is written to the ZIP")

    args = parser.parse_args()
    try:
        if args.command == "prepare-profile":
            prepare_v73_profile(args.converter_root, args.soc_model)
            print(f"Prepared HTP v73 / 8 MB profile under {args.converter_root}")
        else:
            unet_context_bytes = None
            if args.unet_link_job_id:
                import os
                import qai_hub as hub

                api_token = os.environ.get("QAI_HUB_API_TOKEN")
                if not api_token:
                    raise ValueError("QAI_HUB_API_TOKEN must be set when --unet-link-job-id is used")
                client = hub.Client(config=hub.ClientConfig(api_token=api_token, default_project=args.project))
                link_job = client.get_job(args.unet_link_job_id)
                link_status = str(link_job.get_status()).upper()
                if "DONE" not in link_status and "SUCCESS" not in link_status:
                    raise ValueError(f"U-Net link job {args.unet_link_job_id} is not complete: {link_status}")
                print(f"Downloading linked U-Net context {args.unet_link_job_id} into memory for ZIP packaging", flush=True)
                unet_context_bytes = link_job.download_target_model()
                if not isinstance(unet_context_bytes, bytes):
                    raise TypeError(f"Expected in-memory U-Net bytes from AI Hub, got {type(unet_context_bytes).__name__}")
                print(f"Loaded U-Net context in memory: {len(unet_context_bytes)} bytes", flush=True)
            output = package_candidate(
                args.package_dir,
                args.output,
                args.soc_model,
                unet_htp_profiled_sm8550=args.unet_htp_profiled_sm8550,
                vae_encoder_htp_profiled_sm8550=args.vae_encoder_htp_profiled_sm8550,
                vae_decoder_htp_profiled_sm8550=args.vae_decoder_htp_profiled_sm8550,
                unet_context_bytes=unet_context_bytes,
                unet_link_job_id=args.unet_link_job_id,
                vae_encoder_link_job_id=args.vae_encoder_link_job_id,
                vae_decoder_link_job_id=args.vae_decoder_link_job_id,
                consume_source_files=args.consume_source_files,
            )
            print(f"Candidate archive written: {output}")
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
