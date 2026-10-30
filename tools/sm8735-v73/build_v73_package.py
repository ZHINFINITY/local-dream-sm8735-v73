#!/usr/bin/env python3
"""Prepare and package an experimental Local Dream SDXL HTP v73 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import tempfile
import zipfile
from typing import Any

REQUIRED_FILES = {
    "SDXL",
    "tokenizer.json",
    "tokenizer_2.json",
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
CONTENTS_MANIFEST = "PACKAGE_CONTENTS.json"
PROFILE_MANIFEST = "EXPERIMENTAL_V73_UNVALIDATED.json"
COPY_CHUNK_SIZE = 8 * 1024 * 1024


def _write_file_member(
    archive: zipfile.ZipFile, source_path: pathlib.Path, archive_name: str
) -> dict[str, Any]:
    """Write a file without loading it wholly into memory and return its digest."""
    info = zipfile.ZipInfo(archive_name)
    info.compress_type = zipfile.ZIP_DEFLATED
    digest = hashlib.sha256()
    size = 0
    with source_path.open("rb") as source, archive.open(
        info, "w", force_zip64=True
    ) as destination:
        while True:
            chunk = source.read(COPY_CHUNK_SIZE)
            if not chunk:
                break
            destination.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def _write_bytes_member(
    archive: zipfile.ZipFile, data: bytes, archive_name: str
) -> dict[str, Any]:
    """Write an in-memory context in bounded chunks and return its digest."""
    info = zipfile.ZipInfo(archive_name)
    info.compress_type = zipfile.ZIP_DEFLATED
    digest = hashlib.sha256(data)
    with archive.open(info, "w", force_zip64=True) as destination:
        view = memoryview(data)
        for offset in range(0, len(view), COPY_CHUNK_SIZE):
            destination.write(view[offset : offset + COPY_CHUNK_SIZE])
    return {"size_bytes": len(data), "sha256": digest.hexdigest()}


def _link_job_matches_compile_id(link_job: Any, compile_job_id: str) -> bool:
    expected_prefix = f"job_{compile_job_id}_"
    return any(
        str(getattr(model, "name", "")).startswith(expected_prefix)
        for model in getattr(link_job, "models", [])
    )


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
    unet_compile_job_id: str | None = None,
    vae_encoder_compile_job_id: str | None = None,
    vae_decoder_compile_job_id: str | None = None,
    unet_link_job_id: str | None = None,
    vae_encoder_link_job_id: str | None = None,
    vae_decoder_link_job_id: str | None = None,
    consume_source_files: bool = False,
) -> pathlib.Path:
    """Build an atomic, hash-checked package; optional cleanup happens only after verification."""
    package_dir = package_dir.resolve()
    output_path = output_path.resolve()
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Package directory not found: {package_dir}")
    try:
        output_path.relative_to(package_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Output ZIP must be outside package_dir")
    if output_path.exists():
        raise ValueError(f"Refusing to overwrite existing output: {output_path}")
    for label, profiled, compile_job_id, job_id in (
        ("U-Net", unet_htp_profiled_sm8550, unet_compile_job_id, unet_link_job_id),
        (
            "VAE encoder",
            vae_encoder_htp_profiled_sm8550,
            vae_encoder_compile_job_id,
            vae_encoder_link_job_id,
        ),
        (
            "VAE decoder",
            vae_decoder_htp_profiled_sm8550,
            vae_decoder_compile_job_id,
            vae_decoder_link_job_id,
        ),
    ):
        if profiled and not job_id:
            raise ValueError(f"A {label} SM8550 profile claim requires its linked context job ID")
        if profiled and not compile_job_id:
            raise ValueError(f"A {label} SM8550 profile claim requires its source compile job ID")

    found = {name for name in REQUIRED_FILES if (package_dir / name).is_file()}
    if unet_context_bytes is not None:
        if not unet_context_bytes:
            raise ValueError("AI Hub returned an empty U-Net context")
        if (package_dir / "unet.bin").exists():
            raise ValueError("Provide the U-Net context either from AI Hub memory or from package_dir, not both")
        found.add("unet.bin")
    missing = sorted(REQUIRED_FILES - found)
    if missing:
        raise ValueError("Incomplete Local Dream SDXL package; missing: " + ", ".join(missing))

    source_manifest_path = package_dir / "MNN_COMPANIONS_SOURCE.json"
    declared_companions: dict[str, Any] = {}
    if source_manifest_path.is_file():
        try:
            source_manifest = json.loads(source_manifest_path.read_text())
            declared_companions = source_manifest["files"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid companion source manifest: {source_manifest_path}") from error
        if not isinstance(declared_companions, dict):
            raise ValueError("Companion source manifest 'files' must be a JSON object")
        for relative_name, expected in declared_companions.items():
            relative_path = pathlib.PurePosixPath(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe path in companion source manifest: {relative_name}")
            if not isinstance(expected, dict) or "size_bytes" not in expected or "sha256" not in expected:
                raise ValueError(f"Missing size/hash for companion asset: {relative_name}")
            if not (package_dir / pathlib.Path(*relative_path.parts)).is_file():
                raise ValueError(f"Manifest companion asset is missing: {relative_name}")

    local_files: list[tuple[str, pathlib.Path]] = []
    for item in sorted(package_dir.rglob("*")):
        if not item.is_file():
            continue
        relative_name = item.relative_to(package_dir).as_posix()
        if relative_name in {CONTENTS_MANIFEST, PROFILE_MANIFEST}:
            continue
        if unet_context_bytes is not None and relative_name == "unet.bin":
            continue
        local_files.append((relative_name, item))
    local_names = {name for name, _ in local_files}
    missing_declared = sorted(set(declared_companions) - local_names)
    if missing_declared:
        raise ValueError("Companion manifest assets are not package members: " + ", ".join(missing_declared))

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
        "reference_profile_evidence_note": "Compile-job type, status, reference device, and source-model association with the LinkJob are checked by the CLI; LinkJob type, status, reference device, and target model are also checked. Profile booleans remain caller-supplied, and the builder does not independently retrieve or validate inference-profile reports.",
        "sm8550_reference_job_ids": {
            "unet_compile_job_id": unet_compile_job_id,
            "unet_link_job_id": unet_link_job_id,
            "vae_encoder_compile_job_id": vae_encoder_compile_job_id,
            "vae_encoder_link_job_id": vae_encoder_link_job_id,
            "vae_decoder_compile_job_id": vae_decoder_compile_job_id,
            "vae_decoder_link_job_id": vae_decoder_link_job_id,
        },
        "unet_context_sha256": None,
        "sm8735_poco_f7_validated": False,
        "model_capabilities": {
            "text_to_image": True,
            "image_to_image": "vae_encoder.bin" in found,
        },
        "validation": (
            "The caller marked the U-Net and both VAE contexts as separately HTP-profiled on the SM8550 reference; the builder does not independently validate those reports, and the integrated app package and POCO F7 / SM8735 are not device-validated."
            if all_components_profiled
            else "Not validated on POCO F7 / SM8735. Do not infer support from an SM8550 v73 reference profile or the v73 architecture alone."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temporary_path = pathlib.Path(temporary_name)
    source_paths: list[pathlib.Path] = []
    content_entries: dict[str, dict[str, Any]] = {}
    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            for relative_name, item in local_files:
                archive_name = str(PACKAGE_PREFIX / pathlib.PurePosixPath(relative_name))
                metadata = _write_file_member(archive, item, archive_name)
                expected = declared_companions.get(relative_name)
                if expected is not None and (
                    metadata["size_bytes"] != int(expected["size_bytes"])
                    or metadata["sha256"].lower() != str(expected["sha256"]).lower()
                ):
                    raise ValueError(f"Staged companion does not match its source manifest: {relative_name}")
                content_entries[relative_name] = metadata
                source_paths.append(item)

            if unet_context_bytes is not None:
                content_entries["unet.bin"] = _write_bytes_member(
                    archive,
                    unet_context_bytes,
                    str(PACKAGE_PREFIX / "unet.bin"),
                )

            contents = {
                "format_version": 1,
                "package_prefix": str(PACKAGE_PREFIX),
                "files": dict(sorted(content_entries.items())),
            }
            contents_text = json.dumps(contents, indent=2, sort_keys=True) + "\n"
            profile["unet_context_sha256"] = content_entries.get("unet.bin", {}).get("sha256")
            profile["package_contents_sha256"] = hashlib.sha256(contents_text.encode()).hexdigest()
            archive.writestr(str(PACKAGE_PREFIX / CONTENTS_MANIFEST), contents_text)
            archive.writestr(
                str(PACKAGE_PREFIX / PROFILE_MANIFEST),
                json.dumps(profile, indent=2, sort_keys=True) + "\n",
            )

        # Reading every member to EOF verifies its CRC; payloads are also rehashed.
        with zipfile.ZipFile(temporary_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("Final ZIP contains duplicate member names")
            expected_names = {
                str(PACKAGE_PREFIX / pathlib.PurePosixPath(name)) for name in content_entries
            }
            expected_names.update(
                {
                    str(PACKAGE_PREFIX / CONTENTS_MANIFEST),
                    str(PACKAGE_PREFIX / PROFILE_MANIFEST),
                }
            )
            missing_from_zip = sorted(
                str(PACKAGE_PREFIX / name) for name in REQUIRED_FILES if str(PACKAGE_PREFIX / name) not in names
            )
            if missing_from_zip:
                raise ValueError("Final archive is missing: " + ", ".join(missing_from_zip))
            if set(names) != expected_names:
                raise ValueError("Final archive member list differs from the staged content manifest")
            archived_contents = json.loads(archive.read(str(PACKAGE_PREFIX / CONTENTS_MANIFEST)))
            if archived_contents != contents:
                raise ValueError("Final archive content manifest does not match the staged payloads")
            for info in infos:
                digest = hashlib.sha256()
                size = 0
                with archive.open(info, "r") as member:
                    while True:
                        chunk = member.read(COPY_CHUNK_SIZE)
                        if not chunk:
                            break
                        digest.update(chunk)
                        size += len(chunk)
                relative_name = info.filename[len(str(PACKAGE_PREFIX)) + 1 :]
                expected = content_entries.get(relative_name)
                if expected is not None and (
                    size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]
                ):
                    raise ValueError(f"Final archive payload failed hash verification: {relative_name}")

        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    if consume_source_files:
        for item in source_paths:
            if item.relative_to(package_dir).as_posix() in REQUIRED_FILES:
                item.unlink(missing_ok=True)
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
    package_parser.add_argument("--unet-compile-job-id")
    package_parser.add_argument("--vae-encoder-compile-job-id")
    package_parser.add_argument("--vae-decoder-compile-job-id")
    package_parser.add_argument("--unet-link-job-id", help="Download the completed LinkJob context directly into the ZIP in memory")
    package_parser.add_argument("--vae-encoder-link-job-id")
    package_parser.add_argument("--vae-decoder-link-job-id")
    package_parser.add_argument("--project")
    package_parser.add_argument("--consume-source-files", action="store_true", help="Delete local files only after the completed ZIP passes CRC and hash verification")

    args = parser.parse_args()
    try:
        if args.command == "prepare-profile":
            prepare_v73_profile(args.converter_root, args.soc_model)
            print(f"Prepared HTP v73 / 8 MB profile under {args.converter_root}")
        else:
            link_specs = (
                ("U-Net", args.unet_compile_job_id, args.unet_link_job_id, args.unet_htp_profiled_sm8550),
                (
                    "VAE encoder",
                    args.vae_encoder_compile_job_id,
                    args.vae_encoder_link_job_id,
                    args.vae_encoder_htp_profiled_sm8550,
                ),
                (
                    "VAE decoder",
                    args.vae_decoder_compile_job_id,
                    args.vae_decoder_link_job_id,
                    args.vae_decoder_htp_profiled_sm8550,
                ),
            )
            for label, compile_job_id, job_id, profiled in link_specs:
                if profiled and not job_id:
                    raise ValueError(f"A {label} SM8550 profile claim requires its linked context job ID")
                if job_id and not compile_job_id:
                    raise ValueError(f"The {label} link job requires its source compile job ID")

            unet_context_bytes = None
            if any(job_id for _, _, job_id, _ in link_specs):
                import os
                import qai_hub as hub

                api_token = os.environ.get("QAI_HUB_API_TOKEN")
                if not api_token:
                    raise ValueError("QAI_HUB_API_TOKEN must be set when a link-job ID is used")
                client = hub.Client(config=hub.ClientConfig(api_token=api_token, default_project=args.project))
                verified_jobs = {}
                expected_device_name = "Samsung Galaxy S23" if int(args.soc_model) == 43 else None
                for label, compile_job_id, job_id, _ in link_specs:
                    if not job_id:
                        continue
                    compile_job = client.get_job(compile_job_id)
                    if not isinstance(compile_job, hub.CompileJob):
                        raise ValueError(
                            f"{label} source job {compile_job_id} is {type(compile_job).__name__}, not an AI Hub CompileJob"
                        )
                    compile_status = str(compile_job.get_status()).upper()
                    if "DONE" not in compile_status and "SUCCESS" not in compile_status:
                        raise ValueError(f"{label} compile job {compile_job_id} is not complete: {compile_status}")
                    compile_device_name = getattr(compile_job.device, "name", None)
                    if expected_device_name and compile_device_name != expected_device_name:
                        raise ValueError(
                            f"{label} compile job {compile_job_id} targets {compile_device_name!r}, not {expected_device_name}"
                        )
                    link_job = client.get_job(job_id)
                    if not isinstance(link_job, hub.LinkJob):
                        raise ValueError(
                            f"{label} job {job_id} is {type(link_job).__name__}, not an AI Hub LinkJob"
                        )
                    link_status = str(link_job.get_status()).upper()
                    if "DONE" not in link_status and "SUCCESS" not in link_status:
                        raise ValueError(f"{label} link job {job_id} is not complete: {link_status}")
                    device_name = getattr(link_job.device, "name", None)
                    if expected_device_name and device_name != expected_device_name:
                        raise ValueError(
                            f"{label} link job {job_id} targets {device_name!r}, not {expected_device_name}"
                        )
                    if not _link_job_matches_compile_id(link_job, compile_job_id):
                        raise ValueError(
                            f"{label} link job {job_id} does not contain a source model from compile job {compile_job_id}"
                        )
                    if not link_job.models or link_job.get_target_model() is None:
                        raise ValueError(f"{label} link job {job_id} did not produce a linked target model")
                    verified_jobs[label] = link_job
                    print(
                        f"Verified {label} CompileJob {compile_job_id} → LinkJob {job_id} on {device_name or 'unspecified device'}",
                        flush=True,
                    )

                unet_link_job = verified_jobs.get("U-Net")
                if unet_link_job is not None:
                    print(f"Downloading linked U-Net context {args.unet_link_job_id} into memory for ZIP packaging", flush=True)
                    unet_context_bytes = unet_link_job.download_target_model()
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
                unet_compile_job_id=args.unet_compile_job_id,
                vae_encoder_compile_job_id=args.vae_encoder_compile_job_id,
                vae_decoder_compile_job_id=args.vae_decoder_compile_job_id,
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
