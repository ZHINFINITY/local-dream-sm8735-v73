#!/usr/bin/env python3
"""Wait for existing v73 HTP profiles, then build the validated candidate ZIP.

This utility submits no AI Hub jobs. It reads the already-recorded U-Net and VAE
job manifests, requires a Samsung Galaxy S23 / SM8550 reference-only claim and
NPU/HTP-only profile units for all three components, then invokes
build_v73_package.py, which independently validates each CompileJob, LinkJob,
InferenceJob, model association, device, and profile report before packaging.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any

from build_v73_package import _collect_compute_units, _confirms_npu_only

JOB_ID_PATTERN = re.compile(r"/jobs/([A-Za-z0-9]+)/?$")
FAILURE_MARKERS = ("FAILED", "ERROR", "CANCELLED", "CANCELED", "TIMEOUT")


def load_record(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def job_id_from_url(url: str) -> str:
    match = JOB_ID_PATTERN.search(url.strip())
    if not match:
        raise ValueError(f"Invalid AI Hub job URL in manifest: {url!r}")
    return match.group(1)


def status_failure(record: dict[str, Any]) -> str | None:
    for key in (
        "quantize_status",
        "compile_status",
        "compile_statuses",
        "link_status",
        "inference_status",
    ):
        value = record.get(key)
        text = json.dumps(value, sort_keys=True).upper() if isinstance(value, (dict, list)) else str(value or "").upper()
        if any(marker in text for marker in FAILURE_MARKERS):
            return f"{key}={text[:240]}"
    return None


def profile_units(record: dict[str, Any]) -> list[str]:
    return _collect_compute_units({"compute_unit": record.get("profile_compute_units", [])})


def terminal_failure(record: dict[str, Any], label: str) -> str | None:
    device = record.get("device")
    claim = str(record.get("target_claim", ""))
    if device != "Samsung Galaxy S23":
        return f"{label}: expected Samsung Galaxy S23 reference, found {device!r}"
    if "SM8550" not in claim or "not SM8735" not in claim:
        return f"{label}: manifest does not clearly limit its claim to SM8550 reference only"
    failure = status_failure(record)
    if failure:
        return f"{label}: terminal job failure detected ({failure})"

    units = profile_units(record)
    inference_status = str(record.get("inference_status", "")).upper()
    report_is_terminal = "SUCCESS" in inference_status or "DONE" in inference_status
    if (record.get("htp_execution_verified") is True or (report_is_terminal and units)) and not _confirms_npu_only(units):
        return f"{label}: completed profile is not NPU/HTP-only ({units})"
    return None


def component_ready(record: dict[str, Any], label: str) -> tuple[bool, str]:
    compile_urls = record.get("compile_job_urls")
    if not isinstance(compile_urls, list) or not compile_urls:
        return False, f"{label}: no compile-job URL recorded yet"
    if not isinstance(compile_urls[0], str) or not record.get("link_job_url"):
        return False, f"{label}: link-job URL not recorded yet"
    if not record.get("inference_job_url"):
        return False, f"{label}: inference-profile job URL not recorded yet"
    if record.get("htp_execution_verified") is not True:
        return False, f"{label}: HTP execution has not been verified yet"

    units = profile_units(record)
    if not _confirms_npu_only(units):
        return False, f"{label}: profile does not confirm NPU/HTP-only placement ({units})"
    return True, f"{label}: confirmed {units} on Samsung Galaxy S23"


def component_ids(record: dict[str, Any], label: str) -> dict[str, str]:
    urls = record.get("compile_job_urls")
    if not isinstance(urls, list) or len(urls) != 1 or not all(isinstance(url, str) for url in urls):
        raise ValueError(f"{label} must have exactly one recorded compile job URL")
    link_url = record.get("link_job_url")
    profile_url = record.get("inference_job_url")
    if not isinstance(link_url, str) or not isinstance(profile_url, str):
        raise ValueError(f"{label} link/profile job URL is missing")
    return {
        "compile": job_id_from_url(urls[0]),
        "link": job_id_from_url(link_url),
        "profile": job_id_from_url(profile_url),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unet-pipeline",
        type=pathlib.Path,
        default=pathlib.Path("/workspace/sdxl-export/v73-unet-allbiased-allmatmuladd-rescaled-customcal-w8a16/pipeline.json"),
    )
    parser.add_argument(
        "--vae-encoder-jobs",
        type=pathlib.Path,
        default=pathlib.Path("/workspace/sdxl-export/v73-vae-encoder/v73_vae_encoder_jobs.json"),
    )
    parser.add_argument(
        "--vae-decoder-jobs",
        type=pathlib.Path,
        default=pathlib.Path("/workspace/sdxl-export/v73-vae-decoder/v73_vae_decoder_jobs.json"),
    )
    parser.add_argument(
        "--package-dir",
        type=pathlib.Path,
        default=pathlib.Path("/workspace/sdxl-export/local-dream-illustrious-sdxl-v73-candidate"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("/workspace/illustrious_xl_v2_experimental_v73_soc43_candidate.zip"),
    )
    parser.add_argument("--project", default="pg21vxjfj0f")
    parser.add_argument("--soc-model", type=int, default=43)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=86400, help="Use 0 to wait without a time limit")
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.timeout_seconds < 0:
        parser.error("--poll-seconds must be positive and --timeout-seconds must be nonnegative")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing candidate: {args.output}")

    paths = (
        ("U-Net", args.unet_pipeline),
        ("VAE encoder", args.vae_encoder_jobs),
        ("VAE decoder", args.vae_decoder_jobs),
    )
    deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds > 0 else None
    last_report = 0.0
    print("Waiting for existing reference profiles; no AI Hub jobs will be submitted.", flush=True)
    while True:
        records = {label: load_record(path) for label, path in paths}
        reasons = []
        ready = True
        for label, _ in paths:
            record = records[label]
            if record is None:
                ready = False
                reasons.append(f"{label}: manifest not available/readable yet")
                continue
            failure = terminal_failure(record, label)
            if failure:
                print(failure, file=sys.stderr, flush=True)
                return 2
            ok, reason = component_ready(record, label)
            ready = ready and ok
            reasons.append(reason)

        if ready:
            break
        now = time.monotonic()
        if now - last_report >= 300:
            print("\n".join(reasons), flush=True)
            last_report = now
        if deadline is not None and now >= deadline:
            print("Timed out before all three NPU-only reference profiles were ready.", file=sys.stderr)
            return 2
        time.sleep(args.poll_seconds)

    if args.output.exists():
        print(f"Refusing to overwrite candidate created while waiting: {args.output}", file=sys.stderr)
        return 2
    if not args.package_dir.is_dir():
        print(f"Package directory does not exist: {args.package_dir}", file=sys.stderr)
        return 2
    if (args.package_dir / "unet.bin").exists():
        print("Refusing local unet.bin: this workflow streams the validated U-Net LinkJob context directly into the ZIP.", file=sys.stderr)
        return 2
    if not os.environ.get("QAI_HUB_API_TOKEN"):
        print("QAI_HUB_API_TOKEN is required to validate and stream AI Hub artifacts.", file=sys.stderr)
        return 2

    ids = {label: component_ids(records[label], label) for label, _ in paths}
    builder = pathlib.Path(__file__).with_name("build_v73_package.py")
    command = [
        sys.executable,
        str(builder),
        "package",
        "--package-dir",
        str(args.package_dir),
        "--output",
        str(args.output),
        "--soc-model",
        str(args.soc_model),
        "--project",
        args.project,
        "--unet-compile-job-id",
        ids["U-Net"]["compile"],
        "--unet-link-job-id",
        ids["U-Net"]["link"],
        "--unet-profile-job-id",
        ids["U-Net"]["profile"],
        "--vae-encoder-compile-job-id",
        ids["VAE encoder"]["compile"],
        "--vae-encoder-link-job-id",
        ids["VAE encoder"]["link"],
        "--vae-encoder-profile-job-id",
        ids["VAE encoder"]["profile"],
        "--vae-decoder-compile-job-id",
        ids["VAE decoder"]["compile"],
        "--vae-decoder-link-job-id",
        ids["VAE decoder"]["link"],
        "--vae-decoder-profile-job-id",
        ids["VAE decoder"]["profile"],
    ]
    print("All manifests pass the NPU-only reference gate; running the builder’s live job/profile validation.", flush=True)
    print("Job IDs:", json.dumps(ids, sort_keys=True), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode:
        print(f"Package builder failed with exit code {result.returncode}; no candidate is reported as complete.", file=sys.stderr)
        return result.returncode
    if not args.output.is_file() or args.output.stat().st_size == 0:
        print("Builder returned success without a non-empty archive.", file=sys.stderr)
        return 2
    print(f"Reference-profiled candidate ZIP ready: {args.output} ({args.output.stat().st_size} bytes).", flush=True)
    print("The ZIP metadata explicitly does not claim SM8735 / POCO F7 or integrated-app validation.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
