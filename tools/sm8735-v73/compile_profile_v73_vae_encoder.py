#!/usr/bin/env python3
"""Compile, link, and profile the Illustrious SDXL VAE encoder on HTP v73.

The Samsung Galaxy S23 is an SM8550 / HTP-v73 reference device. Success here
is not validation of the POCO F7 / SM8735. This script requests the NPU
explicitly, pins QAIRT 2.51 to match Local Dream's current runtime, and saves
the linked context only after the profile reports HTP execution.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

QAIRT_VERSION = "2.51"


def collect_compute_units(value: object) -> list[str]:
    units: set[str] = set()

    def walk(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = "".join(ch for ch in str(key).lower() if ch.isalnum())
                if "computeunit" in normalized:
                    if isinstance(child, (str, int, float)):
                        units.add(str(child))
                    elif isinstance(child, list):
                        units.update(str(entry) for entry in child if isinstance(entry, (str, int, float)))
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return sorted(units)


def confirms_npu_only(units: list[str]) -> bool:
    # Qualcomm AI Hub's profiler labels Qualcomm Hexagon execution as "NPU".
    # Accept that documented label or explicit HTP, but reject mixed CPU/GPU.
    normalized = {unit.upper().replace(" ", "").replace("_", "").replace("-", "") for unit in units}
    return bool(normalized) and normalized.issubset({"NPU", "HTP", "QNNNPU", "QNNHTP"})


def job_state(job: Any) -> str:
    return str(job.get_status()).upper()


def is_success(state: str) -> bool:
    return "SUCCESS" in state or "DONE" in state


def is_failure(state: str) -> bool:
    return any(token in state for token in ("FAILED", "FAILURE", "CANCELLED", "CANCELED", "ERROR"))


def wait_for_job(job: Any, label: str, timeout: int, poll_seconds: int) -> str:
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        state = job_state(job)
        if state != previous:
            print(f"{label} {job.url}: {state}", flush=True)
            previous = state
        if is_success(state) or is_failure(state):
            return state
        time.sleep(max(1, poll_seconds))
    raise TimeoutError(f"Timed out waiting for {label}: {job.url}")


def save_result(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="VAE encoder ONNX directory, including external data")
    parser.add_argument("--device", default="Samsung Galaxy S23")
    parser.add_argument("--project", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/sdxl-export/v73-vae-encoder"))
    parser.add_argument("--wait-timeout", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not (model_dir / "vae_encoder.onnx").is_file() or not (model_dir / "vae_encoder.onnx.data").is_file():
        parser.error("--model-dir must contain vae_encoder.onnx and vae_encoder.onnx.data")
    if any(path.suffix == ".json" for path in model_dir.iterdir() if path.is_file()):
        parser.error("Remove local manifest JSON from the AI Hub upload directory")

    api_token = os.environ.get("QAI_HUB_API_TOKEN")
    if not api_token:
        parser.error("QAI_HUB_API_TOKEN must be set for AI Hub authentication")

    import qai_hub as hub

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "v73_vae_encoder_jobs.json"
    result: dict[str, Any] = json.loads(result_path.read_text()) if result_path.exists() else {}
    result.update({
        "device": args.device,
        "target_claim": "SM8550 / HTP v73 reference only; not SM8735/POCO F7 validation",
        "qairt_version": QAIRT_VERSION,
        "requested_compute_unit": "npu",
        "graph_name": "vae_encoder",
        "input_contract": {"image": [1, 3, 1024, 1024]},
        "output_contract_in_order": {"mean": [1, 4, 128, 128], "std": [1, 4, 128, 128]},
    })
    save_result(result_path, result)

    client = hub.Client(config=hub.ClientConfig(api_token=api_token, default_project=args.project))
    if result.get("uploaded_model_id"):
        model = client.get_model(result["uploaded_model_id"])
        print(f"Reusing uploaded VAE encoder ONNX model {result['uploaded_model_id']}", flush=True)
    else:
        device = hub.Device(args.device)
        archive_name = model_dir.name + ".zip"
        temp_root = Path(tempfile.gettempdir())
        archives_before = set(temp_root.glob(f"tmp*/{archive_name}"))
        print(f"Uploading VAE encoder ONNX to AI Hub for {device.name}", flush=True)
        model = client.upload_model(
            str(model_dir),
            name="Illustrious XL v2.0 SDXL VAE encoder ONNX",
            project=args.project,
        )
        for archive in set(temp_root.glob(f"tmp*/{archive_name}")) - archives_before:
            archive.unlink(missing_ok=True)
            try:
                archive.parent.rmdir()
            except OSError:
                pass
        result["uploaded_model_id"] = getattr(model, "model_id", None)
        if not result["uploaded_model_id"]:
            raise RuntimeError("AI Hub upload did not return a model ID")
        save_result(result_path, result)

    device = hub.Device(args.device)
    input_specs = [{"image": ((1, 3, 1024, 1024), "float32")}]
    existing_compile_urls = result.get("compile_job_urls") or []
    existing_link_url = result.get("link_job_url")
    if existing_compile_urls and existing_link_url:
        print("Resuming recorded VAE encoder compile/link jobs; no duplicate jobs will be submitted.", flush=True)
        compile_jobs = [client.get_job(url.rstrip("/").split("/")[-1]) for url in existing_compile_urls]
        link_job = client.get_job(existing_link_url.rstrip("/").split("/")[-1])
    else:
        compile_jobs, link_job = client.submit_compile_and_link_jobs(
            models=[model],
            device=device,
            name="Illustrious XL v2.0 SDXL VAE encoder HTP v73 reference QAIRT 2.51",
            input_specs=input_specs,
            graph_names=["vae_encoder"],
            compile_options=[f"--qairt_version {QAIRT_VERSION} --compute_unit npu --qnn_options default_graph_htp_precision=FLOAT16"],
            link_options=f"--qairt_version {QAIRT_VERSION} --qnn_options default_graph_htp_optimizations=O=3",
            project=args.project,
        )
        if not link_job:
            raise RuntimeError("AI Hub did not return a VAE encoder LinkJob")
        result["compile_job_urls"] = [job.url for job in compile_jobs]
        result["link_job_url"] = link_job.url
        save_result(result_path, result)

    compile_states = [wait_for_job(job, "VAE encoder compile", args.wait_timeout, args.poll_seconds) for job in compile_jobs]
    link_state = wait_for_job(link_job, "VAE encoder link", args.wait_timeout, args.poll_seconds)
    result["compile_statuses"] = compile_states
    result["link_status"] = link_state
    save_result(result_path, result)
    if not all(is_success(state) for state in compile_states) or not is_success(link_state):
        log_dir = output_dir / "compile_logs"
        log_dir.mkdir(exist_ok=True)
        for job in [*compile_jobs, link_job]:
            try:
                job.download_job_logs(str(log_dir))
            except Exception as exc:
                result.setdefault("log_download_errors", []).append(f"{job.url}: {type(exc).__name__}: {exc}")
        save_result(result_path, result)
        raise RuntimeError("VAE encoder QNN compile/link failed; no inference was submitted")

    linked_model = link_job.get_target_model()
    if linked_model is None:
        raise RuntimeError("VAE encoder LinkJob succeeded but did not return a target model")
    result["target_model_id"] = getattr(linked_model, "model_id", None)
    rng = np.random.default_rng(8735)
    profile_inputs = {"image": [rng.uniform(-1.0, 1.0, (1, 3, 1024, 1024)).astype(np.float32)]}
    if result.get("inference_job_url"):
        inference_job = client.get_job(result["inference_job_url"].rstrip("/").split("/")[-1])
        print("Resuming recorded VAE encoder inference/profile job; no duplicate inference will be submitted.", flush=True)
    else:
        inference_job = client.submit_inference_job(
            model=linked_model,
            device=device,
            inputs=profile_inputs,
            name="Illustrious XL v2.0 SDXL VAE encoder v73 NPU profile",
            options=f"--qairt_version {QAIRT_VERSION} --compute_unit npu --max_profiler_iterations 1",
            profile=True,
            project=args.project,
        )
        result["inference_job_url"] = inference_job.url
        save_result(result_path, result)

    inference_state = wait_for_job(inference_job, "VAE encoder inference", args.wait_timeout, args.poll_seconds)
    result["inference_status"] = inference_state
    if not is_success(inference_state):
        logs = output_dir / "inference_logs"
        logs.mkdir(exist_ok=True)
        try:
            inference_job.download_job_logs(str(logs))
        except Exception as exc:
            result["inference_log_download_error"] = f"{type(exc).__name__}: {exc}"
        save_result(result_path, result)
        raise RuntimeError("VAE encoder inference/profile failed")

    profile_path = output_dir / "inference_profile.json"
    try:
        profile = inference_job.download_profile(filename=str(profile_path))
        if isinstance(profile, dict):
            profile_data = profile
            profile_path.write_text(json.dumps(profile, indent=2, default=str) + "\n", encoding="utf-8")
        elif isinstance(profile, str) and Path(profile).is_file():
            profile_path = Path(profile)
            profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
        else:
            profile_data = None
        if isinstance(profile_data, dict):
            result["profile_path"] = str(profile_path)
            result["profile_compute_units"] = collect_compute_units(profile_data)
            result["htp_execution_verified"] = confirms_npu_only(result["profile_compute_units"])
    except Exception as exc:
        result["profile_download_error"] = f"{type(exc).__name__}: {exc}"

    logs = output_dir / "inference_logs"
    logs.mkdir(exist_ok=True)
    try:
        inference_job.download_job_logs(str(logs))
    except Exception as exc:
        result["inference_log_download_error"] = f"{type(exc).__name__}: {exc}"
    save_result(result_path, result)
    if not result.get("htp_execution_verified", False):
        raise RuntimeError("VAE encoder inference succeeded but the profile did not confirm HTP execution")

    context_path = output_dir / "vae_encoder.bin"
    if context_path.is_file() and context_path.stat().st_size > 0:
        downloaded_context = context_path
    else:
        downloaded = link_job.download_target_model(filename=str(context_path))
        downloaded_context = Path(downloaded) if isinstance(downloaded, (str, Path)) else context_path
    if not downloaded_context.is_file():
        raise RuntimeError(f"HTP profile succeeded but the linked context download is missing: {downloaded_context}")
    result["context_download_path"] = str(downloaded_context)
    result["context_size_bytes"] = downloaded_context.stat().st_size
    save_result(result_path, result)
    print("FINAL_RESULT", json.dumps(result, indent=2), flush=True)
    print("VAE encoder HTP execution verified on the Samsung Galaxy S23 / SM8550 v73 reference only.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
