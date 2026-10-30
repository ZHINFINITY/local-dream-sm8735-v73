#!/usr/bin/env python3
"""Resume an existing AI Hub W8A16 job, then compile/link/profile on v73.

The Samsung Galaxy S23 is an SM8550 / HTP-v73 reference target only; this
script does not establish compatibility with the POCO F7 / SM8735.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from qai_hub import public_api_pb2 as api_pb

TERMINAL_SUCCESS = "JOB_STATE_DONE"
TERMINAL_FAILURE = "JOB_STATE_FAILED"


def parse_job_response(response: requests.Response) -> tuple[api_pb.Job, str, Any]:
    response.raise_for_status()
    record = api_pb.Job()
    record.ParseFromString(response.content)
    for descriptor, value in record.ListFields():
        if descriptor.name.endswith("_job") and hasattr(value, "job_state"):
            return record, descriptor.name, value
    raise RuntimeError("AI Hub response did not contain a recognized job record")


def job_id_from_url(url: str) -> str:
    match = re.search(r"/jobs/([A-Za-z0-9]+)/?", url)
    if not match:
        raise ValueError(f"Could not extract AI Hub job id from URL: {url}")
    return match.group(1)


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
    # Qualcomm AI Hub reports Qualcomm Hexagon execution with compute_unit=NPU.
    # Require every reported unit to be NPU/HTP so mixed CPU/GPU runs fail closed.
    normalized = {unit.upper().replace(" ", "").replace("_", "").replace("-", "") for unit in units}
    return bool(normalized) and normalized.issubset({"NPU", "HTP", "QNNNPU", "QNNHTP"})


def compile_options_for_level(level: int) -> list[str]:
    if level not in (1, 2, 3):
        raise ValueError("QNN context binary optimization level must be 1, 2, or 3")
    return [f"--qairt_version 2.51 --qnn_context_binary_optimization_level {level}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantize-job-id", required=True)
    parser.add_argument("--project", default=None)
    parser.add_argument("--device", default="Samsung Galaxy S23")
    parser.add_argument("--output-dir", default="/workspace/sdxl-export/v73-unet-allbiased-allmatmuladd-rescaled-customcal-w8a16")
    parser.add_argument("--wait-timeout", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--qnn-context-binary-optimization-level", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--download-context", action="store_true", help="Also save unet.bin locally; by default the package builder streams it from AI Hub directly into the ZIP")
    args = parser.parse_args()

    api_token = os.environ.get("QAI_HUB_API_TOKEN")
    if not api_token:
        parser.error("QAI_HUB_API_TOKEN must be set for AI Hub authentication")

    import qai_hub as hub

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "pipeline.json"
    result: dict[str, Any] = json.loads(result_path.read_text()) if result_path.exists() else {}
    result.update({
        "quantize_job_id": args.quantize_job_id,
        "quantize_job_url": f"https://workbench.aihub.qualcomm.com/jobs/{args.quantize_job_id}/",
        "device": args.device,
        "target_claim": "SM8550 / HTP v73 reference only; not SM8735/POCO F7 validation",
        "input_contract": {
            "sample": [1, 4, 128, 128],
            "timestep": [1, 2],
            "encoder_hidden_states": [1, 77, 2048],
            "text_embeds": [1, 1280],
            "time_ids": [1, 6],
        },
    })
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"token {api_token}",
        "Content-Type": "application/x-protobuf",
    })
    base_url = "https://workbench.aihub.qualcomm.com/api/v1/jobs/"

    def fetch(job_id: str) -> tuple[api_pb.Job, str, Any]:
        response = session.get(f"{base_url}{job_id}/", timeout=(10, 40))
        return parse_job_response(response)

    def wait_for_jobs(job_ids: list[str], label: str) -> dict[str, str]:
        pending = set(job_ids)
        statuses: dict[str, str] = {}
        last_printed: dict[str, str] = {}
        deadline = time.monotonic() + args.wait_timeout
        while pending and time.monotonic() < deadline:
            for job_id in list(pending):
                try:
                    _, kind, job_record = fetch(job_id)
                    state = api_pb.JobState.Name(job_record.job_state)
                    statuses[job_id] = state
                    if state != last_printed.get(job_id):
                        print(f"{label} {job_id}: {state}", flush=True)
                        last_printed[job_id] = state
                    if state in {TERMINAL_SUCCESS, TERMINAL_FAILURE}:
                        pending.remove(job_id)
                except Exception as exc:
                    print(f"STATUS_RETRY {label} {job_id}: {type(exc).__name__}: {exc}", flush=True)
            if pending:
                time.sleep(max(1, args.poll_seconds))
        if pending:
            raise TimeoutError(f"Timed out waiting for {label}: {sorted(pending)}")
        return statuses

    print(f"Monitoring existing quantization job {args.quantize_job_id}; no upload or requantization will be submitted.", flush=True)
    quant_states = wait_for_jobs([args.quantize_job_id], "quantize")
    quant_state = quant_states[args.quantize_job_id]
    result["quantize_status"] = quant_state
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if quant_state != TERMINAL_SUCCESS:
        try:
            quant_job = hub.Client(config=hub.ClientConfig(api_token=api_token, default_project=args.project)).get_job(args.quantize_job_id)
            log_dir = output_dir / "quantize_logs"
            log_dir.mkdir(exist_ok=True)
            quant_job.download_job_logs(str(log_dir))
        except Exception as exc:
            result["quantize_log_download_error"] = f"{type(exc).__name__}: {exc}"
            result_path.write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError(f"Quantization did not succeed: {quant_state}; compile/profile not submitted")

    # Refresh the completed record to obtain its final target model ID.
    quant_record, _, quant_data = fetch(args.quantize_job_id)
    target_model_id = quant_data.target_model.model_id
    if not target_model_id or not quant_data.target_model.file_upload_complete:
        raise RuntimeError("Quantization says DONE but the target model is not fully available")
    result["quantized_model_id"] = target_model_id
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Quantization succeeded; target model {target_model_id}", flush=True)

    client = hub.Client(config=hub.ClientConfig(api_token=api_token, default_project=args.project))
    qmodel = client.get_model(target_model_id)
    device = hub.Device(args.device)
    input_specs = [{
        "sample": ((1, 4, 128, 128), "float16"),
        "timestep": ((1, 2), "float32"),
        "encoder_hidden_states": ((1, 77, 2048), "float16"),
        "text_embeds": ((1, 1280), "float16"),
        "time_ids": ((1, 6), "float16"),
    }]
    existing_compile_urls = result.get("compile_job_urls") or []
    existing_link_url = result.get("link_job_url")
    if existing_compile_urls and existing_link_url:
        print("Resuming the compile/link jobs recorded in pipeline.json; no duplicate jobs will be submitted.", flush=True)
        compile_jobs = [client.get_job(job_id_from_url(url)) for url in existing_compile_urls]
        link_job = client.get_job(job_id_from_url(existing_link_url))
    else:
        compile_options = compile_options_for_level(args.qnn_context_binary_optimization_level)
        link_options = "--qairt_version 2.51 --qnn_options default_graph_htp_optimizations=O=3"
        compile_jobs, link_job = client.submit_compile_and_link_jobs(
            models=[qmodel],
            device=device,
            name=f"Illustrious XL v2 all-biased W8A16 HTP v73 context-opt-L{args.qnn_context_binary_optimization_level}",
            input_specs=input_specs,
            graph_names=["unet"],
            compile_options=compile_options,
            link_options=link_options,
            project=args.project,
        )
        if not link_job:
            raise RuntimeError("AI Hub did not return a link job")
        result["compile_job_urls"] = [job.url for job in compile_jobs]
        result["link_job_url"] = link_job.url
        result["qnn_context_binary_optimization_level"] = args.qnn_context_binary_optimization_level
        result["compile_options"] = compile_options[0]
        result["link_options"] = link_options
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    compile_ids = [job_id_from_url(job.url) for job in compile_jobs]
    link_id = job_id_from_url(link_job.url)
    compile_states = wait_for_jobs(compile_ids + [link_id], "compile/link")
    result["compile_statuses"] = [compile_states[job_id] for job_id in compile_ids]
    result["link_status"] = compile_states[link_id]
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if any(state != TERMINAL_SUCCESS for state in compile_states.values()):
        log_dir = output_dir / "compile_logs"
        log_dir.mkdir(exist_ok=True)
        for job in [*compile_jobs, link_job]:
            try:
                job.download_job_logs(str(log_dir))
            except Exception as exc:
                result.setdefault("log_download_errors", []).append(f"{job.url}: {type(exc).__name__}: {exc}")
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError("QNN compile/link failed; no HTP inference was submitted")

    linked_model = link_job.get_target_model()
    result["target_model_id"] = getattr(linked_model, "model_id", None)
    rng = np.random.default_rng(8735)
    profile_inputs = {
        "sample": [rng.standard_normal((1, 4, 128, 128)).astype(np.float16)],
        "timestep": [np.array([[500.0, 500.0]], dtype=np.float32)],
        "encoder_hidden_states": [rng.standard_normal((1, 77, 2048)).astype(np.float16)],
        "text_embeds": [rng.standard_normal((1, 1280)).astype(np.float16)],
        "time_ids": [np.array([[1024, 1024, 0, 0, 1024, 1024]], dtype=np.float16)],
    }
    existing_inference_url = result.get("inference_job_url")
    if existing_inference_url:
        print("Resuming the inference/profile job recorded in pipeline.json; no duplicate inference will be submitted.", flush=True)
        inference_job = client.get_job(job_id_from_url(existing_inference_url))
    else:
        inference_job = client.submit_inference_job(
            model=linked_model,
            device=device,
            inputs=profile_inputs,
            name="Illustrious XL v2 all-biased W8A16 HTP v73 inference",
            options="--qairt_version 2.51 --compute_unit npu --max_profiler_iterations 1",
            profile=True,
            project=args.project,
        )
        result["inference_job_url"] = inference_job.url
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    inference_id = job_id_from_url(inference_job.url)
    inference_states = wait_for_jobs([inference_id], "inference")
    result["inference_status"] = inference_states[inference_id]
    if result["inference_status"] != TERMINAL_SUCCESS:
        logs = output_dir / "inference_logs"
        logs.mkdir(exist_ok=True)
        try:
            inference_job.download_job_logs(str(logs))
        except Exception as exc:
            result["inference_log_download_error"] = f"{type(exc).__name__}: {exc}"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError("Reference-device inference/profile failed")

    profile_path = output_dir / "inference_profile.json"
    try:
        profile = inference_job.download_profile(filename=str(profile_path))
        profile_data = None
        if isinstance(profile, dict):
            profile_data = profile
            profile_path.write_text(json.dumps(profile, indent=2, default=str) + "\n")
        elif isinstance(profile, str) and Path(profile).is_file():
            profile_path = Path(profile)
            profile_data = json.loads(profile_path.read_text())
        if isinstance(profile_data, dict):
            units = collect_compute_units(profile_data)
            result["profile_path"] = str(profile_path)
            result["profile_compute_units"] = units
            result["htp_execution_verified"] = confirms_npu_only(units)
    except Exception as exc:
        result["profile_download_error"] = f"{type(exc).__name__}: {exc}"
    logs = output_dir / "inference_logs"
    logs.mkdir(exist_ok=True)
    try:
        inference_job.download_job_logs(str(logs))
    except Exception as exc:
        result["inference_log_download_error"] = f"{type(exc).__name__}: {exc}"
    if not result.get("htp_execution_verified", False):
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError("Inference succeeded but the downloaded profile did not confirm HTP compute-unit execution")
    if args.download_context:
        context_path = output_dir / "unet.bin"
        if context_path.is_file() and context_path.stat().st_size > 0:
            downloaded_context = context_path
        else:
            downloaded = link_job.download_target_model(filename=str(context_path))
            downloaded_context = Path(downloaded) if isinstance(downloaded, (str, Path)) else context_path
        if not downloaded_context.is_file():
            raise RuntimeError(f"AI Hub reported NPU execution but the linked context download is missing: {downloaded_context}")
        result["context_download_path"] = str(downloaded_context)
        result["context_size_bytes"] = downloaded_context.stat().st_size
    else:
        result["context_download_skipped_for_streaming"] = True
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print("FINAL_RESULT", json.dumps(result, indent=2), flush=True)
    print("NPU-only execution verified on the Samsung Galaxy S23 / SM8550 v73 reference only.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
