#!/usr/bin/env python3
"""Compile an SDXL U-Net to a v73 QNN context and profile it on real hardware.

The Samsung Galaxy S23 is an SM8550/HTP-v73 reference target. A successful
run does not establish compatibility with the POCO F7 / SM8735.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="AI Hub .onnx directory with one graph and its external weights")
    parser.add_argument("--device", default="Samsung Galaxy S23")
    parser.add_argument("--project", default=None, help="Optional AI Hub project id or name")
    parser.add_argument("--output-dir", default="/workspace/sdxl-export/v73-unet-test")
    parser.add_argument("--wait-timeout", type=int, default=7200)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir() or not model_dir.name.endswith(".onnx"):
        parser.error("--model-dir must be an existing directory ending in .onnx")
    model_files = sorted(p.name for p in model_dir.iterdir() if p.is_file())
    if not any(name.endswith(".onnx") for name in model_files):
        parser.error("model directory has no .onnx graph")
    if any(name.endswith(".json") for name in model_files):
        parser.error("remove the local manifest JSON from the AI Hub upload directory")

    import qai_hub as hub

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = hub.Device(args.device)
    input_specs = [{
        "sample": ((1, 4, 128, 128), "float16"),
        "timestep": ((1, 2), "float32"),
        "encoder_hidden_states": ((1, 77, 2048), "float16"),
        "text_embeds": ((1, 1280), "float16"),
        "time_ids": ((1, 6), "float16"),
    }]
    api_token = os.environ.get("QAI_HUB_API_TOKEN")
    if not api_token:
        parser.error("QAI_HUB_API_TOKEN must be set for AI Hub authentication")
    client = hub.Client(config=hub.ClientConfig(api_token=api_token, default_project=args.project))
    archive_name = model_dir.name + ".zip"
    temp_root = Path(tempfile.gettempdir())
    archive_glob = f"tmp*/{archive_name}"
    archives_before = set(temp_root.glob(archive_glob))
    print(f"Uploading the ONNX directory to AI Hub for {device.name}", flush=True)
    uploaded_model = client.upload_model(
        str(model_dir),
        name="Illustrious XL v2.0 SDXL U-Net FP16 ONNX dtype-castfix",
        project=args.project,
    )
    uploaded_model_id = getattr(uploaded_model, "model_id", None)
    for archive in set(temp_root.glob(archive_glob)) - archives_before:
        print(f"Removing temporary upload archive {archive}", flush=True)
        archive.unlink(missing_ok=True)
        try:
            archive.parent.rmdir()
        except OSError:
            pass
    print(f"Uploaded model id: {uploaded_model_id}", flush=True)
    print(f"Submitting QNN HTP compile/link for {device.name}", flush=True)
    compile_jobs, link_job = client.submit_compile_and_link_jobs(
        models=[uploaded_model],
        device=device,
        name="Illustrious XL v2.0 SDXL U-Net v73 reference dtype-castfix",
        input_specs=input_specs,
        graph_names=["unet"],
        compile_options=["--qairt_version 2.51 --qnn_options default_graph_htp_precision=FLOAT16"],
        link_options="--qairt_version 2.51 --qnn_options default_graph_htp_optimizations=O=3",
        project=args.project,
    )
    result = {
        "device": device.name,
        "project": args.project,
        "target_claim": "SM8550 / HTP v73 reference only; not SM8735/POCO F7 validation",
        "uploaded_model_id": uploaded_model_id,
        "compile_job_urls": [job.url for job in compile_jobs],
        "link_job_url": link_job.url if link_job else None,
    }
    (output_dir / "compile_jobs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not link_job:
        raise RuntimeError("AI Hub did not return a LinkJob")

    for job in compile_jobs:
        status = job.wait(timeout=args.wait_timeout)
        print(f"compile {job.url}: {status}", flush=True)
    link_status = link_job.wait(timeout=args.wait_timeout)
    print(f"link {link_job.url}: {link_status}", flush=True)
    result["compile_statuses"] = [str(job.get_status()) for job in compile_jobs]
    result["link_status"] = str(link_job.get_status())
    if "SUCCESS" not in result["link_status"].upper():
        log_dir = output_dir / "compile_logs"
        log_dir.mkdir(exist_ok=True)
        for job in compile_jobs:
            job.download_job_logs(str(log_dir))
        link_job.download_job_logs(str(log_dir))
        (output_dir / "compile_jobs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError("QNN compile/link did not succeed; see compile_jobs.json and compile_logs")

    result["target_model_id"] = getattr(link_job.get_target_model(), "model_id", None)
    print("Submitting profiled inference on the reference device NPU", flush=True)
    rng = np.random.default_rng(8735)
    profile_inputs = {
        "sample": [rng.standard_normal((1, 4, 128, 128)).astype(np.float16)],
        "timestep": [np.array([[500.0, 500.0]], dtype=np.float32)],
        "encoder_hidden_states": [rng.standard_normal((1, 77, 2048)).astype(np.float16)],
        "text_embeds": [rng.standard_normal((1, 1280)).astype(np.float16)],
        "time_ids": [np.array([[1024, 1024, 0, 0, 1024, 1024]], dtype=np.float16)],
    }
    inference_job = client.submit_inference_job(
        model=link_job.get_target_model(),
        device=device,
        inputs=profile_inputs,
        name="Illustrious XL v2.0 SDXL U-Net v73 HTP dtype-castfix inference",
        options="--qairt_version 2.51 --compute_unit npu --max_profiler_iterations 1",
        profile=True,
    )
    result["inference_job_url"] = inference_job.url
    (output_dir / "compile_jobs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    inference_status = inference_job.wait(timeout=args.wait_timeout)
    result["inference_status"] = str(inference_status)
    profile_path = output_dir / "inference_profile.json"
    profile = inference_job.download_profile(filename=str(profile_path))
    profile_data = None
    if isinstance(profile, str) and Path(profile).is_file():
        result["profile_path"] = profile
        profile_data = json.loads(Path(profile).read_text(encoding="utf-8"))
    elif isinstance(profile, dict):
        profile_data = profile
        profile_path.write_text(json.dumps(profile, indent=2, default=str) + "\n", encoding="utf-8")
        result["profile_path"] = str(profile_path)
    if isinstance(profile_data, dict):
        result["profile_keys"] = list(profile_data.keys())
        units = collect_compute_units(profile_data)
        result["profile_compute_units"] = units
        result["htp_execution_verified"] = confirms_npu_only(units)
    logs = output_dir / "inference_logs"
    logs.mkdir(exist_ok=True)
    inference_job.download_job_logs(str(logs))
    (output_dir / "compile_jobs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if "SUCCESS" not in result["inference_status"].upper():
        raise RuntimeError("QNN inference/profile failed; see inference logs and profile")
    if not result.get("htp_execution_verified", False):
        raise RuntimeError("Inference completed but profile did not confirm HTP compute-unit execution")
    print("Inference/profile completed with HTP compute-unit evidence.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
