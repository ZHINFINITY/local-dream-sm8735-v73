#!/usr/bin/env python3
"""Export the matching SDXL VAE encoder for a v73 QNN compile experiment.

Local Dream's QNN SDXL runtime consumes one image input and two outputs in
order: posterior mean, then posterior standard deviation. The graph is static
for 1024x1024 images (1x3x1024x1024 -> two 1x4x128x128 tensors).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_REPO = "NyuromonX/Illustrious-XL-v2.0-Diffusers-fp16"
DEFAULT_REVISION = "c3cd0edb4c4ebfc6c86011039fb2e81b865b5d85"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", required=True, help="Output directory ending in .onnx")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    if not output_dir.name.endswith(".onnx"):
        parser.error("--output directory name must end with .onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "vae_encoder.onnx"
    if model_path.exists():
        parser.error(f"refusing to overwrite existing output: {model_path}")

    import torch
    import onnx
    from diffusers import AutoencoderKL
    from diffusers.models.attention_processor import AttnProcessor
    from huggingface_hub import hf_hub_download

    torch.set_num_threads(max(1, args.threads))
    torch.set_grad_enabled(False)
    config_path = hf_hub_download(args.repo, "vae/config.json", revision=args.revision)
    hf_hub_download(args.repo, "vae/diffusion_pytorch_model.bin", revision=args.revision)
    print(f"Loading SDXL VAE from {args.repo}@{args.revision}", flush=True)
    vae = AutoencoderKL.from_pretrained(
        Path(config_path).parent,
        torch_dtype=torch.float32,
        use_safetensors=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    vae.eval()
    vae.set_attn_processor(AttnProcessor())

    class VaeEncoder(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, image):
            posterior = self.model.encode(image, return_dict=False)[0]
            return posterior.mean, posterior.std

    wrapper = VaeEncoder(vae).eval()
    example = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)
    print("Exporting static SDXL VAE encoder: 1x3x1024x1024 -> mean/std 1x4x128x128", flush=True)
    torch.onnx.export(
        wrapper,
        (example,),
        str(model_path),
        input_names=["image"],
        output_names=["mean", "std"],
        opset_version=18,
        export_params=True,
        external_data=True,
        dynamo=True,
        optimize=False,
        training=torch.onnx.TrainingMode.EVAL,
    )
    onnx.checker.check_model(str(model_path), full_check=False)
    model = onnx.load(str(model_path), load_external_data=False)
    input_names = [value.name for value in model.graph.input]
    output_names = [value.name for value in model.graph.output]
    if input_names != ["image"] or output_names != ["mean", "std"]:
        raise RuntimeError(f"Unexpected VAE encoder graph I/O: inputs={input_names}, outputs={output_names}")

    manifest = {
        "source_repo": args.repo,
        "source_revision": args.revision,
        "source_component": "vae/diffusion_pytorch_model.bin",
        "source_provenance": "Name-matched Illustrious XL Diffusers mirror; exact identity with the upstream Onoma checkpoint is not independently verified.",
        "input_name": "image",
        "input_shape": [1, 3, 1024, 1024],
        "output_names_in_order": ["mean", "std"],
        "output_shapes": [[1, 4, 128, 128], [1, 4, 128, 128]],
        "local_dream_contract": "QnnModel::executeVaeEncoderGraphsSDXL copies one input and reads outputs[0] as mean, outputs[1] as std.",
        "target_profile": "Compile/link for HTP v73 reference; not validated for SM8735 or POCO F7.",
        "graph_name_for_local_dream": "vae_encoder",
    }
    manifest_path = output_dir.parent / (output_dir.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Export complete: {model_path}; manifest: {manifest_path}", flush=True)
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"{path.name}: {path.stat().st_size} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
