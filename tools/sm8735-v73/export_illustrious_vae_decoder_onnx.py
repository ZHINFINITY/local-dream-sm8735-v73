#!/usr/bin/env python3
"""Export the matching SDXL VAE decoder for a v73 QNN compile experiment.

The decoder is exported at 1024x1024 (latent input 1x4x128x128) from the same
pinned Diffusers revision as the experimental Illustrious XL U-Net. Its output
is intended for `vae_decoder.bin` in Local Dream; HTP execution is not established
until the AI Hub compile/link and profile steps succeed.
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
    model_path = output_dir / "vae_decoder.onnx"
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

    class VaeDecoder(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, latent_sample):
            # The shared Local Dream pipeline applies SDXL's reciprocal
            # scaling_factor (1 / 0.13025) before calling this decoder.
            return self.model.decode(latent_sample, return_dict=False)[0]

    wrapper = VaeDecoder(vae).eval()
    example = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
    print("Exporting static SDXL VAE decoder: 1x4x128x128 -> 1x3x1024x1024", flush=True)
    torch.onnx.export(
        wrapper,
        (example,),
        str(model_path),
        input_names=["latent_sample"],
        output_names=["sample"],
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
    if input_names != ["latent_sample"] or output_names != ["sample"]:
        raise RuntimeError(f"Unexpected VAE graph I/O: inputs={input_names}, outputs={output_names}")

    manifest = {
        "source_repo": args.repo,
        "source_revision": args.revision,
        "source_component": "vae/diffusion_pytorch_model.bin",
        "source_provenance": "Name-matched Illustrious XL Diffusers mirror; exact identity with the upstream Onoma checkpoint is not independently verified.",
        "input_name": "latent_sample",
        "input_shape": [1, 4, 128, 128],
        "output_name": "sample",
        "output_shape": [1, 3, 1024, 1024],
        "latent_scaling": "Local Dream applies reciprocal SDXL VAE scaling (1/0.13025) before decoder invocation; this graph does not rescale latents.",
        "target_profile": "Compile/link for HTP v73 reference; not validated for SM8735 or POCO F7.",
        "graph_name_for_local_dream": "vae_decoder",
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
