#!/usr/bin/env python3
"""Experimental static-shape SDXL U-Net ONNX export using torch.export/Dynamo.

This avoids a full numerical forward pass during legacy TorchScript tracing where
possible. The resulting graph is intended for 1024x1024 SDXL latent input and
must still be compiled/profiled on a v73 NPU before any compatibility claim.
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
    model_path = output_dir / "unet.onnx"
    if model_path.exists():
        parser.error(f"refusing to overwrite existing output: {model_path}")

    import torch
    import onnx
    from diffusers import UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor
    from huggingface_hub import hf_hub_download

    torch.set_num_threads(max(1, args.threads))
    torch.set_grad_enabled(False)
    config_path = hf_hub_download(args.repo, "unet/config.json", revision=args.revision)
    hf_hub_download(args.repo, "unet/diffusion_pytorch_model.bin", revision=args.revision)
    print(f"Loading SDXL U-Net from {args.repo}@{args.revision}", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        Path(config_path).parent,
        torch_dtype=torch.float16,
        use_safetensors=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    unet.eval()
    unet.set_attn_processor(AttnProcessor())

    class LocalDreamUnet(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            # SDXL's time projection is FP32; promote pooled conditioning before
            # concat so ONNX never sees mixed FP16/FP32 Concat inputs.
            text_embeds = text_embeds.to(dtype=torch.float32)
            time_ids = time_ids.to(dtype=torch.float32)
            return self.model(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                return_dict=False,
            )[0]

    wrapper = LocalDreamUnet(unet).eval()
    example_inputs = (
        torch.zeros((1, 4, 128, 128), dtype=torch.float16),
        torch.tensor(1.0, dtype=torch.float32),
        torch.zeros((1, 77, 2048), dtype=torch.float16),
        torch.zeros((1, 1280), dtype=torch.float16),
        torch.zeros((1, 6), dtype=torch.float16),
    )
    print("Exporting static 128x128 latent graph using torch.onnx.export(dynamo=True)", flush=True)
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(model_path),
        input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=["out_sample"],
        opset_version=18,
        export_params=True,
        external_data=True,
        dynamo=True,
        optimize=False,
        training=torch.onnx.TrainingMode.EVAL,
    )
    # Pass a path so ONNX resolves external tensor data beside the graph file.
    onnx.checker.check_model(str(model_path), full_check=False)
    metadata = {
        "source_repo": args.repo,
        "source_revision": args.revision,
        "source_component": "unet/diffusion_pytorch_model.bin",
        "source_provenance": "The model card has no source-checkpoint statement; weight identity is unverified.",
        "input_latent_shape": [1, 4, 128, 128],
        "output_latent_shape": [1, 4, 128, 128],
        "intended_image_size": "1024x1024 SDXL",
        "target_profile": "Compile/link for a v73 reference device; not validated for SM8735 or POCO F7.",
        "precision": "FP16 weights and inputs; no weight quantization applied.",
        "conditioning_cast": "text_embeds and time_ids are promoted to FP32 before SDXL conditioning concat.",
        "graph_name_for_local_dream": "unet",
    }
    manifest = output_dir.parent / (output_dir.name + ".manifest.json")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Export complete: {model_path}; manifest: {manifest}", flush=True)
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"{path.name}: {path.stat().st_size} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
