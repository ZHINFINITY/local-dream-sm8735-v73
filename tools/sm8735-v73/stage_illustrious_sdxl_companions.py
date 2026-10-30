#!/usr/bin/env python3
"""Range-extract only the published Illustrious SDXL CLIP assets for Local Dream.

This deliberately does not download the 4.1 GB MNN archive or its MNN U-Net/VAE.
It uses ZIP byte ranges and writes the two MNN text encoders plus tokenizer and
embedding files under the exact filenames expected by Local Dream's SDXL loader.
Requires the small Python package `remotezip`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

REPO = "Mouserat/Illustrious-XL-v2.0-diffusers-mnn"
REVISION = "922516ef832c1559e6e4e144945152eebad8218b"
DIFFUSERS_REVISION = "c3cd0edb4c4ebfc6c86011039fb2e81b865b5d85"
ARCHIVE = "Illustrious-XL-v2.0-diffusers-mnn-unet-int8-block32-asym-clipvae-int4.zip"
ASSET_MAP = {
    "clip1.mnn": "clip.mnn",
    "clip1.mnn.weight": "clip.mnn.weight",
    "clip2.mnn": "clip_2.mnn",
    "clip2.mnn.weight": "clip_2.mnn.weight",
    "pos_emb1.bin": "pos_emb.bin",
    "token_emb1.bin": "token_emb.bin",
    "pos_emb2.bin": "pos_emb_2.bin",
    "token_emb2.bin": "token_emb_2.bin",
    "tokenizer.json": "tokenizer.json",
    "tokenizer_2.json": "tokenizer_2.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Local Dream model directory")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise SystemExit("Install the small range reader first: python3 -m pip install remotezip") from exc

    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{ARCHIVE}"
    with RemoteZip(url) as archive:
        by_basename = {}
        for info in archive.infolist():
            basename = PurePosixPath(info.filename).name
            if basename in ASSET_MAP:
                if basename in by_basename:
                    raise RuntimeError(f"Archive has duplicate basename {basename!r}; select an exact member path")
                by_basename[basename] = info
        missing = sorted(set(ASSET_MAP) - set(by_basename))
        if missing:
            raise RuntimeError("Pinned MNN archive is missing required assets: " + ", ".join(missing))

        manifest = {
            "source_repo": REPO,
            "source_revision": REVISION,
            "source_archive": ARCHIVE,
            "source_archive_url": url,
            "extraction_method": "HTTP byte-range ZIP member reads; the 4.1 GB archive is not downloaded as a whole.",
            "source_note": "The model card identifies this as an MNN conversion of OnomaAIResearch/Illustrious-XL-v2.0 and declares OpenRAIL-M with attribution required. The MNN CLIP assets are companions; the QNN U-Net/VAE contexts come from the pinned Diffusers mirror and are experimental.",
            "runtime_compatibility_note": "The published CLIP-G graph expects input_embedding and eos_index inputs and returns last_hidden_state and text_embeds; Local Dream needs the forked PipelineSdxl.hpp compatibility branch. The unmodified app has not been verified with these assets.",
            "files": {},
        }
        buffer_size = 8 * 1024 * 1024
        for source_name, target_name in ASSET_MAP.items():
            info = by_basename[source_name]
            target = output / target_name
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite {target}")
            print(f"Extracting {info.filename} -> {target_name} ({info.file_size} bytes)", flush=True)
            digest = hashlib.sha256()
            written = 0
            with archive.open(info.filename) as source, target.open("wb") as destination:
                while True:
                    chunk = source.read(buffer_size)
                    if not chunk:
                        break
                    destination.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if written and written % (256 * 1024 * 1024) < len(chunk):
                        print(f"  {target_name}: {written}/{info.file_size} bytes", flush=True)
            if written != info.file_size:
                target.unlink(missing_ok=True)
                raise IOError(f"Size mismatch for {target_name}: {written} != {info.file_size}")
            manifest["files"][target_name] = {
                "archive_member": info.filename,
                "size_bytes": written,
                "sha256": digest.hexdigest(),
            }

    manifest_path = output / "MNN_COMPANIONS_SOURCE.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    source_note = output / "MODEL_SOURCE.md"
    source_note.write_text(
        "# Illustrious XL v2.0 v73 candidate: model sources and limits\n\n"
        "The CLIP/MNN companion files were extracted from the pinned Mouserat archive listed in `MNN_COMPANIONS_SOURCE.json`. "
        "Its model card identifies the source as OnomaAIResearch/Illustrious-XL-v2.0 and declares OpenRAIL-M with attribution required. "
        "See the [Mouserat model card](https://huggingface.co/Mouserat/Illustrious-XL-v2.0-diffusers-mnn/blob/main/README.md) and the [upstream checkpoint](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0).\n\n"
        f"The experimental QNN U-Net and VAE components are exported from the name-matched Diffusers mirror pinned at `{DIFFUSERS_REVISION}` of "
        "[NyuromonX/Illustrious-XL-v2.0-Diffusers-fp16](https://huggingface.co/NyuromonX/Illustrious-XL-v2.0-Diffusers-fp16). The mirror is not independently verified as bit-identical to the upstream checkpoint.\n\n"
        "**App compatibility:** the published CLIP-G MNN graph expects `input_embedding` and `eos_index` inputs and returns `last_hidden_state` and `text_embeds`. This package requires a Local Dream build containing the fork's `PipelineSdxl.hpp` compatibility branch; the unmodified app has not been verified with these assets.\n\n"
        "**Hardware status:** the candidate targets 1024x1024 SDXL on HTP v73 with 8 MB VTCM. Any AI Hub compile/profile on the Samsung Galaxy S23 is an SM8550 reference result only. This package is not validated on the POCO F7 / SM8735; a matching v73 label alone does not prove a cross-SoC context will run.\n",
        encoding="utf-8",
    )
    print(f"Wrote companion manifest {manifest_path}", flush=True)
    print(f"Wrote model source note {source_note}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
