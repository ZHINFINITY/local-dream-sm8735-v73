# Experimental SDXL QNN v73 candidate for POCO F7 evaluation

**Status: conversion scaffolding only. No model binary has been generated or tested on the POCO F7.** This folder prepares an experimental conversion attempt; it does not establish SM8735 support.

The intended target is the POCO F7’s Snapdragon 8s Gen 4 (SM8735), whose device panel reports Hexagon HTP v73 and 8 MB VTCM. The helper uses Local Dream’s official SDXL conversion scripts and creates an **HTP v73 / 8 MB VTCM** conversion profile. Because Qualcomm’s published QNN device table does not list SM8735, the default `soc_model=43` is the known SM8550 / Snapdragon 8 Gen 2 reference profile. A successful conversion with that profile would only be a same-HTP-generation experiment—not an SM8735-specific compile or a guarantee that it runs on the POCO F7.

This deliberately does **not** add SM8735 to Local Dream’s built-in SDXL allowlist. The app’s built-in SDXL packages are compiled for the 8gen3/v75 tier, and relabelling those binaries as v73 would be unsafe. Local Dream already recognizes a custom SDXL model folder/package by its `SDXL` marker; the custom NPU package contract also needs `tokenizer.json`, `clip.mnn`, `clip_2.mnn`, the two CLIP embedding/token files, `unet.bin`, and the VAE QNN binaries. See the [native package contract](https://github.com/xororz/local-dream/blob/master/app/src/main/cpp/src/main.cpp) and [custom model scanner](https://github.com/xororz/local-dream/blob/master/app/src/main/java/io/github/xororz/localdream/data/Model.kt).

## What the helper does

[`build_sdxl_v73_candidate.sh`](./build_sdxl_v73_candidate.sh) downloads the official Local Dream SDXL converter archive, prepares `dsp_arch: v73`, `vtcm_mb: 8`, and a configurable `soc_model`, then runs the upstream model preparation, ONNX export, quantization, and QNN context-binary conversion. It validates that the resulting archive contains the files Local Dream expects and labels it `EXPERIMENTAL_UNVALIDATED`.

The input must be the **original SDXL checkpoint** (for example, the `.safetensors` checkpoint for Illustrious XL v2), not the MNN package from Nezumi AI. The converter is currently pinned by Local Dream’s guide to Qualcomm AI Engine Direct SDK 2.28.

## Build prerequisites

Use Linux or WSL with the Qualcomm AI Engine Direct SDK 2.28 installed, `uv`, Python 3.10.17 support, the original SDXL checkpoint, and the account/software access needed to obtain Qualcomm’s SDK. Local Dream’s [official guide](https://ld-guide.chino.icu/conversion/sdxl) calls the conversion experimental and recommends about 64 GB RAM plus swap and 100 GB or more disk for high-resolution work. The current Cue sandbox has about 23 GiB RAM and 14 GB free disk, with no QNN converter or `QNN_SDK_ROOT`, so the model build was not run here.

## Run on a suitable Linux/WSL build host

```bash
MODEL_PATH=/absolute/path/Illustrious-XL-v2.0.safetensors \
MODEL_NAME=illustrious_xl_v2 \
QNN_SDK_ROOT=/absolute/path/to/qairt/2.28.0.241029 \
BUILD_ROOT=/absolute/path/to/a/location/with/100GB-free \
bash tools/sm8735-v73/build_sdxl_v73_candidate.sh
```

The default `SOC_MODEL=43` is Qualcomm’s listed SM8550 reference profile, not an SM8735 identifier. Do not substitute a guessed SoC number. If Qualcomm later publishes an SM8735 `soc_model`, pass that documented value explicitly; otherwise keep the candidate labelled as an unvalidated v73 experiment.

The output is named like `illustrious_xl_v2_experimental_v73_soc43_candidate.zip`. It is formatted for Local Dream custom-model import, but must first be tested on the actual POCO F7. A same-v73 build from another SoC does not prove SM8735 compatibility. Qualcomm’s [HTP documentation](https://docs.qualcomm.com/nav/home/htp_backend.html?product=1601111740009302) warns that context binaries compiled for a different HTP architecture can yield indeterminate results; the SM8735’s exact supported SoC target and driver behavior still need on-device validation.

## Memory-bandwidth note

Qualcomm’s [Snapdragon 8s Gen 4 brief](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Product-Brief-Snapdragon-8s-Gen-4.pdf) and [Snapdragon 8 Gen 3 brief](https://docs.qualcomm.com/doc/87-71408-1/87-71408-1_REV_G_Snapdragon_8_gen_3_Mobile_Platform_Product_Brief.pdf) both list LP-DDR5x up to 4800 MHz. Qualcomm describes improved shared-memory bandwidth relative to the previous 8s Gen 3, but its published briefs do not show a higher maximum memory interface speed than 8 Gen 3. In any case, memory bandwidth does not replace matching the HTP target and validating the compiled context.


## Local checks

Run `python3 -m unittest discover -s tools/sm8735-v73 -p 'test_*.py' -v` to verify the v73/8 MB profile generation and candidate ZIP contents. These tests do not run the Qualcomm converter or validate inference on an NPU.
