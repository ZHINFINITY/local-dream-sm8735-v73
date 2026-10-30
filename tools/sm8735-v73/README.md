# Experimental Illustrious SDXL candidate for HTP v73

**This is an experimental conversion, not a POCO F7-validated model.** The user-provided POCO F7 diagnostic reports HTP v73 and 8 MB VTCM, but Qualcomm’s published device list does not identify SM8735 as an AI Hub target. The Samsung Galaxy S23 / SM8550 / SoC model 43 is being used only as an HTP-v73 reference. A context built or profiled on that reference does not prove it will run safely on the POCO F7.

## Current state

The Illustrious-name-matched SDXL U-Net was exported to ONNX at the static 128×128 latent size (1024×1024 output equivalent). The graph was adjusted for AI Hub’s FP16 scale-floor constraint and the custom W8A16 quantization job `jpy2qd6rg` completed successfully. Calibration used only three synthetic diagnostic tensors, so this is not accuracy-validated and a successful NPU profile would prove execution, not image quality. Its reference-device QNN compile/link is still in progress; no successful U-Net HTP inference has yet been confirmed.

Matching VAE decoder and encoder ONNX graphs were also exported from the same pinned Diffusers mirror and passed full ONNX validation. On the SM8550 Galaxy S23 v73 reference, decoder compile/link/profile succeeded (`jp37xmxzg`, `jgowovodp`, `jp8r6748g`) and all 161 profile entries reported `compute_unit=NPU`; encoder compile/link/profile succeeded (`jp11787lp`, `jgd48v8lg`, `jgk0oy9o5`) and all 128 entries reported `compute_unit=NPU`. Their linked context binaries were downloaded (decoder 165,076,992 bytes; encoder 103,813,120 bytes). This is NPU execution on the reference, not SM8735/POCO F7 validation.

The pinned MNN companion archive’s CLIP, tokenizer, and embedding files have been range-extracted into a candidate model folder. Both CLIP graphs were opened and run once with their sidecars using host MNN 3.6.2: CLIP-L accepted float32 `[1,77,768]` and returned finite `[1,77,768]`; CLIP-G accepted float32 `[1,77,1280]` plus int32 `eos_index [1]` and returned finite `last_hidden_state [1,77,1280]` and `text_embeds [1,1280]`. These are graph-level CPU checks with test inputs, not app integration or NPU evidence. The candidate ZIP must not be described as ready until the U-Net QNN context is present and its profile confirms NPU execution. Even then, on-device import and generation on the POCO F7 remain untested.

There is an additional app-code compatibility requirement: the archive’s CLIP-G MNN graph expects `input_embedding` plus an int32 `eos_index`, and returns `last_hidden_state` plus an already EOS-selected/projected `text_embeds` output. Local Dream’s previous loader expected a per-token `pooled_output` instead. The fork’s `PipelineSdxl.hpp` now supports both interfaces, but that C++ change has not yet been built into an APK: this sandbox has no Android SDK/NDK or the QNN SDK 2.51 required by the native CMake build. The MNN submodule is initialized, but the remaining native submodules are not. The package therefore requires a build containing the fork patch; do not assume the unmodified app can load it.

## Model sources and limits

The U-Net and VAE exporters use `NyuromonX/Illustrious-XL-v2.0-Diffusers-fp16` at revision `c3cd0edb4c4ebfc6c86011039fb2e81b865b5d85`. This is a name-matched mirror, not an independently verified bit-for-bit copy of the upstream [OnomaAIResearch Illustrious XL v2.0 checkpoint](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0).

Local Dream’s SDXL text-encoder companions come from the pinned [Mouserat MNN conversion](https://huggingface.co/Mouserat/Illustrious-XL-v2.0-diffusers-mnn), which its model card identifies as converted from that Onoma checkpoint. The card declares OpenRAIL-M and attribution requirements; see the included `MODEL_SOURCE.md` and `MNN_COMPANIONS_SOURCE.json` in a staged package. The model identity across these separate sources has not been independently proven.

## Reproducible component flow

`export_illustrious_unet_onnx_dynamo.py` exports the SDXL U-Net. `rescale_fp16_biased_ops.py` adjusts direct biased ops and fusable MatMul-plus-Add pairs to address AI Hub W8A16’s FP16 activation-scale floor. The accepted W8A16 quantization job is resumed with `continue_v73_quantize_job.py`; that script reuses saved job IDs rather than submitting duplicate compile, link, or inference jobs.

The VAE components are exported with `export_illustrious_vae_encoder_onnx.py` and `export_illustrious_vae_decoder_onnx.py`. `compile_profile_v73_vae_encoder.py` and `compile_profile_v73_vae_decoder.py` pin QAIRT 2.51, explicitly request NPU compile/inference, wait for successful linking, require all profile compute-unit labels to be NPU/HTP, and only then download a context binary. AI Hub’s Qualcomm reference profile labels HTP placement as `NPU`. The encoder’s graph contract is one `[1,3,1024,1024]` input and outputs ordered as mean then standard deviation, each `[1,4,128,128]`, matching Local Dream’s positional QNN binding.

`stage_illustrious_sdxl_companions.py` extracts only the CLIP models, MNN weight sidecars, embeddings, and tokenizer files using HTTP byte ranges. It avoids downloading the archive’s MNN U-Net/VAE. A staged Local Dream model folder has the `SDXL` marker plus `clip.mnn`, `clip.mnn.weight`, `clip_2.mnn`, `clip_2.mnn.weight`, the two embedding/token pairs, `unet.bin`, `vae_encoder.bin`, and `vae_decoder.bin`. The encoder is included because Local Dream’s default img2img preference requires it.

Once all three reference profiles have confirmed HTP, the package builder can fetch the completed U-Net context directly into memory and stream it into the final ZIP, avoiding another multi-gigabyte disk copy. Example command (replace the job IDs with the successful recorded IDs; pass each `--*-htp-profiled-sm8550` flag only after that component’s profile explicitly confirms HTP):

```bash
QAI_HUB_API_TOKEN=... python3 tools/sm8735-v73/build_v73_package.py package \
  --package-dir /path/to/local-dream-model \
  --output /path/to/illustrious_xl_v2_experimental_v73_soc43_candidate.zip \
  --soc-model 43 --project <ai-hub-project-id> \
  --unet-link-job-id <successful-unet-link-job-id> \
  --vae-encoder-link-job-id <successful-encoder-link-job-id> \
  --vae-decoder-link-job-id <successful-decoder-link-job-id> \
  --unet-htp-profiled-sm8550 \
  --vae-encoder-htp-profiled-sm8550 \
  --vae-decoder-htp-profiled-sm8550
```

`--consume-source-files` is optional and destructive: it removes local staged package files as they are written into the ZIP. Do not use it until the ZIP has been verified or the source assets can be recovered.

The package manifest always marks `sm8735_poco_f7_validated` as false. Only profile flags backed by actual HTP reports should be supplied. A successful S23 profile proves HTP execution on that reference, not compatibility with the POCO F7.

## Qualcomm target caution

The reference profile uses `dsp_arch=v73`, 8 MB VTCM, and `soc_model=43`, which Qualcomm lists for SM8550. Do not substitute a guessed SM8735 SoC ID. Qualcomm’s [HTP documentation](https://docs.qualcomm.com/nav/home/htp_backend.html?product=1601111740009302) warns that loading a context binary compiled for a different HTP architecture can produce indeterminate results; a same-v73 label is not evidence that a cross-SoC context is supported. The only definitive completion of the user’s requirement is a successful native QNN/HTP inference and image generation on the actual POCO F7 / SM8735.

## Local checks

Run `python3 -m unittest discover -s tools/sm8735-v73 -p 'test_*.py' -v` to test v73 profile generation, package validation, and in-memory U-Net context packaging. These unit tests do not validate Qualcomm compilation or device inference.
