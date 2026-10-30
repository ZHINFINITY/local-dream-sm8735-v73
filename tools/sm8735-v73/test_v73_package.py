import hashlib
import json
import pathlib
import tempfile
import unittest
import zipfile

from build_v73_package import (
    PACKAGE_PREFIX,
    REQUIRED_FILES,
    _collect_compute_units,
    _confirms_npu_only,
    _link_job_matches_compile_id,
    package_candidate,
    prepare_v73_profile,
)


class V73ProfileTests(unittest.TestCase):
    def test_link_job_must_reference_the_declared_compile_job(self):
        matching = type("Model", (), {"name": "job_compile123_optimized_dlc"})()
        unrelated = type("Model", (), {"name": "job_other456_optimized_dlc"})()
        link_job = type("LinkJob", (), {"models": [matching]})()

        self.assertTrue(_link_job_matches_compile_id(link_job, "compile123"))
        link_job.models = [unrelated]
        self.assertFalse(_link_job_matches_compile_id(link_job, "compile123"))

    def test_profile_parser_rejects_any_non_npu_compute_unit(self):
        report = {"layer_profile": [{"compute_unit": "NPU"}, {"compute_unit": ["HTP"]}]}
        units = _collect_compute_units(report)
        self.assertEqual(units, ["HTP", "NPU"])
        self.assertTrue(_confirms_npu_only(units))
        self.assertFalse(_confirms_npu_only(["NPU", "CPU"]))

    def test_profile_targets_v73_with_8mb_vtcm_and_sm8550_reference_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            scripts = root / "scripts"
            scripts.mkdir()
            export = root / "export_sdxl.sh"
            export.write_text(
                "model_path=~/Downloads/anythingxl.safetensors # Path\n"
                "model_name=anythingxl # Name\n"
                'soc_versions=("8gen3")\n'
                "python prepare_data_sdxl.py --model_path $model_path\n"
                "python export_onnx_sdxl.py --model_path $model_path\n"
            )
            convert = scripts / "convert_all_sdxl.sh"
            convert.write_text("QNN_SDK_ROOT=/data/qairt/2.28.0.241029\ncd $QNN_SDK_ROOT/bin\n")

            prepare_v73_profile(root, 43)
            config = json.loads((root / "htp_config_8gen2.json").read_text())
            backend = json.loads((root / "htp_backend_8gen2.json").read_text())
            self.assertEqual(config["devices"][0]["dsp_arch"], "v73")
            self.assertEqual(config["devices"][0]["soc_model"], 43)
            self.assertEqual(config["graphs"][0]["vtcm_mb"], 8)
            self.assertEqual(backend["backend_extensions"]["config_file_path"], "./htp_config_8gen2.json")
            self.assertIn('soc_versions=("8gen2")', export.read_text())
            self.assertIn('model_path="${MODEL_PATH:?Set MODEL_PATH}"', export.read_text())
            self.assertIn("export QNN_SDK_ROOT", convert.read_text())

            # The helper may be rerun after an interrupted conversion.
            prepare_v73_profile(root, 43)

    def test_package_contains_expected_flattened_files_and_unvalidated_note(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            package_dir = root / "model"
            package_dir.mkdir()
            for name in REQUIRED_FILES:
                (package_dir / name).write_bytes(b"")
            output = root / "candidate.zip"

            package_candidate(package_dir, output, 43)

            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                for name in REQUIRED_FILES:
                    self.assertIn(str(PACKAGE_PREFIX / name), names)
                metadata = json.loads(archive.read(str(PACKAGE_PREFIX / "EXPERIMENTAL_V73_UNVALIDATED.json")))
            self.assertEqual(metadata["htp_arch"], "v73")
            self.assertEqual(metadata["vtcm_mb"], 8)
            self.assertEqual(metadata["qnn_soc_model"], 43)
            self.assertEqual(metadata["status"], "EXPERIMENTAL_UNVALIDATED")
            self.assertIn("not an SM8735-specific target", metadata["target_note"])

    def test_package_rejects_incomplete_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            package_dir = root / "model"
            package_dir.mkdir()
            (package_dir / "SDXL").touch()
            with self.assertRaisesRegex(ValueError, "Incomplete Local Dream SDXL package"):
                package_candidate(package_dir, root / "candidate.zip", 43)

    def test_profile_claim_requires_each_component_link_job_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            package_dir = root / "model"
            package_dir.mkdir()
            for name in REQUIRED_FILES:
                (package_dir / name).write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "VAE decoder SM8550 profile claim requires"):
                package_candidate(
                    package_dir,
                    root / "candidate.zip",
                    43,
                    vae_decoder_htp_profiled_sm8550=True,
                    vae_decoder_compile_job_id="decoder-compile",
                    vae_decoder_profile_job_id="decoder-profile",
                )

    def test_profile_claim_requires_source_compile_job_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            package_dir = root / "model"
            package_dir.mkdir()
            for name in REQUIRED_FILES:
                (package_dir / name).write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "requires its source compile job ID"):
                package_candidate(
                    package_dir,
                    root / "candidate.zip",
                    43,
                    unet_htp_profiled_sm8550=True,
                    unet_link_job_id="unet-link",
                )

    def test_profile_claim_requires_profile_job_and_validated_npu_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            package_dir = root / "model"
            package_dir.mkdir()
            for name in REQUIRED_FILES:
                (package_dir / name).write_bytes(b"")

            shared = dict(
                unet_htp_profiled_sm8550=True,
                unet_compile_job_id="unet-compile",
                unet_link_job_id="unet-link",
                unet_profile_job_id="unet-profile",
            )
            with self.assertRaisesRegex(ValueError, "requires matching validated profile evidence"):
                package_candidate(package_dir, root / "missing-evidence.zip", 43, **shared)

            evidence = {
                "unet": {
                    "profile_job_id": "unet-profile",
                    "status": "SUCCESS",
                    "device": "Samsung Galaxy S23",
                    "compute_units": ["NPU", "CPU"],
                }
            }
            with self.assertRaisesRegex(ValueError, "requires an NPU/HTP-only profile"):
                package_candidate(
                    package_dir,
                    root / "mixed-compute.zip",
                    43,
                    reference_profile_evidence=evidence,
                    **shared,
                )

    def test_package_streams_unet_context_from_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            package_dir = root / "model"
            package_dir.mkdir()
            for name in REQUIRED_FILES - {"unet.bin"}:
                (package_dir / name).write_bytes(b"asset")
            output = root / "candidate.zip"
            context = b"linked QNN context bytes"

            package_candidate(
                package_dir,
                output,
                43,
                unet_context_bytes=context,
                unet_link_job_id="jtestlink",
            )

            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.read(str(PACKAGE_PREFIX / "unet.bin")), context)
                contents = json.loads(archive.read(str(PACKAGE_PREFIX / "PACKAGE_CONTENTS.json")))
                metadata = json.loads(archive.read(str(PACKAGE_PREFIX / "EXPERIMENTAL_V73_UNVALIDATED.json")))
            self.assertEqual(metadata["unet_context_sha256"], hashlib.sha256(context).hexdigest())
            self.assertEqual(contents["files"]["unet.bin"]["sha256"], hashlib.sha256(context).hexdigest())
            self.assertEqual(metadata["sm8550_reference_job_ids"]["unet_link_job_id"], "jtestlink")

    def test_companion_hash_mismatch_fails_without_deleting_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            package_dir = root / "model"
            package_dir.mkdir()
            for name in REQUIRED_FILES:
                (package_dir / name).write_bytes(b"asset")
            (package_dir / "clip.mnn").write_bytes(b"actual bytes")
            declared = b"different bytes"
            manifest = {
                "files": {
                    "clip.mnn": {
                        "size_bytes": len(declared),
                        "sha256": hashlib.sha256(declared).hexdigest(),
                    }
                }
            }
            (package_dir / "MNN_COMPANIONS_SOURCE.json").write_text(json.dumps(manifest))
            output = root / "candidate.zip"

            with self.assertRaisesRegex(ValueError, "does not match its source manifest"):
                package_candidate(package_dir, output, 43, consume_source_files=True)

            self.assertTrue((package_dir / "clip.mnn").is_file())
            self.assertFalse(output.exists())

    def test_output_inside_package_dir_is_rejected_without_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            package_dir = pathlib.Path(temp) / "model"
            package_dir.mkdir()
            for name in REQUIRED_FILES:
                (package_dir / name).write_bytes(b"asset")
            output = package_dir / "candidate.zip"

            with self.assertRaisesRegex(ValueError, "outside package_dir"):
                package_candidate(package_dir, output, 43, consume_source_files=True)

            self.assertTrue(all((package_dir / name).is_file() for name in REQUIRED_FILES))
            self.assertFalse(output.exists())

    def test_consume_happens_only_after_archive_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            package_dir = root / "model"
            package_dir.mkdir()
            for name in REQUIRED_FILES:
                (package_dir / name).write_bytes(b"asset")
            (package_dir / "MODEL_SOURCE.md").write_text("source provenance")
            (package_dir / "unrelated-note.txt").write_text("keep this file")
            output = root / "candidate.zip"

            package_candidate(package_dir, output, 43, consume_source_files=True)

            self.assertTrue(output.is_file())
            self.assertIsNone(zipfile.ZipFile(output).testzip())
            self.assertFalse(any((package_dir / name).exists() for name in REQUIRED_FILES))
            self.assertTrue((package_dir / "MODEL_SOURCE.md").is_file())
            self.assertTrue((package_dir / "unrelated-note.txt").is_file())

    def test_reference_profiles_do_not_claim_sm8735_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            package_dir = root / "model"
            package_dir.mkdir()
            for name in REQUIRED_FILES:
                (package_dir / name).write_bytes(b"")
            output = root / "candidate.zip"

            package_candidate(
                package_dir,
                output,
                43,
                unet_htp_profiled_sm8550=True,
                vae_encoder_htp_profiled_sm8550=True,
                vae_decoder_htp_profiled_sm8550=True,
                unet_compile_job_id="unet-compile",
                vae_encoder_compile_job_id="encoder-compile",
                vae_decoder_compile_job_id="decoder-compile",
                unet_link_job_id="unet-link",
                vae_encoder_link_job_id="encoder-link",
                vae_decoder_link_job_id="decoder-link",
                unet_profile_job_id="unet-profile",
                vae_encoder_profile_job_id="encoder-profile",
                vae_decoder_profile_job_id="decoder-profile",
                reference_profile_evidence={
                    "unet": {
                        "profile_job_id": "unet-profile",
                        "status": "SUCCESS",
                        "device": "Samsung Galaxy S23",
                        "model_id": "unet-target",
                        "model_name": "job_unet-link_linked_bin",
                        "compute_units": ["NPU"],
                    },
                    "vae_encoder": {
                        "profile_job_id": "encoder-profile",
                        "status": "SUCCESS",
                        "device": "Samsung Galaxy S23",
                        "model_id": "encoder-target",
                        "model_name": "job_encoder-link_linked_bin",
                        "compute_units": ["NPU"],
                    },
                    "vae_decoder": {
                        "profile_job_id": "decoder-profile",
                        "status": "SUCCESS",
                        "device": "Samsung Galaxy S23",
                        "model_id": "decoder-target",
                        "model_name": "job_decoder-link_linked_bin",
                        "compute_units": ["NPU"],
                    },
                },
            )

            with zipfile.ZipFile(output) as archive:
                metadata = json.loads(archive.read(str(PACKAGE_PREFIX / "EXPERIMENTAL_V73_UNVALIDATED.json")))
            self.assertEqual(metadata["status"], "COMPONENTS_PROFILED_SM8550_ONLY")
            self.assertTrue(metadata["sm8550_reference_component_profiles"]["unet_htp_profiled_on_sm8550"])
            self.assertTrue(metadata["sm8550_reference_component_profiles"]["vae_encoder_htp_profiled_on_sm8550"])
            self.assertTrue(metadata["sm8550_reference_component_profiles"]["vae_decoder_htp_profiled_on_sm8550"])
            self.assertFalse(metadata["sm8735_poco_f7_validated"])
            self.assertIn("not device-validated", metadata["validation"])
            self.assertEqual(
                metadata["sm8550_reference_job_ids"]["unet_compile_job_id"],
                "unet-compile",
            )
            self.assertEqual(
                metadata["sm8550_reference_job_ids"]["unet_profile_job_id"],
                "unet-profile",
            )
            self.assertEqual(
                metadata["sm8550_reference_profile_evidence"]["unet"]["compute_units"],
                ["NPU"],
            )


if __name__ == "__main__":
    unittest.main()
