import hashlib
import json
import pathlib
import tempfile
import unittest
import zipfile

from build_v73_package import PACKAGE_PREFIX, REQUIRED_FILES, package_candidate, prepare_v73_profile


class V73ProfileTests(unittest.TestCase):
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
                metadata = json.loads(archive.read(str(PACKAGE_PREFIX / "EXPERIMENTAL_V73_UNVALIDATED.json")))
            self.assertEqual(metadata["unet_context_sha256"], hashlib.sha256(context).hexdigest())
            self.assertEqual(metadata["sm8550_reference_job_ids"]["unet_link_job_id"], "jtestlink")

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
            )

            with zipfile.ZipFile(output) as archive:
                metadata = json.loads(archive.read(str(PACKAGE_PREFIX / "EXPERIMENTAL_V73_UNVALIDATED.json")))
            self.assertEqual(metadata["status"], "COMPONENTS_PROFILED_SM8550_ONLY")
            self.assertTrue(metadata["sm8550_reference_component_profiles"]["unet_htp_profiled_on_sm8550"])
            self.assertTrue(metadata["sm8550_reference_component_profiles"]["vae_encoder_htp_profiled_on_sm8550"])
            self.assertTrue(metadata["sm8550_reference_component_profiles"]["vae_decoder_htp_profiled_on_sm8550"])
            self.assertFalse(metadata["sm8735_poco_f7_validated"])
            self.assertIn("not device-validated", metadata["validation"])


if __name__ == "__main__":
    unittest.main()
