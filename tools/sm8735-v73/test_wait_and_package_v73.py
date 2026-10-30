import unittest

from wait_and_package_v73 import component_ids, component_ready, job_id_from_url, terminal_failure


class WaitAndPackageGateTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "device": "Samsung Galaxy S23",
            "target_claim": "SM8550 / HTP v73 reference only; not SM8735/POCO F7 validation",
            "compile_job_urls": ["https://workbench.aihub.qualcomm.com/jobs/compile123/"],
            "link_job_url": "https://workbench.aihub.qualcomm.com/jobs/link123/",
            "inference_job_url": "https://workbench.aihub.qualcomm.com/jobs/profile123/",
            "inference_status": "JOBSTATUS CODE SUCCESS",
            "profile_compute_units": ["NPU"],
            "htp_execution_verified": True,
        }

    def test_ready_requires_successful_reference_npu_profile(self):
        self.assertIsNone(terminal_failure(self.record, "U-Net"))
        ready, detail = component_ready(self.record, "U-Net")
        self.assertTrue(ready)
        self.assertIn("NPU", detail)

    def test_missing_profile_waits_without_claiming_ready(self):
        record = dict(self.record)
        record.pop("inference_job_url")
        record["htp_execution_verified"] = False
        self.assertIsNone(terminal_failure(record, "U-Net"))
        ready, detail = component_ready(record, "U-Net")
        self.assertFalse(ready)
        self.assertIn("inference-profile job URL", detail)

    def test_mixed_cpu_npu_profile_is_terminal_failure(self):
        record = dict(self.record, profile_compute_units=["NPU", "CPU"])
        self.assertIn("not NPU/HTP-only", terminal_failure(record, "U-Net"))

    def test_wrong_reference_device_is_terminal_failure(self):
        record = dict(self.record, device="POCO F7")
        self.assertIn("expected Samsung Galaxy S23", terminal_failure(record, "U-Net"))

    def test_failed_job_is_terminal_failure(self):
        record = dict(self.record, compile_statuses=["CODE: FAILED; MESSAGE: optimizer error"])
        self.assertIn("terminal job failure", terminal_failure(record, "U-Net"))

    def test_job_ids_are_extracted_from_recorded_urls(self):
        self.assertEqual(job_id_from_url(self.record["link_job_url"]), "link123")
        self.assertEqual(
            component_ids(self.record, "U-Net"),
            {"compile": "compile123", "link": "link123", "profile": "profile123"},
        )

    def test_invalid_job_url_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Invalid AI Hub job URL"):
            job_id_from_url("https://example.test/not-a-job")


if __name__ == "__main__":
    unittest.main()
