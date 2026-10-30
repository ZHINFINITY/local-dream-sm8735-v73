import unittest

from continue_v73_quantize_job import htp_options_for_level


class CompileOptimizationOptionTests(unittest.TestCase):
    def test_level_one_uses_documented_ai_hub_qnn_options(self):
        self.assertEqual(
            htp_options_for_level(1),
            ["--qairt_version 2.51 --qnn_options default_graph_htp_optimizations=O=1"],
        )

    def test_levels_one_through_three_are_supported(self):
        for level in (1, 2, 3):
            with self.subTest(level=level):
                self.assertIn(
                    f"default_graph_htp_optimizations=O={level}",
                    htp_options_for_level(level)[0],
                )

    def test_unsupported_optimization_levels_are_rejected(self):
        for level in (0, 4, -1):
            with self.subTest(level=level):
                with self.assertRaisesRegex(ValueError, "must be 1, 2, or 3"):
                    htp_options_for_level(level)


class ModelVariantNamingTests(unittest.TestCase):
    def test_w4a16_compile_and_inference_names_are_distinct(self):
        from continue_v73_quantize_job import compile_job_name, inference_job_name

        self.assertEqual(
            compile_job_name("W4A16", 1),
            "Illustrious XL v2 all-biased W4A16 HTP v73 HTP-opt-L1",
        )
        self.assertEqual(
            inference_job_name("W4A16"),
            "Illustrious XL v2 all-biased W4A16 HTP v73 inference",
        )

    def test_w8a16_default_names_remain_unchanged(self):
        from continue_v73_quantize_job import compile_job_name, inference_job_name

        self.assertEqual(
            compile_job_name("W8A16", 1),
            "Illustrious XL v2 all-biased W8A16 HTP v73 HTP-opt-L1",
        )
        self.assertEqual(
            inference_job_name("W8A16"),
            "Illustrious XL v2 all-biased W8A16 HTP v73 inference",
        )

    def test_new_w4_pipeline_is_accepted(self):
        from continue_v73_quantize_job import validate_pipeline_identity

        validate_pipeline_identity({}, "W4A16", 1)

    def test_recorded_variant_mismatch_is_rejected(self):
        from continue_v73_quantize_job import validate_pipeline_identity

        with self.assertRaisesRegex(ValueError, "already bound to model variant"):
            validate_pipeline_identity({"model_variant": "W8A16"}, "W4A16", 1)

    def test_legacy_pipeline_with_jobs_can_only_resume_as_w8a16(self):
        from continue_v73_quantize_job import validate_pipeline_identity

        validate_pipeline_identity({"compile_job_urls": ["existing"]}, "W8A16", 1)
        with self.assertRaisesRegex(ValueError, "only W8A16 can safely resume"):
            validate_pipeline_identity({"compile_job_urls": ["existing"]}, "W4A16", 1)

    def test_recorded_optimization_level_mismatch_is_rejected(self):
        from continue_v73_quantize_job import validate_pipeline_identity

        with self.assertRaisesRegex(ValueError, "already bound to HTP optimization level"):
            validate_pipeline_identity(
                {"model_variant": "W4A16", "htp_finalize_optimization_level": 3},
                "W4A16",
                1,
            )


if __name__ == "__main__":
    unittest.main()
