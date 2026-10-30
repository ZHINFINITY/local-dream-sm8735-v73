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


if __name__ == "__main__":
    unittest.main()
