import unittest

from continue_v73_quantize_job import compile_options_for_level


class CompileOptimizationOptionTests(unittest.TestCase):
    def test_level_one_uses_ai_hub_context_optimization_flag(self):
        self.assertEqual(
            compile_options_for_level(1),
            ["--qairt_version 2.51 --qnn_context_binary_optimization_level 1"],
        )

    def test_levels_one_through_three_are_supported(self):
        for level in (1, 2, 3):
            with self.subTest(level=level):
                self.assertIn(
                    f"--qnn_context_binary_optimization_level {level}",
                    compile_options_for_level(level)[0],
                )

    def test_unsupported_optimization_levels_are_rejected(self):
        for level in (0, 4, -1):
            with self.subTest(level=level):
                with self.assertRaisesRegex(ValueError, "must be 1, 2, or 3"):
                    compile_options_for_level(level)


if __name__ == "__main__":
    unittest.main()
