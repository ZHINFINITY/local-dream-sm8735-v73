import pathlib
import tempfile
import unittest

from extract_qairt_v73_subset import _ensure_safe_parent


class SafeSdkDestinationTests(unittest.TestCase):
    def test_creates_normal_parent_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp) / "sdk"
            root.mkdir()

            parent = _ensure_safe_parent(root, pathlib.Path("include/QNN/API/header.h"))

            self.assertEqual(parent, root / "include" / "QNN" / "API")
            self.assertTrue(parent.is_dir())

    def test_rejects_symlinked_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp) / "sdk"
            outside = pathlib.Path(temp) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "include").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(FileExistsError, "symlinked SDK parent"):
                _ensure_safe_parent(root, pathlib.Path("include/QNN/API/header.h"))

            self.assertFalse((outside / "QNN").exists())

    def test_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp) / "sdk"
            root.mkdir()

            with self.assertRaisesRegex(ValueError, "Unsafe SDK archive path"):
                _ensure_safe_parent(root, pathlib.Path("../outside/header.h"))


if __name__ == "__main__":
    unittest.main()
