import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
_spec = importlib.util.spec_from_file_location("checked_updates", Path(__file__).parents[1] / "stereo" / "updates.py")
_updates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_updates)
select_release, allowed_url = _updates.select_release, _updates.allowed_url
download_release, cleanup = _updates.download_release, _updates.cleanup


def metadata(data=b"installer"):
    return dict(tag_name="v0.13.0", draft=False, prerelease=False, assets=[dict(name="StereoMap-Setup.exe", state="uploaded", size=len(data), digest="sha256:" + hashlib.sha256(data).hexdigest(), browser_download_url="https://github.com/dservEXE/StereoMap/releases/download/v0.13.0/StereoMap-Setup.exe")])


class UpdateChecks(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(select_release(metadata(), "0.12.1")["version"], "0.13.0")
        self.assertIsNone(select_release(metadata(), "0.13.0"))
        data = metadata(); data["prerelease"] = True
        self.assertIsNone(select_release(data, "0.12.1"))

    def test_trust(self):
        self.assertFalse(allowed_url("https://github.com.evil.invalid/installer"))
        self.assertFalse(allowed_url("http://github.com/file"))
        for key, value in (("size", -1), ("digest", "sha256:bad"), ("browser_download_url", "https://elsewhere.invalid/file")):
            data = metadata(); data["assets"][0][key] = value
            with self.assertRaises(ValueError):
                select_release(data, "0.12.1")

    def test_download_and_tamper_cleanup(self):
        data = b"safe installer" * 200
        release = select_release(metadata(data), "0.12.1")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = download_release(release, root=root, opener=lambda url: io.BytesIO(data))
            self.assertEqual(target.read_bytes(), data)
            cleanup(target.parent, root)
            self.assertFalse(target.parent.exists())
            with self.assertRaises(ValueError):
                download_release(release, root=root, opener=lambda url: io.BytesIO(data[:-1] + b"x"))
            self.assertEqual(list(root.iterdir()), [])
            unrelated = root / "user-files"; unrelated.mkdir()
            file = unrelated / "StereoMap-Setup.exe"; file.write_bytes(data)
            cleanup(unrelated, root)
            self.assertTrue(file.exists())


if __name__ == "__main__":
    unittest.main()
