import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock

_spec = importlib.util.spec_from_file_location("checked_updates", Path(__file__).parents[1] / "stereo" / "updates.py")
_updates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_updates)
select_release, allowed_url = _updates.select_release, _updates.allowed_url
download_release, cleanup = _updates.download_release, _updates.cleanup


def metadata(data=b"installer", tag="v0.13.1"):
    return dict(tag_name=tag, body="Changes\nStereoMapper-Runtime: 1\n", draft=False, prerelease=False,
        assets=[dict(name=_updates.INSTALLER, state="uploaded", size=len(data),
        digest="sha256:" + hashlib.sha256(data).hexdigest(),
        browser_download_url=f"https://github.com/dservEXE/StereoMapper/releases/download/{tag}/{_updates.INSTALLER}")])


class UpdateChecks(unittest.TestCase):
    def test_versions_and_skipped_prerelease(self):
        self.assertEqual(select_release(metadata(), "0.13.0")["version"], "0.13.1")
        self.assertIsNone(select_release(metadata(), "0.13.1"))
        self.assertIsNone(select_release(metadata(), "0.13.2"))
        data = metadata(); data["prerelease"] = True
        self.assertIsNone(select_release(data, "0.13.0"))
        data = metadata(); data["draft"] = True
        self.assertIsNone(select_release(data, "0.13.0"))

    def test_changed_or_missing_runtime_requires_full_installer(self):
        for body in ("StereoMapper-Runtime: 2", "", "Release notes only"):
            data = metadata(); data["body"] = body
            release = select_release(data, "0.13.0")
            self.assertTrue(release["requires_setup"])
            self.assertEqual(release["page"], "https://github.com/dservEXE/StereoMapper/releases/tag/v0.13.1")
            self.assertNotIn("url", release)

    def test_repository_and_strict_metadata(self):
        self.assertEqual(_updates.REPOSITORY, "dservEXE/StereoMapper")
        for key, value in (("size", -1), ("size", True), ("size", _updates.MAX_PACKAGE + 1),
                           ("digest", "sha256:bad"), ("digest", None),
                           ("browser_download_url", "https://github.com/dservEXE/StereoMap/releases/download/v0.13.1/StereoMap-Setup.exe")):
            data = metadata(); data["assets"][0][key] = value
            with self.assertRaises(ValueError): select_release(data, "0.13.0")
        for assets in ([], [metadata()["assets"][0]] * 2):
            data = metadata(); data["assets"] = assets
            with self.assertRaises(ValueError): select_release(data, "0.13.0")
        for value in (None, [], {"tag_name":"../outside"}):
            with self.assertRaises(ValueError): select_release(value, "0.13.0")

    def test_restricted_redirects(self):
        for url in ("https://github.com.evil.invalid/installer", "http://github.com/file",
                    "https://user@github.com/file", "https://github.com:444/file",
                    "https://github.com/file#fragment", "https://github.com:bad/file"):
            self.assertFalse(allowed_url(url), url)
        self.assertTrue(allowed_url("https://release-assets.githubusercontent.com/file?signature=x"))

    def test_exact_api_endpoint_not_source_branch(self):
        opener = Mock(return_value=io.BytesIO(json.dumps(metadata()).encode()))
        self.assertEqual(_updates.check_release("0.13.0", opener)["version"], "0.13.1")
        opener.assert_called_once_with("https://api.github.com/repos/dservEXE/StereoMapper/releases/latest")
        missing = urllib.error.HTTPError("https://api.github.com",404,"not found",None,None)
        self.assertIsNone(_updates.check_release("0.13.0", Mock(side_effect=missing)))
        rate = urllib.error.HTTPError("https://api.github.com",403,"rate limit",None,None)
        with self.assertRaises(urllib.error.HTTPError): _updates.check_release("0.13.0",Mock(side_effect=rate))
        with self.assertRaises(ValueError): _updates.check_release("0.13.0",lambda _:io.BytesIO(b"x"*(_updates.MAX_METADATA+1)))

    def test_download_and_tamper_cleanup(self):
        data = b"safe installer" * 200
        release = select_release(metadata(data), "0.13.0")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress = []
            target = download_release(release, root=root, opener=lambda url: io.BytesIO(data), progress=progress.append)
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(target.name, "StereoMapper-Update.exe")
            self.assertEqual(progress[-1],100)
            cleanup(target.parent, root)
            self.assertFalse(target.parent.exists())
            for damaged in (data[:-1] + b"x", data[:-1], data + b"x"):
                with self.assertRaises(ValueError):
                    download_release(release, root=root, opener=lambda url: io.BytesIO(damaged))
                self.assertEqual(list(root.iterdir()), [])
            unrelated = root / "user-files"; unrelated.mkdir()
            file = unrelated / _updates.INSTALLER; file.write_bytes(data)
            cleanup(unrelated, root)
            self.assertTrue(file.exists())

    def test_cancel_before_and_during_download_leaves_no_package(self):
        data = b"a" * 200000
        release = select_release(metadata(data), "0.13.0")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            opener = Mock(return_value=io.BytesIO(data))
            with self.assertRaises(_updates.Cancelled):
                download_release(release, root=root, opener=opener, cancelled=lambda:True)
            opener.assert_not_called()
            calls = []
            with self.assertRaises(_updates.Cancelled):
                download_release(release, root=root, opener=lambda _:io.BytesIO(data),
                                 cancelled=lambda:bool(calls),progress=lambda _:calls.append(True))
            self.assertEqual(list(root.iterdir()), [])

    def test_untrusted_direct_download_is_rejected_before_io(self):
        release = select_release(metadata(), "0.13.0"); release["url"] = "https://github.com/another/project/file"
        opener = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError): download_release(release,root=temporary,opener=opener)
            opener.assert_not_called()
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__": unittest.main()
