"""Exercise the Windows installer transaction with small synthetic layouts.

These tests validate ownership, payload boundaries and preservation. A full
release additionally runs its bundled interpreter after an actual installation.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile

SOURCE = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "nt" and shutil.which("powershell.exe"), "Windows installer")
class InstallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (SOURCE / "tmp").mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="installer-tests-", dir=SOURCE / "tmp")
        cls.root = Path(cls.temporary.name)
        cls.full_files = {name: "synthetic fixture" for name in (
            "StereoMapper.exe", "main.py", "app_bootstrap.py", "stereo/app.py", "stereo/version.py",
            "runtime/python.exe", "runtime/pythonw.exe", "runtime/Lib/os.py",
            "runtime/Lib/site-packages/cv2/__init__.py", "runtime/Lib/site-packages/numpy/__init__.py",
            "runtime/Lib/site-packages/PIL/__init__.py")}
        cls.full = cls.build("full", cls.full_files)
        cls.code = cls.build("code", {"stereo/app.py": "new app", "stereo/version.py": 'VERSION="0.13.0"'}, code=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def build(cls, name, files, code=False):
        folder = cls.root / name
        folder.mkdir()
        payload = folder / "payload.zip"
        with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative, value in files.items():
                archive.writestr(relative, value)
        arguments = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                     str(SOURCE / "packaging/build-setup.ps1"), "-Version", "0.13.0", "-Payload", str(payload),
                     "-OutputDirectory", str(folder), "-SkipVerify"]
        if code:
            arguments.append("-CodeOnly")
        result = subprocess.run(arguments, capture_output=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace"))
        return folder / ("StereoMapper-Update.exe" if code else "StereoMapper-Setup.exe")

    def run_installer(self, binary, destination):
        report = self.root / (destination.name + "-result.txt")
        result = subprocess.run([str(binary), "--install", "--directory", str(destination), "--no-launch",
                                 "--no-shortcut", "--no-registry", "--result", str(report)], timeout=30)
        return result.returncode, report.read_text(encoding="utf-8-sig")

    def test_fresh_install_has_private_python_and_opencv(self):
        target = self.root / "Новая программа с пробелами"
        code, result = self.run_installer(self.full, target)
        self.assertEqual(code, 0, result)
        self.assertTrue((target / "runtime/Lib/site-packages/cv2/__init__.py").is_file())
        self.assertFalse((target / "data").exists())
        self.assertFalse((target / "settings.json").exists())

    def test_full_upgrade_preserves_research_and_settings(self):
        target = self.root / "full-upgrade"
        self.assertEqual(self.run_installer(self.full, target)[0], 0)
        preserved = {"data/calibration.npz": b"my calibration", "output/study/model.ply": b"my model",
                     "settings.json": b"my settings", "screen_board.json": b"monitor scale"}
        for name, contents in preserved.items():
            file = target / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(contents)
        (target / "stereo/app.py").write_text("old version", encoding="utf-8")
        code, result = self.run_installer(self.full, target)
        self.assertEqual(code, 0, result)
        for name, contents in preserved.items():
            self.assertEqual((target / name).read_bytes(), contents)
        self.assertEqual((target / "stereo/app.py").read_text(), "synthetic fixture")

    def test_code_update_preserves_runtime_and_data(self):
        target = self.root / "code-update"
        self.assertEqual(self.run_installer(self.full, target)[0], 0)
        (target / "data").mkdir()
        (target / "data/calibration.npz").write_bytes(b"calibration")
        native = (target / "runtime/pythonw.exe").read_bytes()
        code, result = self.run_installer(self.code, target)
        self.assertEqual(code, 0, result)
        self.assertEqual((target / "stereo/app.py").read_text(), "new app")
        self.assertEqual((target / "data/calibration.npz").read_bytes(), b"calibration")
        self.assertEqual((target / "runtime/pythonw.exe").read_bytes(), native)

    def test_code_update_refuses_first_install(self):
        target = self.root / "not-installed"
        code, result = self.run_installer(self.code, target)
        self.assertNotEqual(code, 0, result)
        self.assertFalse(target.exists())

    def test_unrelated_destination_is_not_touched(self):
        target = self.root / "my-files"
        target.mkdir()
        (target / "keep.txt").write_text("keep")
        code, result = self.run_installer(self.full, target)
        self.assertNotEqual(code, 0, result)
        self.assertEqual(list(target.iterdir()), [target / "keep.txt"])

    def test_full_payload_cannot_include_personal_data(self):
        files = dict(self.full_files, **{"data/calibration.npz": "must not ship"})
        installer = self.build("personal-data-rejected", files)
        target = self.root / "reject-data"
        code, result = self.run_installer(installer, target)
        self.assertNotEqual(code, 0, result)
        self.assertFalse(target.exists())

    def test_full_payload_requires_opencv(self):
        files = {name: value for name, value in self.full_files.items() if "/cv2/" not in name}
        installer = self.build("missing-opencv", files)
        target = self.root / "reject-no-opencv"
        code, result = self.run_installer(installer, target)
        self.assertNotEqual(code, 0, result)
        self.assertFalse(target.exists())

    def test_wrapper_folder_zip_is_supported(self):
        installer = self.build("wrapped", {"StereoMapper/" + name: value for name, value in self.full_files.items()})
        target = self.root / "wrapped-result"
        code, result = self.run_installer(installer, target)
        self.assertEqual(code, 0, result)
        self.assertTrue((target / "stereo/app.py").is_file())
        self.assertFalse((target / "StereoMapper").exists())

    def test_path_traversal_is_rejected(self):
        installer = self.build("traversal", dict(self.full_files, **{"../escape.txt": "invalid"}))
        target = self.root / "reject-traversal"
        code, result = self.run_installer(installer, target)
        self.assertNotEqual(code, 0, result)
        self.assertFalse((self.root / "escape.txt").exists())
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
