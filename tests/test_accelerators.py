"""Accelerator packs: choosing, downloading, swapping at the next start, choosing where models run."""
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from stereo import gpu_pack
from stereo.gpu_pack import (AcceleratorPack, DownloadInterrupted, MARKER, PACKS, Wheel, activate_pending, gpu_pack_ready,
                             install_pack, installed_pack, pack_advice, pending_pack, recommended_pack, request_removal)
from stereo.people_detection import select_device


def wheel_bytes(name="torch/__init__.py"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, "")
    return stream.getvalue()


def serving(*payloads):
    """An opener serving these wheels by URL, honouring continued downloads."""
    table = {f"https://example.test/{index}": data for index, data in enumerate(payloads)}

    def opener(request, timeout=None):
        return io.BytesIO(table[request.full_url][int((request.get_header("Range") or "bytes=0-")[6:-1]):])
    return opener


def pack_of(*payloads, key="directml"):
    wheels = tuple(Wheel(f"https://example.test/{index}", len(data), hashlib.sha256(data).hexdigest())
                   for index, data in enumerate(payloads))
    return AcceleratorPack(key, "Тест", "", wheels)


class RecommendationTests(unittest.TestCase):
    def test_each_vendor_gets_its_pack_and_nvidia_wins_in_a_laptop_with_two_cards(self):
        cases = (("NVIDIA GeForce RTX 4060 Ti | NVIDIA", "nvidia"), ("AMD Radeon RX 7800 | Advanced Micro Devices, Inc.", "directml"),
                 ("Intel(R) Arc(TM) A770 | Intel Corporation", "directml"), ("Microsoft Basic Display | Microsoft", None))
        for line, expected in cases:
            self.assertEqual(recommended_pack([line]), expected, line)
        self.assertEqual(recommended_pack(["AMD Radeon(TM) Graphics | Advanced Micro Devices, Inc.",
                                           "NVIDIA GeForce RTX 4060 Ti | NVIDIA"]), "nvidia")

    def test_no_information_recommends_nothing(self):
        self.assertIsNone(recommended_pack([]))

    def test_integrated_graphics_alone_recommends_nothing_and_says_why(self):
        for line in ("AMD Radeon(TM) Graphics | Advanced Micro Devices, Inc.", "AMD Radeon 780M Graphics | Advanced Micro Devices, Inc.",
                     "Intel(R) UHD Graphics 770 | Intel Corporation", "Intel(R) Iris(R) Xe Graphics | Intel Corporation"):
            advice = pack_advice([line])
            self.assertIsNone(advice.pack, line)
            self.assertIn("встроенная", advice.text)

    def test_discrete_laptop_radeon_is_not_mistaken_for_integrated(self):
        self.assertEqual(recommended_pack(["AMD Radeon RX 7800M XT | Advanced Micro Devices, Inc."]), "directml")

    def test_advice_names_this_computers_card_and_the_size_to_download(self):
        advice = pack_advice(["AMD Radeon(TM) Graphics | Advanced Micro Devices, Inc.", "NVIDIA GeForce RTX 4060 Ti | NVIDIA"])
        self.assertEqual(advice.pack, "nvidia")
        self.assertIn("RTX 4060 Ti", advice.text)
        self.assertNotIn("Radeon", advice.text)
        self.assertIn("2,6 ГБ", advice.text)

    def test_advice_stops_offering_once_the_right_pack_is_there(self):
        advice = pack_advice(["NVIDIA GeForce RTX 4060 Ti | NVIDIA"], "nvidia")
        self.assertIsNone(advice.pack)
        self.assertIn("уже установлен", advice.text)

    def test_advice_offers_a_replacement_for_the_wrong_pack(self):
        self.assertEqual(pack_advice(["NVIDIA GeForce RTX 4060 Ti | NVIDIA"], "directml").pack, "nvidia")

    def test_advice_admits_when_the_cards_are_unknown(self):
        self.assertIsNone(pack_advice([]).pack)
        self.assertIsNone(pack_advice(["Microsoft Basic Display | Microsoft"]).pack)

    def test_packs_are_pinned_by_size_and_hash(self):
        for pack in PACKS.values():
            for wheel in pack.wheels:
                self.assertRegex(wheel.sha256, r"^[0-9a-f]{64}$")
                self.assertGreater(wheel.size, 0)
                self.assertTrue(wheel.url.startswith("https://"))


    def test_nvidia_pack_is_the_same_pytorch_version_as_the_application(self):
        self.assertIn("torch-2.14.1", PACKS["nvidia"].wheels[0].url)


class InstallTests(unittest.TestCase):
    def test_new_pack_waits_for_the_next_start_and_only_then_becomes_active(self):
        data = wheel_bytes()
        with TemporaryDirectory() as temp:
            install_pack(temp, "directml", opener=serving(data), pack=pack_of(data))
            self.assertEqual(pending_pack(temp), "directml")
            self.assertIsNone(installed_pack(temp))
            self.assertTrue(gpu_pack_ready(temp))
            active = activate_pending(temp)
            self.assertEqual(active, Path(temp) / "gpu_runtime")
            self.assertEqual(installed_pack(temp), "directml")
            self.assertIsNone(pending_pack(temp))

    def test_several_wheels_end_up_in_one_folder(self):
        first, second = wheel_bytes(), wheel_bytes("torch_directml/__init__.py")
        with TemporaryDirectory() as temp:
            install_pack(temp, "directml", opener=serving(first, second), pack=pack_of(first, second))
            activate_pending(temp)
            self.assertTrue((Path(temp) / "gpu_runtime" / "torch_directml" / "__init__.py").is_file())

    def test_switching_replaces_the_previous_pack(self):
        data = wheel_bytes()
        with TemporaryDirectory() as temp:
            install_pack(temp, "nvidia", opener=serving(data), pack=pack_of(data, key="nvidia"))
            activate_pending(temp)
            (Path(temp) / "gpu_runtime" / "old_file").write_text("x", encoding="utf-8")
            install_pack(temp, "directml", opener=serving(data), pack=pack_of(data))
            self.assertEqual(installed_pack(temp), "nvidia")
            activate_pending(temp)
            self.assertEqual(installed_pack(temp), "directml")
            self.assertFalse((Path(temp) / "gpu_runtime" / "old_file").exists())

    def test_corrupt_file_is_rejected_and_nothing_is_activated(self):
        data = wheel_bytes()
        pack = AcceleratorPack("directml", "Тест", "", (Wheel("https://example.test/0", len(data), "0" * 64),))
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "контрольная"):
                install_pack(temp, "directml", opener=serving(data), pack=pack)
            self.assertFalse(gpu_pack_ready(temp))
            self.assertIsNone(activate_pending(temp))

    def test_dropped_connection_keeps_what_arrived_and_continues(self):
        data = wheel_bytes()
        pack = pack_of(data)
        with TemporaryDirectory() as temp:
            with self.assertRaises(DownloadInterrupted):
                install_pack(temp, "directml", opener=lambda request, timeout=None: io.BytesIO(data[:40]), pack=pack)
            partial = Path(temp) / "gpu_runtime.directml.0.download"
            self.assertEqual(partial.stat().st_size, 40)
            ranges = []

            class Answer(io.BytesIO):
                status = 206

            def opener(request, timeout=None):
                ranges.append(request.get_header("Range"))
                return Answer(data[40:])
            install_pack(temp, "directml", opener=opener, pack=pack)
            self.assertEqual(ranges, ["bytes=40-"])
            self.assertEqual(pending_pack(temp), "directml")

    def test_removal_happens_at_the_next_start(self):
        data = wheel_bytes()
        with TemporaryDirectory() as temp:
            install_pack(temp, "directml", opener=serving(data), pack=pack_of(data))
            activate_pending(temp)
            request_removal(temp)
            self.assertEqual(installed_pack(temp), "directml")
            self.assertIsNone(activate_pending(temp))
            self.assertIsNone(installed_pack(temp))

    def test_removal_also_drops_a_pack_waiting_for_its_turn(self):
        data = wheel_bytes()
        with TemporaryDirectory() as temp:
            install_pack(temp, "directml", opener=serving(data), pack=pack_of(data))
            request_removal(temp)
            self.assertFalse(gpu_pack_ready(temp))

    def test_pack_installed_by_an_older_version_counts_as_nvidia(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "gpu_runtime" / "torch"
            folder.mkdir(parents=True)
            (folder / "__init__.py").write_text("", encoding="utf-8")
            self.assertEqual(installed_pack(temp), "nvidia")

    def test_folder_with_a_damaged_marker_is_still_usable(self):
        data = wheel_bytes()
        with TemporaryDirectory() as temp:
            install_pack(temp, "directml", opener=serving(data), pack=pack_of(data))
            activate_pending(temp)
            (Path(temp) / "gpu_runtime" / MARKER).write_text("not json", encoding="utf-8")
            self.assertIsNotNone(installed_pack(temp))

    def test_this_module_imports_nothing_heavy_since_it_runs_before_pytorch(self):
        source = Path(gpu_pack.__file__).read_text(encoding="utf-8")
        for name in ("import torch", "import cv2", "import numpy"):
            self.assertNotIn(name, source)


class DeviceChoiceTests(unittest.TestCase):
    def torch(self, cuda=False):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda index: "RTX"))

    def directml(self, *names):
        return SimpleNamespace(device_count=lambda: len(names), device_name=lambda index: names[index] + "\0",
                               device=lambda index: f"dml:{index}")

    def choose(self, *names):
        with patch.dict("sys.modules", torch_directml=self.directml(*names)):
            return select_device(self.torch())

    def test_nvidia_uses_cuda_first(self):
        self.assertEqual(select_device(self.torch(True)), ("cuda", "RTX"))

    def test_processor_when_no_pack_is_installed(self):
        with patch.dict("sys.modules", torch_directml=None):
            self.assertEqual(select_device(self.torch())[0], "cpu")

    def test_discrete_card_is_preferred_over_an_integrated_one_listed_first(self):
        self.assertEqual(self.choose("AMD Radeon(TM) Graphics", "AMD Radeon RX 7800 XT"), ("dml:1", "AMD Radeon RX 7800 XT"))
        self.assertEqual(self.choose("Intel(R) UHD Graphics 770", "Intel(R) Arc(TM) A770")[0], "dml:1")

    def test_integrated_card_is_used_when_it_is_the_only_one(self):
        self.assertEqual(self.choose("AMD Radeon(TM) Graphics")[0], "dml:0")

    def test_no_adapters_means_processor(self):
        self.assertEqual(self.choose()[0], "cpu")


if __name__ == "__main__":
    unittest.main()
