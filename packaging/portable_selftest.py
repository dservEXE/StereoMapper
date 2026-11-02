"""Offline checks of a relocated StereoMapper distribution; never opens USB devices."""
from pathlib import Path
import argparse
import datetime
import json
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
from unittest.mock import patch


def fixture():
    import cv2
    import numpy as np
    from stereo.calibration import Calibration
    from stereo.camera import CameraConfig
    size = (384, 240)
    k = np.array([[240., 0, 192], [0, 240., 120], [0, 0, 1]])
    distortion, rotation = np.zeros(5), np.eye(3)
    translation = np.array([[-.06], [0], [0]])
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        k, distortion, k, distortion, size, rotation, translation, alpha=0)
    calibration = Calibration(size, k, distortion, k, distortion, rotation, translation,
                              r1, r2, p1, p2, q, np.array(roi1), np.array(roi2),
                              .1, .1, CameraConfig().signature())
    gray = np.random.default_rng(72).integers(0, 256, (240, 384), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), .6)
    left = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    right = np.zeros_like(left)
    right[:, :-16] = left[:, 16:]
    return calibration, left, right


def simulated_camera(config, buffer, lock, sequence, timestamp, notices, stop):
    """Spawn target uses only generated pixels, with the real capture IPC protocol."""
    import numpy as np
    _, left, right = fixture()
    image = np.frombuffer(buffer, np.uint8).reshape(2, config.height, config.width, 3)
    notices.put(("info", {"source": "self-test synthetic camera", "size": (384, 240)}))
    while not stop.is_set():
        with lock:
            image[0], image[1] = left, right
            sequence.value += 1
            timestamp.value = time.perf_counter()
        stop.wait(.03)


def _inside(path, directory):
    try:
        Path(path).resolve().relative_to(directory.resolve())
        return True
    except (ValueError, OSError):
        return False


def main(root, report_path):
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    started = time.perf_counter()
    report = {"root": str(root), "python": sys.version, "executable": sys.executable,
              "prefix": sys.prefix, "sys_path": sys.path[:], "hardware_camera_opened": False,
              "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "checks": {}}

    def check(name, operation):
        begin = time.perf_counter()
        try:
            detail = operation()
            report["checks"][name] = dict(status="passed", seconds=round(time.perf_counter()-begin, 3),
                                           **(detail or {}))
        except Exception as error:
            report["checks"][name] = {"status": "failed", "seconds": round(time.perf_counter()-begin, 3),
                                      "error": str(error), "traceback": traceback.format_exc()}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def dependency_paths():
        import cv2
        import numpy
        import PIL
        import sqlite3
        import ssl
        import tkinter
        runtime = root / "runtime"
        locations = {module.__name__: str(module.__file__) for module in
                     (cv2, numpy, PIL, sqlite3, ssl, tkinter)}
        if runtime.exists():
            assert Path(sys.prefix).resolve() == runtime.resolve(), "Interpreter was not relocated"
            assert _inside(sys.executable, runtime), "External interpreter used"
            assert all(_inside(value, runtime) for value in locations.values()), locations
            assert all(_inside(value, root) for value in sys.path if value), sys.path
        return {"versions": {"opencv": cv2.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__},
                "locations": locations, "portable_runtime_checked": runtime.exists()}

    def cpu_depth():
        import numpy as np
        from stereo.depth import DepthEngine
        calibration, left, right = fixture()
        engine = DepthEngine(calibration, disparities=64, acceleration="cpu", estimate_surfaces=False)
        try:
            frame = engine.compute(left, right)
            fraction = float(frame.valid.mean())
            z = float(np.median(frame.points[frame.valid, 2]))
            assert fraction > .45, f"Valid fraction: {fraction}"
            assert abs(z - .9) < .02, f"Expected 0.9m, got {z}m"
            assert np.isnan(frame.points[~frame.valid]).all(), "Invalid depths must remain unknown"
            return {"valid_fraction": fraction, "median_z_metres": z, "expected_z_metres": .9}
        finally:
            engine.close()

    def gpu_depth():
        import cv2
        import numpy as np
        from stereo.gpu_stereo import CudaSGM, runtime
        try:
            cp, _ = runtime()
        except Exception as error:
            return {"available": False, "reason": str(error), "cpu_fallback_available": True}
        _, left, right = fixture()
        result = CudaSGM(64, ambiguity_guard=True).compute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                                                         cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)) / 16
        region = result[12:-12, 80:-20]
        valid = region > .5
        median = float(np.median(region[valid]))
        assert valid.mean() > .8, f"CUDA returned too few samples: {valid.mean()}"
        assert abs(median - 16) < .25, f"Expected 16px disparity, got {median}"
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode()
        if (root / "runtime").exists():
            assert _inside(cp.__file__, root / "runtime"), "External CuPy installation used"
        return {"available": True, "device": name, "median_disparity_pixels": median,
                "valid_fraction": float(valid.mean()), "cupy_location": str(cp.__file__)}

    def spawned_capture():
        import numpy as np
        from stereo.camera import CameraConfig, CapturePump
        previous = {child.pid for child in multiprocessing.active_children()}
        config = CameraConfig(width=384, height=240, fps=30)
        pump = CapturePump(config, process_target=simulated_camera, startup_timeout=15.)
        pump.start()
        try:
            packet = pump.frames.get(timeout=20.)
            pair, _, count = packet
            _, left, right = fixture()
            np.testing.assert_array_equal(pair[0], left)
            np.testing.assert_array_equal(pair[1], right)
            assert count >= 1
        finally:
            pump.stop_event.set()
            pump.join(5.)
        assert not pump.is_alive(), "Capture supervisor failed to stop"
        assert pump.child is None and pump.ipc is None, "Camera subprocess was not disposed"
        leaked = [child.pid for child in multiprocessing.active_children() if child.pid not in previous]
        assert not leaked, f"Leaked child processes: {leaked}"
        return {"frames": count, "process_stopped": True, "source": "generated stereo pair"}

    def mono_photo():
        import numpy as np
        from stereo.mono import MonoDepth
        _, left, _ = fixture()
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}), \
             patch("urllib.request.urlopen", side_effect=AssertionError("Self-test forbids network downloads")):
            model = MonoDepth(root)
            frame, inverse = model.compute(left)
        assert inverse.shape == left.shape[:2], inverse.shape
        assert np.isfinite(inverse).all(), "Photo depth contains nonfinite values"
        assert frame.units == "relative", "Single photograph must not claim metric accuracy"
        runtime = root / "runtime"
        import torch
        import transformers
        if runtime.exists():
            assert _inside(torch.__file__, runtime)
            assert _inside(transformers.__file__, runtime)
        return {"shape": list(inverse.shape), "units": frame.units, "offline": True,
                "torch": torch.__version__, "transformers": transformers.__version__}

    def interface():
        import tkinter as tk
        from stereo.app import StereoApp
        from stereo.calibration_help import CalibrationHelpWindow
        window = tk.Tk()
        window.withdraw()
        try:
            # Avoid touching the installed application's settings or studies.
            with tempfile.TemporaryDirectory(prefix="stereo-selftest-") as temporary:
                app = StereoApp(window, Path(temporary))
                guide = CalibrationHelpWindow(window, root)
                guide.window.withdraw()
                window.update()
                assert guide.photo is not None, "Calibration illustration missing"
                assert app.worker is None, "Self-test must not open cameras"
                guide.window.destroy()
                app.close()
                window.update()
                # close() is completed by the application's scheduled poll.
                end = time.perf_counter() + .3
                while window.winfo_exists() and time.perf_counter() < end:
                    window.update()
                    time.sleep(.01)
        except tk.TclError as error:
            if "application has been destroyed" not in str(error):
                raise
        finally:
            try:
                window.destroy()
            except tk.TclError:
                pass
        return {"tk": tk.TkVersion, "calibration_illustration_loaded": True, "camera_opened": False}

    check("dependency_paths", dependency_paths)
    check("cpu_metric_depth", cpu_depth)
    check("gpu_stereo", gpu_depth)
    check("multiprocessing_capture", spawned_capture)
    check("single_photo_offline", mono_photo)
    check("tk_interface_and_resources", interface)
    report["passed"] = all(item["status"] == "passed" for item in report["checks"].values())
    report["seconds"] = round(time.perf_counter() - started, 3)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--report", type=Path, default=Path.cwd() / "portable-selftest.json")
    args = parser.parse_args()
    if not (args.root / "stereo").exists() and (args.root.parent / "stereo").exists():
        args.root = args.root.parent
    sys.path.insert(0, str(args.root.resolve()))
    raise SystemExit(main(args.root, args.report))
