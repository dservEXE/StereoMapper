"""Read a few frames from each candidate index without storing camera images."""
import argparse
import json
import time
import sys

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8")

from stereo.camera import CameraConfig, StereoCamera


def probe(index, backend, width, height):
    camera = None
    try:
        config = CameraConfig(left=index, mode="sbs", width=width // 2, height=height, backend=backend)
        # Use the single capture directly: no stereo split or left/right assertion here.
        camera = StereoCamera(config)
        sizes = []
        started = time.perf_counter()
        for _ in range(8):
            ok, frame = camera.caps[0].read()
            if not ok or frame is None:
                raise RuntimeError("Нет изображения")
            sizes.append([frame.shape[1], frame.shape[0]])
        return {"index": index, "backend": backend, "ok": True, "size": sizes[-1],
                "read_fps": round(8 / (time.perf_counter() - started), 1)}
    except Exception as error:
        return {"index": index, "backend": backend, "ok": False, "error": str(error)}
    finally:
        if camera:
            camera.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка доступных USB-камер")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--backend", choices=["DSHOW", "MSMF", "AUTO"], default="DSHOW")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    for index in range(args.count):
        print(json.dumps(probe(index, args.backend, args.width, args.height), ensure_ascii=False), flush=True)
