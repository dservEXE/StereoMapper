"""Bounded live streaming diagnostic, without saving camera images."""
import argparse
import json
import sys
import time
from queue import Empty

from stereo.camera import CapturePump, CameraConfig


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps",type=int,default=30,choices=[15,25,30,50,60])
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--backend", default="AUTO", choices=["AUTO", "MSMF", "DSHOW"])
    args = parser.parse_args()
    pump = CapturePump(CameraConfig(width=args.width, height=args.height, fps=args.fps, backend=args.backend),
                       status=lambda message: print(message, flush=True))
    pump.start()
    try:
        first = pump.frames.get(timeout=45)
        end = time.perf_counter() + args.seconds
        last, next_report, delivered, maximum_gap = first[1], time.perf_counter(), 1, 0.
        while time.perf_counter() < end:
            if pump.error:
                raise pump.error
            try:
                _, captured, _ = pump.frames.get(timeout=.25)
                maximum_gap = max(maximum_gap, captured - last)
                last, delivered = captured, delivered + 1
            except Empty:
                pass
            if time.perf_counter() >= next_report:
                print(json.dumps({"state": pump.state, "fps": round(pump.fps, 1), "captured": pump.count,
                                  "delivered": delivered, "max_gap_s": round(maximum_gap, 3),
                                  "restarts": pump.restart_count, "source": pump.info}, ensure_ascii=False), flush=True)
                next_report = time.perf_counter() + 10
        print("STREAM CHECK COMPLETE", flush=True)
    finally:
        pump.stop_event.set()
        pump.join(5)
        print("Capture stopped:", not pump.is_alive(), flush=True)


if __name__ == "__main__":
    main()
