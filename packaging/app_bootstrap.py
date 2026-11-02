"""Start the portable application with its own interpreter and import paths."""
import os
from pathlib import Path
import runpy
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    root = ROOT
    # Ignore unrelated CUDA/Python/Tcl installations on the destination PC.
    for name in ("CUDA_PATH", "CUDA_HOME", "CUPY_CUDA_PATH", "TCL_LIBRARY", "TK_LIBRARY"):
        os.environ.pop(name, None)
    os.environ["TCL_LIBRARY"] = str(root / "runtime/tcl/tcl8.6")
    os.environ["TK_LIBRARY"] = str(root / "runtime/tcl/tk8.6")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.dont_write_bytecode = True
    logs = root / "data/logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "startup.log").open("a", encoding="utf-8", buffering=1) as stream:
        sys.stdout = sys.stderr = stream
        try:
            runpy.run_path(str(root / "main.py"), run_name="__main__")
        except Exception:
            traceback.print_exc()
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None, "Не удалось запустить программу. Подробности: " + str(logs / "startup.log"),
                "Стереокамера", 0x10)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
