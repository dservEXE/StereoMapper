"""Check source dependencies without loading stereo or opening USB devices."""
import argparse
import importlib
import json
import struct
import sys


def probe_runtime(import_module=importlib.import_module):
    problems = []
    if struct.calcsize("P") != 8:
        problems.append("Нужен 64-разрядный Python.")
    if sys.version_info[:2] not in ((3, 12), (3, 13), (3, 14)):
        problems.append("Для исходников поддерживается Python 3.12–3.14 x64.")
    loaded = {}
    for name in ("cv2", "numpy", "PIL.Image", "PIL.ImageTk", "tkinter"):
        try:
            loaded[name] = import_module(name)
        except (ImportError, OSError) as error:
            problems.append(f"{name}: {error}")
    cv2 = loaded.get("cv2")
    extra = getattr(cv2, "ximgproc", None)
    if cv2 is not None and not all(callable(getattr(extra, name, None)) for name in
                                   ("createRightMatcher", "createDisparityWLSFilter")):
        problems.append("Нужен opencv-contrib-python с модулем ximgproc; обычного opencv-python недостаточно.")
    return {"ready": not problems, "problems": problems, "python": sys.executable,
            "version": list(sys.version_info[:3]), "bits": struct.calcsize("P") * 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = probe_runtime()
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    elif not args.quiet:
        if result["ready"]:
            print("Компоненты готовы.")
        else:
            print("Не хватает компонентов приложения:", file=sys.stderr)
            for problem in result["problems"]:
                print(" • " + problem, file=sys.stderr)
            print("Запустите Install.cmd для установки готовой программы или start.cmd для восстановления исходников.", file=sys.stderr)
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
