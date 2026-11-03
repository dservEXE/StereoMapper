from pathlib import Path
import multiprocessing
import sys

APP_DIRECTORY = Path(__file__).resolve().parent
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))


def main():
    multiprocessing.freeze_support()
    try:
        import tkinter as tk
        from stereo.app import StereoApp
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] not in {"cv2", "numpy", "PIL", "tkinter", "_tkinter"}:
            raise
        print("Не установлены компоненты программы (" + str(error.name) + ").\n"
              "Для обычной установки запустите Install.cmd или StereoMapper-Setup.exe.\n"
              "Для запуска исходников используйте start.cmd: он восстановит библиотеки.")
        return 1
    root = tk.Tk()
    StereoApp(root, APP_DIRECTORY)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
