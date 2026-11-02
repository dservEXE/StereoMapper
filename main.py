from pathlib import Path
import multiprocessing
import tkinter as tk

from stereo.app import StereoApp


if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    StereoApp(root, Path(__file__).resolve().parent)
    root.mainloop()
