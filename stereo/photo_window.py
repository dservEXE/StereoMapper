from pathlib import Path
from queue import Queue, Empty
from threading import Thread
import json
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from .calibration import Calibration, read_image, write_image
from .depth import DepthEngine, measure
from .mapping import export_ply
from .model3d import surface_from_frame, export_surface
from .mono import MonoDepth, MODEL_ID, MODEL_REVISION
from .viewer import CloudViewer
from .worker import stamp


class PhotoWindow:
    def __init__(self, parent, directory):
        from .app import ImagePanel
        self.directory = Path(directory)
        self.window = tk.Toplevel(parent)
        self.window.title("Глубина по фотографии")
        self.window.geometry("1240x800")
        self.window.minsize(1000, 650)
        self.events = Queue()
        self.busy = False
        self.window.storage_busy=lambda:self.busy
        self.closed = False
        self.result = None
        self.mono = None
        self.paths = None
        self.kind = "mono"
        self.calibration_path = self.directory / "data" / "calibration.npz"
        self.rectified = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Откройте обычное фото или левое и правое изображения стереопары.")
        self.measurement = tk.StringVar(value="Одно фото: относительная глубина. Стереопара с калибровкой: метры.")
        self.source_text = tk.StringVar(value="Изображения пока не выбраны")
        self.calibration_text = tk.StringVar(value=f"Калибровка стереопары: {self.calibration_path.name}")
        top = ttk.Frame(self.window, padding=10)
        top.pack(fill="x")
        ttk.Button(top, text="Открыть одно фото…", command=self.load_one).pack(side="left", padx=3)
        ttk.Button(top, text="Открыть стереопару…", command=self.load_pair).pack(side="left", padx=3)
        ttk.Button(top, text="Калибровка стереопары…", command=self.load_calibration).pack(side="left", padx=3)
        self.compute_button = ttk.Button(top, text="Рассчитать глубину", command=self.compute)
        self.compute_button.pack(side="left", padx=3)
        ttk.Checkbutton(self.window, text="Стереофото уже выровнены (например, left.png / right.png из сохранённого снимка)",
                        variable=self.rectified).pack(anchor="w", padx=14)
        ttk.Label(self.window, textvariable=self.source_text, wraplength=1200).pack(anchor="w", padx=14, pady=4)
        ttk.Label(self.window, textvariable=self.calibration_text, wraplength=1200).pack(anchor="w", padx=14)
        views = ttk.Frame(self.window, padding=10)
        views.pack(fill="both", expand=True)
        views.columnconfigure(0, weight=1)
        views.columnconfigure(1, weight=1)
        views.rowconfigure(0, weight=1)
        self.left_panel = ImagePanel(views, "Исходное изображение", self.pick)
        self.depth_panel = ImagePanel(views, "Глубина · синий ближе, красный дальше", self.pick)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=4)
        self.depth_panel.grid(row=0, column=1, sticky="nsew", padx=4)
        bar = ttk.Frame(self.window, padding=10)
        bar.pack(fill="x")
        ttk.Button(bar, text="Сохранить глубину и 3D-модель", command=self.save).pack(side="left", padx=3)
        ttk.Button(bar, text="Покрутить 3D-модель", command=self.view).pack(side="left", padx=3)
        ttk.Label(self.window, textvariable=self.measurement, wraplength=1200).pack(anchor="w", padx=14, pady=4)
        ttk.Label(self.window, textvariable=self.status, wraplength=1200).pack(anchor="w", padx=14, pady=(0, 10))
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.after(50, self.poll)

    def choose_image(self, title):
        return filedialog.askopenfilename(parent=self.window, title=title,
                                          filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Все файлы", "*.*")])

    def load_one(self):
        if self.busy:
            return
        path = self.choose_image("Одно обычное фото")
        if path:
            self.paths, self.kind, self.result = (path,), "mono", None
            self.source_text.set(path)
            self.preview(path)
            self.status.set("Одно фото: будет получена приблизительная относительная глубина, без метров.")

    def load_pair(self):
        if self.busy:
            return
        left = self.choose_image("Левое изображение стереопары")
        if not left:
            return
        right = self.choose_image("Правое изображение той же стереопары")
        if right:
            self.paths, self.kind, self.result = (left, right), "stereo", None
            self.source_text.set(f"Левое: {left}\nПравое: {right}")
            self.preview(left)
            self.status.set("Нужна калибровка этой камеры для того же размера изображений.")

    def preview(self, path):
        try:
            image = read_image(path)
            if image is None:
                raise ValueError("Не удалось прочитать изображение.")
            self.left_panel.set(image)
            self.depth_panel.set(np.full_like(image, (30, 25, 21)))
        except (OSError, ValueError) as error:
            self.status.set(str(error))

    def load_calibration(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self.window, filetypes=[("Калибровка", "*.npz")],
                                          initialdir=self.directory / "data")
        if path:
            self.calibration_path = Path(path)
            self.calibration_text.set(f"Калибровка стереопары: {path}")

    def start_job(self, action):
        if self.busy:
            return
        self.busy = True
        self.compute_button.configure(state="disabled")

        def run():
            try:
                action()
            except Exception as error:
                self.events.put(("error", str(error)))
            finally:
                self.events.put(("done", None))

        Thread(target=run, daemon=True).start()

    def compute(self):
        if not self.paths:
            self.status.set("Сначала выберите изображение.")
            return
        paths, kind = self.paths, self.kind
        calibration_path, rectified = self.calibration_path, self.rectified.get()
        self.status.set("Вычисляю глубину…")

        def job():
            started = time.perf_counter()
            left = read_image(paths[0])
            if left is None:
                raise ValueError("Не удалось открыть фотографию.")
            calibration = raw_calibration = inverse = None
            if kind == "mono":
                if self.mono is None:
                    self.mono = MonoDepth(self.directory, lambda message: self.events.put(("status", message)))
                frame, inverse = self.mono.compute(left)
            else:
                right = read_image(paths[1])
                if right is None:
                    raise ValueError("Не удалось открыть правую фотографию.")
                raw_calibration = Calibration.load(calibration_path)
                engine = DepthEngine(raw_calibration, 20., 128, quality="high", processing_width=1280,
                                     temporal=False, near_priority=True, wide_view=not rectified)
                frame = engine.compute(left, right, rectified=rectified)
                calibration = engine.calibration
            self.events.put(("result", {"frame": frame, "inverse": inverse, "calibration": calibration,
                                         "raw_calibration": raw_calibration, "paths": paths,
                                         "elapsed": time.perf_counter() - started}))

        self.start_job(job)

    def pick(self, x, y):
        if not self.result:
            return
        frame = self.result["frame"]
        value = measure(frame, x, y)
        if value is None:
            self.measurement.set("В выбранном участке нет достоверной глубины.")
        elif frame.units == "relative":
            self.measurement.set(f"Условная глубина: {value['z']:.2f}. Это оценка нейросети; метрический масштаб неизвестен.")
        else:
            self.measurement.set(f"Глубина Z: {value['z']:.2f} м · расстояние от левой камеры: {value['range']:.2f} м")

    def save(self):
        if not self.result:
            self.status.set("Сначала рассчитайте глубину.")
            return
        result = self.result

        def job():
            directory = self.directory / "output" / f"photo_{stamp()}"
            directory.mkdir(parents=True)
            frame = result["frame"]
            write_image(directory / "source.png", frame.left)
            write_image(directory / "depth_color.png", frame.color)
            if frame.preview_color is not None:
                write_image(directory / "depth_preview.png", frame.preview_color)
                write_image(directory / "depth_interpolated_mask.png", np.uint8(frame.preview_filled) * 255)
            np.save(directory / ("depth_metres.npy" if frame.units == "metres" else "depth_relative.npy"),
                    frame.points[:, :, 2])
            export_ply(directory / "cloud.ply", frame.points[frame.valid], frame.left[frame.valid][:, ::-1], frame.units)
            points, colors, faces = surface_from_frame(frame)
            export_surface(directory / "surface.ply", points, colors, faces, frame.units)
            metadata = {"units": frame.units, "method": frame.method, "source_paths": result["paths"],
                        "approximate_projection": frame.units == "relative",
                        "depth_color_range": frame.color_limits,
                        "vertical_alignment_error_px": frame.alignment_error,
                        "depth_quality_warning": frame.quality_warning,
                        "preview_interpolation": "display only; excluded from depth arrays and 3D models"}
            if frame.units == "relative":
                np.save(directory / "inverse_depth_raw.npy", result["inverse"])
                metadata.update(model=MODEL_ID, revision=MODEL_REVISION,
                                warning="Relative depth and illustrative projection; NOT metric distances")
            else:
                result["calibration"].save(directory / "calibration.npz")
                result["raw_calibration"].save(directory / "raw_calibration.npz")
            (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            self.events.put(("status", f"Глубина и модели сохранены: {directory}"))

        self.start_job(job)

    def view(self):
        if self.result:
            f = self.result["frame"]
            CloudViewer(self.window, f.points[f.valid], f.left[f.valid][:, ::-1], f.units,
                        "3D по фотографии" + (" — условный масштаб" if f.units == "relative" else ""))

    def poll(self):
        if self.closed:
            return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "result":
                    self.result = value
                    f = value["frame"]
                    self.left_panel.set(f.left)
                    self.depth_panel.set(f.preview_color if f.preview_color is not None else f.color)
                    self.measurement.set("Относительная глубина: метрический масштаб неизвестен." if f.units == "relative"
                                         else "Стереоглубина в метрах; щёлкните по объекту для измерения.")
                    self.status.set(f"Готово за {value['elapsed']:.1f} с · {f.left.shape[1]} × {f.left.shape[0]}")
                    if f.quality_warning:
                        self.status.set(f.quality_warning)
                elif kind in ("status", "error"):
                    self.status.set(("Ошибка: " if kind == "error" else "") + value)
                elif kind == "done":
                    self.busy = False
                    self.compute_button.configure(state="normal")
        except Empty:
            pass
        self.window.after(50, self.poll)

    def close(self):
        self.closed = True
        self.window.destroy()
