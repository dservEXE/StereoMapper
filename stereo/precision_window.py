from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from .calibration import write_image
from .help_ui import help_icon


class PrecisionWindow:
    def __init__(self, parent, directory, result):
        from .app import ImagePanel
        self.root, self.result = Path(directory), result
        self.window = tk.Toplevel(parent)
        self.window.title("Листья и мелкие детали · точный снимок")
        self.window.geometry("1180x900")
        self.points, self.crop, self.value, self.saved = [], None, None, None
        self.window.storage_paths=lambda:[self.saved] if self.saved else []
        self.plant = tk.StringVar(value="plant_01")
        self.leaf = tk.StringVar(value="leaf_01")
        self.kind = tk.StringVar(value="Длина")
        self.reference = tk.StringVar()
        self.message = tk.StringVar(value="Выберите две точки на снимке. Колесо — увеличение, «Весь кадр» — сброс.")
        header = ttk.Frame(self.window, padding=12)
        header.pack(fill="x")
        ttk.Label(header, text="Измерение листа", font=("Segoe UI", 18, "bold")).pack(side="left")
        help_icon(header, "precision").pack(side="left", padx=8)
        ttk.Button(header, text="Весь кадр", command=self.reset_zoom).pack(side="right")
        ttk.Button(header, text="Выбрать точки заново", command=self.clear).pack(side="right", padx=6)
        self.panel = ImagePanel(self.window, "Полное разрешение · две точки задают прямой отрезок в 3D", self.pick)
        self.panel.pack(fill="both", expand=True, padx=12)
        self.panel.canvas.bind("<MouseWheel>", self.zoom)
        controls = ttk.Frame(self.window, padding=12)
        controls.pack(fill="x")
        for title, variable in (("Растение", self.plant), ("Лист", self.leaf)):
            ttk.Label(controls, text=title).pack(side="left", padx=(0, 4))
            ttk.Entry(controls, textvariable=variable, width=16).pack(side="left", padx=(0, 12))
        ttk.Combobox(controls, textvariable=self.kind, values=("Длина", "Ширина", "Эталон"), width=12,
                     state="readonly").pack(side="left", padx=6)
        ttk.Label(controls, text="Настоящий размер эталона, мм:").pack(side="left", padx=4)
        ttk.Entry(controls, textvariable=self.reference, width=8).pack(side="left")
        ttk.Button(controls, text="Сохранить измерение", command=self.save).pack(side="right")
        ttk.Label(self.window, textvariable=self.message, wraplength=1140, foreground="#146f72").pack(anchor="w", padx=14, pady=5)
        ttk.Label(self.window, text="Цель: различать 1 мм. Точность пока не подтверждена эталоном. Повторяемость пяти кадров не равна абсолютной точности.\n"
                  "Прямой отрезок не равен длине изогнутого листа по поверхности. Сглаженные и предполагаемые точки исключены.",
                  wraplength=1140, foreground="#5f7185").pack(anchor="w", padx=14, pady=(0, 12))
        self.redraw()

    def redraw(self):
        image = self.result.frame.left.copy()
        for i, point in enumerate(self.points):
            cv2.drawMarker(image, point, (0, 230, 255), cv2.MARKER_CROSS, 18, 2)
            if i:
                cv2.line(image, self.points[i-1], point, (0, 230, 255), 2)
        if self.crop:
            x, y, w, h = self.crop
            image = image[y:y+h, x:x+w]
        self.panel.set(image)

    def clear(self):
        self.points, self.value = [], None
        self.redraw()

    def reset_zoom(self):
        self.crop = None
        self.redraw()

    def zoom(self, event):
        height, width = self.result.frame.valid.shape
        x, y, w, h = self.crop or (0, 0, width, height)
        factor = .8 if event.delta > 0 else 1.25
        nw, nh = min(width, max(60, round(w * factor))), min(height, max(40, round(h * factor)))
        if self.panel.geometry:
            ox, oy, dw, dh, _, _ = self.panel.geometry
            u, v = np.clip((event.x-ox)/dw, 0, 1), np.clip((event.y-oy)/dh, 0, 1)
        else:
            u = v = .5
        nx = int(np.clip(x + u*w - u*nw, 0, width-nw))
        ny = int(np.clip(y + v*h - v*nh, 0, height-nh))
        self.crop = (nx, ny, nw, nh)
        self.redraw()
        return "break"

    def pick(self, x, y):
        if self.crop:
            x, y = x+self.crop[0], y+self.crop[1]
        if len(self.points) >= 2:
            self.points = []
        self.points.append((x, y))
        self.value = None
        if len(self.points) == 2:
            try:
                self.value = self.result.length(*self.points)
                r = self.value
                repeat = f"{r['repeatability_mm']:.3f} мм" if r["repeatability_mm"] is not None else "не оценена"
                self.message.set(f"Отрезок: {r['length_mm']:.2f} мм · повторяемость: {repeat} · кадров: {r['samples']}. "
                                 "Для проверки точности выберите «Эталон» и введите его размер.")
            except ValueError as error:
                self.message.set(str(error))
        self.redraw()

    def save(self):
        if self.value is None:
            self.message.set("Сначала выберите две точки с принятой глубиной.")
            return
        try:
            reference = float(self.reference.get().replace(",", ".")) if self.kind.get() == "Эталон" else None
            if reference is not None and (not np.isfinite(reference) or reference <= 0):
                raise ValueError("Введите положительный известный размер эталона.")
            if not self.plant.get().strip() or not self.leaf.get().strip():
                raise ValueError("Введите обозначения растения и листа.")
            if self.saved is None:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                self.saved = self.root / "output" / f"precision_{stamp}"
                self.saved.mkdir(parents=True)
                f = self.result.frame
                for name, image in (("raw_left.png", self.result.raw[0]), ("raw_right.png", self.result.raw[1]),
                                    ("left.png", f.left), ("depth_color.png", f.color)):
                    write_image(self.saved/name, image)
                np.save(self.saved/"depth_metres.npy", f.points[:,:,2])
                self.result.calibration.save(self.saved/"calibration.npz")
                self.result.source_calibration.save(self.saved/"raw_calibration.npz")
            error = self.value["length_mm"] - reference if reference is not None else None
            row = {"time_utc": datetime.now(timezone.utc).isoformat(), "plant": self.plant.get().strip(),
                   "leaf": self.leaf.get().strip(), "kind": self.kind.get(), **self.value,
                   "reference_mm": reference, "error_mm": error, "snapshot": str(self.saved),
                   "method": "straight 3D chord from measured points; accuracy not certified"}
            path = self.root / "output" / "leaf_measurements.csv"
            exists = path.exists()
            with path.open("a", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            (self.saved/f"measurement_{datetime.now(timezone.utc).strftime('%H%M%S_%f')}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            self.message.set(f"Сохранено в {path}" + (f" · ошибка на этом эталоне {error:+.2f} мм" if error is not None else ""))
        except (ValueError, OSError) as error:
            self.message.set(str(error))
