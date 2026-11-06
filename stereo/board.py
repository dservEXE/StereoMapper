from dataclasses import dataclass
from pathlib import Path
import json
import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image

from .calibration_ui import CalibrationShortcuts, StereoPreview


@dataclass(frozen=True)
class BoardSpec:
    columns: int = 9
    rows: int = 6
    square_mm: float = 25.

    def validate(self):
        if not 3 <= self.columns <= 20 or not 3 <= self.rows <= 20 or self.columns == self.rows:
            raise ValueError("Нужно разное число столбцов и строк внутренних углов, от 3 до 20.")
        if not math.isfinite(self.square_mm) or not .5 <= self.square_mm <= 200:
            raise ValueError("Размер клетки должен быть от 0,5 до 200 мм.")
        return self


def board_image(spec, cell_pixels, border_pixels=None):
    spec.validate()
    if cell_pixels < 4:
        raise ValueError("Клетка слишком мала: нужно хотя бы 4 пикселя.")
    border = max(8, cell_pixels // 2) if border_pixels is None else border_pixels
    width, height = (spec.columns + 1) * cell_pixels + 2 * border, (spec.rows + 1) * cell_pixels + 2 * border
    if width * height > 40_000_000:
        raise ValueError("Рисунок слишком большой. Уменьшите клетку или число углов.")
    image = np.full((height, width), 255, np.uint8)
    for y in range(spec.rows + 1):
        for x in range(spec.columns + 1):
            if (x + y) % 2 == 0:
                image[border + y * cell_pixels:border + (y + 1) * cell_pixels,
                      border + x * cell_pixels:border + (x + 1) * cell_pixels] = 0
    return image


class BoardWindow:
    def __init__(self, parent, directory, spec, on_apply=None, on_capture=None, counter_var=None,
                 status_var=None, shortcuts=None):
        self.directory, self.on_apply, self.on_capture = Path(directory), on_apply, on_capture
        self.window = tk.Toplevel(parent)
        self.window.title("Экранная калибровочная доска")
        self.window.geometry("1180x900")
        self.window.minsize(760, 600)
        self.fullscreen = False
        self.closed = False
        self.capture_status = status_var if status_var is not None else tk.StringVar(master=self.window)
        self.spec = spec.validate()
        self.columns, self.rows = tk.StringVar(value=str(spec.columns)), tk.StringVar(value=str(spec.rows))
        self.square = tk.StringVar(value=f"{spec.square_mm:g}")
        self.measured = tk.StringVar(value="100")
        self.message = tk.StringVar(value="Измерьте контрольную линию линейкой и уточните масштаб экрана.")
        self.fit_message = ""
        self.ppm = self.window.winfo_fpixels("1i") / 25.4
        self.scale_verified = False
        self.reference_pixels = round(100 * self.ppm)
        self.load_scale()
        self.controls = ttk.Frame(self.window, padding=10)
        self.controls.pack(fill="x")
        row = ttk.Frame(self.controls)
        row.pack(fill="x")
        for label, variable in (("Углов по горизонтали", self.columns), ("по вертикали", self.rows),
                                 ("Клетка, мм", self.square)):
            ttk.Label(row, text=label).pack(side="left", padx=(0, 4))
            ttk.Entry(row, textvariable=variable, width=7).pack(side="left", padx=(0, 12))
        buttons = ttk.Frame(self.controls)
        buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(buttons, text="Построить", command=self.generate).pack(side="left", padx=3)
        ttk.Button(buttons, text="Вписать в экран", command=self.fit).pack(side="left", padx=3)
        ttk.Button(buttons, text="Полный экран · F11", command=self.toggle_fullscreen).pack(side="left", padx=3)
        scale = ttk.Frame(self.controls)
        scale.pack(fill="x", pady=8)
        ttk.Label(scale, text="Реальная длина контрольной линии, мм:").pack(side="left")
        ttk.Entry(scale, textvariable=self.measured, width=8).pack(side="left", padx=6)
        ttk.Button(scale, text="Уточнить масштаб по линейке", command=self.calibrate_scale).pack(side="left", padx=3)
        action = ttk.Frame(self.controls)
        action.pack(fill="x")
        ttk.Button(action, text="Применить для калибровки", command=self.apply).pack(side="left", padx=3)
        ttk.Button(action, text="Сохранить PNG для печати", command=self.save_png).pack(side="left", padx=3)
        if on_capture:
            self.capture_button = ttk.Button(action, text="Снять пару · пробел / S", command=self.capture)
            self.capture_button.pack(side="left", padx=3)
        if counter_var is not None:
            ttk.Label(action, textvariable=counter_var).pack(side="left", padx=12)
        self.message_label = ttk.Label(self.controls, textvariable=self.message, wraplength=1100)
        self.message_label.pack(anchor="w", pady=(7, 0))
        self.controls.bind("<Configure>", lambda event: self.message_label.configure(wraplength=max(200, event.width - 30)))
        self.canvas = tk.Canvas(self.window, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.preview = StereoPreview(self.canvas, counter_var)
        self.preview.place(relx=1, x=-12, y=12, anchor="ne")
        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.window.bind("<F11>", lambda event: self.toggle_fullscreen())
        self.window.bind("<Escape>", lambda event: self.toggle_fullscreen() if self.fullscreen else self.close())
        self.shortcuts = shortcuts or CalibrationShortcuts(
            self.window, lambda window: self.capture(), lambda window: window is self.window)
        self.shortcuts.add_window(self.window)
        self.status_trace = self.capture_status.trace_add("write", lambda *args: self.draw())
        self.window.bind("<Destroy>", self.on_destroy, add="+")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.draw()
        self.window.after_idle(self.canvas.focus_set)

    def load_scale(self):
        try:
            data = json.loads((self.directory / "screen_board.json").read_text(encoding="utf-8"))
            screen = [self.window.winfo_screenwidth(), self.window.winfo_screenheight()]
            if data["screen"] == screen and .5 <= data["pixels_per_mm"] <= 30:
                self.ppm, self.scale_verified = float(data["pixels_per_mm"]), True
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def parse(self):
        return BoardSpec(int(self.columns.get()), int(self.rows.get()),
                         float(self.square.get().replace(",", "."))).validate()

    @property
    def cell_pixels(self):
        return max(4, round(self.spec.square_mm * self.ppm))

    @property
    def effective_spec(self):
        return BoardSpec(self.spec.columns, self.spec.rows, self.cell_pixels / self.ppm)

    def generate(self):
        try:
            self.spec = self.parse()
            self.draw()
            self.message.set("Доска построена. Проверьте размер клетки линейкой; примените его перед съёмкой.")
            return True
        except (ValueError, tk.TclError) as error:
            self.message.set(str(error))
            return False

    def draw(self):
        if self.closed:
            return
        self.canvas.delete("all")
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        cell = self.cell_pixels
        grid_w, grid_h = (self.spec.columns + 1) * cell, (self.spec.rows + 1) * cell
        ox, oy = round((width - grid_w) / 2), round((height - 100 - grid_h) / 2)
        fits = grid_w + cell <= width - 16 and grid_h + cell <= height - 100
        for y in range(self.spec.rows + 1):
            for x in range(self.spec.columns + 1):
                if (x + y) % 2 == 0:
                    self.canvas.create_rectangle(ox + x * cell, oy + y * cell,
                                                 ox + (x + 1) * cell, oy + (y + 1) * cell,
                                                 fill="black", outline="")
        self.reference_pixels = round(100 * self.ppm)
        start = round((width - self.reference_pixels) / 2)
        baseline = height - 48
        self.canvas.create_line(start, baseline, start + self.reference_pixels, baseline, fill="#222", width=2)
        for x in (start, start + self.reference_pixels):
            self.canvas.create_line(x, baseline - 7, x, baseline + 7, fill="#222", width=2)
        verified = "масштаб проверен" if self.scale_verified else "масштаб требует проверки линейкой"
        self.canvas.create_text(width / 2, baseline - 19, text=f"Контрольная линия ≈100 мм · {verified}", fill="#333")
        self.fit_message = "" if fits else "НЕ ПОМЕЩАЕТСЯ: полный экран или «Вписать в экран»"
        caption = (f"{self.spec.columns} × {self.spec.rows} внутренних углов · клетка ≈{self.effective_spec.square_mm:.2f} мм "
                   f"({cell} px) · пробел / S: снимок · F11: полный экран · Esc: назад")
        caption = self.capture_status.get() or caption
        if self.fit_message:
            caption = self.fit_message
        self.canvas.create_text(width / 2, height - 20, text=caption, width=max(120, width - 32),
                                fill="#a00000" if not fits else "#333")

    def fit(self):
        try:
            spec = self.parse()
            cell = int(min((self.canvas.winfo_width() - 32) / (spec.columns + 2),
                           (self.canvas.winfo_height() - 110) / (spec.rows + 2)))
            if cell < 4:
                raise ValueError("Увеличьте окно.")
            self.square.set(f"{cell / self.ppm:.5f}")
            self.generate()
            self.message.set("Размер клетки изменён, чтобы доска поместилась. Примените новый размер для калибровки.")
        except (ValueError, tk.TclError) as error:
            self.message.set(str(error))

    def calibrate_scale(self):
        try:
            measured = float(self.measured.get().replace(",", "."))
            if not math.isfinite(measured) or not 20 <= measured <= 300:
                raise ValueError("Введите измеренную длину контрольной линии, от 20 до 300 мм.")
            self.ppm = self.reference_pixels / measured
            self.scale_verified = True
            data = {"pixels_per_mm": self.ppm,
                    "screen": [self.window.winfo_screenwidth(), self.window.winfo_screenheight()]}
            (self.directory / "screen_board.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.generate()
            self.message.set("Масштаб уточнён. Перепроверьте линию и клетку; при переносе на другой монитор настройте заново.")
        except (ValueError, OSError) as error:
            self.message.set(str(error))

    def apply(self):
        if not self.generate():
            return False
        if self.fit_message:
            self.message.set("Сначала разместите всю доску в окне: полный экран или «Вписать в экран».")
            return False
        if self.on_apply:
            self.on_apply(self.effective_spec)
        self.message.set(f"Для калибровки установлены {self.spec.columns} × {self.spec.rows} углов "
                         f"и клетка {self.effective_spec.square_mm:.3f} мм. Физический размер проверьте линейкой.")
        return True

    def capture(self):
        if not self.apply():
            self.capture_status.set(self.message.get())
            return False
        if self.on_capture:
            return self.on_capture()
        return False

    def save_png(self):
        if not self.generate():
            return
        try:
            image = board_image(self.spec, self.cell_pixels)
            directory = self.directory / "output" / "boards"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"board_{self.spec.columns}x{self.spec.rows}_{self.effective_spec.square_mm:.3f}mm.png"
            Image.fromarray(image).save(path, dpi=(self.ppm * 25.4,) * 2)
            path.with_suffix(".json").write_text(json.dumps({"inner_corners": [self.spec.columns, self.spec.rows],
                "square_mm": self.effective_spec.square_mm, "cell_pixels": self.cell_pixels,
                "dpi": self.ppm * 25.4, "screen_scale_verified": self.scale_verified}, indent=2), encoding="utf-8")
            self.message.set(f"Сохранено: {path}. При печати выберите масштаб 100% и проверьте клетку линейкой.")
        except (OSError, ValueError) as error:
            self.message.set(str(error))

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.controls.pack_forget()
        else:
            self.controls.pack(fill="x", before=self.canvas)
        self.window.attributes("-fullscreen", self.fullscreen)
        self.window.lift()
        self.canvas.focus_set()

    def on_destroy(self, event):
        if event.widget is self.window:
            self.closed = True
            self.capture_status.trace_remove("write", self.status_trace)

    def close(self):
        self.closed = True
        self.window.destroy()
