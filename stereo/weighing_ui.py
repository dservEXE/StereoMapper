"""The weighing tab: starts by itself when opened and stops when left, to free the processor."""
from datetime import datetime
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from .body_measure import MIN_KEYPOINT_SCORE, MIN_SAMPLES, SKELETON
from .people_detection import DOWNLOAD_SIZE_MB, download_models, models_ready
from .ui import note

OUTLINE_OPACITY = .35
PART_NAMES = (("torso", "туловище"), ("upper_arm", "плечо"), ("forearm", "предплечье"),
              ("thigh", "бедро"), ("shin", "голень"), ("head", "голова"))


def draw_people(image, state):
    """Overlay labels and joints. Text is Latin because OpenCV fonts have no Cyrillic."""
    canvas = image.copy()
    for person in state.people:
        # What the volume is actually computed from; it trails the picture by a moment.
        if person.outline is not None and person.outline.shape == canvas.shape[:2]:
            tint = np.array(person.color, np.float32)
            canvas[person.outline] = (canvas[person.outline] * (1 - OUTLINE_OPACITY) + tint * OUTLINE_OPACITY).astype(np.uint8)
        x0, y0, x1, y1 = (int(value) for value in person.box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), person.color, 2)
        for a, b in SKELETON:
            if min(person.keypoint_scores[a], person.keypoint_scores[b]) >= MIN_KEYPOINT_SCORE:
                cv2.line(canvas, tuple(int(v) for v in person.keypoints[a]),
                         tuple(int(v) for v in person.keypoints[b]), person.color, 2)
        estimate = person.estimate
        text = f"#{person.track_id}"
        if estimate.mass_kg is not None:
            text += f"  ~{estimate.mass_kg:.0f} kg"
        elif estimate.height_m is not None:
            text += f"  {estimate.height_m:.2f} m"
        cv2.rectangle(canvas, (x0, max(0, y0 - 24)), (x0 + 12 * len(text) + 8, y0), person.color, -1)
        cv2.putText(canvas, text, (x0 + 4, max(14, y0 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def person_report(track_id, estimate) -> str:
    lines = [f"Человек №{track_id}"]
    lines.append(f"Рост: {estimate.height_m * 100:.0f} см" if estimate.height_m else "Рост: ещё не измерен")
    if estimate.mass_kg is not None:
        lines.append(f"Вес: около {estimate.mass_kg:.0f} кг ({estimate.mass_low_kg:.0f}–{estimate.mass_high_kg:.0f})")
    for part, name in PART_NAMES:
        if part in estimate.widths_m:
            side = estimate.side_widths_m.get(part)
            suffix = f", бок {side * 100:.0f}" if side else ""
            lines.append(f"{name}: {estimate.widths_m[part] * 100:.0f} см{suffix}")
    views = estimate.view_coverage
    lines.append(f"Кадров: {estimate.samples}; спереди {views.get('front', 0)}, боком {views.get('side', 0)}")
    lines.append(estimate.message)
    return "\n".join(lines)


class WeighingTab:
    def __init__(self, app, page, sidebar, view, footer, heading, action, panel_class):
        self.app, self.page = app, page
        self.active = False
        self.shown = None
        self.downloads = Queue()
        self.downloading = False
        self.message = tk.StringVar(value="Откройте вкладку после подключения и калибровки камер.")
        self.report = tk.StringVar(value="")
        heading(sidebar, "Взвесим людей", "Встаньте целиком в кадр в 2–3 м от камеры и в метре от стены, "
                "разведите руки и медленно повернитесь на месте. Нужна облегающая одежда: свободная завышает вес.")
        note(sidebar, "Вес — приблизительная оценка по объёму тела, с весами не сверялась. "
                      "Номера действуют только в этом сеансе; лица не распознаются.")
        ttk.Label(sidebar, textvariable=self.message, style="Section.TLabel", wraplength=280).pack(anchor="w", pady=(8, 4))
        ttk.Label(sidebar, textvariable=self.report, wraplength=280, justify="left").pack(anchor="w", fill="x")
        self.download_button = action(footer, f"Скачать модели ({DOWNLOAD_SIZE_MB} МБ)", self.start_download, primary=True)
        self.reset_button = action(footer, "Начать замеры заново", lambda: app.send("weighing_reset"), "calibrated")
        self.save_button = action(footer, "Сохранить результат", self.save, "calibrated")
        self.panel = panel_class(view, "Люди в кадре", empty="Здесь появятся найденные люди",
                                 hint="Подключите камеры и встаньте в кадр")
        self.panel.pack(fill="both", expand=True)
        self.available = models_ready(app.directory)
        app.workflow.bind("<<NotebookTabChanged>>", self.tab_changed, add="+")
        self.show_download_state()

    def ready(self) -> bool:
        return self.available

    def show_download_state(self) -> None:
        self.available = models_ready(self.app.directory)
        if self.available:
            self.download_button.pack_forget()
        elif not self.downloading:
            self.message.set(f"Для поиска людей нужны две модели, {DOWNLOAD_SIZE_MB} МБ. Они скачиваются один раз "
                             "и работают на этом компьютере; кадры никуда не передаются.")

    def selected(self) -> bool:
        return self.app.workflow.select() == str(self.page)

    def tab_changed(self, event=None) -> None:
        self.sync()

    def sync(self) -> None:
        """Run only while this tab is open, the camera is connected and the models are here."""
        worker = self.app.worker
        wanted = bool(self.selected() and self.app.connected and worker and getattr(worker, "engine", None)
                      and self.ready())
        if wanted != self.active:
            self.active = wanted
            # After a disconnect the worker is gone and there is nothing left to stop.
            if self.app.connected:
                self.app.send("weighing", wanted)
            if not wanted:
                self.report.set("")
        if self.selected() and self.ready() and not wanted:
            self.message.set("Подключите камеры и подготовьте калибровку на первых вкладках.")

    def start_download(self) -> None:
        if self.downloading:
            return
        self.downloading = True
        self.download_button.configure(state="disabled")
        self.message.set("Скачиваю модели…")

        def work():
            try:
                download_models(self.app.directory, lambda text: self.downloads.put(("status", text)))
                self.downloads.put(("done", None))
            except RuntimeError as error:
                self.downloads.put(("error", str(error)))
        Thread(target=work, daemon=True, name="model-download").start()

    def poll(self, packet) -> None:
        try:
            while True:
                kind, value = self.downloads.get_nowait()
                if kind == "status":
                    self.message.set(value)
                else:
                    self.downloading = False
                    self.download_button.configure(state="normal")
                    self.message.set("Модели скачаны." if kind == "done" else value)
                    self.show_download_state()
                    self.sync()
        except Empty:
            pass
        if self.selected() and not self.downloading:
            self.sync()
        if not self.active:
            return
        state = packet.get("weighing") if packet else None
        if state is None or state is self.shown:
            return
        self.shown = state
        self.message.set(state.message + (f" · {state.fps:.1f} кадра/с" if state.fps else ""))
        self.report.set("\n\n".join(person_report(person.track_id, person.estimate) for person in state.people)
                        or f"Нужно минимум {MIN_SAMPLES} кадров с видимым туловищем.")
        if state.image is not None:
            self.panel.set(draw_people(state.image, state))

    def save(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.app.send("weighing_save", self.app.directory / "output" / f"weighing_{stamp}" / "result.json")
