"""The pig weighing tab: photographs, video files and the connected camera, analysed only while the tab is open."""
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np

from .people_detection import download_models, models_ready
from .pig_detection import SOURCES, missing_size_mb
from .pig_weighing import PICTURE, SEGMENT, SEGMENT_FAILED, STEREO, PigSession, write_result
from .ui import note
from .version import VERSION

NO_VALUE = "—"
OUTLINE_OPACITY = .35
SEGMENT_COLOR = (60, 220, 255)
# Labels grow with the picture, which is shown reduced: 1400 pixels of width per unit of font size.
LABEL_WIDTH_PER_UNIT = 1400
PHOTO_TYPES = [("Фото", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Все файлы", "*.*")]
VIDEO_TYPES = [("Видео", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"), ("Все файлы", "*.*")]
COLUMNS = (("pig", "№", 46), ("mass", "Вес, кг", 80), ("range", "Диапазон, кг", 110), ("length", "Длина, см", 90),
           ("girth", "Обхват груди, см", 130), ("frames", "Кадров", 70), ("note", "Примечание", 300))
TABLE_ROWS = 5
WRAP = 280
SCALE_TEXTS = {
    PICTURE: "Расстояния оценены по самому снимку, вес приблизительный. Отрезок известной длины его уточнит.",
    SEGMENT: "Масштаб уточнён по отмеченному отрезку.",
    STEREO: "Расстояния измерены стереокамерой, отрезок не нужен.",
    SEGMENT_FAILED: "Отрезок измерить не удалось. Отметьте его на неподвижном, хорошо видном предмете.",
}
NO_CAMERA = "Сначала подключите камеры на вкладке «1 Камеры»."


def parse_number(text) -> float | None:
    """A positive number typed with a comma or a point; anything else is no value."""
    try:
        value = float(str(text).strip().replace(",", "."))
    except ValueError:
        return None
    return value if value > 0 else None


def draw_pigs(state, segment=None, points=()) -> np.ndarray:
    """Outlines, body lines and labels. Text is Latin because OpenCV fonts have no Cyrillic."""
    canvas = state.image.copy()
    size = max(.5, canvas.shape[1] / LABEL_WIDTH_PER_UNIT)
    weight = 2 if size >= .8 else 1
    for pig in state.pigs:
        if pig.outline is not None and pig.outline.shape == canvas.shape[:2]:
            tint = np.array(pig.color, np.float32)
            canvas[pig.outline] = (canvas[pig.outline] * (1 - OUTLINE_OPACITY) + tint * OUTLINE_OPACITY).astype(np.uint8)
    for pig in state.pigs:
        x, y = int(pig.box[0]), int(pig.box[1]) + 20
        if pig.shape is not None:
            neck, rump = (tuple(int(value) for value in point) for point in (pig.shape.neck, pig.shape.rump))
            cv2.line(canvas, neck, rump, (255, 255, 255), 2, cv2.LINE_AA)
            x, y = (neck[0] + rump[0]) // 2, (neck[1] + rump[1]) // 2
        text = f"#{pig.track_id}" + (f" {pig.estimate.mass_kg:.0f} kg" if pig.estimate else "")
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, weight)
        left = max(0, min(canvas.shape[1] - width - 8, x - width // 2 - 4))
        top = max(0, min(canvas.shape[0] - height - 8, y - height - 6))
        cv2.rectangle(canvas, (left, top), (left + width + 8, top + height + 8), pig.color, -1)
        cv2.putText(canvas, text, (left + 4, top + height + 3), cv2.FONT_HERSHEY_SIMPLEX, size, (20, 20, 20), weight, cv2.LINE_AA)
    marks = [tuple(int(value) for value in point) for point in (segment or points)]
    if len(marks) == 2:
        cv2.line(canvas, marks[0], marks[1], SEGMENT_COLOR, 2, cv2.LINE_AA)
    for mark in marks:
        cv2.circle(canvas, mark, 5, SEGMENT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, mark, 6, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def pig_row(pig) -> tuple:
    """One table line: number, weight, range, length, girth, frames, and why a value is missing."""
    estimate = pig.estimate
    if estimate is None:
        return (pig.track_id, *(NO_VALUE,) * 4, 0, pig.note)
    return (pig.track_id, f"{estimate.mass_kg:.0f}", f"{estimate.mass_low_kg:.0f}–{estimate.mass_high_kg:.0f}",
            f"{estimate.length_m * 100:.0f}", f"{estimate.girth_m * 100:.0f}", estimate.samples,
            f"этот кадр пропущен: {pig.note}" if pig.note else "")


def time_text(seconds) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


class PigTab:
    def __init__(self, app, page, sidebar, view, footer, heading, section, action, panel_class):
        self.app, self.page = app, page
        self.session = None
        self.shown = -1
        self.state = None
        self.downloads = Queue()
        self.downloading = False
        self.suspending = False
        self.active = False
        self.marking = False
        self.points = []
        self.segment = None
        self.sent_reference = None
        self.sink_worker = None
        self.message = tk.StringVar(value="Откройте вкладку, чтобы начать.")
        self.scale_text = tk.StringVar()
        self.metric_count = tk.StringVar(value=NO_VALUE)
        self.metric_mean = tk.StringVar(value=NO_VALUE)
        self.length = tk.StringVar()
        heading(sidebar, "Взвесим свиней", "Откройте фото или видео или включите камеру. Программа сама оценит "
                "расстояние до каждой свиньи и взвесит тех, кто стоит прямо и виден целиком.")
        self.build_metrics(sidebar)
        ttk.Label(sidebar, textvariable=self.message, style="Muted.TLabel", wraplength=WRAP, justify="left").pack(anchor="w", fill="x", pady=(0, 8))
        self.sources = ttk.Frame(sidebar)
        self.sources.pack(fill="x")
        self.sources.columnconfigure((0, 1, 2), weight=1, uniform="source")
        self.photo_button = ttk.Button(self.sources, text="Фото…", command=self.open_photo)
        self.video_button = ttk.Button(self.sources, text="Видео…", command=self.open_video)
        self.camera_button = ttk.Button(self.sources, text="Камера", command=self.open_camera)
        for column, button in enumerate((self.photo_button, self.video_button, self.camera_button)):
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 3, 0 if column == 2 else 3), pady=2)
        self.controls = ttk.Frame(sidebar)
        self.controls.columnconfigure((0, 1), weight=1, uniform="video")
        self.play_button = ttk.Button(self.controls, text="Пауза", command=self.toggle_play)
        self.play_button.grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=2)
        self.rewind_button = ttk.Button(self.controls, text="Сначала", command=self.rewind)
        self.rewind_button.grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=2)
        self.controls_shown = False
        section(sidebar, "Точность")
        note(sidebar, "Чтобы вес был точнее, отметьте на кадре отрезок, длину которого знаете: ширину кормушки, "
                      "станка или линейку.")
        self.mark_button = action(sidebar, "Отметить отрезок на кадре", self.start_marking)
        form = ttk.Frame(sidebar)
        form.pack(fill="x", pady=(4, 0))
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Длина отрезка, см").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(form, textvariable=self.length, width=8).grid(row=0, column=1, sticky="e", pady=2)
        ttk.Label(sidebar, textvariable=self.scale_text, style="Muted.TLabel", wraplength=WRAP, justify="left").pack(anchor="w", fill="x", pady=(4, 0))
        ttk.Separator(sidebar).pack(fill="x", pady=10)
        note(sidebar, "Вес — приблизительная оценка, на весах она пока не проверена. Номера действуют, пока открыт "
                      "этот источник. Сохраняются только числа, без снимков.")
        self.download_button = action(footer, "Скачать модели", self.start_download, primary=True)
        self.reset_button = action(footer, "Начать замеры заново", self.reset)
        self.save_button = action(footer, "Сохранить результат", self.save)
        self.panel = panel_class(view, "Свиньи в кадре", self.clicked, empty="Здесь появятся найденные свиньи",
                                 hint="Откройте фото или видео или включите камеру")
        self.panel.pack(fill="both", expand=True)
        results = ttk.Frame(view)
        results.pack(fill="x", pady=(8, 0))
        self.table = ttk.Treeview(results, columns=[key for key, _, _ in COLUMNS], show="headings", height=TABLE_ROWS,
                                  selectmode="none")
        for key, title, width in COLUMNS:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor="w" if key == "note" else "center", stretch=key == "note")
        scroll = ttk.Scrollbar(results, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.table.pack(side="left", fill="x", expand=True)
        self.length.trace_add("write", lambda *_: self.apply_reference())
        app.workflow.bind("<<NotebookTabChanged>>", lambda event: self.sync(), add="+")
        self.show_download_state()

    def build_metrics(self, parent) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(4, 10))
        row.columnconfigure((0, 1), weight=1, uniform="metric")
        for column, (variable, caption) in enumerate(((self.metric_count, "свиней в кадре"), (self.metric_mean, "средний вес"))):
            ttk.Label(row, textvariable=variable, style="Metric.TLabel").grid(row=0, column=column, sticky="w")
            ttk.Label(row, text=caption, style="MetricCaption.TLabel").grid(row=1, column=column, sticky="w")

    # Models.
    def ready(self) -> bool:
        return models_ready(self.app.directory, SOURCES)

    def show_download_state(self) -> None:
        if self.ready():
            self.download_button.pack_forget()
        elif not self.downloading:
            size = missing_size_mb(self.app.directory)
            self.download_button.configure(text=f"Скачать модели ({size} МБ)")
            self.message.set(f"Для поиска свиней нужны модели, {size} МБ. Они скачиваются один раз и работают "
                             "на этом компьютере; кадры никуда не передаются.")

    def start_download(self) -> None:
        if self.downloading:
            return
        self.downloading = True
        self.download_button.configure(state="disabled")
        self.message.set("Скачиваю модели…")

        def work():
            try:
                download_models(self.app.directory, lambda text: self.downloads.put(("status", text)), sources=SOURCES)
                self.downloads.put(("done", "Модели скачаны."))
            except Exception as error:  # noqa: BLE001 - the button must never stay blocked.
                self.downloads.put(("done", str(error)))
        Thread(target=work, daemon=True, name="pig-model-download").start()

    # Running only while the tab is open.
    def selected(self) -> bool:
        return self.app.workflow.select() == str(self.page)

    def sync(self) -> None:
        selected = self.selected()
        if selected and self.session is None and self.ready():
            from .pig_detection import PigDetector
            self.session = PigSession(lambda status: PigDetector(self.app.directory, status))
            self.session.start()
        if self.session is not None and selected != self.active:
            self.session.set_active(selected)
        self.active = selected
        connected = bool(self.app.connected)
        # While this tab works, the cameras stay connected but depth and maps are not computed.
        wanted = selected and connected
        if wanted != self.suspending:
            self.suspending = wanted
            if connected:
                self.app.send("suspend", wanted)
        if not connected:
            self.sink_worker = None
        elif self.camera_on() and self.app.worker is not self.sink_worker:
            # The cameras were reconnected: the new camera thread does not know this tab yet.
            self.attach_camera()

    def close(self) -> None:
        if self.session is not None:
            self.session.stop()

    # Sources.
    def need_session(self) -> bool:
        if self.session is None:
            self.message.set("Сначала скачайте модели кнопкой внизу." if not self.ready() else "Модели ещё загружаются.")
            return False
        return True

    def camera_on(self) -> bool:
        return self.session is not None and self.session.kind == "camera"

    def attach_camera(self) -> None:
        self.sink_worker = self.app.worker
        self.app.send("frame_sink", self.session)

    def open_photo(self) -> None:
        if not self.need_session():
            return
        path = filedialog.askopenfilename(parent=self.page, title="Фото со свиньями", filetypes=PHOTO_TYPES)
        if path:
            self.forget_segment()
            self.session.post("photo", Path(path))

    def open_video(self) -> None:
        if not self.need_session():
            return
        path = filedialog.askopenfilename(parent=self.page, title="Видео со свиньями", filetypes=VIDEO_TYPES)
        if path:
            self.forget_segment()
            self.session.post("video", Path(path))

    def open_camera(self) -> None:
        if not self.need_session():
            return
        if not self.app.connected:
            self.message.set(NO_CAMERA)
            return
        self.forget_segment()
        self.session.post("camera")
        self.attach_camera()

    def toggle_play(self) -> None:
        if self.session is not None and self.state is not None:
            self.session.post("play", not self.state.playing)

    def rewind(self) -> None:
        if self.session is not None:
            self.session.post("rewind")

    def reset(self) -> None:
        if self.session is not None:
            self.session.post("reset")

    # Scale.
    def forget_segment(self) -> None:
        """Another picture has another scale."""
        self.segment, self.points, self.marking = None, [], False
        self.apply_reference()

    def start_marking(self) -> None:
        if self.state is None or self.state.image is None:
            self.message.set("Сначала откройте фото или видео или включите камеру.")
            return
        self.marking, self.points = True, []
        if self.state.playing:
            self.session.post("play", False)
        self.message.set("Щёлкните на кадре по двум концам отрезка, длину которого знаете.")

    def clicked(self, x, y) -> None:
        if not self.marking:
            return
        self.points.append((x, y))
        if len(self.points) == 2:
            self.segment, self.points, self.marking = tuple(self.points), [], False
            self.apply_reference()
            if not parse_number(self.length.get()):
                self.message.set("Отрезок отмечен. Введите его длину в сантиметрах.")
        self.render()

    def reference(self) -> tuple | None:
        length = parse_number(self.length.get())
        return (self.segment, length / 100) if self.segment and length else None

    def apply_reference(self) -> None:
        reference = self.reference()
        if self.session is not None and reference != self.sent_reference:
            self.sent_reference = reference
            self.session.post("reference", reference)

    # Showing.
    def render(self) -> None:
        if self.state is not None and self.state.image is not None:
            self.panel.set(draw_pigs(self.state, self.segment, self.points))

    def show_controls(self, shown) -> None:
        if shown != self.controls_shown:
            self.controls_shown = shown
            if shown:
                self.controls.pack(fill="x", after=self.sources)
            else:
                self.controls.pack_forget()

    def poll(self) -> None:
        try:
            while True:
                kind, value = self.downloads.get_nowait()
                self.message.set(value)
                if kind == "done":
                    self.downloading = False
                    self.download_button.configure(state="normal")
                    self.show_download_state()
        except Empty:
            pass
        self.sync()
        if self.session is None or not self.active:
            return
        state = self.session.state
        if state.serial == self.shown:
            return
        self.shown, self.state = state.serial, state
        video = state.kind == "video"
        self.show_controls(video)
        self.play_button.configure(text="Пауза" if state.playing else "Продолжить")
        if not self.marking:
            progress = f" · {time_text(state.position_s)} из {time_text(state.duration_s)}" if video and state.duration_s else ""
            speed = f" · {state.fps:.1f} кадра/с" if state.fps and (state.playing or state.kind == "camera") else ""
            lost = state.kind == "camera" and not self.app.connected
            self.message.set("Камеры отключены. " + NO_CAMERA if lost else state.message + progress + speed)
        self.scale_text.set(SCALE_TEXTS[state.scale_source] if state.image is not None else "")
        top = self.table.yview()[0]
        self.table.delete(*self.table.get_children())
        # Measured animals first: with a dozen in view, the ones with a weight must not hide below the fold.
        for pig in sorted(state.pigs, key=lambda pig: (pig.estimate is None, pig.track_id)):
            self.table.insert("", "end", values=pig_row(pig))
        self.table.yview_moveto(top)
        masses = [pig.estimate.mass_kg for pig in state.pigs if pig.estimate is not None]
        self.metric_count.set(str(len(state.pigs)) if state.image is not None else NO_VALUE)
        self.metric_mean.set(f"~{sum(masses) / len(masses):.0f} кг" if masses else NO_VALUE)
        if state.image is None:
            self.panel.clear()
        else:
            self.render()

    def save(self) -> None:
        state = self.state
        if state is None or not any(pig.estimate is not None for pig in state.pigs):
            self.message.set("Сохранять пока нечего: нет измеренных свиней.")
            return
        folder = f"pigs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        setup = dict(version=VERSION, source=state.kind, scale=state.scale_source)
        try:
            count = write_result(self.app.directory / "output" / folder / "result.json", state, setup)
        except OSError as error:
            self.message.set(f"Не удалось сохранить: {error}")
            return
        self.message.set(f"Сохранено свиней: {count}. Папка output\\{folder}.")
