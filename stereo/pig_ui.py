"""The pig weighing tab: photographs, video files and the connected camera, analysed only while the tab is open."""
from datetime import datetime
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import filedialog, ttk

import cv2

from .gpu_pack import installed_pack
from .people_detection import download_models, models_ready
from .pig_database import PigDatabase, correction, export_csv, make_entry, summary_text
from .pig_detection import SOURCES, missing_size_mb
from .pig_drawing import draw_marks, kg_text, label_text
from .pig_weighing import (LEARNED, PICTURE, SEGMENT, SEGMENT_FAILED, STEREO, WEIGHED, WEIGHT, WEIGHT_FAILED, PigSession,
                           result_folder, write_result)
from .ui import note
from .version import VERSION

NO_VALUE = "—"
SEGMENT_COLOR = (60, 220, 255)
PHOTO_TYPES = [("Фото", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Все файлы", "*.*")]
VIDEO_TYPES = [("Видео", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"), ("Все файлы", "*.*")]
COLUMNS = (("pig", "№", 46), ("mass", "Вес, кг", 80), ("range", "Диапазон, кг", 110), ("volume", "Объём, л", 80),
           ("length", "Длина, см", 90), ("girth", "Обхват груди, см", 130), ("frames", "Кадров", 70),
           ("note", "Примечание", 280))
TABLE_ROWS = 5
WRAP = 280
SCALE_TEXTS = {
    PICTURE: "Вес приблизительный: масштаб угадан нейросетью по самому снимку и может ошибаться в разы{height}. "
             "Введите вес одной свиньи с весов или отметьте отрезок известной длины.",
    SEGMENT: "Масштаб уточнён по отмеченному отрезку.",
    STEREO: "Расстояния измерены стереокамерой, отрезок не нужен.",
    SEGMENT_FAILED: "Отрезок измерить не удалось. Отметьте его на неподвижном, хорошо видном предмете.",
    WEIGHT: "Масштаб уточнён по весу свиньи №{pig}: остальные пересчитаны по ней.",
    WEIGHT_FAILED: "Свинья с этим номером ещё не измерена, по ней масштаб не уточнить. Выберите номер с весом из таблицы.",
    LEARNED: "Масштаб поправлен по базе: взвешенных свиней {learned}. Для другого места или камеры поправка может не "
             "подойти: тогда взвесьте здесь одну свинью.",
}
NO_CAMERA = "Сначала подключите камеры на вкладке «1 Камеры»."
ACCELERATION = "комплект на вкладке «Дополнительно»"


def parse_number(text) -> float | None:
    """A positive number typed with a comma or a point; anything else is no value."""
    try:
        value = float(str(text).strip().replace(",", "."))
    except ValueError:
        return None
    return value if value > 0 else None


def draw_pigs(state, segment=None, points=()):
    """Outlines, body lines, labels and the marked segment over a copy of the frame."""
    canvas = draw_marks(state.image.copy(), state.pigs, [label_text(pig.track_id, pig.estimate) for pig in state.pigs])
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
        return (pig.track_id, *(NO_VALUE,) * 5, 0, pig.note)
    if pig.note == WEIGHED:
        remark = WEIGHED
    elif pig.note and getattr(pig, "in_view", True):
        remark = f"этот кадр пропущен: {pig.note}"
    else:
        remark = ""
    volume = NO_VALUE if estimate.volume_l is None else kg_text(estimate.volume_l)
    return (pig.track_id, kg_text(estimate.mass_kg), f"{kg_text(estimate.mass_low_kg)}–{kg_text(estimate.mass_high_kg)}",
            volume, f"{estimate.length_m * 100:.0f}", f"{estimate.girth_m * 100:.0f}", estimate.samples, remark)


def time_text(seconds) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def scale_text(state) -> str:
    height = "" if state.camera_height_m is None else f": по нему камера в {state.camera_height_m * 100:.0f} см над полом"
    return SCALE_TEXTS[state.scale_source].format(height=height, pig=state.weighed_pig, learned=state.learned_pigs)


def device_text(device, pack) -> str:
    """Where the neural networks run, and what to do when the graphics card stays idle."""
    if not device:
        return ""
    if not device.startswith("процессор"):
        return f"Считает видеокарта: {device}."
    if pack:
        return "Считает процессор: видеокарта для этих нейросетей не подошла."
    return f"Считает процессор, поэтому медленно. Видеокарту подключает {ACCELERATION}."


def progress_text(state) -> str:
    """How fast the video goes against its own rate, how fast the networks look, and how long is left."""
    if state.rendering is not None:
        return f" Записываю видео с разметкой: {state.rendering * 100:.0f} %."
    if state.kind == "camera" and state.fps:
        return f" {state.fps:.1f} кадра/с."
    if state.kind != "video" or state.finished or not state.playing or not state.fps:
        return ""
    left = max(0, state.frames - state.frame) / state.fps
    networks = f", нейросети {state.analysis_fps:.1f} кадра/с" if state.analysis_fps else ""
    return f" Показ {state.fps:.0f} из {state.video_fps:.0f} кадров/с{networks}, осталось около {time_text(left)}."


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
        self.device = tk.StringVar()
        self.known_pig = tk.StringVar()
        self.known_mass = tk.StringVar()
        self.sent_known = None
        self.database = PigDatabase(app.directory)
        self.database_text = tk.StringVar()
        self.sent_learned = None
        # A new pack starts working only after a restart, so this cannot change while the tab is open.
        self.pack = installed_pack(app.directory)
        heading(sidebar, "Взвесим свиней", "Откройте фото или видео или включите камеру. Программа сама оценит "
                "расстояние до каждой свиньи и взвесит тех, кто стоит прямо и виден целиком.")
        self.build_metrics(sidebar)
        ttk.Label(sidebar, textvariable=self.message, style="Muted.TLabel", wraplength=WRAP, justify="left").pack(anchor="w", fill="x", pady=(0, 4))
        ttk.Label(sidebar, textvariable=self.device, style="Muted.TLabel", wraplength=WRAP, justify="left").pack(anchor="w", fill="x", pady=(0, 8))
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
        note(sidebar, "Обычная камера не знает, насколько далеко свиньи, и без подсказки вес может ошибаться в разы. "
                      "Проще всего взвесить одну свинью на весах и ввести её номер и вес: остальные пересчитаются по ней.")
        known = ttk.Frame(sidebar)
        known.pack(fill="x", pady=(4, 0))
        known.columnconfigure(1, weight=1)
        ttk.Label(known, text="Номер свиньи").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(known, textvariable=self.known_pig, width=8).grid(row=0, column=1, sticky="e", pady=2)
        ttk.Label(known, text="Её вес на весах, кг").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(known, textvariable=self.known_mass, width=8).grid(row=1, column=1, sticky="e", pady=2)
        database = ttk.Frame(sidebar)
        database.pack(fill="x", pady=(4, 0))
        database.columnconfigure((0, 1), weight=1, uniform="database")
        self.add_button = ttk.Button(database, text="Добавить в базу", command=self.add_to_database)
        self.add_button.grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=2)
        self.undo_button = ttk.Button(database, text="Убрать запись", command=self.remove_last)
        self.undo_button.grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=2)
        self.export_button = ttk.Button(database, text="Сохранить базу в CSV", command=self.export_database)
        self.export_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Label(sidebar, textvariable=self.database_text, style="Muted.TLabel", wraplength=WRAP, justify="left").pack(anchor="w", fill="x", pady=(2, 6))
        note(sidebar, "Или отметьте на кадре отрезок, длину которого знаете: ширину кормушки, станка или линейку.")
        self.mark_button = action(sidebar, "Отметить отрезок на кадре", self.start_marking)
        form = ttk.Frame(sidebar)
        form.pack(fill="x", pady=(4, 0))
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Длина отрезка, см").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(form, textvariable=self.length, width=8).grid(row=0, column=1, sticky="e", pady=2)
        ttk.Label(sidebar, textvariable=self.scale_text, style="Muted.TLabel", wraplength=WRAP, justify="left").pack(anchor="w", fill="x", pady=(4, 0))
        ttk.Separator(sidebar).pack(fill="x", pady=10)
        note(sidebar, "Вес — оценка по форме тела, на весах она пока не проверена. Номера действуют, пока открыт этот "
                      "источник. Видео обрабатывается целиком, кадр за кадром, и записывается заново с разметкой в папку "
                      "output. Кроме него сохраняются только числа.")
        self.download_button = action(footer, "Скачать модели", self.start_download, primary=True)
        self.video_actions = ttk.Frame(footer)
        self.video_actions.columnconfigure((0, 1), weight=1, uniform="marked")
        self.render_button = ttk.Button(self.video_actions, text="Записать заново", command=self.render_video)
        self.render_button.grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=2)
        self.open_video_button = ttk.Button(self.video_actions, text="Открыть видео", command=self.open_marked_video)
        self.open_video_button.grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=2)
        self.video_actions_shown = False
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
        for variable in (self.known_pig, self.known_mass):
            variable.trace_add("write", lambda *_: self.apply_known())
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
            self.session = PigSession(lambda status: PigDetector(self.app.directory, status), self.app.directory / "output")
            self.session.start()
            self.sent_reference = self.sent_known = None
            self.apply_reference()
            self.apply_known()
            self.apply_learned()
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

    def render_video(self) -> None:
        if self.session is not None:
            self.session.post("render")

    def shown_path(self, path) -> str:
        """A saved file named from the program folder, which is how the owner finds it."""
        try:
            return str(Path(path).relative_to(self.app.directory))
        except ValueError:
            return str(path)

    def open_marked_video(self) -> None:
        path = self.state.video_file if self.state is not None else ""
        if not path or not Path(path).is_file():
            self.message.set("Видео с разметкой ещё не записано.")
            return
        if hasattr(os, "startfile"):
            os.startfile(path)

    # Scale.
    def forget_segment(self) -> None:
        """Another picture has another scale and other numbers."""
        self.segment, self.points, self.marking = None, [], False
        self.apply_reference()
        self.known_pig.set("")
        self.known_mass.set("")

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

    # The pig database.
    def apply_learned(self) -> None:
        """Tell the session how the pictures of earlier videos were corrected, and show what the base holds."""
        try:
            entries = self.database.entries()
        except Exception as error:  # noqa: BLE001 - a damaged base must not stop the weighing.
            self.database_text.set(f"Базу свиней не прочитать: {error}")
            return
        self.database_text.set(summary_text(entries, PICTURE))
        learned = correction(entries, PICTURE)
        if self.session is not None:
            self.sent_learned = learned
            self.session.post("learned", learned)

    def add_to_database(self) -> None:
        known = self.known()
        if known is None or self.state is None:
            self.message.set("Введите номер свиньи с кадра и её вес с весов, затем добавьте в базу.")
            return
        row = next((pig for pig in self.state.animals if pig.track_id == known[0]), None)
        source = self.state.kind or ""
        try:
            entry = make_entry(row, str(known[1]), f"№{known[0]}", source, VERSION)
            self.database.add(entry)
        except Exception as error:  # noqa: BLE001 - every failure is told, none is fatal.
            self.message.set(str(error))
            return
        self.message.set(f"Свинья №{known[0]} добавлена в базу: {kg_text(entry.real_weight_kg)} кг на весах, "
                         f"{kg_text(entry.estimated_weight_kg)} кг по снимку без поправки.")
        self.apply_learned()

    def remove_last(self) -> None:
        try:
            entries = self.database.entries()
            if not entries:
                self.message.set("База свиней пуста.")
                return
            self.database.delete(entries[0].id)
        except Exception as error:  # noqa: BLE001 - every failure is told, none is fatal.
            self.message.set(f"Не удалось изменить базу: {error}")
            return
        self.message.set(f"Из базы убрана запись {entries[0].label or entries[0].id}.")
        self.apply_learned()

    def export_database(self) -> None:
        path = self.app.directory / "output" / f"pigs_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            count = export_csv(self.database.entries(), path)
        except Exception as error:  # noqa: BLE001 - every failure is told, none is fatal.
            self.message.set(f"Не удалось сохранить базу: {error}")
            return
        self.message.set(f"Записей в CSV: {count}. Файл {self.shown_path(path)}.")

    def known(self) -> tuple | None:
        """(number, kilograms) of the animal weighed on scales, once both are typed."""
        number, mass = parse_number(self.known_pig.get()), parse_number(self.known_mass.get())
        return (int(number), mass) if number and number == int(number) and mass else None

    def apply_known(self) -> None:
        known = self.known()
        if self.session is not None and known != self.sent_known:
            self.sent_known = known
            self.session.post("known", known)

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
        if shown != self.video_actions_shown:
            self.video_actions_shown = shown
            if shown:
                self.video_actions.pack(fill="x", before=self.reset_button.master)
            else:
                self.video_actions.pack_forget()

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
        self.play_button.configure(text="Пауза" if state.playing else "Продолжить",
                                   state="disabled" if state.finished else "normal")
        busy = state.rendering is not None
        self.render_button.configure(state="normal" if state.finished and not busy else "disabled")
        self.open_video_button.configure(state="normal" if state.video_file and not busy else "disabled")
        self.device.set(device_text(state.device, self.pack))
        if not self.marking:
            lost = state.kind == "camera" and not self.app.connected
            saved = f" Видео с разметкой: {self.shown_path(state.video_file)}" if state.video_file and not busy else ""
            self.message.set("Камеры отключены. " + NO_CAMERA if lost else state.message + progress_text(state) + saved)
        self.scale_text.set(scale_text(state) if state.image is not None else "")
        top = self.table.yview()[0]
        self.table.delete(*self.table.get_children())
        # Measured animals first: with a dozen in view, the ones with a weight must not hide below the fold.
        for pig in sorted(state.animals, key=lambda pig: (pig.estimate is None, pig.track_id)):
            self.table.insert("", "end", values=pig_row(pig))
        self.table.yview_moveto(top)
        masses = [pig.estimate.mass_kg for pig in state.animals if pig.estimate is not None]
        self.metric_count.set(str(len(state.pigs)) if state.image is not None else NO_VALUE)
        self.metric_mean.set(f"~{kg_text(sum(masses) / len(masses))} кг" if masses else NO_VALUE)
        if state.image is None:
            self.panel.clear()
        else:
            self.render()

    def save(self) -> None:
        state = self.state
        if state is None or not any(pig.estimate is not None for pig in state.animals):
            self.message.set("Сохранять пока нечего: нет измеренных свиней.")
            return
        folder = result_folder(self.app.directory / "output")
        setup = dict(version=VERSION, source=state.kind, scale=state.scale_source)
        try:
            count = write_result(folder / "result.json", state.animals, setup)
        except OSError as error:
            self.message.set(f"Не удалось сохранить: {error}")
            return
        self.message.set(f"Сохранено свиней: {count}. Папка output\\{folder.name}.")
