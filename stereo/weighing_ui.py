"""The weighing tab: starts by itself when opened and stops when left, to free the processor."""
from datetime import datetime
import json
from pathlib import Path
from queue import Empty, Queue
import sqlite3
from types import SimpleNamespace
from threading import Thread
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np

from .body_measure import MIN_KEYPOINT_SCORE, MIN_SAMPLES, SKELETON
from .gpu_pack import PACKS, graphics_adapters, install_pack, installed_pack, pack_advice, pending_pack
from .experiment_files import RESULT_FILE, attach_real_data, experiment_setup, write_experiment
from .people_database import CLOTHING, PeopleDatabase, SEXES, export_csv, make_entry, summarize, summary_text
from .people_detection import DOWNLOAD_SIZE_MB, download_models, models_ready
from .ui import fit_wrap, note
from .version import VERSION
from .weighing import person_result

NO_VALUE = "—"
OUTLINE_OPACITY = .35
SHOWN_EXPERIMENTS = 30
HISTORY_REFRESH_MS = 700
# The real-data form beside the database table, and the text width inside it.
FORM_WIDTH, FORM_WRAP = 290, 268
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


def saved_experiments(root, limit=SHOWN_EXPERIMENTS) -> list:
    """Saved results, newest first, as (folder name, parsed result). Unreadable ones are skipped."""
    results = []
    for folder in sorted((Path(root) / "output").glob("weighing_*"), key=lambda path: path.name, reverse=True):
        try:
            data = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("people"), list):
            results.append((folder.name, data))
        if len(results) >= limit:
            break
    return results


def experiment_title(name, data) -> str:
    """Short list line: when it was made, and the mass of each person."""
    stamp = name.removeprefix("weighing_")
    try:
        when = datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%d.%m %H:%M:%S")
    except ValueError:
        when = stamp
    masses = [f"{person['mass_kg']:.0f} кг" for person in data["people"] if person.get("mass_kg")]
    return f"{when}  ·  " + (", ".join(masses) if masses else "без оценки")


def experiment_report(data) -> str:
    """The same report as for a live person, from a saved result."""
    parts = []
    for person in data["people"]:
        side = person.get("side_observed")
        estimate = SimpleNamespace(height_m=person.get("height_m"), mass_kg=person.get("mass_kg"),
                                   mass_low_kg=person.get("mass_low_kg"), mass_high_kg=person.get("mass_high_kg"),
                                   widths_m=person.get("widths_m") or {}, side_widths_m=person.get("side_widths_m") or {},
                                   view_coverage=person.get("views") or {}, samples=person.get("samples", 0),
                                   message="Бок измерен." if side else "Бок не измерен.")
        parts.append(person_report(person.get("label", "?"), estimate))
    return "\n\n".join(parts) or "В этом результате нет людей."


class WeighingTab:
    def __init__(self, app, page, sidebar, view, footer, heading, action, panel_class):
        self.app, self.page = app, page
        self.active = False
        self.shown = None
        self.downloads = Queue()
        self.downloading = False
        self.database = PeopleDatabase(app.directory)
        self.database_entries = []
        self.sources = {}
        self.history_items = []
        self.message = tk.StringVar(value="Откройте вкладку после подключения и калибровки камер.")
        self.report = tk.StringVar(value="")
        self.metric_weight = tk.StringVar(value=NO_VALUE)
        self.metric_height = tk.StringVar(value=NO_VALUE)
        heading(sidebar, "Взвесим людей", "Встаньте целиком в кадр в 2–3 м от камеры и в метре от стены, "
                "разведите руки и медленно повернитесь на месте. Нужна облегающая одежда: свободная завышает вес.")
        self.build_metrics(sidebar)
        ttk.Label(sidebar, textvariable=self.message, style="Muted.TLabel", wraplength=280, justify="left").pack(anchor="w", fill="x", pady=(0, 6))
        ttk.Label(sidebar, textvariable=self.report, wraplength=280, justify="left").pack(anchor="w", fill="x")
        ttk.Separator(sidebar).pack(fill="x", pady=12)
        action(sidebar, "Ввести настоящие данные →", lambda: self.views.select(self.database_page))
        note(sidebar, "Вес — приблизительная оценка по объёму тела. Номера действуют только в этом сеансе; "
                      "лица не распознаются, снимки не сохраняются.")
        self.gpu_advice = tk.StringVar()
        ttk.Label(sidebar, textvariable=self.gpu_advice, style="Muted.TLabel", wraplength=280, justify="left").pack(anchor="w", fill="x", pady=(6, 0))
        self.download_button = action(footer, f"Скачать модели ({DOWNLOAD_SIZE_MB} МБ)", self.start_download, primary=True)
        self.reset_button = action(footer, "Начать замеры заново", lambda: app.send("weighing_reset"), "calibrated")
        self.save_button = action(footer, "Сохранить результат", self.save, "calibrated")
        self.gpu_button = action(footer, "Ускорить на этом компьютере", self.start_gpu_install)
        self.gpu_button_place = self.gpu_button.pack_info()
        self.gpu_button.pack_forget()
        self.recommended_pack = None
        self.views = ttk.Notebook(view, style="Inner.TNotebook")
        self.views.pack(fill="both", expand=True)
        camera_page = ttk.Frame(self.views)
        self.views.add(camera_page, text="Камера")
        self.panel = panel_class(camera_page, "Люди в кадре", empty="Здесь появятся найденные люди",
                                 hint="Подключите камеры и встаньте в кадр")
        self.panel.pack(fill="both", expand=True)
        self.build_history_page(action)
        self.build_database_page(action)
        self.refresh_sources()
        self.available = models_ready(app.directory)
        app.workflow.bind("<<NotebookTabChanged>>", self.tab_changed, add="+")
        self.show_download_state()
        present = pending_pack(app.directory) or installed_pack(app.directory)
        # Asking Windows about the graphics cards takes a moment.
        Thread(target=lambda: self.downloads.put(("advice", pack_advice(graphics_adapters(), present))), daemon=True).start()

    def build_metrics(self, parent) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(4, 10))
        row.columnconfigure((0, 1), weight=1, uniform="metric")
        for column, (variable, caption) in enumerate(((self.metric_weight, "вес, оценка"), (self.metric_height, "рост"))):
            ttk.Label(row, textvariable=variable, style="Metric.TLabel").grid(row=0, column=column, sticky="w")
            ttk.Label(row, text=caption, style="MetricCaption.TLabel").grid(row=1, column=column, sticky="w")

    def build_entry_form(self, parent, action) -> None:
        """Real height and weight for a measurement, to build the database."""
        ttk.Label(parent, text="Настоящие данные", style="Section.TLabel").pack(anchor="w")
        note(parent, "Камера сравнит свой замер с реальностью, и эти записи пойдут на улучшение расчёта. "
                     "Записывайте человека только с его согласия; вместо имени подойдёт код.", width=FORM_WRAP)
        form = ttk.Frame(parent)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self.entry_source = tk.StringVar()
        self.entry_label = tk.StringVar()
        self.entry_height = tk.StringVar()
        self.entry_weight = tk.StringVar()
        self.entry_sex = tk.StringVar(value=SEXES[0])
        self.entry_clothing = tk.StringVar(value=CLOTHING[0])
        self.entry_scales = tk.BooleanVar(value=True)
        self.source_box = ttk.Combobox(form, textvariable=self.entry_source, state="readonly", postcommand=self.refresh_sources)
        fields = (("Замер", self.source_box),
                  ("Метка", ttk.Entry(form, textvariable=self.entry_label)),
                  ("Рост, см", ttk.Entry(form, textvariable=self.entry_height)),
                  ("Вес, кг", ttk.Entry(form, textvariable=self.entry_weight)),
                  ("Пол", ttk.Combobox(form, textvariable=self.entry_sex, values=SEXES, state="readonly")),
                  ("Одежда", ttk.Combobox(form, textvariable=self.entry_clothing, values=CLOTHING, state="readonly")))
        for row, (caption, control) in enumerate(fields):
            ttk.Label(form, text=caption).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
            control.grid(row=row, column=1, sticky="ew", pady=3)
        ttk.Checkbutton(form, text="Вес взят с весов только что", variable=self.entry_scales).grid(row=len(fields), column=0, columnspan=2, sticky="w", pady=(4, 2))
        self.entry_feedback = tk.StringVar()
        self.add_button = action(parent, "Добавить в базу", self.add_to_database, primary=True)
        self.add_button.pack_configure(pady=(6, 2))
        ttk.Label(parent, textvariable=self.entry_feedback, wraplength=FORM_WRAP, justify="left").pack(anchor="w", fill="x")

    def build_history_page(self, action) -> None:
        page = ttk.Frame(self.views, padding=8)
        self.views.add(page, text="Эксперименты")
        table = ttk.Frame(page)
        table.pack(side="left", fill="both", expand=True)
        self.history_list = ttk.Treeview(table, columns=("title",), show="headings", selectmode="browse")
        self.history_list.heading("title", text="Сохранённые замеры, сначала новые", anchor="w")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.history_list.yview)
        self.history_list.configure(yscrollcommand=scroll.set)
        self.history_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        self.history_list.bind("<<TreeviewSelect>>", self.show_experiment)
        details = ttk.Frame(page, padding=(14, 0, 4, 0), width=330)
        details.pack(side="left", fill="y")
        details.pack_propagate(False)
        action(details, "Записать этот замер в базу", self.use_experiment, primary=False)
        self.history_text = tk.StringVar(value="Выберите эксперимент, чтобы увидеть его числа.")
        ttk.Label(details, textvariable=self.history_text, wraplength=300, justify="left").pack(anchor="nw", fill="x", pady=(10, 0))
        self.refresh_history()

    def build_database_page(self, action) -> None:
        page = ttk.Frame(self.views, padding=8)
        self.database_page = page
        self.views.add(page, text="База людей")
        form = ttk.Frame(page, padding=(14, 0, 4, 0), width=FORM_WIDTH)
        form.pack(side="right", fill="y")
        form.pack_propagate(False)
        self.build_entry_form(form, action)
        self.database_summary = tk.StringVar()
        summary = ttk.Label(page, textvariable=self.database_summary, wraplength=FORM_WIDTH, justify="left")
        summary.pack(anchor="w", fill="x", pady=(0, 8))
        fit_wrap(page, summary, margin=FORM_WIDTH + 32)
        buttons = ttk.Frame(page)
        buttons.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Удалить запись", style="Danger.TButton", command=self.delete_entry).pack(side="left")
        ttk.Button(buttons, text="Сохранить CSV", command=self.export_database).pack(side="right")
        table = ttk.Frame(page)
        table.pack(fill="both", expand=True)
        columns = (("created", "Дата", 114), ("label", "Метка", 74), ("sex", "Пол", 62), ("real_height", "Рост, см", 70),
                   ("camera_height", "Камера, см", 86), ("real_weight", "Вес, кг", 62), ("camera_weight", "Камера, кг", 86),
                   ("error", "Ошибка, кг", 86), ("clothing", "Одежда", 70))
        self.database_table = ttk.Treeview(table, columns=[column[0] for column in columns], show="headings", selectmode="browse")
        for name, title, width in columns:
            self.database_table.heading(name, text=title, anchor="w")
            self.database_table.column(name, width=width, minwidth=width, anchor="w", stretch=name == "label")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.database_table.yview)
        # On a narrow window the columns keep their width and the table scrolls sideways.
        sideways = ttk.Scrollbar(table, orient="horizontal", command=self.database_table.xview)
        self.database_table.configure(yscrollcommand=scroll.set, xscrollcommand=sideways.set)
        sideways.pack(side="bottom", fill="x")
        self.database_table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        self.refresh_database()

    def refresh_sources(self) -> None:
        """Measurements that can be paired with real data: live people first, then saved ones.

        Each is (numbers, experiment folder); a live person has no folder until the real data is added.
        """
        sources = {}
        known = self.shown.known if self.shown is not None else {}
        for track_id, estimate in sorted(known.items()):
            if estimate.samples:
                # No mass in this name: it changes while real data is being typed.
                sources[f"Сейчас · №{track_id}"] = (person_result(track_id, estimate), None)
        for name, data in self.history_items:
            when = experiment_title(name, dict(people=[])).split("  ")[0]
            folder = self.app.directory / "output" / name
            for person in data["people"]:
                if person.get("samples"):
                    mass = f"{person['mass_kg']:.0f} кг" if person.get("mass_kg") else "без оценки"
                    sources[f"{when} · №{person.get('label', '?')} · {mass}"] = (person, folder)
        self.sources = sources
        self.source_box.configure(values=list(sources))
        if self.entry_source.get() not in sources:
            self.entry_source.set(next(iter(sources), ""))

    def add_to_database(self) -> None:
        chosen = self.entry_source.get()
        self.refresh_sources()
        if chosen and chosen not in self.sources:
            # Silently using another measurement would put a wrong person into the database.
            self.entry_feedback.set("Выбранного замера больше нет. Выберите замер снова.")
            return
        worker = self.app.worker
        camera = getattr(getattr(worker, "config", None), "mode", "")
        measurement, folder = self.sources.get(self.entry_source.get(), (None, None))
        try:
            entry = make_entry(measurement, self.entry_label.get(), self.entry_height.get(),
                               self.entry_weight.get(), self.entry_sex.get(), self.entry_clothing.get(),
                               self.entry_scales.get(), camera, VERSION)
            self.database.add(entry)
        except ValueError as error:
            self.entry_feedback.set(str(error))
            return
        except (sqlite3.Error, OSError) as error:
            self.entry_feedback.set(f"Не удалось записать в базу: {error}")
            return
        estimate = entry.estimated_weight_kg
        stored = self.store_in_experiment(folder, measurement.get("label"), entry)
        self.entry_feedback.set((f"Записано. Камера {estimate:.1f} кг, реально {entry.real_weight_kg:g} кг, "
                                 f"ошибка {entry.weight_error_kg:+.1f} кг." if estimate is not None else
                                 "Записано; в этом замере нет оценки веса.") + " " + stored)
        self.entry_label.set("")
        self.entry_height.set("")
        self.entry_weight.set("")
        self.refresh_database()

    def store_in_experiment(self, folder, person_label, entry) -> str:
        """Put the real data into the measurement's own result.json; a live one is saved there first."""
        try:
            if folder is None:
                folder = self.save_live_experiment()
            attach_real_data(folder / RESULT_FILE, person_label, entry)
        except (OSError, ValueError) as error:
            return f"В папку замера не записано: {error}"
        self.refresh_history()
        return f"Папка замера: {folder}"

    def save_live_experiment(self) -> Path:
        """Everyone measured now, written by the interface so the real data can follow at once."""
        known = self.shown.known if self.shown is not None else {}
        people = [person_result(track_id, estimate) for track_id, estimate in sorted(known.items()) if estimate.samples]
        worker = self.app.worker
        folder = self.app.directory / "output" / f"weighing_{datetime.now():%Y%m%d_%H%M%S}"
        write_experiment(folder / RESULT_FILE, people, experiment_setup(getattr(worker, "config", None),
                                                                       getattr(worker, "calibration", None)))
        return folder

    def refresh_database(self) -> None:
        try:
            self.database_entries = self.database.entries()
        except (sqlite3.Error, OSError) as error:
            self.database_entries = []
            self.database_summary.set(f"Не удалось открыть базу: {error}")
            return
        self.database_summary.set(summary_text(summarize(self.database_entries)))
        self.database_table.delete(*self.database_table.get_children())
        number = lambda value, digits=1: f"{value:.{digits}f}" if value is not None else NO_VALUE
        for entry in self.database_entries:
            error = entry.weight_error_kg
            self.database_table.insert("", "end", iid=str(entry.id), values=(
                entry.created.replace("T", " ")[:16], entry.label or NO_VALUE, entry.sex, f"{entry.real_height_cm:g}",
                number(entry.estimated_height_cm, 0), f"{entry.real_weight_kg:g}", number(entry.estimated_weight_kg),
                f"{error:+.1f}" if error is not None else NO_VALUE, entry.clothing))

    def delete_entry(self) -> None:
        selected = self.database_table.selection()
        if not selected:
            self.database_summary.set("Выберите запись в таблице, чтобы удалить её.")
            return
        if not messagebox.askyesno("База людей", "Удалить эту запись из базы? Вернуть её будет нельзя.", parent=self.page):
            return
        try:
            self.database.delete(int(selected[0]))
        except (sqlite3.Error, OSError) as error:
            self.database_summary.set(f"Не удалось удалить запись: {error}")
            return
        self.refresh_database()

    def export_database(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.app.directory / "output" / f"people_database_{stamp}.csv"
        try:
            count = export_csv(self.database_entries, path)
        except OSError as error:
            self.database_summary.set(f"Не удалось сохранить файл: {error}")
            return
        self.database_summary.set(f"Сохранено записей: {count}. Файл: {path}")

    def refresh_history(self) -> None:
        selected = self.history_list.selection()
        self.history_items = saved_experiments(self.app.directory)
        self.history_list.delete(*self.history_list.get_children())
        names = {name for name, _ in self.history_items}
        for name, data in self.history_items:
            self.history_list.insert("", "end", iid=name, values=(experiment_title(name, data),))
        if selected and selected[0] in names:
            self.history_list.selection_set(selected[0])

    def selected_experiment(self):
        selected = self.history_list.selection()
        return next((item for item in self.history_items if selected and item[0] == selected[0]), None)

    def show_experiment(self, event=None) -> None:
        experiment = self.selected_experiment()
        if experiment:
            self.history_text.set(experiment_report(experiment[1]))

    def use_experiment(self) -> None:
        """Put the chosen experiment into the form, where the real data is typed."""
        experiment = self.selected_experiment()
        if not experiment:
            self.history_text.set("Сначала выберите эксперимент в списке.")
            return
        self.refresh_sources()
        when = experiment_title(experiment[0], dict(people=[])).split("  ")[0]
        match = next((title for title in self.sources if title.startswith(when + " ")), None)
        if match is None:
            self.history_text.set("В этом эксперименте нет людей с измерениями.")
            return
        self.entry_source.set(match)
        self.entry_feedback.set("Замер выбран. Введите настоящие рост и вес.")
        self.views.select(self.database_page)

    def start_gpu_install(self) -> None:
        if self.downloading:
            return
        if pending_pack(self.app.directory) == self.recommended_pack:  # Chosen on the settings page meanwhile.
            self.gpu_button.pack_forget()
            self.gpu_advice.set("")
            self.message.set("Ускорение установлено. Перезапустите программу, чтобы оно заработало.")
            return
        self.downloading = True
        self.gpu_button.configure(state="disabled")
        key = self.recommended_pack
        self.gpu_advice.set("")

        def work():
            try:
                install_pack(self.app.directory, key, lambda text: self.downloads.put(("status", text)))
                self.downloads.put(("gpu_done", None))
            except Exception as error:  # noqa: BLE001 - the button must never stay blocked.
                self.downloads.put(("error", str(error)))
        Thread(target=work, daemon=True, name="gpu-download").start()

    def ready(self) -> bool:
        return self.available

    def show_download_state(self) -> None:
        self.available = models_ready(self.app.directory)
        # One main button at a time: the footer stays visible over the form while models are missing.
        self.add_button.configure(style="Accent.TButton" if self.available else "TButton")
        if self.available:
            self.download_button.pack_forget()
        elif not self.downloading:
            self.message.set(f"Для поиска людей нужны три модели, {DOWNLOAD_SIZE_MB} МБ. Они скачиваются один раз "
                             "и работают на этом компьютере; кадры никуда не передаются.")

    def selected(self) -> bool:
        return self.app.workflow.select() == str(self.page)

    def tab_changed(self, event=None) -> None:
        if self.selected():
            self.refresh_history()
            self.refresh_sources()
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
                self.metric_weight.set(NO_VALUE)
                self.metric_height.set(NO_VALUE)
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
            except Exception as error:  # noqa: BLE001 - the button must never stay blocked.
                self.downloads.put(("error", str(error)))
        Thread(target=work, daemon=True, name="model-download").start()

    def poll(self, packet) -> None:
        try:
            while True:
                kind, value = self.downloads.get_nowait()
                if kind == "status":
                    self.message.set(value)
                elif kind == "advice":
                    self.recommended_pack = value.pack
                    if value.pack:
                        self.gpu_button.configure(text=f"Ускорить этот компьютер ({PACKS[value.pack].size_text})")
                        self.gpu_button.pack(**self.gpu_button_place)
                        self.gpu_advice.set(value.text)
                elif kind == "gpu_done":
                    self.downloading = False
                    self.gpu_button.pack_forget()
                    self.message.set("Ускорение установлено. Перезапустите программу, чтобы оно заработало.")
                else:
                    self.gpu_button.configure(state="normal")
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
        self.show_metrics(state)
        self.report.set("\n\n".join(person_report(person.track_id, person.estimate) for person in state.people)
                        or f"Нужно минимум {MIN_SAMPLES} кадров с видимым туловищем.")
        if state.image is not None:
            self.panel.set(draw_people(state.image, state))

    def show_metrics(self, state) -> None:
        """Large figures for the person measured most; several people are listed in the report."""
        people = sorted(state.people, key=lambda person: -person.estimate.samples)
        estimate = people[0].estimate if people else None
        self.metric_weight.set(f"~{estimate.mass_kg:.0f} кг" if estimate and estimate.mass_kg else NO_VALUE)
        self.metric_height.set(f"{estimate.height_m * 100:.0f} см" if estimate and estimate.height_m else NO_VALUE)

    def save(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.app.send("weighing_save", self.app.directory / "output" / f"weighing_{stamp}" / "result.json")
        self.app.root.after(HISTORY_REFRESH_MS, self.refresh_history)
