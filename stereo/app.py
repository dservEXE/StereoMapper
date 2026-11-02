from pathlib import Path
from queue import Empty
from threading import Thread
import json
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from .camera import CameraConfig
from .depth import measure
from .worker import CameraWorker
from .viewer import CloudViewer
from .board import BoardSpec, BoardWindow
from .help_ui import help_icon, HELP
from .presets import PRESETS
from .ui import NIGHT


class ImagePanel(ttk.Frame):
    def __init__(self, parent, title, click=None, empty="Ожидание изображения",hint="Изображение появится после начала съёмки"):
        super().__init__(parent, style="Card.TFrame", padding=8)
        self.title_label = ttk.Label(self, text=title, style="CardTitle.TLabel", wraplength=380)
        self.title_label.pack(anchor="w", pady=(0, 7))
        self.bind("<Configure>", lambda event: self.title_label.configure(wraplength=max(140, event.width-20)))
        self.empty,self.empty_hint=empty,hint
        self.canvas = tk.Canvas(self, background=NIGHT, highlightthickness=0, width=400, height=270)
        self.canvas.pack(fill="both", expand=True)
        self.image = None
        self.source = None
        self.geometry = None
        self.click = click
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        self.canvas.bind("<Map>",lambda event:self.redraw())
        if click:
            self.canvas.bind("<Button-1>", self.on_click)

    def set(self, bgr):
        self.source = bgr
        self.redraw()

    def redraw(self):
        if self.source is None:
            self.canvas.delete("all")
            w,h=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
            if w<140 or h<100:return
            x,y=w/2,max(40,h/2-55)
            self.canvas.create_rectangle(x-34,y-20,x+34,y+20,outline="#4e6b83",width=2)
            for dx in (-15,15):self.canvas.create_oval(x+dx-8,y-8,x+dx+8,y+8,outline="#79afb9",width=2)
            self.canvas.create_text(x,y+48,text=self.empty,fill="#d6e5f1",font=("Segoe UI",12,"bold"),width=max(120,w-40),justify="center")
            self.canvas.create_text(x,y+94,text=self.empty_hint,fill="#90a8bf",font=("Segoe UI",10),width=max(120,w-50),justify="center")
            return
        if not self.winfo_viewable():return
        h, w = self.source.shape[:2]
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        factor = min(cw / w, ch / h)
        dw, dh = max(1, int(w * factor)), max(1, int(h * factor))
        resized = cv2.resize(self.source, (dw, dh), interpolation=cv2.INTER_AREA)
        self.image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.image)
        self.geometry = (ox, oy, dw, dh, w, h)

    def clear(self):
        self.source=self.geometry=None
        self.redraw()

    def on_click(self, event):
        if self.geometry:
            ox, oy, dw, dh, w, h = self.geometry
            if ox <= event.x < ox + dw and oy <= event.y < oy + dh:
                self.click(int((event.x - ox) * w / dw), int((event.y - oy) * h / dh))


class StereoApp:
    def __init__(self, root, directory):
        self.root, self.directory = root, Path(directory)
        self.worker = None
        self.packet = None
        self.preview_last = None
        self.board_window = None
        self.scan_window=None
        self.displayed_depth = self.displayed_map = None
        self.selected = None
        self.settings_controls = []
        self.connected = False
        self.closing = False
        self.hidden_for_capture = False
        self.capture_generation = 0
        self.status = tk.StringVar(value="F1 — краткая инструкция · ? — подсказка к настройке")
        self.calibration_text = tk.StringVar(value="Калибровки нет — расстояния пока недоступны")
        self.measurement = tk.StringVar(value="Щёлкните по объекту в левом изображении")
        self.tracking = tk.StringVar(value="Текущая область: проекция измеренных поверхностей")
        self.count = tk.StringVar(value="Снято пар: 0 / минимум 15")
        self.calibration_hint=tk.StringVar(value="Добавьте наклоны доски влево/вправо и вверх/вниз; одного изменения расстояния недостаточно.")
        self.stats = tk.StringVar(value="")
        self.map_stats = tk.StringVar(value="Получено стереопар: 0 · добавлено в карту кадров: 0")
        self.range_text = tk.StringVar(value="Нижняя граница фильтра: 5 см. Рабочий минимум определяется стереопарой и калибровкой.")
        self.mapping = tk.BooleanVar(value=False)
        self.depth_enabled = tk.BooleanVar(value=False)
        self.smooth_depth = tk.BooleanVar(value=False)
        self.depth_legend = tk.StringVar(value="Тёмные участки: достоверной глубины нет")
        self.guides = tk.BooleanVar(value=True)
        self.hide_capture = tk.BooleanVar(value=True)
        self.vars = {"left": tk.StringVar(value="0"), "right": tk.StringVar(value="1"),
                     "width": tk.StringVar(value="640"), "height": tk.StringVar(value="480"),
                     "fps": tk.StringVar(value="15"), "backend": tk.StringVar(value="AUTO"),
                     "mode": tk.StringVar(value="Две камеры"), "swap": tk.BooleanVar(value=False),
                     "columns": tk.StringVar(value="9"), "rows": tk.StringVar(value="6"),
                     "square_mm": tk.StringVar(value="25"), "max_depth": tk.StringVar(value="8"),
                     "disparities": tk.StringVar(value="128"), "quality": tk.StringVar(value="Сбалансированная"),
                     "processing_width": tk.StringVar(value="640"), "temporal": tk.BooleanVar(value=True),
                     "near_priority": tk.BooleanVar(value=True), "wide_view": tk.BooleanVar(value=True),
                     "voxel_mm": tk.StringVar(value="5"),"map_stride":tk.StringVar(value="2"),
                     "preview_points":tk.StringVar(value="250000"),"near_minimum_cm":tk.StringVar(value="10"),
                     "profile":tk.StringVar(value="Свои настройки"),"estimate_surfaces":self.smooth_depth,
                     "acceleration":tk.StringVar(value="Авто · NVIDIA / CPU")}
        self.vars['ghost_filter']=tk.StringVar(value="Обычная")
        self.load_settings()
        self.build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id=self.root.after(40, self.poll)
        self.root.bind("<Destroy>",self.on_destroy,add="+")

    def load_settings(self):
        try:
            data = json.loads((self.directory / "settings.json").read_text(encoding="utf-8"))
            if data.get("settings_version", 0) < 3:
                backup = self.directory / "settings_before_v03.json"
                if not backup.exists():
                    backup.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                data.update(width="640", height="480", fps="15", backend="AUTO",
                            quality="Сбалансированная", processing_width="640")
            for key, value in data.items():
                if key in self.vars:
                    if key == "mode":
                        value = {"dual": "Две камеры", "sbs": "Два вида рядом"}.get(value, value)
                    self.vars[key].set(value)
        except (OSError, ValueError, tk.TclError):
            pass

    def entry(self, parent, text, key, row, values=None):
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text=text, wraplength=140).grid(row=row, column=0, sticky="w", pady=5)
        widget = (ttk.Combobox(parent, textvariable=self.vars[key], values=values,
                               state="readonly", width=10) if values else
                  ttk.Entry(parent, textvariable=self.vars[key], width=10))
        widget.grid(row=row, column=1, sticky="e", pady=3)
        help_icon(parent, key).grid(row=row, column=2, padx=(7, 0))
        self.settings_controls.append((widget, "readonly" if values else "normal"))

    def check(self, parent, text, variable, key, command=None, setting=False):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        control = ttk.Checkbutton(row, text=text, variable=variable, command=command)
        control.pack(side="left")
        help_icon(row, key).pack(side="right", padx=2)
        if setting:
            self.settings_controls.append((control, "normal"))
        return control

    def button(self, parent, text, command, key=None, accent=False):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        control = ttk.Button(row, text=text, command=command, style="Accent.TButton" if accent else "TButton")
        control.pack(side="left", fill="x", expand=True)
        if key:
            help_icon(row, key).pack(side="right", padx=(7, 0))
        return control

    def build(self):
        from .main_ui import build
        build(self)

    def show_page(self,key):
        self.workflow.select(self.workflow_pages[key])

    def toggle_depth(self):
        self.depth_enabled.set(not self.depth_enabled.get())
        self.send("depth",self.depth_enabled.get())
        self.sync_ui(force=True)

    def sync_ui(self,force=False):
        if not hasattr(self,"workflow") or self.closing:return
        worker=self.worker
        calibrated=bool(self.connected and worker and getattr(worker,"engine",None))
        frame=bool(worker and getattr(worker,"frame",None) is not None)
        scan_active=bool(worker and getattr(worker,"scan",None) is not None) or bool(self.scan_window and self.scan_window.window.winfo_exists() and self.scan_window.active)
        signature=(self.connected,self.connection_pending,calibrated,frame,self.depth_enabled.get(),
                   self.mapping.get(),self.calibration_pairs,self.calibration_busy,scan_active,self.workflow.select())
        if signature==self.ui_signature and not force:return
        self.ui_signature=signature
        available=dict(connected=self.connected and not self.calibration_busy,
                       calibrated=calibrated and not self.calibration_busy,
                       capture=self.connected and not self.calibration_busy and not self.hidden_for_capture,
                       calibrate=self.connected and self.calibration_pairs>=15 and not self.calibration_busy,
                       frame=frame)
        for control,capability in self.ui_actions:
            control.configure(state="normal" if available[capability] and not scan_active and not self.connection_pending else "disabled")
        self.connect_button.configure(state="disabled" if self.connection_pending or scan_active else "normal",
            text=("Отключение…" if self.connected else "Подключение…") if self.connection_pending else
                 "Отключить камеры" if self.connected else "Подключить камеры")
        self.depth_button.configure(text="Выключить глубину" if self.depth_enabled.get() else "Включить глубину")
        self.calibrate_button.configure(text="Рассчитываю калибровку…" if self.calibration_busy else "Рассчитать калибровку")
        self.calibration_progress.configure(value=min(15,self.calibration_pairs))
        self.connection_badge.set("Камеры подключены" if self.connected else "Подключаю камеры…" if self.connection_pending else "Камеры не подключены")
        self.calibration_badge.set("Калибровка готова" if calibrated else "Калибровка не загружена")
        self.connection_label.configure(style="Success.TLabel" if self.connected else "Badge.TLabel")
        self.calibration_label.configure(style="Success.TLabel" if calibrated else "Badge.TLabel")
        if calibrated:self.camera_measurement.grid()
        else:self.camera_measurement.grid_remove()
        if not self.connected:
            hint="Подключите камеры на первой вкладке. Сохранённые карты можно открыть без камер."
        elif not calibrated:
            hint="Следующий шаг: вкладка «Калибровка» — загрузите готовую или снимите доску."
        elif not self.depth_enabled.get():
            hint="Всё готово: на вкладке «Сканирование» включите глубину или создайте новый 3D-скан."
        elif scan_active:
            hint="Идёт сканирование. Завершите и сохраните его в окне 3D-скана."
        else:
            hint="Проверьте глубину и создайте 3D-скан. Чёрные участки означают, что достоверного измерения нет."
        self.next_hint.set(hint)
        self.scan_hint.set("Камеры и калибровка готовы." if calibrated else
                           "Сначала подключите камеры и подготовьте калибровку на первых двух вкладках.")

    def parse(self):
        val = lambda key: self.vars[key].get()
        config = CameraConfig(left=int(val("left")), right=int(val("right")), width=int(val("width")),
                              height=int(val("height")), fps=int(val("fps")),
                              mode={"Две камеры": "dual", "Два вида рядом": "sbs"}.get(val("mode"), val("mode")),
                              swap=bool(val("swap")), backend=val("backend"))
        pattern = int(val("columns")), int(val("rows"))
        square, maximum = float(val("square_mm").replace(",", ".")), float(val("max_depth").replace(",", "."))
        disparities = int(val("disparities"))
        if min(config.left, config.right) < 0 or min(config.width, config.height) < 64 or not 1 <= config.fps <= 60:
            raise ValueError("Проверьте индексы, разрешение и частоту кадров.")
        if min(pattern) < 3 or pattern[0] == pattern[1] or max(pattern) > 20 or not 0 < square <= 1000:
            raise ValueError("Доска: разные числа столбцов и строк от 3 до 20; клетка 0–1000 мм.")
        if not 0.05 < maximum <= 100 or disparities < 16 or disparities % 16:
            raise ValueError("Максимальная дальность больше 0,05 м и до 100 м; поиск сдвига кратен 16 пикселям.")
        if not .25<=float(val("voxel_mm"))<=100 or int(val("map_stride")) not in (1,2,3,4):
            raise ValueError("Шаг карты 0,25–100 мм; выбор пикселей от 1 до 4.")
        if not 10000<=int(val("preview_points"))<=2000000 or not 5<=float(val("near_minimum_cm"))<=50:
            raise ValueError("Просмотр: 10 000–2 000 000 точек. Ближний поиск: 5–50 см.")
        return config, pattern, square, maximum, disparities

    def depth_options(self):
        width = self.vars["processing_width"].get()
        return {"quality": {"Быстрая": "fast", "Сбалансированная": "balanced", "Подробная": "high"}[
                    self.vars["quality"].get()],
                "processing_width": None if width == "Оригинал" else int(width),
                "temporal": self.vars["temporal"].get(), "near_priority": self.vars["near_priority"].get(),
                "voxel_mm": float(self.vars["voxel_mm"].get()), "wide_view": self.vars["wide_view"].get(),
                "map_stride":int(self.vars["map_stride"].get()),"preview_points":int(self.vars["preview_points"].get()),
                "estimate_surfaces":self.smooth_depth.get(),"near_minimum":float(self.vars["near_minimum_cm"].get())/100,
                "acceleration":{"Авто · NVIDIA / CPU":"auto","Процессор":"cpu","NVIDIA CUDA":"cuda"}[self.vars["acceleration"].get()],
                "ghost_filter":{"Обычная":"normal","Строгая":"strict","Выключена":"off"}[self.vars['ghost_filter'].get()]}

    def storage_protected(self):
        paths=[]
        if self.worker:
            for store in (self.worker.cloud,self.worker.estimated_cloud):
                if store.path is not None:paths.append(store.path.parent)
            if self.worker.collector:paths.append(self.worker.collector.directory)
        for child in self.root.winfo_children():
            getter=getattr(child,'storage_paths',None)
            if getter:paths.extend(getter())
        return paths

    def storage_busy(self):
        if self.worker and (self.worker.is_alive() or self.worker.export_thread and self.worker.export_thread.is_alive()):return True
        return any(getattr(child,'storage_busy',lambda:False)() for child in self.root.winfo_children())

    def open_storage(self):
        from .storage_window import StorageWindow
        StorageWindow(self.root,self.directory,self.storage_protected,self.storage_busy,on_open_cloud=self.open_model)

    def apply_preset(self):
        if self.connected:
            self.status.set("Отключите камеры перед сменой профиля.")
            return
        preset=PRESETS.get(self.vars["profile"].get())
        if preset:
            for key,value in preset.items():
                self.vars[key].set(value)
            self.status.set("Профиль применён. Подключите камеры; для метров нужна калибровка 1280 × 720.")

    def update_surface_mode(self):
        self.refresh_depth()
        if self.connected:
            self.send("surfaces",self.smooth_depth.get())

    def open_map_archive(self):
        from .map_store import VoxelMap
        path=filedialog.askopenfilename(title="Архив всей карты",initialdir=self.directory/"data"/"maps",
                                        filetypes=[("Архив карты","*.sqlite")])
        if not path:
            return
        store=None
        try:
            store=VoxelMap(path=path,readonly=True)
            points,colors=store.arrays(int(self.vars["preview_points"].get()))
            if not len(points):
                raise ValueError("В этом архиве пока нет точек.")
            CloudViewer(self.root,points,colors,"metres",title="Архив карты",total_count=len(store),
                        max_points=int(self.vars["preview_points"].get()),archive_path=path)
        except Exception as error:
            messagebox.showerror("Архив",str(error))
        finally:
            if store is not None:
                store.close()

    def show_performance(self):
        packet=self.packet or {}
        stages=packet.get("stage_ms",{})
        labels={"rectify_ms":"Выравнивание","alignment_ms":"Проверка стереопары","stereo_ms":"Поиск соответствий",
                "filter_ms":"Фильтры и геометрия","preview_ms":"Цвет и оценка поверхностей"}
        lines=[f"{labels.get(key,key)}: {value:.1f} мс" for key,value in stages.items()]
        lines.extend([f"Слежение: {packet.get('tracking_ms',0):.1f} мс",f"Запись карты: {packet.get('insert_ms',0):.1f} мс",
                      f"Показ карты в фоне: {packet.get('map_render_ms',0):.1f} мс", "",
                      packet.get("acceleration") or "Ускоритель ещё не выбран: подключите камеры.",
                      packet.get("acceleration_warning") or "CPU обслуживает захват, фильтры, слежение и запись; CUDA ускоряет поиск стереосоответствий.",
                      "Для скорости: выключите оценку поверхностей, уменьшите ширину расчёта; для комнаты выключите ближний поиск.",
                      "Мелкий шаг карты и каждый пиксель повышают плотность, но увеличивают нагрузку на запись."])
        messagebox.showinfo("Производительность последнего кадра","\n".join(lines))

    def toggle_connect(self):
        if self.worker and self.worker.is_alive():
            self.worker.stop_event.set()
            self.connection_pending=True
            self.sync_ui(force=True)
            self.connect_button.configure(state="disabled", text="Отключение…")
            return
        if any(getattr(child,'storage_busy',lambda:False)() for child in self.root.winfo_children()):
            self.status.set("Дождитесь завершения операции с исследованиями.");return
        try:
            config, pattern, square, maximum, disparities = self.parse()
            options = self.depth_options()
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Настройки", str(error))
            return
        (self.directory / "settings.json").write_text(
            json.dumps(dict(settings_version=3, **{key: var.get() for key, var in self.vars.items()}),
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        self.selected = None
        self.connection_pending=True
        self.calibration_pairs=0; self.calibration_busy=False
        self.packet = None
        self.preview_last = None
        self.displayed_depth = self.displayed_map = None
        self.mapping.set(False)
        self.depth_enabled.set(False)
        self.calibration_text.set("Калибровки нет — расстояния пока недоступны")
        for control, _ in self.settings_controls:
            control.configure(state="disabled")
        self.connect_button.configure(state="disabled", text="Подключение…")
        self.status.set("Открываю камеры…")
        self.worker = CameraWorker(self.directory, config, pattern, square, maximum, disparities, **options)
        self.worker.start()
        self.sync_ui(force=True)

    def send(self, name, value=None):
        if name in ("export", "view") and self.worker and not self.worker.is_alive() and self.worker.frame is not None:
            def action():
                try:
                    self.worker.command(name, value)
                except Exception as error:
                    self.worker.emit("error", str(error))
            Thread(target=action, daemon=True).start()
            return
        if not self.connected or not self.worker or not self.worker.is_alive():
            self.mapping.set(False)
            if name == "depth":
                self.depth_enabled.set(False)
            self.status.set("Сначала подключите камеру.")
            return
        self.worker.commands.put((name, value))

    def open_object_scan(self):
        if self.scan_window and self.scan_window.window.winfo_exists():
            self.scan_window.window.lift();return
        from .scan_window import ObjectScanWindow
        self.scan_window=ObjectScanWindow(self)

    def open_calibration_help(self):
        from .calibration_help import CalibrationHelpWindow
        CalibrationHelpWindow(self.root,self.directory)

    def load_calibration(self):
        path = filedialog.askopenfilename(title="Калибровка этой стереопары", filetypes=[("Калибровка", "*.npz")],
                                          initialdir=self.directory / "data")
        if path:
            self.send("load", path)

    def capture_calibration(self):
        if not self.connected:
            self.status.set("Сначала подключите камеры.")
            return False
        self.status.set("Остановите камеру и доску. Снимок будет сделан через секунду.")
        self.hidden_for_capture = self.hide_capture.get()
        if self.hidden_for_capture:
            self.root.withdraw()
        self.send("capture_after", time.perf_counter() + 1.)
        self.capture_generation += 1
        generation = self.capture_generation
        self.root.after(10000, lambda: self.capture_timeout(generation))
        return True

    def restore_visibility(self):
        if self.hidden_for_capture:
            self.hidden_for_capture = False
            self.root.deiconify()
            if self.board_window and not self.board_window.closed:
                self.board_window.window.lift()

    def capture_timeout(self, generation):
        if self.hidden_for_capture and generation == self.capture_generation:
            self.send("cancel_capture")
            self.restore_visibility()
            self.status.set("Съёмка заняла слишком долго. Пара отменена; проверьте камеру и повторите.")

    def restore_series(self):
        path = filedialog.askdirectory(title="Папка сохранённой серии калибровки", initialdir=self.directory / "data")
        if path:
            self.send("restore_series", path)

    def open_model(self,path=None):
        if path is None:path = filedialog.askopenfilename(title="Сохранённая 3D-карта", initialdir=self.directory / "output",
                                          filetypes=[("Облака и проекты 3D", "*.ply *.sqlite *.sqlite3 *.db *.cloudproj *.npz"),
                                                     ("Облако PLY", "*.ply"),
                                                     ("Архив карты", "*.sqlite *.sqlite3 *.db"),
                                                     ("Проект редактирования", "*.cloudproj *.npz")])
        if path:
            try:
                budget=int(self.vars['preview_points'].get())
                # The viewer loads the complete file in the background. The point
                # budget affects drawing only, never the editable document.
                CloudViewer(self.root,np.empty((0,3),np.float64),np.empty((0,3),np.uint8),
                            title=Path(path).name,max_points=budget,source_path=path)
            except (OSError, ValueError) as error:
                messagebox.showerror("3D-модель", str(error))

    def open_photos(self):
        from .photo_window import PhotoWindow
        PhotoWindow(self.root, self.directory)

    def make_board(self):
        try:
            if self.board_window and not self.board_window.closed:
                self.board_window.window.lift()
                return
            spec = BoardSpec(int(self.vars["columns"].get()), int(self.vars["rows"].get()),
                             float(self.vars["square_mm"].get().replace(",", "."))).validate()
            self.board_window = BoardWindow(self.root, self.directory, spec, self.apply_board,
                                             self.capture_calibration, self.count)
        except (ValueError, OSError, tk.TclError) as error:
            messagebox.showerror("Доска", str(error))

    def apply_board(self, spec):
        self.vars["columns"].set(str(spec.columns))
        self.vars["rows"].set(str(spec.rows))
        self.vars["square_mm"].set(f"{spec.square_mm:.5f}")
        if self.connected:
            self.send("board_settings", (spec.columns, spec.rows, spec.square_mm))

    def open_output(self):
        path = self.directory / "output"
        path.mkdir(exist_ok=True)
        os.startfile(str(path))

    def pick(self, x, y):
        self.selected = (x, y)
        self.update_measurement()

    def pick_live(self, x, y):
        preview = self.preview_last
        if not preview or preview["calibration_token"] is None or not self.connected:
            self.measurement.set("Для расстояния сначала выполните калибровку.")
            return
        self.selected = None
        self.measurement.set("Вычисляю расстояние на выбранном кадре…")
        self.send("measure_snapshot", dict(preview, point=(x, y)))

    def update_measurement(self):
        f = self.packet.get("depth") if self.packet else None
        if f is None:
            self.measurement.set("Для расстояния сначала выполните калибровку")
        elif self.selected:
            result = measure(f, *self.selected)
            if result:
                self.measurement.set(f"Глубина Z: {result['z']:.2f} м    |    "
                                     f"Расстояние от левой камеры: {result['range']:.2f} м    |    "
                                     "Оценка; проверьте точность линейкой")
            else:
                x, y = self.selected
                filled = f.preview_filled
                if filled is not None and 0 <= y < filled.shape[0] and 0 <= x < filled.shape[1] and filled[y, x]:
                    self.measurement.set("Здесь показана оценённая поверхность; достоверного измерения нет.")
                else:
                    self.measurement.set("В выбранной точке нет достоверной глубины. Выберите участок с деталями.")

    def refresh_depth(self):
        frame = self.packet.get("depth") if self.packet else None
        if frame is None:
            return
        smooth = self.smooth_depth.get() and frame.preview_color is not None
        color = (frame.preview_color if smooth else frame.color).copy()
        if self.selected:
            cv2.drawMarker(color, self.selected, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        self.panels[2].set(color)
        low, high = frame.color_limits or (.05, self.worker.max_depth)
        extra = f" · оценено {frame.preview_filled.mean():.1%}" if smooth and frame.preview_filled is not None else ""
        self.depth_legend_label.configure(foreground="#a02b20" if frame.quality_warning else "#1b293a")
        self.depth_legend.set(frame.quality_warning or
                             f"Цвета: {low:.2f}–{high:.2f} м · измерено {frame.valid.mean():.0%}{extra}")

    def poll(self):
        self.poll_id=None
        if self.worker:
            try:
                while True:
                    kind, value = self.worker.events.get_nowait()
                    if kind in ("status", "error"):
                        if kind == "error":
                            self.restore_visibility()
                        self.status.set(("Ошибка: " if kind == "error" else "") + value)
                        if kind=='error' and self.scan_window and self.scan_window.window.winfo_exists() and self.worker.scan is None:
                            self.scan_window.set_active(False)
                            self.scan_window.message.set(value)
                    elif kind=='calibration_error':
                        self.status.set("Калибровка рассчитана и отклонена: " + value)
                        messagebox.showerror("Калибровка не прошла проверку",value+"\n\nИсходные пары сохранены. "
                            "Для Full HD начните с 30 кадров/с; держите доску неподвижно при каждой съёмке. "
                            "Проверьте её плоскость, отсутствие бликов и разные наклоны.",parent=self.root)
                    elif kind=='calibration_busy':
                        self.calibration_busy=bool(value)
                        self.calibrate_button.configure(state='disabled' if value else 'normal',
                            text='Калибровка рассчитывается…' if value else 'Рассчитать калибровку')
                    elif kind in ('scan_started','scan_progress','scan_finished'):
                        if self.scan_window:self.scan_window.update(kind,value)
                    elif kind == "connected":
                        self.connection_pending=False
                        self.connected = True
                        self.connect_button.configure(state="normal", text="Отключить")
                        self.stats.set(f"Реальный размер одного кадра: {value[0]} × {value[1]}")
                    elif kind == "disconnected":
                        self.connection_pending=False; self.calibration_busy=False
                        self.calibrate_button.configure(state='normal',text='Рассчитать калибровку')
                        self.restore_visibility()
                        self.connected = False
                        self.mapping.set(False)
                        self.depth_enabled.set(False)
                        self.preview_last = None
                        self.selected = None
                        self.measurement.set("Камеры отключены — показан последний кадр")
                        self.stats.set("Камеры отключены; последнюю карту можно сохранить или открыть в 3D.")
                        self.connect_button.configure(state="normal", text="Подключить")
                        for control, state in self.settings_controls:
                            control.configure(state=state)
                    elif kind == "count":
                        self.calibration_pairs=int(value)
                        self.count.set(f"Снято пар: {value} / минимум 15")
                    elif kind=='calibration_guidance':
                        message=(f"Наклонных ракурсов: {value['tilted']} / минимум {value['required_tilted']} · "
                            f"углы доски покрывают {value['covered_cells']} из 12 участков кадра.")
                        if value['tilted']<value['required_tilted']:
                            message+='\nНаклоняйте доску в обе стороны вокруг горизонтальной и вертикальной осей.'
                        else:
                            message+='\nДобавьте положения по краям кадра и разные расстояния; замирайте перед снимком.'
                        self.calibration_hint.set(message)
                    elif kind == "capture_finished":
                        self.restore_visibility()
                    elif kind == "calibration":
                        self.selected = None
                        self.measurement.set("Щёлкните по объекту в левом изображении или на карте глубины.")
                        self.calibration_text.set(f"Калибровка: база {value.baseline * 1000:.1f} мм · "
                                                  f"RMS {value.rms:.2f} px · "
                                                  f"ошибка выравнивания {value.epipolar_error:.2f} px")
                        engine = self.worker.engine
                        if engine:
                            self.range_text.set(f"Нижняя граница фильтра: 5 см · текущий поиск глубины примерно от "
                                                f"{engine.search_minimum * 100:.0f} см · "
                                                f"геометрический предел перекрытия примерно {engine.geometric_minimum * 100:.0f} см"+
                                                (". Для 10–15 см включите ближний поиск или выберите профиль с 15 см до подключения."
                                                 if engine.search_minimum>.15 else ""))
                    elif kind == "mapping":
                        self.mapping.set(value)
                    elif kind == "depth_enabled":
                        self.depth_enabled.set(value)
                    elif kind == "calibration_invalidated":
                        self.packet = None
                        self.preview_last = None
                        self.selected = None
                        self.calibration_text.set(f"Размер изменился на {value[0]} × {value[1]} — нужна новая калибровка")
                        self.range_text.set("Нижняя граница фильтра: 5 см. Рабочий минимум определяется калибровкой.")
                        self.measurement.set("Для этого разрешения выполните калибровку заново.")
                        self.panels[2].clear()
                        self.depth_legend.set("Для нового разрешения нужна калибровка")
                    elif kind == "snapshot_measurement":
                        self.selected = None
                        if value:
                            self.measurement.set(f"На выбранном кадре: глубина Z {value['z']:.2f} м · "
                                                 f"расстояние {value['range']:.2f} м · проверьте линейкой")
                        else:
                            self.measurement.set("На выбранном кадре нет достоверной глубины в этой точке.")
                    elif kind == "model":
                        CloudViewer(self.root, *value)
                    elif kind == "estimated_model":
                        CloudViewer(self.root, *value, title="3D с оценёнными поверхностями · оценки не являются измерениями")
                    elif kind == "map_model":
                        CloudViewer(self.root,value["points"],value["colors"],"metres",title=value["title"],
                                    total_count=value["total"],max_points=value["preview_limit"],archive_path=value["archive"])
                    elif kind == "precision":
                        from .precision_window import PrecisionWindow
                        PrecisionWindow(self.root, self.directory, value)
                    elif kind == "distance_report":
                        lines = ["Теоретическая оценка текущего режима. Это НЕ подтверждение точности 1 мм.", ""]
                        for row in value:
                            lines.append(f"{row['distance_m'] * 100:.0f} см: общая область {row['overlap_fraction']:.0%} · "
                                         f"поиск {'есть' if row['search_available'] else 'недостаточен'} · "
                                         f"{row['lateral_mm_per_pixel']:.3f} мм/пиксель")
                        lines.extend(["", "Точный снимок листа использует полное разрешение.",
                                      "Проверьте известный размер линейкой в том же ракурсе и на той же дистанции."])
                        messagebox.showinfo("Дистанция и мелкие детали", "\n".join(lines))
            except Empty:
                pass
            try:
                packet = self.worker.frames.get_nowait()
                if self.connected:
                    self.packet = packet
                    f = packet["depth"]
                    if f and (f is not self.displayed_depth or self.selected):
                        self.refresh_depth()
                    elif f is None and self.displayed_depth is not False:
                        self.panels[2].clear()
                    self.displayed_depth = f if f else False
                    if packet["map"] is not self.displayed_map:
                        self.panels[3].set(packet["map"])
                        self.displayed_map = packet["map"]
                    self.tracking.set(packet["tracking"] if packet["mapping"] else
                                      "Текущая область: измеренные поверхности. Пустые участки неизвестны.")
                    valid = f" · достоверная глубина {f.valid.mean():.0%}" if f else " · глубина пока недоступна"
                    capture = packet.get("capture", {})
                    source_size = capture.get("size", (packet["raw"][0].shape[1], packet["raw"][0].shape[0]))
                    processing = f"расчёт {f.left.shape[1]} × {f.left.shape[0]} · " if f else ""
                    nearest = f" · ближняя глубина ≈{np.percentile(f.points[f.valid, 2], .5):.2f} м" if f and f.valid.any() else ""
                    depth_status = f"глубина {packet['fps']:.1f} кадров/с" if packet.get("depth_enabled") else "расчёт глубины выключен"
                    self.stats.set(f"{capture.get('backend', '')} · вход {source_size[0]} × {source_size[1]} · " +
                                   ("NVIDIA CUDA · " if str(packet.get('acceleration','')).startswith('NVIDIA') else "CPU · ")+
                                   processing +
                                   f"захват {packet.get('capture_fps', 0):.1f} пар/с · "
                                   f"{depth_status}{valid}{nearest}")
                    self.map_stats.set(f"Получено пар: {packet.get('received_frames', 0)} · "
                                       f"кадров глубины: {packet.get('depth_frames', 0)} · "
                                       f"в карту: {packet.get('map_frames', 0)} · "
                                       f"пропущено при слежении: {packet.get('map_rejected', 0)} · "
                                       f"пропущено в очереди карты: {packet.get('map_skipped',0)} · "
                                       f"не подтверждено точек: {packet.get('unconfirmed_points',0)} · "
                                       f"добавление {packet.get('map_fps', 0):.1f} кадров/с · "
                                       f"точек в архиве {packet['map_count']:,} · "
                                       f"оценённых точек {packet.get('estimated_map_count', 0)} · "
                                       f"задержка {packet.get('latency_ms', 0):.0f} мс")
                    if self.selected:
                        self.update_measurement()
            except Empty:
                pass
            self.show_preview()
        if self.closing:
            if not self.worker or not self.worker.is_alive():
                self.root.destroy()
                return
        self.sync_ui()
        self.poll_id=self.root.after(40, self.poll)

    def show_preview(self):
        try:
            raw, captured, sequence = self.worker.previews.get_nowait()
        except Empty:
            return
        if not self.connected:
            return
        engine = self.worker.engine
        calibration = engine.source_calibration if engine else None
        token = None
        if engine and calibration and calibration.size == (raw[0].shape[1], raw[0].shape[0]):
            # Maps point into the full source frames and produce the processing size.
            images = [cv2.remap(image, *maps, cv2.INTER_LINEAR)
                      for image, maps in zip(raw, engine.maps)]
            token = id(engine)
        else:
            images = [image.copy() for image in raw]
        if self.guides.get():
            for image in images:
                for y in range(40, image.shape[0], 40):
                    cv2.line(image, (0, y), (image.shape[1], y), (70, 220, 70), 1)
        self.panels[0].set(images[0])
        self.panels[1].set(images[1])
        self.calibration_panel.set(images[0])
        self.preview_last = {"raw": raw, "calibration_token": token, "captured": captured,
                             "sequence": sequence}

    def close(self):
        self.closing = True
        if self.worker:
            self.worker.stop_event.set()
        self.status.set("Освобождаю камеры и закрываю приложение…")

    def on_destroy(self,event):
        if event.widget is self.root:
            self.closing=True
            if self.poll_id:
                try:self.root.after_cancel(self.poll_id)
                except tk.TclError:pass
                self.poll_id=None
