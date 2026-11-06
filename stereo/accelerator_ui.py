"""Settings panel: see the graphics cards here and download the pack that makes use of them."""
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import messagebox, ttk

from .gpu_pack import (PACKS, gpu_pack_ready, graphics_adapters, install_pack, installed_pack, pack_advice, pending_pack,
                       request_removal)
from .ui import note

POLL_MS = 200


def state_text(root) -> str:
    waiting, active = pending_pack(root), installed_pack(root)
    if waiting:
        return f"Установлено: {PACKS[waiting].title}. Заработает после перезапуска программы."
    if active:
        return f"Сейчас используется: {PACKS[active].title}."
    return "Нейросети сейчас работают на процессоре."


class AcceleratorPanel:
    def __init__(self, parent, app):
        self.app = app
        self.events = Queue()
        self.busy = False
        self.poll_id = None
        self.adapters = None
        self.recommended = None
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x")
        note(self.frame, "Необязательные комплекты для нейросетей взвешивания. Скачиваются один раз и работают только "
                         "на этом компьютере. Программа сама проверит, что быстрее: видеокарта или процессор.")
        self.state = tk.StringVar(value=state_text(app.directory))
        self.found = tk.StringVar(value="Определяю видеокарты…")
        self.progress = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.state, wraplength=270, justify="left").pack(anchor="w", fill="x")
        ttk.Label(self.frame, textvariable=self.found, style="Muted.TLabel", wraplength=270, justify="left").pack(anchor="w", fill="x", pady=(4, 6))
        self.buttons = {}
        for key, pack in PACKS.items():
            button = ttk.Button(self.frame, text=f"{pack.title}, {pack.size_text}", command=lambda key=key: self.install(key))
            button.pack(fill="x", pady=2)
            self.buttons[key] = button
            ttk.Label(self.frame, text=pack.summary, style="Muted.TLabel", wraplength=270, justify="left").pack(anchor="w", fill="x")
        self.remove_button = ttk.Button(self.frame, text="Убрать ускорение", style="Danger.TButton", command=self.remove)
        self.remove_button.pack(fill="x", pady=(8, 2))
        ttk.Label(self.frame, textvariable=self.progress, wraplength=270, justify="left").pack(anchor="w", fill="x")
        note(self.frame, "Глубину со стереокамеры на видеокарте считает только NVIDIA, это уже встроено. "
                         "На других картах она идёт на процессоре.")
        self.frame.bind("<Destroy>", self.closed, add="+")
        self.refresh()
        self.start(lambda: self.events.put(("adapters", graphics_adapters())))

    def start(self, work) -> None:
        Thread(target=work, daemon=True, name="accelerator-panel").start()
        self.schedule()

    def schedule(self) -> None:
        if self.poll_id is None:
            self.poll_id = self.frame.after(POLL_MS, self.poll)

    def closed(self, event) -> None:
        if event.widget is self.frame and self.poll_id is not None:
            self.frame.after_cancel(self.poll_id)
            self.poll_id = None

    def refresh(self) -> None:
        root = self.app.directory
        self.state.set(state_text(root))
        present = pending_pack(root) or installed_pack(root)
        if self.adapters is not None:
            advice = pack_advice(self.adapters, present)
            self.recommended = advice.pack
            self.found.set(advice.text)
        for key, button in self.buttons.items():
            button.configure(state="disabled" if self.busy or key == present else "normal",
                             style="Accent.TButton" if key == self.recommended and key != present else "TButton")
        self.remove_button.configure(state="normal" if gpu_pack_ready(root) and not self.busy else "disabled")

    def install(self, key) -> None:
        if self.busy:
            return
        self.busy = True
        self.refresh()
        self.progress.set("Начинаю скачивание…")

        def work():
            try:
                install_pack(self.app.directory, key, lambda text: self.events.put(("progress", text)))
                self.events.put(("done", "Готово. Перезапустите программу, чтобы ускорение заработало."))
            except Exception as error:  # noqa: BLE001 - the buttons must never stay blocked.
                self.events.put(("done", str(error)))
        self.start(work)

    def remove(self) -> None:
        if not messagebox.askyesno("Ускорение", "Убрать комплект ускорения? Нейросети снова будут работать "
                                                "на процессоре. Комплект можно скачать заново.", parent=self.frame):
            return
        try:
            request_removal(self.app.directory)
            self.progress.set("Ускорение будет убрано при следующем запуске программы.")
        except OSError as error:
            self.progress.set(f"Не удалось убрать: {error}")
        self.refresh()

    def poll(self) -> None:
        self.poll_id = None
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "adapters":
                    self.adapters = value
                    self.refresh()
                elif kind == "progress":
                    self.progress.set(value)
                else:
                    self.busy = False
                    self.progress.set(value)
                    self.refresh()
        except Empty:
            pass
        if self.busy or self.adapters is None:
            self.schedule()
