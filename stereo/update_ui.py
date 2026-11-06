"""Optional updates: networking stays in workers and Tk stays on its main thread."""
import json
import math
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
from threading import Event, Lock, Thread
import time
import tkinter as tk
from tkinter import ttk
import urllib.error
import webbrowser

from . import ui
from .version import VERSION
from .updates import (Cancelled, check_release, download_release, cleanup, state_root,
                      update_root, owned_directory, REPOSITORY)

INTERVAL = 6 * 3600


class UpdateController:
    def __init__(self, app, parent, status_parent=None):
        self.app, self.root = app, app.root
        self.file = state_root() / "updates-settings.json"
        try:
            self.settings = json.loads(self.file.read_text(encoding="utf8"))
        except (OSError, ValueError):
            self.settings = {}
        if not isinstance(self.settings, dict):
            self.settings = {}
        last = self.settings.get("last", 0)
        if type(last) not in (int, float) or not math.isfinite(last) or last > time.time() + 300:
            self.settings["last"] = 0
        self.events, self.available, self.prepared = Queue(), None, None
        self.working, self.closed, self.poll_id = False, False, None
        self.cancelled = Event()
        self.events_lock = Lock()
        self.thread = None
        self.auto = tk.BooleanVar(master=self.root, value=self.settings.get("auto") is not False)
        self.text = tk.StringVar(master=self.root, value=f"Версия {VERSION}")
        # The version and the check live quietly in the status bar when one is given.
        self.frame = ttk.Frame(status_parent or parent)
        if status_parent is None:
            self.frame.pack(fill="x", pady=(0, 10), before=app.workflow)
        else:
            self.frame.pack(side="right", padx=(ui.SPACE_L, 0))
        self.label = ttk.Label(self.frame, textvariable=self.text, style="CardInfo.TLabel", wraplength=300)
        self.label.pack(side="left", fill="x", expand=True)
        self.check_button = ttk.Button(self.frame, text="Проверить обновления", style="Small.Quiet.TButton", command=lambda: self.check(True))
        self.check_button.pack(side="right")
        self.auto_button = ttk.Checkbutton(self.frame, text="Автопроверка", style="Small.TCheckbutton", variable=self.auto, command=self.configure)
        self.auto_button.pack(side="right", padx=ui.SPACE_M)
        colors = ui.palette()
        self.banner = tk.Frame(parent, background=colors.surface, padx=ui.SPACE_L, pady=ui.SPACE_S)
        self.banner_text = tk.Label(self.banner, text="", background=colors.surface, foreground=colors.accent, font=ui.SECTION, wraplength=550, justify="left", anchor="w")
        self.banner_text.pack(side="left", fill="x", expand=True)
        self.later_button = ttk.Button(self.banner, text="Позже", style="Toolbar.TButton", command=self.later)
        self.later_button.pack(side="right", padx=(ui.SPACE_S, 0))
        self.install_button = ttk.Button(self.banner, text="Обновить и перезапустить", style="Accent.TButton", command=self.install)
        self.install_button.pack(side="right")
        self.startup_id = self.root.after(1500, self.startup)
        self.schedule_id = self.root.after(INTERVAL * 1000, self.scheduled)
        self.startup_cleanup()

    def save(self):
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.file.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.settings), encoding="utf8")
            temporary.replace(self.file)
            return True
        except OSError:
            # Failed preference storage must never prevent using the camera.
            return False

    def configure(self):
        self.settings["auto"] = self.auto.get()
        self.save()
        if not self.auto.get() and not self.working:
            self.text.set(f"Версия {VERSION} · автопроверка выключена")

    def later(self):
        if self.closed or self.working:
            return
        if self.available:
            self.settings["dismissed"] = self.available["tag"]
            self.save()
            self.banner.pack_forget()
            self.text.set("Обновление доступно через «Проверить обновления».")

    def startup(self):
        self.startup_id = None
        self.check(False)

    def scheduled(self):
        self.schedule_id = None
        if self.closed:
            return
        self.check(False)
        self.schedule_id = self.root.after(INTERVAL * 1000, self.scheduled)

    def job(self, kind, function):
        if self.closed or self.working:
            return
        self.working = True
        self.check_button.configure(state="disabled")
        self.install_button.configure(state="disabled")
        self.later_button.configure(state="disabled")

        def work():
            try:
                value = function()
                with self.events_lock:
                    if self.cancelled.is_set():
                        if kind == "downloaded":
                            cleanup(value[0].parent)
                        return
                    self.events.put((kind, value))
            except Cancelled:
                if not self.cancelled.is_set():
                    self.events.put(("error", "Загрузка отменена"))
            except Exception as error:
                if not self.cancelled.is_set():
                    self.events.put(("error", self.error_message(error)))

        self.thread = Thread(target=work, daemon=True, name="github-update-" + kind)
        self.thread.start()
        self.poll_id = self.root.after(100, self.poll)

    @staticmethod
    def error_message(error):
        if isinstance(error, urllib.error.HTTPError) and error.code in (403, 429):
            return "GitHub временно ограничил проверки. Попробуйте позже."
        if isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError)):
            return "Нет связи с GitHub. Программой можно пользоваться без обновления."
        return str(error)

    def check(self, manual=False):
        if self.closed or self.working or self.app.closing:
            return
        if not manual and (not self.auto.get() or time.time() - self.settings.get("last", 0) < INTERVAL):
            return
        self.text.set("Проверяю обновления на GitHub…")
        self.job("release", lambda: (check_release(VERSION), manual))

    def busy(self):
        if any(bool(getattr(self.app, name, False)) for name in ("connected", "connection_pending", "calibration_busy")):
            return True
        worker = getattr(self.app, "worker", None)
        if worker:
            if getattr(worker, "scan", None) is not None or getattr(worker, "calibration_busy", False):
                return True
            for name in ("export_thread", "calibration_job"):
                thread = getattr(worker, name, None)
                if thread and thread.is_alive():
                    return True
        for child in self.root.winfo_children():
            check = getattr(child, "storage_busy", None)
            if callable(check) and check():
                return True
        return False

    def install(self):
        if self.closed or self.working or not self.available:
            return
        if self.busy():
            self.text.set("Сначала отключите камеры и дождитесь завершения сохранения и расчётов.")
            return
        root = self.app.directory.resolve()
        if self.available.get("requires_setup") or not (root / "runtime/pythonw.exe").is_file() or not (root / "StereoMapper.exe").is_file():
            # A complete installer is needed for changed runtime requirements or sources.
            webbrowser.open(self.available.get("page", f"https://github.com/{REPOSITORY}/releases/latest"))
            self.text.set("Открыта страница версии. Скачайте StereoMapper-Setup.exe и запустите его.")
            return
        if self.prepared:
            self.launch(*self.prepared)
            return
        release = self.available.copy()
        self.text.set("Загружаю обновление…")
        self.job("downloaded", lambda: (download_release(release,
            lambda percent: self.events.put(("progress", percent)), cancelled=self.cancelled.is_set), root))

    def launch(self, installer, root):
        if self.closed:
            cleanup(installer.parent)
            return
        if self.busy():
            self.prepared = (installer, root)
            self.text.set("Обновление загружено. Отключите камеры и завершите обработку, затем нажмите «Обновить».")
            return
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            subprocess.Popen([str(installer), "--update", str(os.getpid()), sys.executable,
                              str(root), str(installer.parent)], creationflags=flags,
                             close_fds=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # The installer now owns this file; close() must not remove it.
            self.prepared = None
            self.app.close()
        except OSError as error:
            cleanup(installer.parent)
            self.prepared = None
            self.text.set("Не удалось запустить обновление: " + str(error))

    def poll(self):
        self.poll_id = None
        if self.closed:
            return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "progress":
                    self.text.set(f"Загружаю обновление: {value}%")
                    continue
                # Exactly one terminal message per job; no 'done' race with a new job.
                self.working = False
                self.check_button.configure(state="normal")
                self.install_button.configure(state="normal")
                self.later_button.configure(state="normal")
                if kind == "release":
                    available, manual = value
                    if self.prepared and (not available or not self.available or available["tag"] != self.available["tag"]):
                        cleanup(self.prepared[0].parent)
                        self.prepared = None
                    self.available = available
                    self.settings["last"] = time.time()
                    if self.available:
                        if manual:
                            self.settings["dismissed"] = None
                        self.text.set("Доступна версия " + self.available["version"])
                        self.banner_text.configure(text="Новая версия " + self.available["version"] + " готова к установке")
                        self.install_button.configure(text="Скачать установщик" if self.available.get("requires_setup") else "Обновить и перезапустить")
                        if self.settings.get("dismissed") != self.available["tag"]:
                            self.banner.pack(fill="x", before=self.app.workflow)
                    else:
                        self.banner.pack_forget()
                        self.text.set("Обновлений пока нет · версия " + VERSION)
                    self.save()
                elif kind == "downloaded":
                    self.prepared = value
                    self.launch(*value)
                elif kind == "error":
                    self.text.set("Обновление недоступно: " + value)
                if self.closed:
                    return
        except Empty:
            pass
        if self.working:
            self.poll_id = self.root.after(100, self.poll)

    def close(self):
        if self.closed:
            return
        with self.events_lock:
            self.closed = True
            self.cancelled.set()
        for name in ("startup_id", "schedule_id", "poll_id"):
            token = getattr(self, name, None)
            if token:
                try:
                    self.root.after_cancel(token)
                except tk.TclError:
                    pass
                setattr(self, name, None)
        if self.prepared:
            cleanup(self.prepared[0].parent)
            self.prepared = None
        # A completed worker may have queued a download just before close().
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "downloaded":
                    cleanup(value[0].parent)
        except Empty:
            pass

    def startup_cleanup(self):
        # Only this updater's UUID cache and known filenames are ever removed.
        try:
            root = update_root()
            for directory in root.iterdir():
                if owned_directory(directory) and time.time() - directory.stat().st_mtime > 86400:
                    cleanup(directory)
        except OSError:
            pass
        if "--cleanup-update" not in sys.argv:
            return
        try:
            i = sys.argv.index("--cleanup-update")
            directory, pid = Path(sys.argv[i + 1]), int(sys.argv[i + 2])
            if pid <= 0 or not owned_directory(directory):
                return

            def wait():
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                kernel.WaitForSingleObject.restype = wintypes.DWORD
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.OpenProcess(0x100000, False, pid)
                if handle:
                    result = kernel.WaitForSingleObject(handle, 60000)
                    kernel.CloseHandle(handle)
                    if result != 0:
                        return
                cleanup(directory)
            Thread(target=wait, daemon=True, name="github-update-cleanup").start()
        except (ValueError, IndexError, OSError):
            pass
