"""Stereo aiming preview and capture shortcuts shared by calibration windows."""
import ctypes
import os
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import ui


class StereoPreview(tk.Frame):
    def __init__(self, parent, counter_var=None):
        colors = ui.palette()
        super().__init__(parent, background=colors.image_bg, highlightthickness=1,
                         highlightbackground=colors.image_line, padx=6, pady=6)
        self.frames = None
        self.image = None
        self.hidden = False
        self.placement = None
        header = tk.Frame(self, background=colors.image_bg)
        header.pack(fill="x")
        for text in ("Левая камера", "Правая камера"):
            tk.Label(header, text=text, background=colors.image_bg, foreground=colors.image_ink, font=ui.CAPTION).pack(
                side="left", expand=True)
        self.canvas = tk.Canvas(self, width=448, height=168, background="#000000",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_text(224, 84, text="Подключите камеры", fill=colors.image_muted, font=ui.BODY)
        footer = tk.Frame(self, background=colors.image_bg)
        footer.pack(fill="x", pady=(4, 0))
        tk.Label(footer, text="Пробел / S — снять пару", background=colors.image_bg,
                 foreground=colors.image_ink, font=ui.CAPTION).pack(side="left")
        if counter_var is not None:
            tk.Label(footer, textvariable=counter_var, background=colors.image_bg,
                     foreground=colors.image_muted, font=ui.CAPTION).pack(side="right", padx=(12, 0))
        self.bind("<Map>", lambda event: self.draw())

    def set_frames(self, frames):
        self.frames = frames
        self.draw()

    def draw(self):
        if self.hidden or self.frames is None or not self.winfo_viewable():
            return
        combined = np.zeros((168, 448, 3), np.uint8)
        for index, frame in enumerate(self.frames):
            height, width = frame.shape[:2]
            factor = min(224 / width, 168 / height)
            size = (max(1, round(width * factor)), max(1, round(height * factor)))
            resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            x, y = index * 224 + (224 - size[0]) // 2, (168 - size[1]) // 2
            combined[y:y + size[1], x:x + size[0]] = resized
        self.image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)),
                                       master=self.canvas)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.image, anchor="nw")

    def set_hidden(self, hidden):
        if hidden == self.hidden:
            return
        self.hidden = hidden
        if hidden:
            self.placement = self.place_info()
            self.place_forget()
        elif self.placement:
            self.place(**self.placement)
            self.draw()


class CalibrationShortcuts:
    def __init__(self, root, on_capture, is_active):
        self.root, self.on_capture, self.is_active = root, on_capture, is_active
        self.tag = f"CalibrationCapture{id(self)}"
        self.windows, self.held = [], set()
        self.poll_id = None
        self.closed = False
        self.press_id = root.bind_class(self.tag, "<KeyPress>", self.press)
        self.release_id = root.bind_class(self.tag, "<KeyRelease>", self.release)
        self.user32 = None
        if os.name == "nt":
            self.user32 = ctypes.WinDLL("user32", use_last_error=True)
            self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
            self.user32.GetAsyncKeyState.restype = ctypes.c_short
            self.user32.GetForegroundWindow.restype = ctypes.c_void_p
            self.user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            self.user32.GetAncestor.restype = ctypes.c_void_p
            self.poll_id = root.after(30, self.poll)
        root.bind("<Destroy>", self.on_destroy, add="+")

    def add_window(self, window):
        self.windows.append(window)
        widgets = [window]
        while widgets:
            widget = widgets.pop()
            widget.bindtags((self.tag,) + tuple(tag for tag in widget.bindtags() if tag != self.tag))
            widgets.extend(widget.winfo_children())

    @staticmethod
    def capture_key(event):
        if os.name == "nt" and event.keycode in (32, 83):
            return event.keycode
        return {"space": 32, "s": 83, "S": 83, "Cyrillic_yeru": 83,
                "Cyrillic_YERU": 83}.get(event.keysym)

    def press(self, event):
        key = self.capture_key(event)
        window = event.widget.winfo_toplevel()
        if key is None or not self.is_active(window):
            return
        if key not in self.held:
            self.held.add(key)
            self.on_capture(window)
        return "break"

    def release(self, event):
        key = self.capture_key(event)
        self.held.discard(key)
        if key is not None and self.is_active(event.widget.winfo_toplevel()):
            return "break"

    def poll(self):
        """Also accept physical keys when Windows has not assigned Tk child focus."""
        self.poll_id = None
        if self.closed:
            return
        self.poll_id = self.root.after(30, self.poll)
        states = {key: self.user32.GetAsyncKeyState(key) for key in (32, 83)}
        down = {key for key, state in states.items() if state & 0x8000}
        # The transition bit also catches taps released between timer ticks.
        pressed = {key for key, state in states.items() if state & 0x8001 and key not in self.held}
        self.held = down
        self.windows = [window for window in self.windows if window.winfo_exists()]
        if pressed:
            foreground = self.user32.GetForegroundWindow()
            for window in self.windows:
                if (window.winfo_viewable() and self.is_active(window)
                        and self.user32.GetAncestor(window.winfo_id(), 2) == foreground):
                    self.on_capture(window)
                    break

    def on_destroy(self, event):
        if event.widget is self.root:
            self.closed = True
            if self.poll_id is not None:
                self.root.after_cancel(self.poll_id)
                self.poll_id = None
            for sequence, command in (("<KeyPress>", self.press_id), ("<KeyRelease>", self.release_id)):
                self.root.unbind_class(self.tag, sequence)
                # bind_class registers the command on the Tk root, even from a Toplevel.
                self.root._root().deletecommand(command)
