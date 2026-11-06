"""Shared visual language: palette tokens, short transitions, window helpers and common controls."""
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk


@dataclass(frozen=True)
class Palette:
    """Every colour of the interface. Widgets name a role here, never a hex value."""
    bg: str              # window background behind cards
    surface: str         # cards, sidebars, header and status bar
    raised: str          # controls and chips standing on a surface
    field: str           # inset area of inputs and tables
    border: str          # hairlines between areas
    control_line: str    # outline of raised controls
    border_strong: str   # outline of inputs and check boxes, scrollbar thumbs: at least 3:1 on the surface
    ink: str             # main text
    muted: str           # secondary text
    disabled: str        # text of unavailable controls
    disabled_fill: str   # fill of unavailable controls
    accent: str          # the one interactive colour
    accent_hover: str
    accent_pressed: str
    on_accent: str       # text on the accent colour
    accent_muted: str    # fill of an unavailable primary button
    on_accent_muted: str
    accent_soft: str     # selection background
    success: str
    success_bg: str
    warning: str
    warning_bg: str
    danger: str
    danger_bg: str
    danger_line: str
    image_bg: str        # image wells stay dark in both themes: they frame camera pictures
    image_ink: str
    image_muted: str
    image_line: str


DARK = Palette(
    bg="#10151A", surface="#1A2129", raised="#252E38", field="#12171D",
    border="#303A46", control_line="#3D4855", border_strong="#5F6C7C",
    ink="#E8EDF3", muted="#A4B0BD", disabled="#8896A5", disabled_fill="#1F262E",
    accent="#2DD4BF", accent_hover="#5EEAD4", accent_pressed="#14B8A6", on_accent="#04211D",
    accent_muted="#1D3634", on_accent_muted="#8DB5AE", accent_soft="#17403F",
    success="#7EE2A8", success_bg="#173229", warning="#F5C451", warning_bg="#3A2E12",
    danger="#FF9090", danger_bg="#3C1F24", danger_line="#6E3238",
    image_bg="#101418", image_ink="#E8EDF3", image_muted="#A4B0BD", image_line="#3D4855")

LIGHT = Palette(
    bg="#EDF0F4", surface="#FFFFFF", raised="#F1F4F7", field="#FFFFFF",
    border="#D5DBE3", control_line="#C3CBD5", border_strong="#8A95A2",
    ink="#17202A", muted="#55616E", disabled="#5F6B78", disabled_fill="#EEF1F4",
    accent="#0A7A70", accent_hover="#096A61", accent_pressed="#075951", on_accent="#FFFFFF",
    accent_muted="#D9E8E6", on_accent_muted="#3F6B66", accent_soft="#D2ECE8",
    success="#14663F", success_bg="#E2F4EA", warning="#7A4F00", warning_bg="#FDF1D3",
    danger="#AE2A22", danger_bg="#FCEBE9", danger_line="#EDBDB9",
    image_bg="#101418", image_ink="#E8EDF3", image_muted="#A4B0BD", image_line="#3D4855")

THEME_LABELS = {"dark": "Тёмная", "light": "Светлая"}
_PALETTES = {"dark": DARK, "light": LIGHT}
_active = DARK

# The current values of the main roles, for code that only reads them once the theme is chosen.
BG, SURFACE, INK, MUTED, ACCENT, BORDER, NIGHT = (
    DARK.bg, DARK.surface, DARK.ink, DARK.muted, DARK.accent, DARK.border, DARK.image_bg)

FONT, FONT_STRONG = "Segoe UI", "Segoe UI Semibold"
CAPTION, BODY, BODY_STRONG = (FONT, 9), (FONT, 10), (FONT_STRONG, 10)
SECTION, HEADING, DISPLAY = (FONT_STRONG, 11), (FONT_STRONG, 14), (FONT_STRONG, 20)

MIN_WRAP, WRAP_STEP = 160, 4
# How far a danger button moves towards its text colour; its text stays above 4.5:1 on both.
DANGER_HOVER, DANGER_PRESSED = .08, .15

# Spacing rhythm, in pixels.
SPACE_XS, SPACE_S, SPACE_M, SPACE_L = 4, 8, 12, 16
# Keeps rectangular children of a card clear of its rounded corners.
CARD_INSET = 3

# Transition timings: about 120-180 ms in total, only ever started by an event.
HOVER_STEP_MS, PRESS_STEP_MS, FADE_STEP_MS = 45, 70, 50
FLAGS = ("user1", "user2", "user3")
HOVER_IN = (("user1",), ("user2",), ("user1", "user2"))
HOVER_OUT = tuple(reversed(HOVER_IN))
PRESS = (("user3",),)
FADE_IN = (("user1",), ("user2",))


def palette() -> Palette:
    return _active


def theme_name(value) -> str:
    """Only an explicit light choice is light: old settings without the key open dark."""
    return "light" if str(value).strip().lower() in ("светлая", "light") else "dark"


def use_theme(value) -> Palette:
    """Choose the palette. Call before the interface is built; widgets read it when created."""
    global _active, BG, SURFACE, INK, MUTED, ACCENT, BORDER, NIGHT
    _active = _PALETTES[theme_name(value)]
    BG, SURFACE, INK, MUTED, ACCENT, BORDER, NIGHT = (
        _active.bg, _active.surface, _active.ink, _active.muted, _active.accent, _active.border, _active.image_bg)
    return _active


def _channels(color) -> tuple:
    value = color.lstrip("#")
    return int(value[:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def mix(first, second, amount) -> str:
    """The colour `amount` of the way from first to second."""
    return "#%02X%02X%02X" % tuple(round(a + (b - a) * amount) for a, b in zip(_channels(first), _channels(second)))


def luminance(color) -> float:
    def linear(channel):
        channel /= 255
        return channel / 12.92 if channel <= .03928 else ((channel + .055) / 1.055) ** 2.4
    red, green, blue = _channels(color)
    return .2126 * linear(red) + .7152 * linear(green) + .0722 * linear(blue)


def contrast(first, second) -> float:
    """WCAG contrast ratio of two colours."""
    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + .05) / (darker + .05)


class Animator:
    """Short ttk state transitions. Runs only after an event: nothing is scheduled while idle.

    The theme maps the user1-user3 state flags to in-between colours, so a transition touches
    one widget and never its style, command or any camera picture.
    """

    def __init__(self, root):
        self.root = root
        self.enabled = True
        self.running = {}
        self.watched = set()

    def play(self, widget, frames, step=HOVER_STEP_MS) -> None:
        """Show each frame of state flags for `step` ms, then return to the resting look."""
        self.stop(widget)
        if self.enabled:
            self._watch(widget)
            self._show(widget, tuple(frames), 0, step)

    def stop(self, widget) -> None:
        self._cancel(str(widget))
        self._set_flags(widget, ())

    def stop_all(self) -> None:
        for widget, _ in list(self.running.values()):
            self.stop(widget)

    def _show(self, widget, frames, index, step) -> None:
        key = str(widget)
        self.running.pop(key, None)
        done = index >= len(frames)
        if self._set_flags(widget, () if done else frames[index]) and not done:
            self.running[key] = (widget, self.root.after(step, self._show, widget, frames, index + 1, step))

    @staticmethod
    def _set_flags(widget, flags) -> bool:
        try:
            widget.state([flag if flag in flags else "!" + flag for flag in FLAGS])
            return True
        except tk.TclError:
            return False

    def _cancel(self, key) -> None:
        entry = self.running.pop(key, None)
        if entry:
            try:
                self.root.after_cancel(entry[1])
            except tk.TclError:
                pass

    def _watch(self, widget) -> None:
        key = str(widget)
        if key not in self.watched:
            self.watched.add(key)
            widget.bind("<Destroy>", lambda event: self._forget(key), add="+")

    def _forget(self, key) -> None:
        self._cancel(key)
        self.watched.discard(key)

    def attach(self) -> None:
        """Follow every button and notebook of this root, whoever creates them."""
        bind = self.root.bind_class
        bind("TButton", "<Enter>", lambda event: self._button(event, HOVER_IN, HOVER_STEP_MS), add="+")
        bind("TButton", "<Leave>", lambda event: self._button(event, HOVER_OUT, HOVER_STEP_MS), add="+")
        bind("TButton", "<ButtonPress-1>", lambda event: self._button(event, PRESS, PRESS_STEP_MS), add="+")
        bind("TButton", "<ButtonRelease-1>", lambda event: self._button(event, PRESS, PRESS_STEP_MS), add="+")
        bind("TNotebook", "<<NotebookTabChanged>>", self._notebook, add="+")

    def _button(self, event, frames, step) -> None:
        widget = event.widget
        # A button's own command may already have destroyed it when this runs.
        if isinstance(widget, str) or not self.enabled:
            return
        try:
            if widget.winfo_exists() and not widget.instate(["disabled"]):
                self.play(widget, frames, step)
        except tk.TclError:
            pass

    def _notebook(self, event) -> None:
        if not isinstance(event.widget, str):
            self.play(event.widget, FADE_IN, FADE_STEP_MS)


def animator(widget):
    return getattr(widget._root(), "_stereo_animator", None)


def set_animations(widget, enabled) -> None:
    """Takes effect at once; switching off finishes running transitions immediately."""
    engine = animator(widget)
    if engine:
        engine.enabled = bool(enabled)
        if not enabled:
            engine.stop_all()


def set_badge_style(label, style) -> None:
    """Change a status chip to another style, fading in when the state really changed."""
    if str(label.cget("style")) == style:
        return
    label.configure(style=style)
    engine = animator(label)
    if engine:
        engine.play(label, FADE_IN, FADE_STEP_MS)


def apply_theme(widget) -> None:
    root = widget._root()
    if getattr(root, "_stereo_theme", False):
        return
    root._stereo_theme = True
    from .ui_styles import build_styles
    root.configure(background=palette().bg)
    # Tk keeps only image names, so the image objects must stay alive with the root.
    root._stereo_images = build_styles(root, palette())
    root._stereo_animator = Animator(root)
    root._stereo_animator.attach()


def _dark_title_bar(window) -> None:
    """Match the Windows title bar to the dark theme. Older systems simply keep their own."""
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL("user32")
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        dwm = ctypes.WinDLL("dwmapi")
        dwm.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        handle = user32.GetAncestor(window.winfo_id(), 2)
        enabled = ctypes.c_int(1)
        # 20 is the documented attribute; builds before Windows 10 20H1 used 19.
        if dwm.DwmSetWindowAttribute(handle, 20, ctypes.byref(enabled), ctypes.sizeof(enabled)):
            dwm.DwmSetWindowAttribute(handle, 19, ctypes.byref(enabled), ctypes.sizeof(enabled))
        red, green, blue = _channels(palette().surface)
        caption = wintypes.DWORD(red | green << 8 | blue << 16)
        dwm.DwmSetWindowAttribute(handle, 35, ctypes.byref(caption), ctypes.sizeof(caption))
    except (AttributeError, OSError, ValueError, tk.TclError):
        pass


def title_bar(window) -> None:
    """Re-apply the title bar colour; Tk recreates the native frame when it changes full-screen mode."""
    if palette() is DARK:
        _dark_title_bar(window)


def style_window(window) -> None:
    """Prepare any top-level window: the theme, its background and a matching title bar."""
    apply_theme(window)
    window.configure(background=palette().surface)
    # The native frame exists only once the window is on screen.
    window.bind("<Map>", lambda event: title_bar(window) if event.widget is window else None, add="+")


def note(parent, text=None, variable=None, width=280, **kwargs):
    style = kwargs.pop("style", "Muted.TLabel")
    label = ttk.Label(parent, text=text, textvariable=variable, style=style,
                      wraplength=width, justify="left", **kwargs)
    label.pack(fill="x", pady=(SPACE_XS, SPACE_S))
    return label


def fit_wrap(frame, *labels, margin=0) -> None:
    """Let labels wrap to the current width of the frame or window that holds them.

    A wrapped label asks for its wrap width, which can feed back into a grid column. Small
    changes are ignored, because a pixel of rounding could otherwise repeat for ever and
    freeze the window; where a column is shared, also pass a margin wider than its padding.
    """
    def resize(event):
        if event.widget is not frame:
            return
        width = max(MIN_WRAP, event.width - margin)
        for label in labels:
            if abs(int(str(label.cget("wraplength"))) - width) >= WRAP_STEP:
                label.configure(wraplength=width)
    frame.bind("<Configure>", resize, add="+")


class ScrolledFrame(ttk.Frame):
    """Wheel handling is local to this pane; never replaces global bindings."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, background=palette().surface, highlightthickness=0, width=310)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.scroll.pack(side="right", fill="y", pady=SPACE_XS)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.content = ttk.Frame(self.canvas, padding=(SPACE_L, 14, SPACE_S, SPACE_M))
        item = self.canvas.create_window(0, 0, window=self.content, anchor="nw")
        self.content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(item, width=e.width))
        # A unique bindtag adds local scrolling without stealing a 3D wheel or a
        # combobox's native choice handling in another window.
        self.tag = f"ScrollPane{id(self)}"
        self.canvas.bind_class(self.tag, "<MouseWheel>", self._wheel)
        self.bind("<Destroy>", self._destroy, add="+")

    def attach_wheel(self):
        def attach(widget):
            if not isinstance(widget, (ttk.Combobox, ttk.Entry, tk.Entry)):
                tags = widget.bindtags()
                if self.tag not in tags:
                    widget.bindtags((tags[0], self.tag, *tags[1:]))
            for child in widget.winfo_children():
                attach(child)
        attach(self)

    def _wheel(self, event):
        if self.canvas.yview() != (0., 1.):
            self.canvas.yview_scroll(-int(event.delta / 120), "units")
        return "break"

    def _destroy(self, event):
        if event.widget is self:
            self.unbind_class(self.tag, "<MouseWheel>")


class Foldout(ttk.Frame):
    def __init__(self, parent, title, opened=False):
        super().__init__(parent)
        self.title, self.opened = title, opened
        self.toggle_button = ttk.Button(self, style="Quiet.TButton", command=self.toggle)
        self.toggle_button.pack(fill="x")
        self.content = ttk.Frame(self, padding=(0, SPACE_XS, 0, SPACE_S))
        self._update()

    def toggle(self):
        self.opened = not self.opened
        self._update()

    def _update(self):
        self.toggle_button.configure(text=("▾  " if self.opened else "▸  ") + self.title)
        if self.opened:
            self.content.pack(fill="x")
        else:
            self.content.pack_forget()


def quick_guide(parent):
    window = tk.Toplevel(parent)
    style_window(window)
    window.title("С чего начать · StereoMapper")
    window.geometry("640x560")
    window.minsize(500, 440)
    buttons = ttk.Frame(window, padding=(20, SPACE_M, 20, SPACE_L))
    buttons.pack(side="bottom", fill="x")
    ttk.Button(buttons, text="Понятно", style="Accent.TButton", command=window.destroy).pack(side="right")
    ttk.Separator(window).pack(side="bottom", fill="x")
    pane = ScrolledFrame(window)
    pane.pack(fill="both", expand=True)
    frame = pane.content
    ttk.Label(frame, text="Ваша первая 3D-карта", style="Heading.TLabel").pack(anchor="w", pady=(SPACE_XS, SPACE_M))
    for title, text in (
            ("1. Подключите камеры", "На вкладке «Камеры» выберите подключение. Прикройте левый объектив: должно закрыться левое изображение."),
            ("2. Подготовьте калибровку", "Она связывает изображения с реальными метрами. Загрузите готовую для этой пары или снимите доску в 20–30 разных положениях. Подсказки находятся на вкладке «Калибровка»."),
            ("3. Проверьте глубину", "На вкладке «Сканирование» включите глубину. Чёрные участки означают, что измерения нет. Добавьте свет, детали или измените ракурс."),
            ("4. Соберите и очистите карту", "Нажмите «Новый 3D-скан», начните съёмку и медленно перемещайте камеры. Завершите скан и откройте 3D: там можно вращать, вырезать фон и сшивать снимки.")):
        ttk.Label(frame, text=title, style="Section.TLabel").pack(anchor="w", pady=(SPACE_S, 0))
        note(frame, text, width=560)
    note(frame, "Уже есть карта? «Открыть 3D-карту» работает без подключённых камер. Значок ? объясняет соседнюю настройку.", width=560)
    pane.attach_wheel()
