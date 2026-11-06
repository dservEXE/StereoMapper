"""The ttk styles of the interface, drawn from the palette in ui.py."""
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from .ui import BODY, BODY_STRONG, CAPTION, DANGER_HOVER, DANGER_PRESSED, DISPLAY, FONT_STRONG, HEADING, SECTION, mix

CONTROL_RADIUS, CARD_RADIUS = 6, 10
# Sizes of the prepared control images: see _Painter._stretched.
GROW_CONTROL, GROW_CHIP, GROW_TAB, GROW_CARD = (320, 48), (200, 40), (200, 48), (600, 600)


class _Painter:
    """Draws anti-aliased control images. They are opaque, already composed over their parent colour."""
    OVERSAMPLE = 4

    def __init__(self, root):
        from PIL import Image, ImageDraw, ImageTk
        self.Image, self.ImageDraw, self.ImageTk = Image, ImageDraw, ImageTk
        self.root = root
        self.scale = max(1., round(root.winfo_fpixels("1i") / 96, 2))
        self.size = self.OVERSAMPLE * self.scale
        self.images = {}

    def px(self, value) -> int:
        return max(1, round(value * self.scale))

    def _sheet(self, width, height, on):
        w, h = self.px(width), self.px(height)
        image = self.Image.new("RGB", (w * self.OVERSAMPLE, h * self.OVERSAMPLE), on)
        return image, self.ImageDraw.Draw(image), image.width, image.height

    def _keep(self, key, image, width, height, grow=None) -> str:
        small = image.resize((self.px(width), self.px(height)), self.Image.LANCZOS)
        if grow:
            small = self._stretched(self._stretched(small, self.px(grow[0]), 0), self.px(grow[1]), 1)
        self.images[key] = self.ImageTk.PhotoImage(small, master=self.root)
        return str(self.images[key])

    def _stretched(self, image, size, axis):
        """The same ends around a far larger middle.

        Tk fills the middle of an element by drawing copies of the image's middle one at a
        time, so a small middle on a large card costs thousands of draws on every repaint.
        Elements using a grown image must state their own -width and -height.
        """
        width, height = image.size
        length = image.size[axis]
        if size <= length:
            return image
        middle = length // 2
        if axis == 0:
            result = self.Image.new("RGB", (size, height))
            result.paste(image.crop((0, 0, middle, height)), (0, 0))
            result.paste(image.crop((middle, 0, width, height)), (size - (width - middle), 0))
            result.paste(image.crop((middle, 0, middle + 1, height)).resize((size - length, height), self.Image.NEAREST), (middle, 0))
        else:
            result = self.Image.new("RGB", (width, size))
            result.paste(image.crop((0, 0, width, middle)), (0, 0))
            result.paste(image.crop((0, middle, width, height)), (0, size - (height - middle)))
            result.paste(image.crop((0, middle, width, middle + 1)).resize((width, size - length), self.Image.NEAREST), (0, middle))
        return result

    def _rounded(self, draw, wide, tall, radius, fill, line, line_width, ring) -> None:
        corner = radius * self.size
        if line:
            draw.rounded_rectangle((0, 0, wide - 1, tall - 1), corner, fill=line)
            inset = line_width * self.size
            draw.rounded_rectangle((inset, inset, wide - 1 - inset, tall - 1 - inset), max(0, corner - inset), fill=fill)
        else:
            draw.rounded_rectangle((0, 0, wide - 1, tall - 1), corner, fill=fill)
        if ring:
            inset = 2 * self.size
            draw.rounded_rectangle((inset, inset, wide - 1 - inset, tall - 1 - inset), max(0, corner - inset),
                                   outline=ring, width=round(1.4 * self.size))

    def box(self, width, height, radius, fill, on, line=None, line_width=1, ring=None, grow=None, mark=None, points=()) -> str:
        """A rounded box. `line` is its outline, `ring` an outline inset within the fill.

        `mark` draws `points` on top: one point is a dot, several are a stroke.
        """
        key = (width, height, radius, fill, on, line, line_width, ring, grow, mark, tuple(points))
        if key not in self.images:
            image, draw, wide, tall = self._sheet(width, height, on)
            self._rounded(draw, wide, tall, radius, fill, line, line_width, ring)
            if mark and len(points) == 1:
                x, y, dot = points[0][0] * self.size, points[0][1] * self.size, 3.5 * self.size
                draw.ellipse((x - dot, y - dot, x + dot, y + dot), fill=mark)
            elif mark:
                draw.line([(x * self.size, y * self.size) for x, y in points], fill=mark, width=round(2 * self.size), joint="curve")
            self._keep(key, image, width, height, grow)
        return str(self.images[key])

    def tab(self, on, color=None, thickness=0, outline=None) -> str:
        """A tab background: flat, with an optional line along its lower edge."""
        key = ("tab", on, color, thickness, outline)
        if key not in self.images:
            image, draw, wide, tall = self._sheet(24, 24, on)
            if outline:
                draw.rounded_rectangle((0, 0, wide - 1, tall - 1), 4 * self.size, outline=outline, width=round(1.5 * self.size))
            if color:
                margin = 2 * self.size
                draw.rounded_rectangle((margin, tall - thickness * self.size, wide - margin, tall + self.size),
                                       thickness * self.size / 2, fill=color)
            self._keep(key, image, 24, 24, GROW_TAB)
        return str(self.images[key])

    def stroke(self, width, height, on, color, points, line_width) -> str:
        """A short stroke on a plain background: the chevron of a list."""
        key = ("stroke", width, height, on, color, tuple(points), line_width)
        if key not in self.images:
            image, draw, _, _ = self._sheet(width, height, on)
            draw.line([(x * self.size, y * self.size) for x, y in points], fill=color,
                      width=round(line_width * self.size), joint="curve")
            self._keep(key, image, width, height)
        return str(self.images[key])

    def bar(self, width, height, color, on) -> str:
        """A pill with a clear margin, drawn inside a wider element: a scrollbar thumb."""
        key = ("bar", width, height, color, on)
        if key not in self.images:
            image, draw, wide, tall = self._sheet(width, height, on)
            margin = 2 * self.size
            if width < height:
                draw.rounded_rectangle((margin, 0, wide - 1 - margin, tall - 1), (width - 4) * self.size / 2, fill=color)
            else:
                draw.rounded_rectangle((0, margin, wide - 1, tall - 1 - margin), (height - 4) * self.size / 2, fill=color)
            self._keep(key, image, width, height, (width, 400) if width < height else (400, height))
        return str(self.images[key])


def _button_element(style, paint, name, on, fill, hover, pressed, line, disabled_fill, disabled_line,
                    focus_line=None, focus_ring=None) -> None:
    """A rounded button background with in-between colours for the state-flag transitions."""
    def look(color, outline=line, width=1, ring=None):
        return paint.box(20, 20, CONTROL_RADIUS, color, on, outline if outline else None, width, ring, GROW_CONTROL)
    focus = dict(outline=focus_line or line, width=2 if focus_line else 1, ring=focus_ring)
    style.element_create(
        name, "image", look(fill),
        ("disabled", paint.box(20, 20, CONTROL_RADIUS, disabled_fill, on, disabled_line, grow=GROW_CONTROL)),
        ("pressed", "user3", look(mix(hover, pressed, .5))),
        ("pressed", look(pressed)),
        ("user3", look(mix(hover, pressed, .5))),
        ("focus", "active", look(hover, **focus)),
        ("focus", look(fill, **focus)),
        ("user1", "user2", look(mix(fill, hover, .75))),
        ("user2", look(mix(fill, hover, .5))),
        ("user1", look(mix(fill, hover, .25))),
        ("active", look(hover)),
        border=paint.px(8), padding=0, width=paint.px(20), height=paint.px(20), sticky="nsew")


def _button_layout(style, name, element) -> None:
    style.layout(name, [(element, {"sticky": "nsew", "children": [
        ("Button.padding", {"sticky": "nsew", "children": [("Button.label", {"sticky": "nsew"})]})]})])


def _style_buttons(style, paint, c) -> None:
    px = paint.px
    hover_raised = mix(c.raised, c.ink, .08)
    _button_element(style, paint, "sm.button", c.surface, c.raised, hover_raised, mix(c.raised, c.bg, .5),
                    c.control_line, c.disabled_fill, c.disabled_fill, focus_line=c.accent)
    _button_element(style, paint, "sm.accent", c.surface, c.accent, c.accent_hover, c.accent_pressed,
                    None, c.accent_muted, None, focus_ring=c.on_accent)
    _button_element(style, paint, "sm.danger", c.surface, c.danger_bg, mix(c.danger_bg, c.danger, DANGER_HOVER),
                    mix(c.danger_bg, c.danger, DANGER_PRESSED), c.danger_line, c.disabled_fill, c.disabled_fill,
                    focus_line=c.danger)
    _button_element(style, paint, "sm.quiet", c.surface, c.surface, c.raised, mix(c.raised, c.bg, .5),
                    None, c.surface, None, focus_line=c.accent)
    _button_layout(style, "TButton", "sm.button")
    _button_layout(style, "Accent.TButton", "sm.accent")
    _button_layout(style, "Danger.TButton", "sm.danger")
    _button_layout(style, "Quiet.TButton", "sm.quiet")
    style.configure("TButton", font=BODY, padding=(px(14), px(8)), background=c.surface, foreground=c.ink,
                    anchor="center", relief="flat", borderwidth=0, width=-8)
    style.map("TButton", foreground=[("disabled", c.disabled)], background=[], bordercolor=[], relief=[])
    style.configure("Accent.TButton", font=BODY_STRONG, foreground=c.on_accent, padding=(px(14), px(9)))
    style.map("Accent.TButton", foreground=[("disabled", c.on_accent_muted)])
    style.configure("Danger.TButton", foreground=c.danger)
    style.map("Danger.TButton", foreground=[("disabled", c.disabled)])
    style.configure("Quiet.TButton", foreground=c.accent, anchor="w", padding=(px(6), px(6)))
    style.map("Quiet.TButton", foreground=[("disabled", c.disabled)])
    # Lower controls for the header and the status bar; they keep the looks above.
    style.configure("Toolbar.TButton", padding=(px(10), px(6)))
    style.configure("Small.Quiet.TButton", font=CAPTION, anchor="center", padding=(px(6), px(3)))


def _style_inputs(style, paint, c) -> None:
    px = paint.px
    def field(fill, line, width=1):
        return paint.box(16, 16, 5, fill, c.surface, line, width, grow=GROW_CONTROL)
    style.element_create(
        "sm.field", "image", field(c.field, c.border_strong),
        ("disabled", field(c.disabled_fill, c.disabled_fill)),
        ("focus", field(c.field, c.accent, 2)),
        ("hover", field(c.field, c.muted)),
        # The margin keeps the chevron's opaque background clear of the outline.
        border=px(6), padding=px(2), width=px(16), height=px(16), sticky="nsew")
    chevron = ((5, 7), (8, 10), (11, 7))
    style.element_create(
        "sm.chevron", "image", paint.stroke(18, 16, c.field, c.muted, chevron, 1.5),
        ("disabled", paint.stroke(18, 16, c.disabled_fill, c.disabled, chevron, 1.5)),
        ("focus", paint.stroke(18, 16, c.field, c.accent, chevron, 1.5)),
        sticky="", padding=0)
    style.layout("TEntry", [("sm.field", {"sticky": "nsew", "children": [
        ("Entry.padding", {"sticky": "nsew", "children": [("Entry.textarea", {"sticky": "nsew"})]})]})])
    style.layout("TCombobox", [("sm.field", {"sticky": "nsew", "children": [
        ("sm.chevron", {"side": "right", "sticky": "", "border": 2}),
        ("Combobox.padding", {"expand": "1", "sticky": "nsew", "children": [
            ("Combobox.textarea", {"sticky": "nsew"})]})]})])
    common = dict(font=BODY, foreground=c.ink, background=c.surface, fieldbackground=c.field, insertcolor=c.ink,
                  insertwidth=1, selectbackground=c.accent_soft, selectforeground=c.ink)
    style.configure("TEntry", padding=(px(7), px(4)), **common)
    style.map("TEntry", foreground=[("disabled", c.disabled)], fieldbackground=[], background=[],
              selectbackground=[], selectforeground=[], bordercolor=[], lightcolor=[])
    style.configure("TCombobox", padding=(px(7), px(4), 0, px(4)), **common)
    # Choosing a value must not leave the text of a read-only list highlighted.
    style.map("TCombobox", foreground=[("disabled", c.disabled)], fieldbackground=[], background=[],
              selectbackground=[("readonly", c.field), ("disabled", c.disabled_fill)],
              selectforeground=[("disabled", c.disabled), ("readonly", c.ink)], arrowcolor=[], bordercolor=[], lightcolor=[])
    style.configure("ComboboxPopdownFrame", background=c.border_strong, bordercolor=c.border_strong,
                    lightcolor=c.border_strong, darkcolor=c.border_strong, relief="flat", borderwidth=0)


def _style_choices(style, paint, c) -> None:
    px = paint.px
    tick = ((4.5, 9.5), (7.5, 12.5), (13.5, 6.0))
    dash = ((5, 9), (13, 9))
    def box(radius, fill, line, width=1):
        return paint.box(18, 18, radius, fill, c.surface, line, width)
    def marked(radius, points, fill, mark, ring=None):
        return paint.box(18, 18, radius, fill, c.surface, ring=ring, mark=mark, points=points)
    def indicator(name, radius, on_points):
        style.element_create(
            name, "image", box(radius, c.field, c.border_strong),
            ("disabled", "selected", marked(radius, on_points, c.accent_muted, c.on_accent_muted)),
            ("disabled", box(radius, c.disabled_fill, c.control_line)),
            ("selected", "focus", marked(radius, on_points, c.accent, c.on_accent, ring=c.on_accent)),
            ("selected", "active", marked(radius, on_points, c.accent_hover, c.on_accent)),
            ("selected", marked(radius, on_points, c.accent, c.on_accent)),
            ("alternate", marked(radius, dash, c.raised, c.muted)),
            ("focus", box(radius, c.field, c.accent, 2)),
            ("active", box(radius, c.field, c.muted)),
            width=px(26), sticky="w")
    indicator("sm.check", 4, tick)
    indicator("sm.radio", 9, ((9, 9),))
    for name, element in (("TCheckbutton", "sm.check"), ("TRadiobutton", "sm.radio")):
        kind = name[1:]
        style.layout(name, [(kind + ".padding", {"sticky": "nsew", "children": [
            (element, {"side": "left", "sticky": ""}),
            (kind + ".label", {"side": "left", "sticky": "nsew"})]})])
        style.configure(name, font=BODY, background=c.surface, foreground=c.ink, padding=(0, px(3)))
        style.map(name, background=[], foreground=[("disabled", c.disabled)], indicatorcolor=[], indicatorbackground=[])
    style.configure("Small.TCheckbutton", font=CAPTION, foreground=c.muted, padding=0)
    style.map("Small.TCheckbutton", foreground=[("disabled", c.disabled), ("active", c.ink)])


def _style_scrollbars(style, paint, c) -> None:
    px = paint.px
    for name, width, height, border in (("sm_v.thumb", 10, 24, (px(5), px(8))), ("sm_h.thumb", 24, 10, (px(8), px(5)))):
        style.element_create(
            name, "image", paint.bar(width, height, c.border_strong, c.surface),
            # A disabled scrollbar has nothing to scroll: it stays out of sight.
            ("disabled", paint.bar(width, height, c.surface, c.surface)),
            ("pressed", paint.bar(width, height, c.ink, c.surface)),
            ("active", paint.bar(width, height, c.muted, c.surface)),
            border=border, padding=0, width=px(width), height=px(height), sticky="nsew")
    trough = dict(troughcolor=c.surface, background=c.surface, bordercolor=c.surface, lightcolor=c.surface,
                  darkcolor=c.surface, borderwidth=0, gripcount=0, width=px(10), arrowsize=0)
    for orient, element, sticky in (("Vertical", "sm_v.thumb", "ns"), ("Horizontal", "sm_h.thumb", "ew")):
        name = orient + ".TScrollbar"
        style.layout(name, [(orient + ".Scrollbar.trough", {"sticky": sticky, "children": [
            (element, {"expand": "1", "sticky": "nsew"})]})])
        style.configure(name, **trough)
        style.map(name, background=[], troughcolor=[], bordercolor=[], lightcolor=[], darkcolor=[])


def _style_progress(style, paint, c) -> None:
    px = paint.px
    def pill(color, on):
        return paint.box(16, 6, 3, color, on, grow=(400, 6))
    sizes = dict(border=(px(3), 0), padding=0, height=px(6), sticky="nsew")
    style.element_create("sm_p.pbar", "image", pill(c.accent, c.raised), width=px(8), **sizes)
    # One trough for cards and side panels, one for the window background.
    for name, element, on in (("Horizontal.TProgressbar", "sm_p.trough", c.surface),
                              ("App.Horizontal.TProgressbar", "sm_q.trough", c.bg)):
        style.element_create(element, "image", pill(c.raised, on), width=px(16), **sizes)
        style.layout(name, [(element, {"sticky": "nsew", "children": [("sm_p.pbar", {"side": "left", "sticky": "ns"})]})])
        style.configure(name, background=on, thickness=px(6))


def _style_scale(style, paint, c) -> None:
    px = paint.px
    def knob(color, line=None, width=1):
        return paint.box(16, 16, 8, color, c.surface, line, width)
    style.element_create("sm_s.slider", "image", knob(c.accent),
                         ("disabled", knob(c.disabled_fill, c.control_line)),
                         ("pressed", knob(c.accent_pressed)),
                         ("active", knob(c.accent_hover)),
                         ("focus", knob(c.accent, c.ink, 2)), sticky="")
    style.element_create("sm_s.trough", "image", paint.box(16, 4, 2, c.border_strong, c.surface, grow=(400, 4)),
                         border=(px(3), 0), padding=0, width=px(16), height=px(4), sticky="ew")
    style.layout("Horizontal.TScale", [("sm_s.trough", {"sticky": "ew", "children": [
        ("sm_s.slider", {"side": "left", "sticky": ""})]})])
    style.configure("Horizontal.TScale", background=c.surface, troughcolor=c.surface, sliderlength=px(16), borderwidth=0)
    style.map("Horizontal.TScale", background=[])


def _style_tabs(style, paint, c) -> None:
    px = paint.px
    def tab_element(name, on):
        def line(color, thickness, outline=None):
            return paint.tab(on, color, thickness, outline)
        style.element_create(
            name, "image", line(None, 0),
            ("selected", "user1", line(mix(on, c.accent, .45), 1)),
            ("selected", "user2", line(mix(on, c.accent, .8), 2)),
            ("selected", "focus", line(c.accent, 3, c.accent)),
            ("selected", line(c.accent, 3)),
            ("active", line(c.border_strong, 2)),
            border=(px(8), px(6), px(8), px(6)), padding=0, width=px(24), height=px(24), sticky="nsew")
    def tab_layout(name, element):
        style.layout(name, [(element, {"sticky": "nsew", "children": [
            ("Notebook.padding", {"side": "top", "sticky": "nsew", "children": [
                ("Notebook.label", {"side": "top", "sticky": ""})]})]})])
    tab_element("sm.tab", c.surface)
    tab_element("sm.itab", c.bg)
    tab_layout("TNotebook.Tab", "sm.tab")
    tab_layout("Inner.TNotebook.Tab", "sm.itab")
    frame = dict(borderwidth=0, bordercolor=c.bg, lightcolor=c.bg, darkcolor=c.bg, padding=0)
    # The first tab's text lines up with the title above it.
    style.configure("TNotebook", background=c.surface, tabmargins=(px(2), 0, px(2), 0), **frame)
    style.configure("Inner.TNotebook", background=c.bg, tabmargins=(0, 0, 0, px(8)), **frame)
    style.configure("TNotebook.Tab", font=BODY_STRONG, padding=(px(14), px(8)), background=c.surface,
                    foreground=c.muted, borderwidth=0)
    style.map("TNotebook.Tab", foreground=[("disabled", c.disabled), ("selected", c.ink), ("active", c.ink)],
              background=[], padding=[], expand=[], lightcolor=[], bordercolor=[], darkcolor=[])
    style.configure("Inner.TNotebook.Tab", font=BODY_STRONG, padding=(px(12), px(7)), background=c.bg)


def _chip(paint, fill, on, line=None, width=1):
    return paint.box(16, 16, 5, fill, on, line, width, grow=GROW_CHIP)


def _style_labels(style, paint, c) -> None:
    style.configure("TLabel", background=c.surface, foreground=c.ink, font=BODY)
    style.map("TLabel", foreground=[("disabled", c.disabled)])
    style.configure("Muted.TLabel", foreground=c.muted)
    style.configure("Accent.TLabel", foreground=c.accent)
    style.configure("App.TLabel", background=c.bg, foreground=c.muted)
    style.configure("AppSection.TLabel", background=c.bg, foreground=c.ink, font=SECTION)
    style.configure("Title.TLabel", background=c.surface, foreground=c.ink, font=HEADING)
    style.configure("Heading.TLabel", font=HEADING)
    style.configure("Section.TLabel", font=SECTION)
    style.configure("CardTitle.TLabel", background=c.surface, foreground=c.ink, font=BODY_STRONG)
    style.configure("CardInfo.TLabel", background=c.surface, foreground=c.muted, font=CAPTION)
    style.configure("Status.TLabel", background=c.surface, foreground=c.ink, font=CAPTION)
    style.configure("Metric.TLabel", foreground=c.ink, font=DISPLAY)
    style.configure("MetricCaption.TLabel", foreground=c.muted, font=CAPTION)


def _style_chips(style, paint, c) -> None:
    """Status chips: a small dot and text on a tinted rounded box, fading between the two states."""
    px = paint.px

    def chip_style(name, element, fill, text, dot, from_fill, from_text, from_dot):
        def blended(amount):
            return _chip(paint, mix(from_fill, fill, amount), c.surface)
        def dot_image(color, on):
            return paint.box(8, 8, 4, color, on)
        style.element_create(element, "image", _chip(paint, fill, c.surface),
                             ("user1", blended(1 / 3)), ("user2", blended(2 / 3)),
                             border=px(6), padding=0, width=px(16), height=px(16), sticky="nsew")
        style.element_create(element + "_dot", "image", dot_image(dot, fill),
                             ("user1", dot_image(mix(from_dot, dot, 1 / 3), mix(from_fill, fill, 1 / 3))),
                             ("user2", dot_image(mix(from_dot, dot, 2 / 3), mix(from_fill, fill, 2 / 3))),
                             width=px(14), sticky="w")
        style.layout(name, [(element, {"sticky": "nsew", "children": [
            ("Label.padding", {"sticky": "nsew", "children": [
                (element + "_dot", {"side": "left", "sticky": ""}),
                ("Label.label", {"side": "left", "sticky": "nsew"})]})]})])
        style.configure(name, background=c.surface, foreground=text, font=CAPTION, padding=(px(10), px(4)))
        style.map(name, foreground=[("user1", mix(from_text, text, 1 / 3)), ("user2", mix(from_text, text, 2 / 3))])
    chip_style("Badge.TLabel", "sm.badge", c.raised, c.muted, c.muted, c.success_bg, c.success, c.success)
    chip_style("Success.TLabel", "sm.success", c.success_bg, c.success, c.success, c.raised, c.muted, c.muted)

    style.element_create("sm.warning", "image", _chip(paint, c.warning_bg, c.surface), border=px(6), padding=0, width=px(16), height=px(16), sticky="nsew")
    style.layout("Warning.TLabel", [("sm.warning", {"sticky": "nsew", "children": [
        ("Label.padding", {"sticky": "nsew", "children": [("Label.label", {"sticky": "nsew"})]})]})])
    style.configure("Warning.TLabel", background=c.surface, foreground=c.warning, font=BODY, padding=(px(10), px(6)))
    style.map("Warning.TLabel", foreground=[])

    style.element_create(
        "sm.help", "image", _chip(paint, c.raised, c.surface),
        ("focus", _chip(paint, c.raised, c.surface, c.accent, 2)),
        ("hover", _chip(paint, mix(c.raised, c.ink, .08), c.surface, c.control_line)),
        border=px(6), padding=0, width=px(16), height=px(16), sticky="nsew")
    style.layout("Help.TLabel", [("sm.help", {"sticky": "nsew", "children": [
        ("Label.padding", {"sticky": "nsew", "children": [("Label.label", {"sticky": "nsew"})]})]})])
    style.configure("Help.TLabel", background=c.surface, foreground=c.muted, font=(FONT_STRONG, 9), padding=(px(5), px(3)))
    style.map("Help.TLabel", foreground=[("focus", c.accent), ("hover", c.ink)])


def _style_surfaces(style, paint, c) -> None:
    style.configure(".", font=BODY, background=c.surface, foreground=c.ink, troughcolor=c.raised, bordercolor=c.border,
                    lightcolor=c.surface, darkcolor=c.surface, fieldbackground=c.field, insertcolor=c.ink,
                    selectbackground=c.accent_soft, selectforeground=c.ink, focuscolor=c.accent)
    # The base map of clam repaints anything active or disabled in its own light grey.
    style.map(".", background=[], foreground=[("disabled", c.disabled)], selectbackground=[], selectforeground=[])
    style.configure("TFrame", background=c.surface, borderwidth=0)
    style.configure("App.TFrame", background=c.bg)
    style.element_create("sm.card", "image", paint.box(28, 28, CARD_RADIUS, c.surface, c.bg, c.border, grow=GROW_CARD),
                         border=paint.px(11), padding=0, width=paint.px(28), height=paint.px(28), sticky="nsew")
    style.layout("Card.TFrame", [("sm.card", {"sticky": "nsew"})])
    style.configure("Card.TFrame", background=c.bg)
    for orient, sticky, size in (("Horizontal", "ew", (400, 1)), ("Vertical", "ns", (1, 400))):
        element = "sm.line" + orient
        style.element_create(element, "image", paint.box(size[0], size[1], 0, c.border, c.border), padding=0,
                             width=paint.px(min(size[0], 4)), height=paint.px(min(size[1], 4)), sticky=sticky)
        style.layout(orient + ".TSeparator", [(element, {"sticky": sticky})])
    style.configure("TSeparator", background=c.surface)


def _style_tables(style, paint, c) -> None:
    px = paint.px
    style.configure("Treeview", font=BODY, rowheight=px(30), background=c.field, fieldbackground=c.field, foreground=c.ink,
                    bordercolor=c.border, lightcolor=c.field, darkcolor=c.field, borderwidth=1)
    style.map("Treeview", background=[("selected", c.accent_soft)],
              foreground=[("selected", c.ink), ("disabled", c.disabled)], bordercolor=[], lightcolor=[], darkcolor=[])
    style.configure("Treeview.Heading", font=BODY_STRONG, padding=(px(8), px(7)), background=c.raised, foreground=c.muted,
                    relief="flat", borderwidth=0, bordercolor=c.border, lightcolor=c.raised, darkcolor=c.raised)
    style.map("Treeview.Heading", background=[("active", mix(c.raised, c.ink, .08))], foreground=[("active", c.ink)],
              relief=[], lightcolor=[], darkcolor=[], bordercolor=[])
    style.configure("Item", focuscolor=c.accent)


def _named_fonts(root) -> None:
    """Entries and lists take their font from these names rather than from the style."""
    for name, (family, size) in (("TkDefaultFont", BODY), ("TkTextFont", BODY), ("TkMenuFont", BODY),
                                 ("TkHeadingFont", BODY_STRONG), ("TkCaptionFont", SECTION),
                                 ("TkTooltipFont", CAPTION), ("TkSmallCaptionFont", CAPTION)):
        try:
            tkfont.nametofont(name, root=root).configure(family=family, size=size)
        except tk.TclError:
            pass


def _option_defaults(root, c) -> None:
    """Classic Tk widgets read the option database, so they follow the palette when they are created."""
    font = "{%s} %d" % BODY
    for pattern, value in (
            ("*Toplevel.background", c.surface), ("*Frame.background", c.surface),
            ("*Label.background", c.surface), ("*Label.foreground", c.ink),
            ("*Canvas.background", c.surface), ("*Canvas.highlightThickness", 0),
            ("*Listbox.background", c.field), ("*Listbox.foreground", c.ink),
            ("*Listbox.selectBackground", c.accent_soft), ("*Listbox.selectForeground", c.ink),
            ("*Listbox.highlightThickness", 1), ("*Listbox.highlightBackground", c.border),
            ("*Listbox.highlightColor", c.accent), ("*Listbox.borderWidth", 0),
            ("*Listbox.relief", "flat"), ("*Listbox.font", font),
            ("*TCombobox*Listbox.background", c.raised), ("*TCombobox*Listbox.highlightThickness", 0),
            ("*Text.background", c.field), ("*Text.foreground", c.ink), ("*Text.insertBackground", c.ink),
            ("*Text.selectBackground", c.accent_soft), ("*Text.selectForeground", c.ink),
            ("*Text.highlightThickness", 0), ("*Text.borderWidth", 0), ("*Text.relief", "flat")):
        root.option_add(pattern, value)


def build_styles(root, colors) -> dict:
    """Define every style on this root; returns the images, which must outlive the styles."""
    _option_defaults(root, colors)
    _named_fonts(root)
    style = ttk.Style(root)
    style.theme_use("clam")
    paint = _Painter(root)
    for define in (_style_surfaces, _style_labels, _style_chips, _style_buttons, _style_inputs, _style_choices,
                   _style_scrollbars, _style_progress, _style_scale, _style_tabs, _style_tables):
        define(style, paint, colors)
    return paint.images
