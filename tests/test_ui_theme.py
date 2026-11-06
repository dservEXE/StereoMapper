"""Theme tokens, the saved interface preferences and the short transitions, without USB devices.

All classes share one Tk root; tests/conftest.py keeps starting Tk reliable under pytest.
"""
import faulthandler
import gc
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from stereo import ui
from stereo.app import StereoApp
from stereo.viewer import COMPASS_LABEL_HEIGHT, COMPASS_LABEL_WIDTH, compass_label_places


TEXT_ON_BACKGROUND = (
    ("ink", "bg"), ("ink", "surface"), ("ink", "raised"), ("ink", "field"), ("ink", "accent_soft"),
    ("muted", "bg"), ("muted", "surface"), ("muted", "raised"), ("muted", "field"),
    ("disabled", "disabled_fill"), ("disabled", "surface"),
    ("on_accent", "accent"), ("on_accent", "accent_hover"), ("on_accent", "accent_pressed"),
    ("on_accent_muted", "accent_muted"),
    ("accent", "bg"), ("accent", "surface"), ("accent", "raised"),
    ("success", "success_bg"), ("warning", "warning_bg"),
    ("danger", "danger_bg"), ("danger", "surface"),
    ("image_ink", "image_bg"), ("image_muted", "image_bg"))
ICONS_ON_BACKGROUND = (("border_strong", "surface"), ("border_strong", "field"), ("muted", "surface"))


class PaletteTests(unittest.TestCase):
    def test_text_reaches_four_and_a_half_to_one_in_both_themes(self):
        for colors in (ui.DARK, ui.LIGHT):
            for text, background in TEXT_ON_BACKGROUND:
                ratio = ui.contrast(getattr(colors, text), getattr(colors, background))
                self.assertGreaterEqual(ratio, 4.5, f"{text} on {background}: {ratio:.2f}")

    def test_danger_text_stays_readable_on_its_hover_and_pressed_fills(self):
        for colors in (ui.DARK, ui.LIGHT):
            for amount in (ui.DANGER_HOVER, ui.DANGER_PRESSED):
                fill = ui.mix(colors.danger_bg, colors.danger, amount)
                self.assertGreaterEqual(ui.contrast(colors.danger, fill), 4.5)

    def test_icons_and_outlines_reach_three_to_one(self):
        for colors in (ui.DARK, ui.LIGHT):
            for mark, background in ICONS_ON_BACKGROUND:
                ratio = ui.contrast(getattr(colors, mark), getattr(colors, background))
                self.assertGreaterEqual(ratio, 3, f"{mark} on {background}: {ratio:.2f}")

    def test_background_is_dark_but_not_black(self):
        self.assertNotEqual(ui.DARK.bg.lower(), "#000000")
        self.assertLess(ui.luminance(ui.DARK.bg), .02)

    def test_contrast_matches_known_values(self):
        self.assertAlmostEqual(ui.contrast("#000000", "#FFFFFF"), 21, places=3)
        self.assertAlmostEqual(ui.contrast("#123456", "#123456"), 1, places=3)

    def test_only_an_explicit_light_choice_is_light(self):
        self.assertEqual(ui.theme_name("Светлая"), "light")
        self.assertEqual(ui.theme_name("light"), "light")
        for value in ("Тёмная", "", None, "неизвестно"):
            self.assertEqual(ui.theme_name(value), "dark")

    def test_use_theme_updates_the_module_constants(self):
        self.addCleanup(ui.use_theme, "dark")
        ui.use_theme("Светлая")
        self.assertIs(ui.palette(), ui.LIGHT)
        self.assertEqual(ui.BG, ui.LIGHT.bg)
        self.assertEqual(ui.NIGHT, ui.LIGHT.image_bg)
        ui.use_theme("Тёмная")
        self.assertIs(ui.palette(), ui.DARK)
        self.assertEqual(ui.SURFACE, ui.DARK.surface)


class Pending:
    """The Tcl timers currently waiting in an interpreter."""

    @staticmethod
    def timers(root):
        return root.tk.splitlist(root.tk.call("after", "info"))


def new_root():
    root = tk.Tk()
    root.withdraw()
    return root


def close_root(root):
    root.destroy()
    del root
    gc.collect()


shared_root = None


def setUpModule():
    global shared_root
    shared_root = new_root()
    ui.apply_theme(shared_root)


def tearDownModule():
    global shared_root
    root, shared_root = shared_root, None
    close_root(root)


class AnimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = shared_root
        cls.engine = ui.animator(cls.root)

    def setUp(self):
        self.errors = []
        self.root.report_callback_exception = lambda *error: self.errors.append(error)
        ui.set_animations(self.root, True)
        self.button = ttk.Button(self.root, text="Кнопка")
        self.button.pack()
        self.root.update()

    def tearDown(self):
        self.root.update()
        self.engine.stop_all()
        if self.button.winfo_exists():
            self.button.destroy()
        self.assertEqual(self.engine.running, {})
        self.assertEqual(self.errors, [])

    def flags(self, widget):
        return {flag for flag in ui.FLAGS if widget.instate([flag])}

    def pump(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.root.update()
            time.sleep(.005)

    def test_a_transition_passes_its_frames_and_returns_to_rest(self):
        self.engine.play(self.button, ui.HOVER_IN, 10)
        self.assertEqual(self.flags(self.button), {"user1"})
        self.assertIn(str(self.button), self.engine.running)
        self.pump(.25)
        self.assertEqual(self.flags(self.button), set())
        self.assertEqual(self.engine.running, {})
        self.assertEqual(Pending.timers(self.root), ())

    def test_entering_a_button_starts_a_transition_only_while_enabled(self):
        self.button.event_generate("<Enter>")
        self.assertTrue(self.button.instate(["active"]))
        self.assertIn(str(self.button), self.engine.running)
        self.engine.stop(self.button)
        ui.set_animations(self.root, False)
        self.button.event_generate("<Leave>")
        self.button.event_generate("<Enter>")
        self.assertTrue(self.button.instate(["active"]), "the hover state itself must stay immediate")
        self.assertEqual(self.engine.running, {})
        self.assertEqual(self.flags(self.button), set())

    def test_a_disabled_button_does_not_animate(self):
        self.button.state(["disabled"])
        self.button.event_generate("<Enter>")
        self.assertEqual(self.engine.running, {})

    def test_switching_off_finishes_running_transitions_at_once(self):
        self.engine.play(self.button, ui.HOVER_IN, 500)
        self.assertEqual(len(Pending.timers(self.root)), 1)
        ui.set_animations(self.root, False)
        self.assertEqual(self.flags(self.button), set())
        self.assertEqual(self.engine.running, {})
        self.assertEqual(Pending.timers(self.root), ())

    def test_playing_while_switched_off_schedules_nothing(self):
        ui.set_animations(self.root, False)
        self.engine.play(self.button, ui.HOVER_IN, 10)
        self.assertEqual(self.flags(self.button), set())
        self.assertEqual(Pending.timers(self.root), ())
        ui.set_animations(self.root, True)
        self.engine.play(self.button, ui.HOVER_IN, 10)
        self.assertEqual(len(Pending.timers(self.root)), 1)

    def test_destroying_a_widget_cancels_its_timer(self):
        self.engine.play(self.button, ui.HOVER_IN, 10)
        self.assertEqual(len(Pending.timers(self.root)), 1)
        self.button.destroy()
        self.assertEqual(self.engine.running, {})
        self.assertNotIn(str(self.button), self.engine.watched)
        self.assertEqual(Pending.timers(self.root), ())
        self.pump(.05)

    def test_a_new_transition_replaces_the_running_one(self):
        self.engine.play(self.button, ui.HOVER_IN, 500)
        self.engine.play(self.button, ui.HOVER_OUT, 500)
        self.assertEqual(len(Pending.timers(self.root)), 1)
        self.assertEqual(self.flags(self.button), {"user1", "user2"})

    def test_a_button_destroyed_by_its_own_command_raises_nothing(self):
        window = tk.Toplevel(self.root)
        button = ttk.Button(window, text="Закрыть", command=window.destroy)
        button.pack()
        self.root.update()
        button.event_generate("<Enter>")
        button.event_generate("<ButtonPress-1>")
        button.event_generate("<ButtonRelease-1>")
        self.pump(.05)
        self.assertEqual(self.engine.running, {})
        self.assertEqual(Pending.timers(self.root), ())
        self.assertFalse(window.winfo_exists())

    def test_changing_the_tab_animates_once_and_ends(self):
        book = ttk.Notebook(self.root)
        pages = [ttk.Frame(book), ttk.Frame(book)]
        for page in pages:
            book.add(page, text="Вкладка")
        book.pack()
        self.root.update()
        self.pump(.2)
        book.select(pages[1])
        self.root.update()
        self.assertIn(str(book), self.engine.running)
        self.assertTrue(book.instate(["alternate"]), "user1 and user2 mark the first and last tab")
        self.assertFalse(set(ui.TAB_FADE_IN[0] + ui.TAB_FADE_IN[1]) & {"user1", "user2"})
        self.pump(.3)
        self.assertEqual(self.engine.running, {})
        self.assertEqual(Pending.timers(self.root), ())
        book.destroy()

    def test_a_status_chip_fades_only_when_its_state_changes(self):
        label = ttk.Label(self.root, text="Камеры", style="Badge.TLabel")
        label.pack()
        ui.set_badge_style(label, "Badge.TLabel")
        self.assertEqual(self.engine.running, {})
        ui.set_badge_style(label, "Success.TLabel")
        self.assertEqual(str(label.cget("style")), "Success.TLabel")
        self.assertIn(str(label), self.engine.running)
        label.destroy()
        self.assertEqual(self.engine.running, {})

    def test_destroying_the_whole_window_leaves_no_timer(self):
        other = new_root()
        errors = []
        other.report_callback_exception = lambda *error: errors.append(error)
        ui.apply_theme(other)
        button = ttk.Button(other, text="Кнопка")
        button.pack()
        other.update()
        ui.animator(other).play(button, ui.HOVER_IN, 10)
        del button
        close_root(other)
        self.pump(.05)
        self.assertEqual(errors, [])


class SettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = shared_root

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name) / "settings.json"
        self.errors = []
        self.host.report_callback_exception = lambda *error: self.errors.append(error)
        self.window = tk.Toplevel(self.host)

    def tearDown(self):
        self.app.worker = None
        self.app = None
        ui.set_animations(self.host, True)
        self.window.destroy()
        self.window = None
        self.host.update()
        self.temp.cleanup()
        self.assertEqual(self.errors, [])

    def open_app(self):
        self.app = StereoApp(self.window, Path(self.temp.name))
        self.host.update()
        return self.app

    def test_defaults_are_a_dark_theme_with_animations(self):
        app = self.open_app()
        self.assertEqual(app.vars["theme"].get(), "Тёмная")
        self.assertTrue(app.vars["animations"].get())
        self.assertIs(ui.palette(), ui.DARK)
        self.assertTrue(ui.animator(self.host).enabled)

    def test_old_settings_without_the_keys_open_dark(self):
        self.path.write_text(json.dumps(dict(settings_version=3, quality="Быстрая")), encoding="utf-8")
        app = self.open_app()
        self.assertEqual(app.vars["theme"].get(), "Тёмная")
        self.assertTrue(app.vars["animations"].get())
        self.assertEqual(app.vars["quality"].get(), "Быстрая")

    def test_an_unknown_saved_theme_falls_back_to_dark(self):
        self.path.write_text(json.dumps(dict(settings_version=3, theme="Розовая")), encoding="utf-8")
        app = self.open_app()
        self.assertEqual(app.vars["theme"].get(), "Тёмная")

    def test_changing_the_theme_is_saved_at_once_and_says_when_it_applies(self):
        app = self.open_app()
        self.assertFalse(self.path.exists())
        app.vars["theme"].set("Светлая")
        app.change_theme()
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["theme"], "Светлая")
        self.assertEqual(saved["settings_version"], 3)
        self.assertIn("следующем запуске", app.status.get())
        self.assertIs(ui.palette(), ui.DARK, "the running interface keeps its theme")

    def test_saving_a_preference_keeps_the_other_settings(self):
        self.path.write_text(json.dumps(dict(settings_version=3, quality="Быстрая", left="2")), encoding="utf-8")
        app = self.open_app()
        app.vars["animations"].set(False)
        app.change_animations()
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual((saved["quality"], saved["left"], saved["animations"]), ("Быстрая", "2", False))

    def test_a_first_saved_preference_does_not_trigger_the_old_settings_reset(self):
        app = self.open_app()
        app.vars["theme"].set("Светлая")
        app.change_theme()
        app.vars["quality"].set("Подробная")
        app.load_settings()
        self.assertEqual(app.vars["quality"].get(), "Подробная")
        self.assertFalse((Path(self.temp.name) / "settings_before_v03.json").exists())

    def test_the_animation_switch_acts_without_a_restart(self):
        app = self.open_app()
        engine = ui.animator(self.host)
        app.vars["animations"].set(False)
        app.change_animations()
        self.assertFalse(engine.enabled)
        self.assertFalse(json.loads(self.path.read_text(encoding="utf-8"))["animations"])
        app.vars["animations"].set(True)
        app.change_animations()
        self.assertTrue(engine.enabled)

    def test_a_saved_switched_off_state_applies_at_start(self):
        self.path.write_text(json.dumps(dict(settings_version=3, animations=False)), encoding="utf-8")
        self.open_app()
        self.assertFalse(ui.animator(self.host).enabled)

    def test_no_window_width_makes_the_settings_page_lay_out_for_ever(self):
        # 1100 and 1101 used to freeze: a watchdog ends the run instead of hanging it.
        app = self.open_app()
        app.show_page("advanced")
        self.window.geometry("1000x700+10+10")
        self.host.update()
        faulthandler.dump_traceback_later(120, exit=True)
        try:
            for width in range(1000, 1300, 3):
                self.window.geometry(f"{width}x700")
                self.host.update()
            for width in (1099, 1100, 1101, 1102):
                self.window.geometry(f"{width}x700")
                self.host.update()
        finally:
            faulthandler.cancel_dump_traceback_later()

    def test_real_data_form_fits_without_scrolling_and_keeps_its_attributes(self):
        app = self.open_app()
        self.window.geometry("1400x860+10+10")
        app.show_page("weighing")
        tab = app.weighing_tab
        tab.views.select(tab.database_page)
        self.host.update()
        bottom = tab.add_button.winfo_rooty() + tab.add_button.winfo_height()
        self.assertTrue(tab.add_button.winfo_viewable())
        self.assertLessEqual(bottom, tab.database_page.winfo_rooty() + tab.database_page.winfo_height())
        self.assertTrue(str(tab.source_box).startswith(str(tab.database_page)))
        for name in ("entry_source", "entry_label", "entry_height", "entry_weight", "entry_sex", "entry_clothing", "entry_scales"):
            self.assertIsInstance(getattr(tab, name), tk.Variable)
        self.assertTrue(callable(tab.add_to_database))
        columns = sum(tab.database_table.column(name, "width") for name in tab.database_table.cget("columns"))
        self.assertLessEqual(columns, tab.database_table.winfo_width(), "every column shows at this size")

    def test_the_weighing_tab_has_one_main_button_in_each_state(self):
        app = self.open_app()
        tab = app.weighing_tab
        self.assertFalse(tab.available)
        self.assertEqual(str(tab.download_button.cget("style")), "Accent.TButton")
        self.assertEqual(str(tab.add_button.cget("style")), "TButton")
        with patch("stereo.weighing_ui.models_ready", return_value=True):
            tab.show_download_state()
        self.assertEqual(str(tab.add_button.cget("style")), "Accent.TButton")
        self.assertFalse(tab.download_button.winfo_manager())

    def test_choosing_an_experiment_opens_the_form(self):
        app = self.open_app()
        tab = app.weighing_tab
        tab.history_items = [("weighing_20261106_100000", dict(people=[dict(label=1, mass_kg=70., samples=9)]))]
        tab.history_list.insert("", "end", iid="weighing_20261106_100000", values=("замер",))
        tab.history_list.selection_set("weighing_20261106_100000")
        with patch("stereo.weighing_ui.saved_experiments", return_value=tab.history_items):
            tab.use_experiment()
        self.assertEqual(tab.views.select(), str(tab.database_page))
        self.assertIn("06.11 10:00:00", tab.entry_source.get())

    def test_a_resize_repaints_only_the_widgets_that_changed(self):
        # A nested frame used to be repainted once per resized ancestor.
        self.app = SimpleNamespace(worker=None)
        self.window.geometry("600x400+10+10")
        ui.style_window(self.window)
        outer = ttk.Frame(self.window)
        outer.pack(fill="both", expand=True)
        parent = outer
        for _ in range(5):
            parent = ttk.Frame(parent)
            parent.pack(fill="both", expand=True)
        still = ttk.Label(parent, text="Подпись")
        still.place(x=4, y=4)
        self.host.update()
        ui.repaint_only_what_changed(self.window)
        for _ in range(3):
            self.host.update()
        exposed = []
        still.bind("<Expose>", lambda event: exposed.append(event))
        for width in range(600, 620, 2):
            self.window.geometry(f"{width}x400")
            self.host.update()
        self.assertEqual(exposed, [])


    def test_an_unwritable_settings_file_only_shows_a_message(self):
        app = self.open_app()
        self.path.mkdir()
        app.vars["theme"].set("Светлая")
        app.change_theme()
        self.assertIn("Не удалось", app.status.get())
        self.path.rmdir()

    def test_the_settings_page_offers_both_switches(self):
        app = self.open_app()
        app.show_page("advanced")
        self.host.update()

        def walk(widget):
            yield widget
            for child in widget.winfo_children():
                yield from walk(child)
        widgets = list(walk(app.workflow_pages["advanced"]))
        boxes = [w for w in widgets if isinstance(w, ttk.Combobox) and w.cget("textvariable") == str(app.vars["theme"])]
        checks = [w for w in widgets if isinstance(w, ttk.Checkbutton)]
        self.assertEqual(len(boxes), 1)
        self.assertEqual(tuple(boxes[0].cget("values")), tuple(ui.THEME_LABELS.values()))
        self.assertTrue(any(str(w.cget("variable")) == str(app.vars["animations"]) and
                            w.cget("text") == "Анимации интерфейса" for w in checks))
        self.assertTrue(all(widget is not boxes[0] for widget, _ in app.settings_controls),
                        "the theme can be changed while the cameras are connected")


LIGHT_START = """
import json, sys, tempfile, tkinter as tk
from pathlib import Path
from tkinter import ttk
sys.path.insert(0, sys.argv[1])
from stereo import ui
from stereo.app import StereoApp
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    Path(folder, "settings.json").write_text(json.dumps(dict(settings_version=3, theme="Светлая")), encoding="utf-8")
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *error: errors.append(repr(error))
    app = StereoApp(root, Path(folder))
    root.update()
    assert ui.palette() is ui.LIGHT
    assert app.vars["theme"].get() == "Светлая"
    assert ttk.Style(root).lookup("TFrame", "background") == ui.LIGHT.surface
    assert str(app.panels[0].canvas.cget("background")) == ui.LIGHT.image_bg
    assert not errors, errors
    app.close()
    app.updates.close()
    root.destroy()
print("ok")
"""


class LightThemeTests(unittest.TestCase):
    def test_a_saved_light_theme_builds_without_errors(self):
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-c", LIGHT_START, str(project)], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=120, stdin=subprocess.DEVNULL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


class ThemeCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = shared_root

    def test_every_style_the_weighing_tab_relies_on_is_defined(self):
        style = ttk.Style(self.root)
        for name in ("App.TFrame", "Card.TFrame", "TLabel", "Muted.TLabel", "Heading.TLabel", "Section.TLabel",
                     "Badge.TLabel", "Success.TLabel", "Warning.TLabel", "Metric.TLabel", "MetricCaption.TLabel",
                     "Accent.TButton", "Danger.TButton", "Quiet.TButton", "TEntry", "TCombobox", "TCheckbutton",
                     "Treeview", "Treeview.Heading", "TNotebook", "Inner.TNotebook", "TProgressbar", "Vertical.TScrollbar"):
            self.assertTrue(style.lookup(name, "background") or style.lookup(name, "foreground"), name)
        self.assertIn("20", str(style.lookup("Metric.TLabel", "font")))
        self.assertEqual(style.lookup("Warning.TLabel", "foreground"), ui.palette().warning)
        for name in ("Accent.TButton", "Danger.TButton", "Quiet.TButton"):
            ttk.Button(self.root, text=name, style=name).destroy()
        self.root.update()

    def test_the_clam_base_map_no_longer_repaints_controls_grey(self):
        style = ttk.Style(self.root)
        for state in (("active",), ("disabled",)):
            self.assertEqual(style.lookup("TCheckbutton", "background", state), ui.palette().surface)

    def test_wrapping_ignores_tiny_changes_that_could_repeat_for_ever(self):
        frame = ttk.Frame(self.root, width=400, height=40)
        label = ttk.Label(frame, text="Текст", wraplength=280)
        ui.fit_wrap(frame, label, margin=10)
        frame.pack()
        try:
            for width, expected in ((400, 390), (401, 390), (403, 390), (392, 382), (100, ui.MIN_WRAP)):
                frame.event_generate("<Configure>", width=width, height=40)
                self.assertEqual(int(str(label.cget("wraplength"))), expected, width)
        finally:
            frame.destroy()

    def test_a_small_progress_value_is_drawn_as_a_short_pill(self):
        bar = ttk.Progressbar(self.root, maximum=100, length=400)
        bar.pack()
        self.root.deiconify()
        self.root.update()
        try:
            width = bar.winfo_width()
            for value, expected in ((0, 0), (.1, 100 * ui.PROGRESS_MIN_PIXELS / width), (50, 50), (100, 100)):
                ui.show_progress(bar, value)
                self.assertAlmostEqual(float(bar.cget("value")), expected, places=3)
        finally:
            bar.destroy()
            self.root.withdraw()

    def test_compass_labels_never_cover_each_other(self):
        for ends in (((105, 300), (105, 302), (70, 262)),        # the opening view: Right and Forward
                     ((65, 300), (65, 300), (65, 300)),
                     ((90, 280), (92, 290), (100, 285))):
            places = compass_label_places(np.array(ends, float))
            for index, first in enumerate(places):
                for second in places[index + 1:]:
                    apart = (abs(first[0] - second[0]) >= COMPASS_LABEL_WIDTH or
                             abs(first[1] - second[1]) >= COMPASS_LABEL_HEIGHT)
                    self.assertTrue(apart, (ends, places))

    def test_a_label_with_room_stays_where_it_was(self):
        places = compass_label_places(np.array(((100, 300), (300, 100)), float))
        self.assertEqual(places, [(105., 291.), (305., 91.)])

    def test_a_dark_title_bar_call_is_safe_before_and_after_mapping(self):
        window = tk.Toplevel(self.root)
        try:
            ui.style_window(window)
            ui.title_bar(window)
            self.root.update()
            ui.title_bar(window)
            self.assertEqual(str(window.cget("background")), ui.palette().surface)
        finally:
            window.destroy()


if __name__ == "__main__":
    unittest.main()
