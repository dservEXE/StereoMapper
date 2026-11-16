"""Pig weighing: outline geometry, the tape formula, numbering, the session thread and the tab."""
import json
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stereo.app import StereoApp
from stereo.people_detection import SEGMENTER, WEIGHTS_FILE, model_path
from stereo.pig_detection import DETECTOR, keep_distinct, missing_size_mb
from stereo.pig_measure import (BODY_LEVEL_TO_LENGTH, PigAccumulator, PigShape, body_scale, pig_shape, plausible,
                                tape_mass)
from stereo.pig_ui import draw_pigs, parse_number, pig_row, segment_scale
from stereo.pig_weighing import (BAD_POSE, CUT, MEMORY_S, PigSession, PigTracker, fitted, read_image, reading_order,
                                 write_result)

SIZE = 700


def pig_mask(angle=0., tail=True, ears=True, centre=(350, 350), scale=1.) -> np.ndarray:
    """A pig seen from above: trunk 300 x 90, a narrower head on the left, a neck between them."""
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(mask, (350, 350), (150, 45), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (230, 305), (470, 395), 255, -1)
    cv2.ellipse(mask, (150, 350), (55, 30), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (180, 322), (215, 378), 255, -1)
    if ears:
        cv2.line(mask, (165, 330), (150, 290), 255, 7)
        cv2.line(mask, (165, 370), (150, 410), 255, 7)
    if tail:
        cv2.line(mask, (500, 350), (560, 340), 255, 5)
    matrix = cv2.getRotationMatrix2D((350, 350), angle, scale)
    matrix[:, 2] += (centre[0] - 350, centre[1] - 350)
    return cv2.warpAffine(mask, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST) > 0


def bent_mask() -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(mask, (350, 500), (220, 220), 0, 200, 340, 255, 80)
    return mask > 0


def shape(length=280., girth=90., straightness=1.) -> PigShape:
    return PigShape(length, length * 1.4, girth, straightness, (0., 0.), (1., 0.), (2., 0.))


class ShapeTests(unittest.TestCase):
    def test_length_and_thickness_do_not_depend_on_how_the_pig_is_turned(self):
        for angle in (0, 30, 90, 137, 200):
            found = pig_shape(pig_mask(angle))
            self.assertAlmostEqual(found.length_px, 280, delta=8, msg=angle)
            self.assertAlmostEqual(found.girth_px, 90, delta=4, msg=angle)
            self.assertGreater(found.straightness, .98)
            self.assertTrue(plausible(found))

    def test_head_is_the_narrow_end_and_the_neck_lies_between_head_and_shoulders(self):
        found = pig_shape(pig_mask())
        self.assertLess(found.head[0], found.neck[0])
        self.assertLess(found.neck[0], found.rump[0])
        self.assertTrue(180 <= found.neck[0] <= 235, found.neck)
        turned = pig_shape(pig_mask(180))
        self.assertGreater(turned.head[0], turned.rump[0])

    def test_tail_and_ears_do_not_change_the_measurement(self):
        full, bare = pig_shape(pig_mask()), pig_shape(pig_mask(tail=False, ears=False))
        self.assertAlmostEqual(full.length_px, bare.length_px, delta=4)
        self.assertAlmostEqual(full.girth_px, bare.girth_px, delta=2)
        self.assertLess(full.rump[0], 510)

    def test_a_turning_pig_is_measured_along_the_spine_but_not_trusted(self):
        found = pig_shape(bent_mask())
        self.assertIsNotNone(found)
        self.assertLess(found.straightness, .96)
        self.assertFalse(plausible(found))

    def test_specks_and_empty_outlines_give_nothing(self):
        self.assertIsNone(pig_shape(np.zeros((100, 100), bool)))
        speck = np.zeros((100, 100), bool)
        speck[10:15, 10:15] = True
        self.assertIsNone(pig_shape(speck))

    def test_two_merged_animals_are_not_taken_for_one(self):
        self.assertFalse(plausible(shape(length=100., girth=90.)))
        self.assertFalse(plausible(shape(length=600., girth=90.)))


class WeightTests(unittest.TestCase):
    def test_tape_formula_matches_the_farm_table(self):
        # Heart girth 1.12 m and length 1.10 m is a pig of about 96 kg.
        width = .32
        girth, mass = tape_mass(1.10, width, "top")
        self.assertAlmostEqual(girth, 1.11, delta=.03)
        self.assertAlmostEqual(mass, 69.3 * girth * girth * 1.10, places=6)
        self.assertTrue(85 < mass < 105, mass)

    def test_top_and_side_views_of_the_same_body_agree(self):
        top = tape_mass(1., .30, "top")[1]
        side = tape_mass(1., .30 * 1.2, "side")[1]
        self.assertAlmostEqual(top, side, places=6)

    def test_floor_scale_is_reduced_for_a_body_closer_to_the_camera(self):
        self.assertEqual(body_scale(.002, None, 500), .002)
        corrected = body_scale(.002, 2.5, 500)
        self.assertAlmostEqual(corrected, .002 / (1 + .002 * BODY_LEVEL_TO_LENGTH * 500 / 2.5))
        # The same answer the other way round: the body level follows from the corrected length.
        level = BODY_LEVEL_TO_LENGTH * 500 * corrected
        self.assertAlmostEqual(corrected, .002 * (2.5 - level) / 2.5)

    def test_estimate_needs_a_scale_and_a_usable_frame(self):
        accumulator = PigAccumulator()
        self.assertIsNone(accumulator.estimate(.004))
        self.assertFalse(accumulator.add(shape(straightness=.8)))
        self.assertTrue(accumulator.add(shape()))
        self.assertIsNone(accumulator.estimate(None))
        estimate = accumulator.estimate(.004)
        self.assertAlmostEqual(estimate.length_m, 1.12)
        self.assertAlmostEqual(estimate.thickness_m, .36)
        self.assertLess(estimate.mass_low_kg, estimate.mass_kg)
        self.assertGreater(estimate.mass_high_kg, estimate.mass_kg)

    def test_camera_height_lowers_the_weight_seen_from_above_only(self):
        accumulator = PigAccumulator()
        accumulator.add(shape())
        plain = accumulator.estimate(.004).mass_kg
        self.assertLess(accumulator.estimate(.004, 2.5).mass_kg, plain * .7)
        self.assertEqual(accumulator.estimate(.004, 2.5, "side").mass_kg, accumulator.estimate(.004, None, "side").mass_kg)

    def test_one_odd_frame_does_not_move_the_result(self):
        accumulator = PigAccumulator()
        for _ in range(9):
            accumulator.add(shape())
        steady = accumulator.estimate(.004).mass_kg
        accumulator.add(shape(length=380., girth=120.))
        self.assertAlmostEqual(accumulator.estimate(.004).mass_kg, steady, delta=steady * .01)


class DetectionHelperTests(unittest.TestCase):
    def test_one_animal_found_under_two_names_is_kept_once(self):
        boxes = np.array([[0, 0, 100, 50], [2, 1, 101, 52], [200, 0, 300, 50]], float)
        self.assertEqual(keep_distinct(boxes, [.5, .9, .7]), [1, 2])

    def test_download_size_counts_only_what_is_missing(self):
        with TemporaryDirectory() as temp:
            everything = missing_size_mb(temp)
            self.assertEqual(everything, round((DETECTOR.weights_size + SEGMENTER.weights_size) / 1e6))
            folder = model_path(temp, SEGMENTER)
            folder.mkdir(parents=True)
            with (folder / WEIGHTS_FILE).open("wb") as output:
                output.truncate(SEGMENTER.weights_size)
            self.assertEqual(missing_size_mb(temp), round(DETECTOR.weights_size / 1e6))


class TrackerTests(unittest.TestCase):
    def test_numbers_stay_with_animals_that_move_a_little_and_swap_order(self):
        tracker = PigTracker()
        first = tracker.update(np.array([[0, 0, 100, 50], [300, 0, 400, 50]], float), 0.)
        self.assertEqual([track.track_id for track in first], [1, 2])
        second = tracker.update(np.array([[310, 5, 410, 55], [8, 2, 108, 52]], float), .3)
        self.assertEqual([track.track_id for track in second], [2, 1])

    def test_a_newcomer_gets_a_new_number_and_a_long_gone_number_is_not_reused(self):
        tracker = PigTracker()
        tracker.update(np.array([[0, 0, 100, 50]], float), 0.)
        self.assertEqual(tracker.update(np.array([[0, 0, 100, 50], [500, 300, 600, 350]], float), .2)[1].track_id, 2)
        late = tracker.update(np.array([[0, 0, 100, 50]], float), MEMORY_S + 1)
        self.assertEqual(late[0].track_id, 3)

    def test_photograph_numbers_read_left_to_right_in_rows(self):
        boxes = np.array([[500, 10, 600, 60], [10, 400, 110, 450], [10, 20, 110, 70]], float)
        self.assertEqual(list(reading_order(boxes, 600)), [2, 0, 1])
        self.assertEqual(len(reading_order(np.zeros((0, 4)), 600)), 0)


class FakeDetector:
    """Finds white animals on a black picture, so the session can be tested without neural networks."""
    device_label = "проверка"

    def detect(self, image) -> tuple:
        count, _, stats, _ = cv2.connectedComponentsWithStats((image[:, :, 0] > 127).astype(np.uint8))
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h, area in stats[1:count] if area > 500], float).reshape(-1, 4)
        return boxes, np.ones(len(boxes))

    def outlines(self, image, boxes) -> list:
        white = image[:, :, 0] > 127
        masks = []
        for x0, y0, x1, y1 in boxes.astype(int):
            mask = np.zeros(white.shape, bool)
            mask[y0:y1, x0:x1] = white[y0:y1, x0:x1]
            masks.append(mask)
        return masks


def picture(*masks) -> np.ndarray:
    image = np.zeros((SIZE, SIZE, 3), np.uint8)
    for mask in masks:
        image[mask] = 255
    return image


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.session = PigSession(lambda status: FakeDetector())
        self.session.start()
        self.session.set_active(True)
        self.wait(lambda state: state.ready)

    def tearDown(self):
        self.session.stop()
        self.session.join(timeout=5)
        self.temp.cleanup()

    def wait(self, condition, timeout=10.):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if condition(self.session.state):
                return self.session.state
            time.sleep(.01)
        self.fail(f"session state did not arrive: {self.session.state.message}")

    def photo(self, image, name="свиньи.png") -> Path:
        path = Path(self.temp.name) / name
        cv2.imencode(".png", image)[1].tofile(str(path))
        return path

    def video(self, frames, name="pigs.avi") -> Path:
        path = Path(self.temp.name) / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20., (SIZE, SIZE))
        for frame in frames:
            writer.write(frame)
        writer.release()
        return path

    def test_photo_is_measured_once_the_scale_is_known(self):
        small = pig_mask(angle=90, centre=(560, 350), scale=.5)
        self.session.post("photo", self.photo(picture(pig_mask(centre=(300, 350), scale=.8), small)))
        state = self.wait(lambda state: len(state.pigs) == 2)
        self.assertEqual([pig.track_id for pig in state.pigs], [1, 2])
        self.assertTrue(all(pig.estimate is None for pig in state.pigs))
        self.assertIn("отрезок", state.message)
        self.session.post("scale", .004, None, "top")
        state = self.wait(lambda state: all(pig.estimate is not None for pig in state.pigs))
        big, little = (pig.estimate for pig in state.pigs)
        self.assertAlmostEqual(big.length_m, 280 * .8 * .004, delta=.04)
        self.assertGreater(big.mass_kg, little.mass_kg * 3)
        self.assertIn("измерено: 2", state.message)

    def test_animal_cut_by_the_frame_edge_is_numbered_but_not_measured(self):
        self.session.post("scale", .004, None, "top")
        self.session.post("photo", self.photo(picture(pig_mask(centre=(110, 120)), bent_mask())))
        state = self.wait(lambda state: len(state.pigs) == 2)
        notes = {pig.note for pig in state.pigs}
        self.assertEqual(notes, {CUT, BAD_POSE})
        self.assertTrue(all(pig.estimate is None for pig in state.pigs))

    def test_unreadable_photo_is_reported_and_the_session_goes_on(self):
        broken = Path(self.temp.name) / "broken.jpg"
        broken.write_bytes(b"not a picture")
        self.session.post("photo", broken)
        self.wait(lambda state: "Не удалось открыть фото" in state.message)
        self.session.post("photo", self.photo(picture(pig_mask())))
        self.wait(lambda state: len(state.pigs) == 1)

    def test_video_plays_to_the_end_keeps_numbers_and_gathers_frames(self):
        frames = [picture(pig_mask(centre=(250 + 4 * step, 350), scale=.8)) for step in range(30)]
        self.session.post("scale", .004, None, "top")
        self.session.post("video", self.video(frames))
        state = self.wait(lambda state: state.finished)
        self.assertEqual(state.kind, "video")
        self.assertFalse(state.playing)
        self.assertEqual([pig.track_id for pig in state.pigs], [1])
        self.assertGreater(state.pigs[0].estimate.samples, 2)
        self.session.post("play", True)
        self.wait(lambda state: state.playing and not state.finished)
        self.wait(lambda state: state.finished)

    def test_nothing_is_analysed_while_the_tab_is_closed(self):
        frames = [picture(pig_mask(scale=.8))] * 40
        self.session.set_active(False)
        self.session.post("video", self.video(frames))
        self.wait(lambda state: state.kind == "video")
        time.sleep(.5)
        self.assertIsNone(self.session.state.image)
        self.session.set_active(True)
        self.wait(lambda state: state.image is not None)

    def test_pause_stops_the_video_and_reset_gives_numbers_again(self):
        frames = [picture(pig_mask(scale=.8))] * 200
        self.session.post("video", self.video(frames))
        self.wait(lambda state: state.image is not None)
        self.session.post("play", False)
        paused = self.wait(lambda state: not state.playing)
        time.sleep(.4)
        self.assertLess(self.session.state.position_s - paused.position_s, .2)
        self.assertFalse(self.session.state.finished)
        self.session.post("photo", self.photo(picture(pig_mask())))
        self.wait(lambda state: state.kind == "photo" and len(state.pigs) == 1)
        before = self.session.state.serial
        self.session.post("reset")
        state = self.wait(lambda state: state.serial > before and len(state.pigs) == 1)
        self.assertEqual(state.pigs[0].track_id, 1)

    def test_saved_result_holds_numbers_only(self):
        self.session.post("scale", .004, 2.5, "top")
        self.session.post("photo", self.photo(picture(pig_mask())))
        state = self.wait(lambda state: any(pig.estimate for pig in state.pigs))
        target = Path(self.temp.name) / "output" / "pigs_test" / "result.json"
        self.assertEqual(write_result(target, state, dict(view="top")), 1)
        saved = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(saved["view"], "top")
        self.assertEqual(set(saved["pigs"][0]), {"label", "mass_kg", "mass_low_kg", "mass_high_kg", "length_m",
                                                  "thickness_m", "girth_m", "samples"})
        self.assertEqual([path.name for path in target.parent.iterdir()], ["result.json"])


class FileTests(unittest.TestCase):
    def test_photo_with_a_cyrillic_name_opens_and_large_ones_are_reduced(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "свинья сверху.png"
            cv2.imencode(".png", np.zeros((300, 2600, 3), np.uint8))[1].tofile(str(path))
            image = read_image(path)
            self.assertEqual(image.shape[:2], (300, 2600))
            self.assertEqual(fitted(image).shape[:2], (148, 1280))
            small = image[:, :800]
            self.assertIs(fitted(small), small)
            self.assertIsNone(read_image(Path(temp) / "нет такого.png"))


class TabTextTests(unittest.TestCase):
    def test_numbers_are_read_with_a_comma_or_a_point(self):
        self.assertEqual(parse_number(" 2,5 "), 2.5)
        self.assertEqual(parse_number("120"), 120.)
        for text in ("", "abc", "0", "-3"):
            self.assertIsNone(parse_number(text))

    def test_scale_comes_from_the_marked_segment_and_its_real_length(self):
        self.assertAlmostEqual(segment_scale(((0, 0), (300, 400)), 100.), .002)
        self.assertIsNone(segment_scale(None, 100.))
        self.assertIsNone(segment_scale(((0, 0), (300, 400)), None))
        self.assertIsNone(segment_scale(((5, 5), (5, 5)), 100.))

    def test_table_line_gives_the_weight_or_says_why_there_is_none(self):
        accumulator = PigAccumulator()
        accumulator.add(shape())
        estimate = accumulator.estimate(.004)
        row = pig_row(SimpleNamespace(track_id=3, estimate=estimate, note=""))
        self.assertEqual(row[0], 3)
        self.assertEqual(row[1], f"{estimate.mass_kg:.0f}")
        self.assertEqual(row[3:6], ("112", "36", f"{estimate.girth_m * 100:.0f}"))
        self.assertEqual(row[6:], (1, ""))
        self.assertIn(BAD_POSE, pig_row(SimpleNamespace(track_id=3, estimate=estimate, note=BAD_POSE))[7])
        skipped = SimpleNamespace(track_id=4, estimate=None, note=CUT)
        self.assertEqual(pig_row(skipped), (4, "—", "—", "—", "—", "—", 0, CUT))
        waiting = SimpleNamespace(track_id=5, estimate=None, note="")
        self.assertEqual(pig_row(waiting, has_scale=False)[7], "масштаб не задан")

    def test_drawing_marks_animals_and_the_segment_without_touching_the_frame(self):
        mask = pig_mask()
        image = picture(mask)
        pig = SimpleNamespace(track_id=1, box=(95., 290., 505., 410.), color=(80, 200, 255), outline=mask,
                              shape=pig_shape(mask), estimate=None, note="")
        state = SimpleNamespace(image=image, pigs=(pig,))
        before = image.copy()
        canvas = draw_pigs(state, ((20, 20), (120, 20)))
        self.assertTrue(np.array_equal(image, before))
        self.assertFalse(np.array_equal(canvas, image))
        self.assertEqual(tuple(canvas[20, 70]), (60, 220, 255))


class TabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = tk.Tk()
        cls.host.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.host.destroy()

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.errors = []
        self.previous = self.host.report_callback_exception
        self.host.report_callback_exception = lambda *error: self.errors.append(error)
        self.window = tk.Toplevel(self.host)
        self.app = StereoApp(self.window, Path(self.temp.name))
        self.window.geometry("1100x700+10+10")
        self.host.update()
        self.tab = self.app.pig_tab

    def tearDown(self):
        self.tab.close()
        self.app.worker = None
        self.window.destroy()
        self.host.update_idletasks()
        self.host.report_callback_exception = self.previous
        self.temp.cleanup()
        self.assertFalse(self.errors, self.errors)

    def pump(self, condition, timeout=10.):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.host.update()
            self.tab.poll()
            if condition():
                return
            time.sleep(.01)
        self.fail(f"the tab did not get there: {self.tab.message.get()}")

    def connect(self) -> Queue:
        worker = SimpleNamespace(engine=None, frame=None, scan=None, commands=Queue(), events=Queue(), frames=Queue(),
                                 previews=Queue(), stop_event=Event(), is_alive=lambda: True, max_depth=1.2,
                                 near_minimum=.1, ghost_filter="normal", raw=None)
        self.app.worker, self.app.connected = worker, True
        self.app.sync_ui(force=True)
        return worker.commands

    def sent(self, commands) -> list:
        found = []
        while not commands.empty():
            found.append(commands.get_nowait())
        return found

    def open_with_models(self):
        patches = (patch("stereo.pig_ui.models_ready", return_value=True),
                   patch("stereo.pig_detection.PigDetector", lambda root, status: FakeDetector()))
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.app.show_page("pigs")
        self.pump(lambda: self.tab.session is not None and self.tab.session.state.ready)

    def test_tab_comes_after_weighing_and_asks_for_models_before_anything_else(self):
        self.assertEqual(list(self.app.workflow_pages), ["cameras", "calibration", "scan", "weighing", "pigs", "advanced"])
        self.app.show_page("pigs")
        self.host.update()
        self.tab.poll()
        self.assertIsNone(self.tab.session)
        self.assertTrue(self.tab.download_button.winfo_ismapped())
        self.assertIn("МБ", self.tab.download_button.cget("text"))
        self.tab.reset()
        self.tab.toggle_play()
        self.tab.start_marking()
        self.assertIn("Сначала откройте", self.tab.message.get())

    def test_camera_work_is_paused_only_while_the_tab_is_open(self):
        commands = self.connect()
        self.app.show_page("pigs")
        self.host.update()
        self.tab.poll()
        self.assertIn(("suspend", True), self.sent(commands))
        self.tab.poll()
        self.assertEqual(self.sent(commands), [])
        self.app.show_page("scan")
        self.host.update()
        self.tab.poll()
        self.assertIn(("suspend", False), self.sent(commands))

    def test_cameras_connected_while_the_tab_is_open_are_paused_too(self):
        self.app.show_page("pigs")
        self.host.update()
        self.tab.poll()
        commands = self.connect()
        self.tab.poll()
        self.assertIn(("suspend", True), self.sent(commands))

    def test_photo_is_weighed_after_marking_a_segment_and_typing_its_length(self):
        self.open_with_models()
        path = Path(self.temp.name) / "фото.png"
        cv2.imencode(".png", picture(pig_mask(scale=.8)))[1].tofile(str(path))
        self.tab.session.post("photo", path)
        self.pump(lambda: self.tab.state is not None and len(self.tab.state.pigs) == 1)
        self.assertEqual(self.tab.metric_count.get(), "1")
        self.assertEqual(self.tab.metric_mean.get(), "—")
        self.assertIn("не задан", self.tab.scale_text.get())
        self.assertEqual(self.tab.table.item(self.tab.table.get_children()[0], "values")[7], "масштаб не задан")
        self.tab.start_marking()
        self.tab.clicked(100, 600)
        self.tab.clicked(350, 600)
        self.assertIn("Введите его длину", self.tab.message.get())
        self.tab.length.set("100")
        self.pump(lambda: self.tab.metric_mean.get() != "—")
        self.assertIn("1 пиксель = 4.0 мм", self.tab.scale_text.get())
        self.assertIn("Высота камеры не указана", self.tab.scale_text.get())
        row = self.tab.table.item(self.tab.table.get_children()[0], "values")
        self.assertEqual(str(row[0]), "1")
        self.assertNotEqual(row[1], "—")
        heavy = self.tab.state.pigs[0].estimate.mass_kg
        self.tab.camera_height.set("250")
        self.pump(lambda: self.tab.state.pigs[0].estimate.mass_kg < heavy * .8)
        self.tab.view.set("Сбоку")
        self.assertEqual(str(self.tab.height_entry.cget("state")), "disabled")
        self.assertIn("Высота туловища", self.tab.table.heading("thickness", "text"))
        light = self.tab.state.pigs[0].estimate.mass_kg
        self.pump(lambda: self.tab.state.pigs[0].estimate.mass_kg != light)
        self.tab.save()
        self.assertIn("Сохранено свиней: 1", self.tab.message.get())
        saved = list((Path(self.temp.name) / "output").glob("pigs_*/result.json"))
        self.assertEqual(json.loads(saved[0].read_text(encoding="utf-8"))["view"], "side")

    def test_leaving_the_tab_stops_the_video_and_coming_back_continues_it(self):
        self.open_with_models()
        path = Path(self.temp.name) / "pigs.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20., (SIZE, SIZE))
        for _ in range(400):
            writer.write(picture(pig_mask(scale=.8)))
        writer.release()
        self.tab.session.post("video", path)
        self.pump(lambda: self.tab.state is not None and self.tab.state.image is not None)
        self.assertEqual(str(self.tab.play_button.cget("state")), "normal")
        self.app.show_page("cameras")
        self.pump(lambda: not self.tab.active)
        time.sleep(.3)
        stopped = self.tab.session.state.position_s
        time.sleep(.5)
        self.assertEqual(self.tab.session.state.position_s, stopped)
        self.app.show_page("pigs")
        self.pump(lambda: self.tab.session.state.position_s > stopped)
        self.assertLess(self.tab.session.state.position_s, stopped + 1.5)

    def test_new_picture_forgets_the_old_scale(self):
        self.open_with_models()
        self.tab.segment = ((0, 0), (100, 0))
        self.tab.length.set("50")
        self.assertIsNotNone(self.tab.scale())
        self.tab.forget_segment()
        self.assertIsNone(self.tab.scale())
        self.assertIn("не задан", self.tab.scale_text.get())


if __name__ == "__main__":
    unittest.main()
