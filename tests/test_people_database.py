"""The people database: validation, storage, statistics and export."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from stereo.body_measure import SLICES
from stereo.people_database import (CLOTHING, ENOUGH_PEOPLE, PeopleDatabase, SEXES, export_csv, make_entry, parse_number,
                                    safe_cell, summarize, summary_text)

MEASURED = dict(label=1, samples=30, height_m=1.54, mass_kg=53., mass_low_kg=45., mass_high_kg=60., volume_l=50.,
                widths_m=dict(torso=.26), side_widths_m=dict(torso=.2), lengths_m=dict(torso=.44), side_observed=True,
                views=dict(front=9), slices_front_m=[.3] * SLICES, slices_side_m=[.2] * SLICES)


def entry(height="158", weight="53", label="A", measured=None, sex=SEXES[1], scales=True):
    return make_entry(MEASURED if measured is None else measured, label, height, weight, sex, CLOTHING[0], scales, "dual", "0.16.0")


class EntryTests(unittest.TestCase):
    def test_decimal_comma_and_spaces_are_accepted(self):
        self.assertEqual(parse_number(" 53,5 ", "Вес", (3, 300)), 53.5)

    def test_unbelievable_or_broken_numbers_are_refused_with_a_clear_message(self):
        for text in ("", "abc", "0", "9000", "nan", "inf"):
            with self.assertRaisesRegex(ValueError, "Вес"):
                parse_number(text, "Вес", (3., 300.))

    def test_height_and_weight_mixed_up_are_caught_by_the_limits(self):
        with self.assertRaisesRegex(ValueError, "Вес"):
            entry(height="53", weight="580")

    def test_a_measurement_is_required(self):
        for measured in ({}, dict(samples=0)):
            with self.assertRaisesRegex(ValueError, "замер"):
                entry(measured=measured)

    def test_label_is_tidied_and_limited(self):
        self.assertEqual(entry(label="  A   01 ").label, "A 01")
        with self.assertRaisesRegex(ValueError, "Метка"):
            entry(label="x" * 41)

    def test_unknown_choices_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Пол"):
            entry(sex="другое значение")

    def test_error_compares_camera_with_reality(self):
        record = entry(height="158", weight="50")
        self.assertAlmostEqual(record.weight_error_kg, 3.)
        self.assertAlmostEqual(record.height_error_cm, -4.)

    def test_missing_estimates_mean_no_error_rather_than_a_crash(self):
        record = entry(measured=dict(samples=5))
        self.assertIsNone(record.weight_error_kg)
        self.assertIsNone(record.height_error_cm)


class StorageTests(unittest.TestCase):
    def test_records_survive_a_new_connection_newest_first(self):
        with TemporaryDirectory() as temp:
            database = PeopleDatabase(temp)
            first = database.add(entry(label="A"))
            second = database.add(entry(label="B"))
            self.assertGreater(second, first)
            records = PeopleDatabase(temp).entries()
        self.assertEqual([record.label for record in records], ["B", "A"])
        self.assertEqual(records[0].measurement["mass_kg"], 53.)
        self.assertTrue(records[0].weight_from_scales)

    def test_no_file_means_no_records_and_is_not_created_by_reading(self):
        with TemporaryDirectory() as temp:
            database = PeopleDatabase(temp)
            self.assertEqual(database.entries(), [])
            self.assertFalse(database.path.exists())

    def test_it_lives_in_user_data_which_updates_keep(self):
        self.assertEqual(PeopleDatabase("root").path.parts[:2], ("root", "data"))

    def test_delete_removes_only_that_record(self):
        with TemporaryDirectory() as temp:
            database = PeopleDatabase(temp)
            keep = database.add(entry(label="keep"))
            database.delete(database.add(entry(label="drop")))
            self.assertEqual([record.id for record in database.entries()], [keep])

    def test_label_cannot_break_the_database(self):
        with TemporaryDirectory() as temp:
            database = PeopleDatabase(temp)
            database.add(entry(label="x'); DROP TABLE people;--"))
            self.assertEqual(len(database.entries()), 1)

    def test_nan_in_a_measurement_is_stored_as_unknown(self):
        with TemporaryDirectory() as temp:
            database = PeopleDatabase(temp)
            database.add(entry(measured=dict(MEASURED, slices_front_m=[float("nan")] * 3, mass_kg=float("nan"))))
            record = database.entries()[0]
        self.assertEqual(record.measurement["slices_front_m"], [None] * 3)
        self.assertIsNone(record.estimated_weight_kg)

    def test_damaged_measurement_does_not_lose_the_real_data(self):
        with TemporaryDirectory() as temp:
            database = PeopleDatabase(temp)
            database.add(entry())
            connection = database.connect()
            with connection:
                connection.execute("UPDATE people SET measurement = 'not json'")
            connection.close()
            record = database.entries()[0]
        self.assertEqual(record.real_weight_kg, 53.)
        self.assertEqual(record.measurement, {})


class SummaryTests(unittest.TestCase):
    def test_errors_are_averaged_over_records_with_an_estimate(self):
        records = [entry(weight="50", label="A"), entry(weight="56", label="B"), entry(measured=dict(samples=5), label="C")]
        summary = summarize(records)
        self.assertEqual((summary.records, summary.people, summary.compared), (3, 3, 2))
        self.assertAlmostEqual(summary.mean_absolute_error_kg, 3.)
        self.assertAlmostEqual(summary.mean_error_kg, 0.)

    def test_same_label_is_one_person_and_no_label_counts_separately(self):
        summary = summarize([entry(label="A"), entry(label="A"), entry(label=""), entry(label="")])
        self.assertEqual(summary.people, 3)

    def test_empty_database_explains_what_to_do(self):
        self.assertIn("пуста", summary_text(summarize([])))

    def test_few_people_are_flagged_as_not_enough(self):
        self.assertIn("мало", summary_text(summarize([entry()])))
        records = [entry(label=str(index)) for index in range(ENOUGH_PEOPLE)]
        self.assertNotIn("мало", summary_text(summarize(records)))


class ExportTests(unittest.TestCase):
    def test_every_measured_number_is_written_for_later_fitting(self):
        with TemporaryDirectory() as temp:
            database = PeopleDatabase(temp)
            database.add(entry(weight="50"))
            path = Path(temp) / "deep" / "people.csv"
            self.assertEqual(export_csv(database.entries(), path), 1)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter=";"))
        self.assertEqual(rows[0]["real_weight_kg"], "50.0")
        self.assertEqual(rows[0]["weight_error_kg"], "3.0")
        self.assertEqual(rows[0]["torso_front_width_m"], "0.26")
        self.assertEqual(rows[0][f"slice_{SLICES - 1:02d}_side_m"], "0.2")

    def test_label_cannot_run_as_a_spreadsheet_formula(self):
        self.assertEqual(safe_cell("=1+1"), "'=1+1")
        self.assertEqual(safe_cell("A-01"), "A-01")

    def test_old_records_with_fewer_numbers_still_export(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "people.csv"
            export_csv([entry(measured=dict(samples=3))], path)
            self.assertIn("real_weight_kg", path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
