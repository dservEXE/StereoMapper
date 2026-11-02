import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from stereo.cloud_document import CloudDocument, load_cloud
from stereo.map_store import VoxelMap


class CloudDocumentTests(unittest.TestCase):
    def cloud(self, count=8, source=None, units="metres"):
        points = np.arange(count * 3, dtype=float).reshape(-1, 3) / 1000
        colors = (np.arange(count * 3).reshape(-1, 3) % 256).astype(np.uint8)
        return CloudDocument(points, colors, units, source)

    def test_empty_document_supports_async_load_placeholder(self):
        document = CloudDocument(np.empty((0, 3)), np.empty((0, 3), np.uint8))
        self.assertEqual(document.count, 0)
        self.assertEqual(document.selection_count, 0)
        self.assertEqual(document.arrays()[0].shape, (0, 3))
        self.assertEqual(document.arrays(include_ids=True)[2].shape, (0, 2))
        document.select(np.empty(0, bool))
        document.add_cloud([[0, 0, 1]], [[1, 2, 3]], "metres")
        self.assertEqual(document.count, 1)
        document.undo()
        self.assertEqual(document.count, 0)

    def test_full_resolution_edit_and_export_above_preview_budget(self):
        count = 250019
        document = self.cloud(count)
        # This point falls between display-sampled indices in a 250k preview.
        mask = np.zeros(count, bool)
        mask[250017] = True
        document.select(mask)
        document.delete_selected()
        expected_points, expected_colors = document.arrays()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "all_points.ply"
            self.assertEqual(document.export_ply(output), count - 1)
            points, colors, units, source = load_cloud(output)
            self.assertEqual(len(points), count - 1)
            np.testing.assert_array_equal(points, expected_points)
            np.testing.assert_array_equal(colors, expected_colors)
            self.assertEqual(units, "metres")
            self.assertEqual(source, str(output.resolve()))

    def test_selections_keep_stable_ids_and_undo_across_add_and_delete(self):
        document = self.cloud(4)
        original_points = document.layers[0].points
        document.select(np.array([False, True, False, True]))
        self.assertEqual(document.delete_selected(), 2)
        np.testing.assert_array_equal(document.arrays(include_ids=True)[2], [[0, 0], [0, 2]])
        index = document.add_cloud([[1, 2, 3]], [[11, 22, 33]], "m")
        self.assertEqual(index, 1)
        self.assertEqual(document.count, 3)
        document.undo()
        self.assertEqual(document.count, 2)
        self.assertEqual(len(document.layers), 1)
        document.undo()
        self.assertEqual(document.count, 4)
        self.assertEqual(document.selection_count, 2)
        document.redo()
        document.redo()
        self.assertEqual(document.count, 3)
        self.assertIs(document.layers[0].points, original_points)
        self.assertFalse(original_points.flags.writeable)

    def test_keep_add_subtract_invert_and_no_empty_accidental_crop(self):
        document = self.cloud(4)
        with self.assertRaises(ValueError):
            document.keep_selected()
        document.select(np.array([True, False, False, False]))
        document.select(np.array([False, True, False, False]), "add")
        document.select(np.array([True, False, False, False]), "subtract")
        np.testing.assert_array_equal(document.selection, [False, True, False, False])
        document.invert_selection()
        self.assertEqual(document.keep_selected(), 1)
        self.assertEqual(document.count, 3)
        self.assertEqual(document.selection_count, 0)
        np.testing.assert_array_equal(document.arrays(include_ids=True)[2][:, 1], [0, 2, 3])
        document.undo()
        self.assertEqual(document.count, 4)
        self.assertEqual(document.selection_count, 3)
        document.clear_selection()
        document.select(np.array([False, True, False, False]))
        document.delete_selected()
        self.assertFalse(document.can_redo)

    def test_scoped_delete_preserves_other_layer_points_selections_and_history(self):
        document = self.cloud(3)
        document.add_cloud(np.arange(12).reshape(4, 3), np.full((4, 3), 50), "metres")
        selected = np.array([True, False, True, False, True, False, True])
        scope = np.array([False, False, False, True, True, True, True])
        document.select(selected)
        before_points, before_colors, before_ids = document.arrays(include_ids=True)
        self.assertEqual(document.delete_selected(scope), 2)
        points, colors, ids = document.arrays(include_ids=True)
        np.testing.assert_array_equal(ids, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2]])
        np.testing.assert_array_equal(points[:3], before_points[:3])
        np.testing.assert_array_equal(colors[:3], before_colors[:3])
        np.testing.assert_array_equal(document.selection, [True, False, True, False, False])
        document.undo()
        np.testing.assert_array_equal(document.selection, selected)
        np.testing.assert_array_equal(document.arrays(include_ids=True)[2], before_ids)
        document.redo()
        np.testing.assert_array_equal(document.arrays(include_ids=True)[2], ids)
        self.assertEqual(document.selection_count, 2)

    def test_scoped_keep_crops_only_target_layer_with_stale_selection_elsewhere(self):
        document = self.cloud(3)
        document.add_cloud(np.arange(12).reshape(4, 3), np.full((4, 3), 50), "metres")
        selected = np.array([True, False, True, False, True, False, True])
        scope = np.array([False, False, False, True, True, True, True])
        document.select(selected)
        self.assertEqual(document.keep_selected(scope), 2)
        ids = document.arrays(include_ids=True)[2]
        np.testing.assert_array_equal(ids, [[0, 0], [0, 1], [0, 2], [1, 1], [1, 3]])
        np.testing.assert_array_equal(document.selection, [True, False, True, False, False])
        document.undo()
        np.testing.assert_array_equal(document.selection, selected)
        self.assertEqual(document.count, 7)
        document.redo()
        np.testing.assert_array_equal(document.arrays(include_ids=True)[2], ids)
        self.assertEqual(document.selection_count, 2)

    def test_scoped_keep_requires_selection_inside_scope_and_validates_masks(self):
        document = self.cloud(4)
        document.select(np.array([True, False, False, False]))
        scope = np.array([False, False, True, True])
        with self.assertRaises(ValueError):
            document.keep_selected(scope)
        self.assertEqual(document.count, 4)
        self.assertFalse(document.can_undo)
        self.assertEqual(document.delete_selected(scope), 0)
        self.assertFalse(document.can_undo)
        for invalid in (np.array([True, False]), np.array([0, 0, 1, 1])):
            for operation in (document.delete_selected, document.keep_selected):
                with self.subTest(operation=operation.__name__, mask=invalid):
                    with self.assertRaises(ValueError):
                        operation(invalid)

    def test_self_contained_project_reopens_masks_layers_units_and_transforms(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.ply"
            document = self.cloud(4, source)
            document.add_cloud([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], "metres")
            transform = np.array([[0, -1, 0, 7], [1, 0, 0, 8], [0, 0, 1, 9], [0, 0, 0, 1.]])
            document.apply_transform(1, transform)
            document.select(np.array([False, True, False, False, True, False]))
            document.delete_selected()
            document.select(np.array([True, False, False, True]))
            project = Path(folder) / "saved.cloudproj"
            document.save_project(project)
            # Reopening never depends on the existence of the original inputs.
            loaded = CloudDocument.load_project(project)
            self.assertEqual(len(loaded.layers), 2)
            self.assertEqual(loaded.total_count, 6)
            self.assertEqual(loaded.count, 4)
            self.assertEqual(loaded.selection_count, 2)
            self.assertEqual(loaded.layers[0].source, str(source.resolve()))
            np.testing.assert_array_equal(loaded.layers[1].transform, transform)
            for actual, expected in zip(loaded.arrays(include_ids=True), document.arrays(include_ids=True)):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(loaded.selection, document.selection)
            self.assertFalse(loaded.can_undo)
            loaded.save_project(project)
            flat_points, flat_colors, units, _ = load_cloud(project)
            np.testing.assert_array_equal(flat_points, document.arrays()[0])
            np.testing.assert_array_equal(flat_colors, document.arrays()[1])
            self.assertEqual(units, "metres")

    def test_unit_mismatch_rejected_and_relative_not_silently_scaled(self):
        document = self.cloud()
        for units in ("relative", "unknown", "millimetres"):
            with self.assertRaises(ValueError):
                document.add_cloud([[0, 0, 0]], [[0, 0, 0]], units)
        self.assertEqual(len(document.layers), 1)
        self.assertFalse(document.can_undo)
        relative = self.cloud(units="relative")
        relative.add_cloud([[0, 0, 0]], [[0, 0, 0]], "relative")
        self.assertEqual(len(relative.layers), 2)

    def test_original_source_preserved_and_both_save_types_are_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / "original.ply"
            original.write_text("source must survive", encoding="ascii")
            document = self.cloud(source=original)
            with self.assertRaises(ValueError):
                document.export_ply(original)
            with self.assertRaises(ValueError):
                document.save_project(original)
            self.assertEqual(original.read_text(encoding="ascii"), "source must survive")
            original_archive = Path(folder) / "original.sqlite"
            original_archive.write_bytes(b"original archive")
            document.add_cloud([[1, 2, 3]], [[4, 5, 6]], "metres", original_archive)
            with self.assertRaises(ValueError):
                document.save_project(original_archive)
            with self.assertRaises(ValueError):
                document.export_ply(original_archive)
            self.assertEqual(original_archive.read_bytes(), b"original archive")
            project = Path(folder) / "project.cloudproj"
            document.save_project(project)
            before = project.read_bytes()
            with self.assertRaises(ValueError):
                document.export_ply(project)
            self.assertEqual(project.read_bytes(), before)
            with patch("stereo.cloud_document.np.savez_compressed", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    document.save_project(project)
            self.assertEqual(project.read_bytes(), before)
            output = Path(folder) / "edited.ply"
            output.write_text("previous export", encoding="ascii")
            with patch("stereo.cloud_document.np.savetxt", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    document.export_ply(output)
            self.assertEqual(output.read_text(encoding="ascii"), "previous export")
            self.assertEqual(list(Path(folder).glob(".*.tmp")), [])

    def test_transform_is_absolute_rigid_undoable_and_exported_in_world_coordinates(self):
        document = self.cloud(2)
        local = document.layer_points(0, world=False)[0].copy()
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        document.apply_transform(0, transform)
        np.testing.assert_allclose(document.arrays()[0], local + [1, 2, 3])
        transform[0, 3] = 999
        self.assertEqual(document.layers[0].transform[0, 3], 1)
        document.undo()
        np.testing.assert_array_equal(document.arrays()[0], local)
        document.redo()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "world.ply"
            document.export_ply(path)
            np.testing.assert_array_equal(load_cloud(path)[0], document.arrays()[0])
        with self.assertRaises(ValueError):
            document.apply_transform(0, np.diag([2, 2, 2, 1]))

    def test_load_sqlite_reads_all_rows_without_modifying_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.sqlite"
            archive = VoxelMap(voxel=.001, path=path)
            points = np.column_stack((np.arange(101) * .01, np.zeros(101), np.ones(101)))
            colors = np.full((101, 3), 123, np.uint8)
            archive.add_points(points, colors)
            expected = archive.arrays()
            archive.close()
            before = path.read_bytes()
            actual_points, actual_colors, units, _ = load_cloud(path)
            self.assertEqual(len(actual_points), 101)
            self.assertEqual(actual_points.dtype, np.float64)
            np.testing.assert_array_equal(actual_points, expected[0])
            np.testing.assert_array_equal(actual_colors, expected[1])
            self.assertEqual(units, "metres")
            self.assertEqual(path.read_bytes(), before)

    def test_reject_bad_inputs_and_malformed_project_without_partial_document(self):
        with self.assertRaises(ValueError):
            CloudDocument([[np.nan, 0, 1]], [[0, 0, 0]])
        with self.assertRaises(ValueError):
            CloudDocument([[0, 0, 1]], [[0, 0, 999]])
        document = self.cloud(2)
        with self.assertRaises(ValueError):
            document.select(np.array([1, 0]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.cloudproj"
            with path.open("wb") as stream:
                np.savez(stream, metadata=np.asarray(json.dumps({"format": "stereo-cloud-project", "version": 99})))
            with self.assertRaises(ValueError):
                CloudDocument.load_project(path)
            path.write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 1\n", encoding="ascii")
            with self.assertRaises(ValueError):
                load_cloud(path)

    def test_damaged_projects_with_complex_coordinates_colors_or_transforms_are_rejected(self):
        document = self.cloud(2)
        with self.assertRaises(ValueError):
            CloudDocument([[0, 0, 1j]], [[1, 2, 3]])
        with self.assertRaises(ValueError):
            CloudDocument([[0, 0, 1]], [[1, 2, 3j]])
        with self.assertRaises(ValueError):
            document.apply_transform(0, np.eye(4, dtype=complex))
        self.assertFalse(document.can_undo)
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / "source.cloudproj"
            document.save_project(project)
            with np.load(project, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            for field in ("points", "colors", "transform"):
                with self.subTest(field=field):
                    damaged = dict(arrays)
                    damaged["layer_0_" + field] = arrays["layer_0_" + field].astype(complex) + 1j
                    path = Path(folder) / (field + ".cloudproj")
                    with path.open("wb") as stream:
                        np.savez(stream, **damaged)
                    with self.assertRaises(ValueError):
                        CloudDocument.load_project(path)


if __name__ == "__main__":
    unittest.main()
