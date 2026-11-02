from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from stereo.app import StereoApp
from stereo.worker import CameraWorker


class CloudEntrypointTests(unittest.TestCase):
    def test_session_library_opens_exact_model_without_a_second_file_dialog(self):
        app=SimpleNamespace(root=object(),directory=Path('.'),vars={'preview_points':SimpleNamespace(get=lambda:80000)})
        path=Path('scan/measured.sqlite')
        with patch('stereo.app.filedialog.askopenfilename') as dialog,patch('stereo.app.CloudViewer') as viewer:
            StereoApp.open_model(app,path)
            dialog.assert_not_called(); self.assertEqual(viewer.call_args.kwargs['source_path'],path)

    def test_open_saved_cloud_defers_full_source_load_to_viewer(self):
        app = SimpleNamespace(root=object(), directory=Path("."),
                              vars={"preview_points": SimpleNamespace(get=lambda: 80000)})
        for extension in (".ply", ".sqlite", ".cloudproj"):
            path = str(Path("nonexistent-source" + extension).resolve())
            with self.subTest(extension=extension), \
                    patch("stereo.app.filedialog.askopenfilename", return_value=path) as dialog, \
                    patch("stereo.app.CloudViewer") as viewer, \
                    patch("stereo.app.messagebox.showerror") as error:
                StereoApp.open_model(app)
                error.assert_not_called()
                args, kwargs = viewer.call_args
                self.assertIs(args[0], app.root)
                self.assertEqual(args[1].shape, (0, 3))
                self.assertEqual(args[2].shape, (0, 3))
                self.assertEqual(args[1].dtype, np.float64)
                self.assertEqual(args[2].dtype, np.uint8)
                self.assertEqual(kwargs["source_path"], path)
                self.assertEqual(kwargs["max_points"], 80000)
                filters = " ".join(pattern for _, pattern in dialog.call_args.kwargs["filetypes"])
                for pattern in ("*.ply", "*.sqlite", "*.cloudproj"):
                    self.assertIn(pattern, filters)

    def test_cancel_open_cloud_does_not_create_viewer(self):
        with patch("stereo.app.filedialog.askopenfilename", return_value=""), \
                patch("stereo.app.CloudViewer") as viewer:
            StereoApp.open_model(SimpleNamespace(directory=Path(".")))
            viewer.assert_not_called()

    def test_live_composite_and_memory_clouds_keep_all_editable_points(self):
        class StoredCloud:
            def __init__(self, count, path=None):
                self.path, self.calls = path, []
                self.points = np.arange(count * 3, dtype=float).reshape(-1, 3)

            def __len__(self):
                return len(self.points)

            def arrays(self, max_points=None):
                self.calls.append(max_points)
                points = self.points if max_points is None else self.points[:max_points]
                return points, np.full(points.shape, 123, np.uint8)

        events = []
        worker = SimpleNamespace(calibration_busy=False, scan=None, mapping=False, frame=None,
                                 cloud=StoredCloud(23), estimated_cloud=StoredCloud(19),
                                 preview_points=5, emit=lambda kind, value: events.append((kind, value)))
        CameraWorker.command(worker, "view", False)
        self.assertEqual(len(events[-1][1]["points"]), 23)
        self.assertIsNone(events[-1][1]["archive"])
        self.assertIsNone(worker.cloud.calls[-1])
        worker.cloud.path = Path("map.sqlite")
        CameraWorker.command(worker, "view", True)
        self.assertEqual(len(events[-1][1]["points"]), 42)
        self.assertIsNone(events[-1][1]["archive"])
        self.assertIsNone(worker.cloud.calls[-1])
        self.assertIsNone(worker.estimated_cloud.calls[-1])
        CameraWorker.command(worker, "view", False)
        self.assertEqual(len(events[-1][1]["points"]), 5)
        self.assertEqual(events[-1][1]["archive"], str(worker.cloud.path))
        self.assertEqual(worker.cloud.calls[-1], 5)

    def test_current_frame_fallback_does_not_reload_an_empty_map_archive(self):
        class EmptyCloud:
            path = Path("empty.sqlite")

            def __len__(self):
                return 0

            def arrays(self, max_points=None):
                return np.empty((0, 3)), np.empty((0, 3), np.uint8)

        events = []
        frame = SimpleNamespace(valid=np.ones((2, 2), bool), points=np.ones((2, 2, 3)),
                                left=np.full((2, 2, 3), 123, np.uint8))
        worker = SimpleNamespace(calibration_busy=False, scan=None, mapping=False, frame=frame,
                                 cloud=EmptyCloud(), estimated_cloud=EmptyCloud(), preview_points=5,
                                 emit=lambda kind, value: events.append((kind, value)))
        CameraWorker.command(worker, "view", False)
        self.assertEqual(len(events[-1][1]["points"]), 4)
        self.assertIsNone(events[-1][1]["archive"])


if __name__ == "__main__":
    unittest.main()
