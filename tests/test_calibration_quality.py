from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import cv2
import numpy as np

from stereo.calibration import CalibrationSession
from stereo.calibration_quality import pose_diversity,view_geometry
from stereo.worker import CameraWorker
from stereo.camera import CameraConfig
from test_core import fixture


def projected_views(tilted=False):
    calibration,_,_=fixture()
    objects=np.zeros((54,3),np.float32)
    objects[:,:2]=np.mgrid[:9,:6].T.reshape(-1,2)*.025
    pairs=[]
    rng=np.random.default_rng(12)
    for i in range(20):
        rotation=rng.uniform(-.45,.45,3) if tilted else np.array([0.,0.,i*.02])
        translation=np.array([rng.uniform(-.15,-.04),rng.uniform(-.1,-.02),rng.uniform(.55,1.1)])
        a,_=cv2.projectPoints(objects,rotation,translation,calibration.K1,calibration.D1)
        b,_=cv2.projectPoints(objects,rotation,translation+calibration.T.ravel(),calibration.K2,calibration.D2)
        pairs.append((a,b))
    return calibration,pairs


class CalibrationQualityTests(unittest.TestCase):
    def test_translation_distance_and_in_plane_rotation_do_not_replace_tilt(self):
        c,pairs=projected_views()
        summary=pose_diversity(pairs,(9,6),c.size)
        self.assertEqual(summary['tilted'],0)
        self.assertFalse(summary['ready'])
        self.assertLess(max(v['perspective'] for v in summary['views']),1e-5)

    def test_frontal_series_is_rejected_before_fitting_and_report_is_saved(self):
        c,pairs=projected_views()
        with TemporaryDirectory() as d:
            session=CalibrationSession(d);session.size=c.size;session.points=pairs
            with patch.object(session,'_fit') as fit:
                with self.assertRaisesRegex(ValueError,'Недостаточно наклонов'):
                    session.solve()
                fit.assert_not_called()
            report=json.loads((Path(d)/'quality_report.json').read_text(encoding='utf-8'))
            self.assertFalse(report['accepted'])
            self.assertEqual(report['rejection_reason'],'insufficient_pose_diversity')
            self.assertEqual(report['pose_quality']['tilted'],0)

    def test_varied_perspectives_pass_preflight_and_both_cameras_must_observe_tilt(self):
        c,varied=projected_views(True)
        _,frontal=projected_views(False)
        summary=pose_diversity(varied,(9,6),c.size)
        self.assertGreaterEqual(summary['tilted'],5)
        self.assertTrue(summary['ready'])
        inconsistent=[(a[0],b[1]) for a,b in zip(varied,frontal)]
        self.assertEqual(pose_diversity(inconsistent,(9,6),c.size)['tilted'],0)

    def test_error_log_persists_context_without_camera_images(self):
        with TemporaryDirectory() as d:
            worker=CameraWorker(d,CameraConfig(),(9,6),25.,8.,64)
            worker.new_series()
            worker.emit('calibration_error','test vertical error')
            record=json.loads(worker.log_path.read_text(encoding='utf-8'))
            self.assertEqual(record['kind'],'calibration_error')
            self.assertEqual(record['camera']['left'],0)
            self.assertIn('calibration_',record['calibration_series'])
            self.assertNotIn('raw',record)
            worker.close_maps()

    def test_bad_cached_corner_shape_is_rejected_as_data_error(self):
        with self.assertRaisesRegex(ValueError,'Неверный набор углов'):
            view_geometry(np.zeros((12,2)),(9,6),(1280,720))
