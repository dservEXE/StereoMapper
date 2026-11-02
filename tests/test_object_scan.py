from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import time
import cv2
import numpy as np

from stereo.depth import DepthFrame
from stereo.ghost_filter import MultiViewConfirmation
from stereo.refining_map import RefiningMap,ScanCoverage
from stereo.map_store import VoxelMap
from stereo.object_scan import ObjectScan
from stereo.model3d import read_model,render_cloud
from stereo.trackball import arcball_rotation,axis_rotation
from stereo.gpu_stereo import gpu_info,runtime,CudaSGM
from stereo.worker import CameraWorker
from stereo.camera import CameraConfig
from test_core import fixture


def flat(z=.3,image=None):
    h,w=64,96;f=200.
    K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])
    yy,xx=np.indices((h,w))
    depth=np.broadcast_to(z,(h,w)).copy()
    points=np.stack(((xx-w/2)/f*depth,(yy-h/2)/f*depth,depth),axis=2).astype(np.float32)
    if image is None:image=np.full((h,w,3),100,np.uint8)
    return DepthFrame(image,image,f*.06/depth,points,np.ones((h,w),bool),image),K


class ObjectScanTests(unittest.TestCase):
    def test_scan_and_live_map_start_in_raw_left_axes_and_coverage_uses_rectified_origin(self):
        c,a,b=fixture(); c.R1=axis_rotation(2,.24)@axis_rotation(1,-.08)
        expected=np.eye(4); expected[:3,:3]=c.R1.T
        with TemporaryDirectory() as d:
            scan=ObjectScan(Path(d)/'orientation',c,width=384,wide_view=False,max_depth=2.,acceleration='cpu',automatic=False)
            try:
                np.testing.assert_allclose(scan.odometry.world_pose,expected,atol=1e-14)
                disparity=np.full(a.shape[:2],16.,np.float32)
                points=cv2.reprojectImageTo3D(disparity,c.Q)
                frame=DepthFrame(a,b,disparity,points,np.ones(a.shape[:2],bool),a)
                self.assertGreater(scan.coverage.update(frame,expected),.99)
                result=scan.finish()
                np.testing.assert_allclose(result['initial_rectified_camera_to_world'],expected,atol=1e-14)
                self.assertIn('gravity not measured',result['start_orientation'])
            finally:scan.close()
            worker=CameraWorker(Path(d)/'live',CameraConfig(),(9,6),25.,2.,64,wide_view=False)
            try:
                worker.apply_calibration(c)
                np.testing.assert_allclose(worker.odometry.world_pose,expected,atol=1e-14)
                import json
                session=json.loads((worker.cloud.path.parent/'session.json').read_text(encoding='utf-8'))
                np.testing.assert_allclose(session['initial_rectified_camera_to_world'],expected,atol=1e-14)
            finally:
                if worker.engine:worker.engine.close()
                worker.close_maps()

    def test_repeated_free_space_measurements_correct_old_phantom_and_export_has_actual_count(self):
        frame,K=flat()
        with TemporaryDirectory() as d:
            path=Path(d)/'scan.sqlite'
            cloud=RefiningMap(path=path,voxel=.001,K=K,baseline=.06,stride=1)
            cloud.add_points([[0,0,.12],[.03,0,.3]],[[255,0,0],[0,255,0]])
            self.assertEqual(cloud.revise(frame,np.eye(4)),0)
            self.assertEqual(cloud.revise(frame,np.eye(4)),0)
            self.assertEqual(cloud.revise(frame,np.eye(4)),1)
            self.assertEqual(len(cloud),1)
            cloud.add(frame,np.eye(4))
            count=len(cloud)
            model=Path(d)/'cloud.ply';self.assertEqual(cloud.export(model),count)
            cloud.close()
            store=VoxelMap(path=path,readonly=True)
            self.assertEqual(len(store),count)
            self.assertEqual(len(store.arrays()[0]),count)
            self.assertEqual(len(read_model(model,max_points=count+1)[0]),count)
            store.close()

    def test_unknown_and_occluding_depth_do_not_delete_old_surface_or_thin_leaf(self):
        frame,K=flat(z=.1)
        with TemporaryDirectory() as d:
            cloud=RefiningMap(path=Path(d)/'map.sqlite',K=K,baseline=.06,voxel=.001)
            cloud.add_points([[0,0,.3],[.02,0,.3]],[[1,2,3],[4,5,6]])
            for _ in range(5):self.assertEqual(cloud.revise(frame,np.eye(4)),0)
            frame.valid[:]=False;frame.points[:]=np.nan
            for _ in range(5):self.assertEqual(cloud.revise(frame,np.eye(4)),0)
            self.assertEqual(len(cloud),2);cloud.close()

    def test_agreeing_measurement_resets_contradiction_evidence(self):
        bad,K=flat(.3);good,_=flat(.12)
        with TemporaryDirectory() as d:
            cloud=RefiningMap(path=Path(d)/'map.sqlite',K=K,baseline=.06,voxel=.001)
            cloud.add_points([[0,0,.12]],[[1,2,3]])
            cloud.revise(bad,np.eye(4));cloud.revise(bad,np.eye(4))
            cloud.revise(good,np.eye(4))
            cloud.revise(bad,np.eye(4));cloud.revise(bad,np.eye(4))
            self.assertEqual(len(cloud),1);cloud.close()

    def test_coverage_gains_only_from_measured_observations(self):
        frame,K=flat()
        check=ScanCoverage(K,np.ones(frame.valid.shape,bool))
        frame.valid[:,48:]=False
        self.assertAlmostEqual(check.update(frame,np.eye(4)),.5)
        frame.valid=~frame.valid
        self.assertAlmostEqual(check.update(frame,np.eye(4)),1.)
        self.assertTrue(check.known.all())
        frame.valid[:]=False
        check.known[:]=False
        self.assertEqual(check.update(frame,np.eye(4)),0.)

    def test_independent_image_flow_rejects_persistent_wrong_depth_during_small_motion(self):
        rng=np.random.default_rng(19)
        image=cv2.GaussianBlur(rng.integers(0,256,(64,96,3),dtype=np.uint8),(3,3),.6)
        z=np.full((64,96),.3);z[12:54,20:80]=.12
        previous,K=flat(z,image)
        current,_=flat(z,np.roll(image,-4,axis=1))
        pose=np.eye(4);pose[0,3]=.006
        depth_only=MultiViewConfirmation(K,.06)
        checked=MultiViewConfirmation(K,.06,image_motion=True)
        depth_only.filter(previous,np.eye(4));checked.filter(previous,np.eye(4))
        a=depth_only.filter(current,pose)[0]
        b=checked.filter(current,pose)[0]
        self.assertGreater(a.valid[20:45,30:65].mean(),.9)
        self.assertLess(b.valid[20:45,30:65].mean(),.15)
        self.assertGreater(b.valid[:10,20:70].mean(),.5)

    @unittest.skipUnless(gpu_info(),'Optional CUDA runtime unavailable')
    def test_periodic_foreground_cannot_borrow_wrong_depth_from_background_regularization(self):
        rng=np.random.default_rng(8)
        left=rng.integers(0,256,(96,256),dtype=np.uint8)
        right=np.zeros_like(left);right[:,:-4]=left[:,4:]
        tile=rng.integers(0,256,(48,16),dtype=np.uint8)
        repeated=np.tile(tile,(1,5))
        left[24:72,80:160]=repeated;right[24:72,60:140]=repeated
        old=CudaSGM(64).compute(left,right)/16
        guarded=CudaSGM(64,ambiguity_guard=True).compute(left,right)/16
        region=np.s_[35:60,100:140]
        old_wrong=(old[region]>0)&(np.abs(old[region]-20)>1)
        new_wrong=(guarded[region]>0)&(np.abs(guarded[region]-20)>1)
        self.assertGreater(old_wrong.mean(),.7)
        self.assertLess(new_wrong.mean(),.01)
        self.assertGreater((guarded[10:20,80:200]>0).mean(),.9)

    @unittest.skipUnless(gpu_info(),'Optional CUDA runtime unavailable')
    def test_raw_cost_ambiguity_rejects_periodic_key_alias_but_keeps_unique_match(self):
        cp,kernels=runtime();d=32
        costs=np.full((3,d),30,np.uint8)
        costs[:,8]=2;costs[0,20]=2;costs[2,24]=3
        output=cp.asarray(np.full(3,8*16,np.int16))
        kernels['reject_ambiguous']((1,),(32,),(cp.asarray(costs),output,np.int32(3),np.int32(d),np.int32(0)))
        result=cp.asnumpy(output)
        self.assertEqual(result[0],-16);self.assertEqual(result[1],128);self.assertEqual(result[2],-16)

    def test_small_scan_accumulates_saves_reopens_and_does_not_generate_unknown_surface(self):
        c,a,b=fixture()
        with TemporaryDirectory() as d:
            scan=ObjectScan(Path(d)/'photo',c,width=384,max_depth=2.,acceleration='cpu',maximum_seconds=0,automatic=False)
            try:
                for _ in range(5):frame,report=scan.process((a,b))
                self.assertEqual(report['frames'],5)
                self.assertGreater(report['integrated'],0)
                self.assertGreater(report['points'],1000)
                self.assertLess(report['coverage'],.95)
                result=scan.finish()
                self.assertFalse(result['estimated_points'])
                self.assertTrue((Path(result['directory'])/'scan.ply').exists())
                self.assertTrue((Path(result['directory'])/'raw_left.png').exists())
            finally:scan.close()

    def test_calibration_command_runs_in_background_and_reports_rejection_separately(self):
        c,_,_=fixture()
        with TemporaryDirectory() as d:
            worker=CameraWorker(d,CameraConfig(),(9,6),25.,8.,64)
            worker.new_series();worker.collector.points=[None]*15
            def reject(progress):
                progress('stage');time.sleep(.05);raise ValueError('vertical 1.5 px')
            with patch.object(worker.collector,'solve',side_effect=reject):
                worker.command('calibrate')
                self.assertTrue(worker.calibration_busy)
                worker.calibration_job.join(2)
            worker.command(*worker.commands.get(timeout=1))
            self.assertFalse(worker.calibration_busy)
            events=[]
            while not worker.events.empty():events.append(worker.events.get())
            self.assertIn(('calibration_error','vertical 1.5 px'),events)
            worker.close_maps()

    def test_stationary_photo_keeps_pose_fixed_but_cumulative_image_motion_unlocks_it(self):
        c,a,b=fixture()
        disparity=np.full(a.shape[:2],16.,np.float32)
        frame=DepthFrame(a,b,disparity,cv2.reprojectImageTo3D(disparity,c.Q),np.ones(a.shape[:2],bool),a)
        with TemporaryDirectory() as d:
            scan=ObjectScan(Path(d)/'stable',c,width=384,acceleration='cpu',maximum_seconds=0,automatic=False)
            moved=np.eye(4);moved[0,3]=.03
            try:
                with patch.object(scan.odometry,'update',side_effect=[(np.eye(4),'init'),(moved,'moved')]) as update:
                    for _ in range(3):
                        pose,_=scan.estimate_pose(frame)
                        np.testing.assert_array_equal(pose,np.eye(4))
                    self.assertTrue(scan.stationary);self.assertEqual(update.call_count,1)
                    changed=DepthFrame(np.roll(a,-2,axis=1),np.roll(b,-2,axis=1),disparity,frame.points,frame.valid,a)
                    pose,_=scan.estimate_pose(changed)
                    self.assertFalse(scan.stationary);self.assertEqual(update.call_count,2)
                    np.testing.assert_array_equal(pose,moved)
            finally:scan.close()

    def test_automatic_scan_waits_five_seconds_and_enough_frames_but_time_limit_saves_partial(self):
        c,a,b=fixture()
        with TemporaryDirectory() as d:
            scan=ObjectScan(Path(d)/'auto',c,width=384,max_depth=2.,acceleration='cpu',maximum_seconds=0,
                region=(.3,.2,.9,.8))
            try:
                # Exercise stopping rules with a fully measured ground-truth plane;
                # actual stereo density is covered by the separate scan and GPU tests.
                disparity=np.full(a.shape[:2],16.,np.float32)
                truth=DepthFrame(a,b,disparity,cv2.reprojectImageTo3D(disparity,scan.engine.calibration.Q),
                    np.ones(a.shape[:2],bool),a)
                with patch.object(scan.engine,'compute',return_value=truth):
                    for i in range(16):
                        with patch('stereo.object_scan.time.perf_counter',return_value=scan.started+i/3):
                            _,report=scan.process((a,b))
                        if i<15:self.assertIsNone(report['reason'])
                self.assertGreaterEqual(report['integrated'],12)
                self.assertGreaterEqual(report['coverage'],.9)
                self.assertIn('90%',report['reason'])
            finally:scan.close()
            scan=ObjectScan(Path(d)/'partial',c,width=384,max_depth=2.,acceleration='cpu',maximum_seconds=5)
            try:
                with patch('stereo.object_scan.time.perf_counter',return_value=scan.started+5):
                    _,report=scan.process((a,b))
                self.assertEqual(report['integrated'],0)
                self.assertIn('предел времени',report['reason'])
                saved=scan.finish();self.assertEqual(saved['points'],0)
            finally:scan.close()

    def test_free_rotation_roll_and_antipodal_drag_preserve_rotation_geometry(self):
        rotation=np.eye(3)
        for i in range(100):
            rotation=arcball_rotation((120+i,140),(190+i,180),800,600)@rotation
        rotation=axis_rotation(2,np.pi/3)@rotation
        np.testing.assert_allclose(rotation@rotation.T,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation),1.)
        flip=arcball_rotation((0,300),(800,300),800,600)
        np.testing.assert_allclose(flip@[-1,0,0],[1,0,0],atol=1e-12)
        points=np.array([[x,y,z] for x in (-1.,1.) for y in (-.4,.5) for z in (-.3,.6)])
        colors=np.full((8,3),200,np.uint8)
        first=render_cloud(points,colors,500,400,rotation=rotation,axes=True)
        second=render_cloud(points,colors,500,400,rotation=axis_rotation(2,1.)@rotation,axes=True)
        self.assertGreater(np.count_nonzero(first!=second),100)
