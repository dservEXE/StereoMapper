import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from stereo.cloud_orientation import (canonical_view, frame_from_plane, initial_camera_pose,
    rotation_matrix, view_state, save_views, load_views)
from stereo.cloud_geometry import project_points, selection_mask, pick_point, render_points
from stereo.cloud_document import CloudDocument
from stereo.mapping import VisualOdometry
from stereo.refining_map import ScanCoverage
from stereo.trackball import axis_rotation


class OrientationTests(unittest.TestCase):
    def state(self):
        camera=dict(rotation=np.eye(3).tolist(),zoom=1.7,pan=[.1,-.05],center=[1,2,3],extent=5,
                    projection="orthographic")
        return dict(version=1,frame=axis_rotation(2,.3).tolist(),current=camera,
                    bookmarks={"Стол сверху":camera})

    def test_six_directions_are_exact_in_custom_scene_basis(self):
        frame=axis_rotation(2,.3)@axis_rotation(0,-.4)
        for name, axis in (("front",[0,0,1]),("back",[0,0,-1]),("top",[0,1,0]),
                           ("bottom",[0,-1,0]),("right",[-1,0,0]),("left",[1,0,0])):
            view=canonical_view(name,frame)
            np.testing.assert_allclose(view@view.T,np.eye(3),atol=1e-14)
            np.testing.assert_allclose(view[2]@frame.T,axis,atol=1e-14)

    def test_plane_levels_tilted_floor_and_keeps_correct_up_sign(self):
        tilt=axis_rotation(2,.28)@axis_rotation(0,-.35)
        plane=np.array([[0,1,0],[2,1,0],[0,1,3]])@tilt.T
        frame=frame_from_plane(plane,np.eye(3))
        leveled=plane@frame.T
        np.testing.assert_allclose(leveled[:,1],leveled[0,1],atol=1e-14)
        self.assertGreater(-frame[1]@[-tilt[0,1],-tilt[1,1],-tilt[2,1]],.99)
        reverse=frame_from_plane(plane[[0,2,1]],np.eye(3))
        np.testing.assert_allclose(reverse,frame,atol=1e-14)

    def test_plane_rejects_collinear_marks_and_handles_front_looking_down(self):
        for points in ([[0,0,0],[1,0,0],[2,0,0]],[[0,0,0],[0,0,0],[0,1,0]]):
            with self.assertRaises(ValueError):frame_from_plane(points,np.eye(3))
        old=axis_rotation(0,np.pi/2)
        frame=frame_from_plane([[0,0,0],[1,0,0],[0,0,1]],old)
        np.testing.assert_allclose(frame@frame.T,np.eye(3),atol=1e-14)

    def test_initial_pose_cancels_rectification_rotation_without_reflection(self):
        rectification=axis_rotation(2,.21)@axis_rotation(1,-.08)
        pose=initial_camera_pose(SimpleNamespace(R1=rectification))
        raw=np.array([[.1,.2,.7],[-.2,.1,1.3]])
        rectified=raw@rectification.T
        np.testing.assert_allclose(rectified@pose[:3,:3].T,raw,atol=1e-14)
        odometry=VisualOdometry(np.eye(3),initial_pose=pose)
        np.testing.assert_allclose(odometry.world_pose,pose,atol=1e-14)
        pose[0,0]=999
        self.assertNotEqual(odometry.world_pose[0,0],999)

    def test_coverage_stays_in_first_rectified_image_after_world_rotation_and_translation(self):
        y,x=np.indices((12,16)); z=np.ones_like(x)
        K=np.array([[50,0,8],[0,50,6],[0,0,1.]])
        points=np.stack(((x-8)/50,(y-6)/50,z),axis=2)
        frame=SimpleNamespace(points=points,valid=np.ones((12,16),bool))
        pose=np.eye(4); pose[:3,:3]=axis_rotation(2,.45)@axis_rotation(1,.2); pose[:3,3]=[.5,-.2,.8]
        coverage=ScanCoverage(K,frame.valid,reference_pose=pose)
        self.assertEqual(coverage.update(frame,pose),1.)
        coverage.known[:]=False
        world=points.reshape(-1,3)@pose[:3,:3].T+pose[:3,3]
        self.assertEqual(coverage.add_world(world),1.)

    def test_orthographic_render_selection_and_pick_share_constant_scale_and_depth(self):
        points=np.array([[.2,.1,-.4],[.2,.1,.4],[-.2,-.1,0.]])
        colors=np.array([[255,0,0],[0,255,0],[0,0,255]],np.uint8)
        camera=dict(rotation=np.eye(3),center=np.zeros(3),extent=1.,projection="orthographic")
        xy,depth,front=project_points(points,200,200,**camera)
        np.testing.assert_allclose(xy[0],xy[1]); self.assertTrue(front.all())
        self.assertLess(depth[0],depth[1])
        x,y=xy[0]; polygon=[[x-2,y-2],[x+2,y-2],[x+2,y+2],[x-2,y+2]]
        np.testing.assert_array_equal(selection_mask(points,polygon,200,200,**camera),[1,1,0])
        np.testing.assert_array_equal(selection_mask(points,polygon,200,200,visible_only=True,**camera),[1,0,0])
        self.assertEqual(pick_point(points,x,y,200,200,radius=1,**camera),0)
        image=render_points(points,colors,200,200,axes=False,**camera)
        np.testing.assert_array_equal(image[int(round(y)),int(round(x))],[0,0,255])

    def test_views_round_trip_in_project_and_separate_json_without_changing_vertices(self):
        points=np.array([[0,0,0],[1,2,3]],float); colors=np.zeros((2,3),np.uint8)
        document=CloudDocument(points,colors); document.view_state=self.state()
        with TemporaryDirectory() as folder:
            project=Path(folder)/"карта.cloudproj"; sidecar=Path(folder)/"карта.ply.views.json"
            document.save_project(project); loaded=CloudDocument.load_project(project)
            self.assertEqual(loaded.view_state,view_state(self.state()))
            np.testing.assert_array_equal(loaded.arrays()[0],points)
            save_views(sidecar,self.state()); self.assertEqual(load_views(sidecar),loaded.view_state)

    def test_invalid_view_state_rejects_reflections_and_nan_and_bad_bookmarks(self):
        with self.assertRaises(ValueError):rotation_matrix(np.diag([1,-1,1]))
        with self.assertRaises(ValueError):rotation_matrix(np.full((3,3),np.nan))
        for change in (lambda s:s.update(frame=np.zeros((3,3))),
                       lambda s:s['current'].update(zoom=float('nan')),
                       lambda s:s['current'].update(projection='broken'),
                       lambda s:s.update(bookmarks={'':s['current']})):
            state=self.state(); change(state)
            with self.assertRaises(ValueError):view_state(state)


if __name__=="__main__":unittest.main()
