"""Tk integration checks using synthetic clouds and temporary files only."""

from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np

from stereo.cloud_document import CloudDocument, load_cloud
from stereo.cloud_geometry import project_points
from stereo.map_store import VoxelMap
from stereo.registration import transform_points
from stereo.trackball import axis_rotation
from stereo.viewer import CloudViewer


class CloudViewerTests(unittest.TestCase):
    def test_saved_pan_survives_loading_before_window_mapping_and_later_resize(self):
        viewer=CloudViewer(self.root,self.points,self.colors)
        self.viewers.append(viewer)
        state=viewer.capture_camera(); state['pan']=[.12,-.07]
        viewer.apply_camera(state)
        viewer.window.geometry('1240x860+20+20'); viewer.window.deiconify(); self.root.update()
        np.testing.assert_allclose(viewer.capture_camera()['pan'],state['pan'],atol=1e-14)
        self.assertEqual(viewer.zoom,state['zoom'])
        viewer.window.geometry('1050x740+20+20'); self.root.update()
        np.testing.assert_allclose(viewer.capture_camera()['pan'],state['pan'],atol=1e-14)

    def test_set_front_defines_all_six_views_without_transforming_cloud(self):
        viewer=self.viewer(); original=viewer.points.copy()
        frame=axis_rotation(2,.33)@axis_rotation(0,-.41)
        viewer.rotation=frame.copy(); viewer.set_front()
        for name in ('front','back','top','bottom','left','right'):
            viewer.preset(name)
            np.testing.assert_allclose(viewer.rotation@viewer.rotation.T,np.eye(3),atol=1e-14)
        viewer.reset()
        np.testing.assert_allclose(viewer.rotation,frame,atol=1e-14)
        np.testing.assert_array_equal(viewer.points,original)
        viewer.flip_up()
        np.testing.assert_allclose(viewer.scene_frame[1],-frame[1],atol=1e-14)

    def test_three_actual_floor_clicks_level_scene_and_cancel_does_not_clear_selection(self):
        plane=np.array([[-.5,.2,-.2],[.5,.2,-.2],[-.5,.2,.6]])
        tilt=axis_rotation(2,.21)@axis_rotation(0,-.18)
        viewer=self.viewer(plane@tilt.T); original=viewer.points.copy()
        viewer.start_plane()
        for xy in self.screen_points(viewer):
            event=self.event(*xy); viewer.start_drag(event,'left'); viewer.release(event)
        self.assertIsNone(viewer.plane_points)
        leveled=viewer.points@viewer.scene_frame.T
        np.testing.assert_allclose(leveled[:,1],leveled[0,1],atol=1e-14)
        np.testing.assert_array_equal(viewer.points,original)
        viewer.document.select(np.ones(3,bool)); viewer.start_plane(); viewer.cancel()
        self.assertIsNone(viewer.plane_points); self.assertEqual(viewer.document.selection_count,3)
        viewer.start_plane(); viewer.tool.set('brush'); viewer.tool_changed()
        self.assertIsNone(viewer.plane_points)

    def test_bookmarks_and_orientation_survive_project_and_ply_reopening(self):
        viewer=self.viewer(); viewer.rotation=axis_rotation(2,.3); viewer.set_front()
        viewer.projection.set('orthographic'); viewer.zoom=1.45; viewer.pan=[30.,-17.]
        viewer.view_name.set('Точная позиция'); viewer.remember_view()
        expected=viewer.capture_views()
        project=self.directory/'ракурсы.cloudproj'; ply=self.directory/'ракурсы.ply'
        for save,path in ((viewer.save_project,project),(viewer.save_ply,ply)):
            with patch('stereo.viewer.filedialog.asksaveasfilename',return_value=str(path)):save()
            self.wait_for(lambda:not viewer.busy)
            reopened=self.viewer(source_path=path)
            self.wait_for(lambda:not reopened.busy)
            actual=reopened.capture_views()
            self.assertEqual(actual['bookmarks'],expected['bookmarks'])
            np.testing.assert_allclose(actual['frame'],expected['frame'],atol=1e-14)
            np.testing.assert_allclose(actual['current']['pan'],expected['current']['pan'],atol=1e-14)
            self.assertEqual(actual['current']['projection'],'orthographic')
            reopened.preset('top'); reopened.recall_view()
            self.assertEqual(reopened.capture_camera(),expected['current'])

    def test_orthographic_brush_uses_same_projection_and_retains_depth_layers(self):
        points=np.array([[.2,.1,-.4],[.2,.1,.4],[-.2,-.1,0.]])
        viewer=self.viewer(points); viewer.projection.set('orthographic'); viewer.fit()
        viewer.tool.set('brush'); viewer.brush_radius.set('4')
        xy=self.screen_points(viewer); np.testing.assert_allclose(xy[0],xy[1])
        event=self.event(*xy[0]); viewer.start_drag(event,'left'); viewer.release(event)
        np.testing.assert_array_equal(viewer.document.selection,[1,1,0])
        viewer.clear_selection(); viewer.visible_only.set(True)
        viewer.start_drag(event,'left'); viewer.release(event)
        np.testing.assert_array_equal(viewer.document.selection,[1,0,0])

    def test_explicit_views_file_restores_pose_but_cannot_replace_cloud(self):
        source=self.directory/'source.ply'; CloudDocument(self.points,self.colors).export_ply(source)
        viewer=self.viewer(source_path=source); self.wait_for(lambda:not viewer.busy)
        viewer.rotation=axis_rotation(1,.4); viewer.set_front(); expected=viewer.capture_views()
        settings=self.directory/'source.ply.views.json'
        with patch('stereo.viewer.filedialog.asksaveasfilename',return_value=str(settings)):viewer.export_views()
        self.assertTrue(settings.exists())
        viewer.reset_frame()
        with patch('stereo.viewer.filedialog.askopenfilename',return_value=str(settings)):viewer.import_views()
        self.assertEqual(viewer.capture_views(),expected)
        original=source.read_bytes()
        with patch('stereo.viewer.filedialog.asksaveasfilename',return_value=str(source)):viewer.export_views()
        self.assertEqual(source.read_bytes(),original)
        self.assertIn('Ошибка',viewer.message.get())

    def test_brush_paints_on_press_and_accumulates_separate_clicks_before_release(self):
        points=np.array([[-.4,0,0],[0,0,0],[.4,0,0],[0,.5,0]])
        viewer=self.viewer(points); viewer.rotation=np.eye(3); viewer.fit()
        viewer.tool.set('brush'); viewer.brush_radius.set('5')
        viewer.operation.set('Новое')
        xy=self.screen_points(viewer)
        viewer.start_drag(self.event(*xy[0]),'left')
        np.testing.assert_array_equal(viewer.document.selection,[True,False,False,False])
        viewer.release(self.event(*xy[0]))
        viewer.start_drag(self.event(*xy[3]),'left')
        np.testing.assert_array_equal(viewer.document.selection,[True,False,False,True])
        viewer.release(self.event(*xy[3]))
        viewer.start_drag(self.event(*xy[0]),'left')
        viewer.move(self.event(*xy[2]))
        # A single fast motion paints the entire segment without waiting for mouse-up.
        np.testing.assert_array_equal(viewer.document.selection,[True,True,True,True])
        viewer.release(self.event(*xy[2]))
        np.testing.assert_array_equal(viewer.document.selection,[True,True,True,True])

    def test_live_brush_ctrl_erases_locally_and_clear_or_escape_resets_explicitly(self):
        points=np.array([[-.4,0,0],[0,0,0],[.4,0,0],[0,.5,0]])
        viewer=self.viewer(points); viewer.rotation=np.eye(3); viewer.fit()
        viewer.tool.set('brush'); viewer.brush_radius.set('5')
        viewer.document.select(np.ones(4,bool))
        x,y=self.screen_points(viewer)[1]
        erase=SimpleNamespace(x=int(round(x)),y=int(round(y)),state=4)
        viewer.start_drag(erase,'left')
        np.testing.assert_array_equal(viewer.document.selection,[True,False,True,True])
        viewer.release(erase)
        np.testing.assert_array_equal(viewer.document.selection,[True,False,True,True])
        viewer.clear_selection()
        self.assertEqual(viewer.document.selection_count,0)
        viewer.start_drag(self.event(x,y),'left')
        self.assertEqual(viewer.document.selection_count,1)
        viewer.cancel()
        viewer.release(self.event(x,y))
        self.assertEqual(viewer.document.selection_count,0)

    def test_live_brush_respects_visible_surface_and_selected_cloud_scope(self):
        viewer=self.viewer(); viewer.rotation=np.eye(3); viewer.fit()
        viewer.tool.set('brush'); viewer.brush_radius.set('5'); viewer.visible_only.set(True)
        x,y=self.screen_points(viewer)[0]
        viewer.start_drag(self.event(x,y),'left')
        np.testing.assert_array_equal(viewer.document.selection,[True,False,False,False])
        viewer.release(self.event(x,y))
        viewer.document.add_cloud(self.points+[.2,0,0],self.colors,'metres'); viewer.refresh(fit=True)
        viewer.tool.set('brush'); viewer.visible_only.set(False)
        viewer.selection_scope.set('Выбранное облако'); viewer.active_layer.set(viewer.layer_names[1])
        viewer.document.select(np.array([True,False,False,False]+[False]*4))
        x,y=self.screen_points(viewer)[4]
        viewer.start_drag(self.event(x,y),'left')
        self.assertTrue(viewer.document.selection[0])
        self.assertTrue(viewer.document.selection[4])
        self.assertFalse(viewer.document.selection[1:4].any())
        viewer.release(self.event(x,y))

    def test_numlock_does_not_turn_selection_tools_into_rotation(self):
        if self.root.tk.call('tk','windowingsystem')!='win32':
            self.skipTest('Windows Num Lock modifier regression')
        viewer=self.viewer()
        viewer.window.deiconify(); self.root.update()
        for tool in ('rectangle','lasso','brush'):
            with self.subTest(tool=tool):
                viewer.tool.set(tool); viewer.tool_changed()
                before=viewer.rotation.copy()
                # Tk on Windows reports Num Lock as Mod1 (0x0008).
                viewer.canvas.event_generate('<ButtonPress-1>',x=120,y=120,state=0x0008)
                self.assertEqual(viewer.drag['mode'],tool)
                viewer.canvas.event_generate('<B1-Motion>',x=160,y=150,state=0x0108)
                np.testing.assert_array_equal(viewer.rotation,before)
                viewer.canvas.event_generate('<ButtonRelease-1>',x=160,y=150,state=0x0108)
        # Let Tk resolve Alt; do not confuse it with Mod1 / Num Lock.
        viewer.tool.set('rectangle'); viewer.tool_changed()
        before=viewer.rotation.copy()
        viewer.canvas.event_generate('<Alt-ButtonPress-1>',x=250,y=250)
        self.assertEqual(viewer.drag['mode'],'rotate')
        viewer.canvas.event_generate('<B1-Motion>',x=290,y=280,state=0x20100)
        viewer.canvas.event_generate('<ButtonRelease-1>',x=290,y=280,state=0x20100)
        self.assertFalse(np.allclose(viewer.rotation,before))

    def click(self, viewer, xy):
        event=self.event(*xy); viewer.start_drag(event,'left'); viewer.release(event)

    def test_ruler_picks_nearest_point_under_cursor(self):
        points=np.array([[-.3,0.,0.],[.3,.4,0.],[-.25,0.,0.]])
        viewer=self.viewer(points); viewer.tool.set('measure'); viewer.tool_changed()
        xy=self.screen_points(viewer); step=(xy[2]-xy[0])/np.linalg.norm(xy[2]-xy[0])
        self.assertGreater(np.linalg.norm(xy[2]-xy[0]),10)
        self.click(viewer,xy[0]+3*step)
        self.click(viewer,xy[2]-3*step)
        np.testing.assert_array_equal(viewer.measure_points,points[[0,2]])
        viewer.cancel(); self.assertEqual(viewer.measure_points,[])
        self.click(viewer,(5,5))
        self.assertEqual(viewer.measure_points,[]); self.assertIn("нет видимой точки",viewer.message.get())

    def test_ruler_draws_line_with_distance_and_restarts_or_resets(self):
        points=np.array([[-.3,0.,0.],[.3,.4,0.],[0.,-.4,.1]])
        for scale,units in ((1,'metres'),(1000,'millimetres')):
            viewer=self.viewer(points*scale,units=units); viewer.tool.set('measure'); viewer.tool_changed()
            viewer.document.select(np.ones(3,bool))
            xy=self.screen_points(viewer)
            self.click(viewer,xy[0]); self.assertIn("Точка A",viewer.message.get())
            self.click(viewer,xy[1])
            self.assertIn("721.1 мм",viewer.message.get())
            viewer.draw()
            texts=[viewer.canvas.itemcget(item,'text') for item in viewer.canvas.find_all() if viewer.canvas.type(item)=='text']
            self.assertIn("721.1 мм",texts)
            self.assertTrue(any(viewer.canvas.type(item)=='line' and viewer.canvas.itemcget(item,'fill')=="#4ff0ff" for item in viewer.canvas.find_all()))
            rotation=viewer.rotation.copy()
            self.drag(viewer,[xy[2],xy[2]+(40,10),xy[2]+(80,20)])
            self.assertEqual(len(viewer.measure_points),2); self.assertFalse(np.allclose(viewer.rotation,rotation))
            xy=self.screen_points(viewer)
            self.click(viewer,xy[2]); self.assertEqual(len(viewer.measure_points),1)
            viewer.cancel()
            self.assertEqual(viewer.measure_points,[]); self.assertEqual(viewer.document.selection_count,3)
            self.click(viewer,xy[2]); viewer.tool.set('rotate'); viewer.tool_changed()
            self.assertEqual(viewer.measure_points,[]); self.assertFalse(viewer.measure_button.winfo_manager())

    def test_selecting_a_drawing_tool_exits_landmark_picking(self):
        viewer=self.viewer()
        viewer.document.add_cloud(self.points+.1,self.colors,'metres')
        viewer.refresh(fit=True)
        viewer.reference_layer.set(viewer.layer_names[0]); viewer.moving_layer.set(viewer.layer_names[1])
        viewer.start_landmarks()
        self.assertIsNotNone(viewer.landmark_task)
        viewer.tool.set('rectangle'); viewer.tool_changed()
        self.assertIsNone(viewer.landmark_task)
        self.assertIsNone(viewer.view_layer)
        viewer.start_drag(self.event(100,100),'left')
        self.assertEqual(viewer.drag['mode'],'rectangle')

    def test_fit_keeps_rotated_full_cloud_inside_viewport(self):
        viewer=self.viewer(np.array([[x,y,z] for x in (-1.,1.) for y in (-.9,.9) for z in (-.8,.8)]))
        viewer.rotation=axis_rotation(2,.7)@axis_rotation(0,.9)@axis_rotation(1,.6)
        viewer.fit()
        xy=self.screen_points(viewer)
        w,h=viewer.viewport()
        self.assertTrue((xy>=8).all())
        self.assertTrue((xy[:,0]<w-8).all())
        self.assertTrue((xy[:,1]<h-8).all())

    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
            cls.root.withdraw()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk display unavailable: {error}")

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.viewers = []
        self.temporary = TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.errors = []
        self.callback_errors = []
        self.previous_callback_handler = self.root.report_callback_exception
        self.root.report_callback_exception = lambda *args: self.callback_errors.append(args)
        self.error_patch = patch("stereo.viewer.messagebox.showerror", side_effect=lambda *a, **k: self.errors.append(a))
        self.info_patch = patch("stereo.viewer.messagebox.showinfo")
        self.error_patch.start()
        self.info_patch.start()
        # The first two vertices overlap on screen but are separate 3D points.
        self.points = np.array([[0., 0., -.4], [0., 0., .4], [.7, .7, 0.], [-.7, -.7, 0.]])
        self.colors = np.tile([35, 150, 220], (len(self.points), 1)).astype(np.uint8)

    def tearDown(self):
        for viewer in self.viewers:
            if viewer.window.winfo_exists():
                viewer.window.destroy()
        self.root.update_idletasks()
        self.error_patch.stop()
        self.info_patch.stop()
        self.root.report_callback_exception = self.previous_callback_handler
        self.temporary.cleanup()
        self.assertEqual(self.errors, [], "Unexpected error dialogs")
        self.assertEqual(self.callback_errors, [], "Unexpected Tk callback exceptions")

    def viewer(self, points=None, colors=None, **kwargs):
        points = self.points if points is None else points
        colors = np.tile([35, 150, 220], (len(points), 1)).astype(np.uint8) if colors is None else colors
        viewer = CloudViewer(self.root, points, colors, **kwargs)
        self.viewers.append(viewer)
        viewer.window.geometry("1000x850+20+20")
        # Windows needs a map event before the canvas gets its real geometry.
        self.root.update()
        viewer.window.withdraw()
        return viewer

    def wait_for(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.root.update()
            if predicate():
                return
            time.sleep(.01)
        self.fail("Background viewer operation did not finish")

    @staticmethod
    def event(x, y, **kwargs):
        return SimpleNamespace(x=int(round(x)), y=int(round(y)), state=0, **kwargs)

    def screen_points(self, viewer):
        return project_points(viewer.points, *viewer.viewport(), **viewer.camera())[0]

    def drag(self, viewer, path):
        viewer.start_drag(self.event(*path[0]), "left")
        for point in path[1:]:
            viewer.move(self.event(*point))
        viewer.release(self.event(*path[-1]))

    def test_rectangle_selects_full_cloud_through_depth_or_visible_surface(self):
        viewer = self.viewer(max_points=1)
        viewer.rotation = np.eye(3)
        viewer.tool.set("rectangle")
        viewer.visible_only.set(False)
        x, y = self.screen_points(viewer)[0]
        self.drag(viewer, [(x - 6, y - 6), (x + 6, y + 6)])
        np.testing.assert_array_equal(viewer.document.selection, [True, True, False, False])
        self.assertEqual(len(viewer.points), 4)
        viewer.visible_only.set(True)
        self.drag(viewer, [(x - 6, y - 6), (x + 6, y + 6)])
        np.testing.assert_array_equal(viewer.document.selection, [True, False, False, False])

    def test_lasso_and_brush_select_the_points_behind_the_overlay(self):
        viewer = self.viewer()
        viewer.rotation = np.eye(3)
        viewer.visible_only.set(False)
        x, y = self.screen_points(viewer)[0]
        viewer.tool.set("lasso")
        self.drag(viewer, [(x - 12, y - 12), (x + 12, y - 12),
                           (x + 12, y + 12), (x - 12, y + 12), (x - 12, y - 12)])
        np.testing.assert_array_equal(viewer.document.selection, [True, True, False, False])
        viewer.document.clear_selection()
        viewer.tool.set("brush")
        self.drag(viewer, [(x, y), (x + 1, y)])
        np.testing.assert_array_equal(viewer.document.selection, [True, True, False, False])

    def test_delete_keep_undo_and_redo_affect_full_points_and_keep_camera(self):
        viewer = self.viewer(max_points=1)
        before_points = viewer.points.copy()
        before_camera = viewer.camera()
        viewer.apply_selection(np.array([True, False, True, False]))
        viewer.delete_selection()
        np.testing.assert_array_equal(viewer.points, before_points[[1, 3]])
        self.assertEqual(viewer.document.count, 2)
        viewer.undo()
        np.testing.assert_array_equal(viewer.points, before_points)
        viewer.redo()
        np.testing.assert_array_equal(viewer.points, before_points[[1, 3]])
        viewer.undo()
        viewer.apply_selection(np.array([False, True, False, False]))
        viewer.keep_selection()
        np.testing.assert_array_equal(viewer.points, before_points[[1]])
        viewer.undo()
        np.testing.assert_array_equal(viewer.points, before_points)
        after_camera = viewer.camera()
        for name in before_camera:
            if name=="projection":self.assertEqual(after_camera[name],before_camera[name])
            else:np.testing.assert_allclose(after_camera[name], before_camera[name])

    def test_center_and_edge_mouse_drags_rotate_about_different_axes_without_editing_data(self):
        viewer = self.viewer()
        viewer.tool.set("rotate")
        original_points, original_colors = viewer.points.copy(), viewer.colors.copy()
        width, height = viewer.viewport()
        viewer.rotation = np.eye(3)
        self.drag(viewer, [(width / 2, height / 2), (width / 2 + 50, height / 2)])
        center_rotation = viewer.rotation.copy()
        self.assertGreater(abs(center_rotation[0, 2]), .01)
        self.assertLess(abs(center_rotation[0, 1]), .001)
        viewer.rotation = np.eye(3)
        self.drag(viewer, [(width * .95, height * .05), (width * .95, height * .15)])
        self.assertGreater(abs(viewer.rotation[0, 1]), .01)
        self.assertLess(abs(viewer.rotation[0, 2]), 1e-10)
        np.testing.assert_allclose(viewer.rotation @ viewer.rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(viewer.rotation), 1., places=12)
        viewer.wheel(self.event(100, 100, delta=120))
        viewer.preset("top")
        viewer.preset("side")
        viewer.reset()
        np.testing.assert_array_equal(viewer.points, original_points)
        np.testing.assert_array_equal(viewer.colors, original_colors)
        np.testing.assert_array_equal(viewer.document.layers[0].points, original_points)

    def test_background_source_load_replaces_preview_with_every_original_vertex(self):
        count = 1137
        points = np.column_stack((np.arange(count) * .001, np.sin(np.arange(count)) * .01,
                                  np.ones(count)))
        colors = np.tile([40, 80, 160], (count, 1)).astype(np.uint8)
        path = self.directory / "полная_карта.ply"
        CloudDocument(points, colors).export_ply(path)
        viewer = self.viewer(points[::20], colors[::20], max_points=3,
                             total_count=count, source_path=path)
        self.wait_for(lambda: viewer.document.count == count)
        np.testing.assert_allclose(viewer.points, points, atol=1e-14)
        np.testing.assert_array_equal(viewer.colors, colors)
        self.assertEqual(viewer.document.total_count, count)
        viewer.apply_selection(np.arange(count) % 2 == 0)
        viewer.delete_selection()
        self.assertEqual(viewer.document.count, count // 2)
        viewer.undo()
        self.assertEqual(viewer.document.count, count)

    def test_archive_opening_loads_full_map_for_editing(self):
        points = np.column_stack((np.arange(61) * .02, np.zeros(61), np.ones(61)))
        colors = np.tile([30, 160, 80], (len(points), 1)).astype(np.uint8)
        archive = self.directory / "карта.sqlite"
        store = VoxelMap(path=archive, voxel=.001)
        try:
            store.add_points(points, colors)
            full_points, full_colors = store.arrays()
        finally:
            store.close()
        viewer = self.viewer(full_points[::20], full_colors[::20], archive_path=archive,
                             total_count=len(full_points), max_points=2)
        self.wait_for(lambda: viewer.document.count == len(full_points))
        np.testing.assert_array_equal(viewer.points, full_points)
        np.testing.assert_array_equal(viewer.colors, full_colors)

    def test_save_project_and_reopen_retain_all_vertices_and_edit_mask(self):
        points = np.random.default_rng(71).uniform(-1., 1., (117, 3))
        viewer = self.viewer(points, max_points=2)
        viewer.apply_selection(np.arange(len(points)) % 3 == 0)
        viewer.delete_selection()
        expected = viewer.points.copy()
        path = self.directory / "отредактированная_карта.cloudproj"
        with patch("stereo.viewer.filedialog.asksaveasfilename", return_value=str(path)):
            viewer.save_project()
        self.wait_for(lambda: path.exists())
        reopened = self.viewer(max_points=2)
        with patch("stereo.viewer.filedialog.askopenfilename", return_value=str(path)):
            reopened.open_file()
        self.wait_for(lambda: reopened.document.total_count == len(points))
        self.assertEqual(reopened.document.count, len(expected))
        np.testing.assert_array_equal(reopened.points, expected)
        np.testing.assert_array_equal(reopened.document.layers[0].points, points)

    def test_add_three_landmark_pairs_icp_undo_redo_and_merged_export(self):
        reference = np.random.default_rng(91).uniform(-.4, .4, (350, 3))
        pose = np.eye(4)
        pose[:3, :3] = axis_rotation(2, .35) @ axis_rotation(1, -.2)
        pose[:3, 3] = [.35, -.1, .15]
        original_moving = transform_points(reference, pose)
        path = self.directory / "другой_ракурс.ply"
        colors = np.tile([140, 60, 200], (len(reference), 1)).astype(np.uint8)
        CloudDocument(original_moving, colors).export_ply(path)
        viewer = self.viewer(reference)
        with patch("stereo.viewer.filedialog.askopenfilename", return_value=str(path)):
            viewer.add_file()
        self.wait_for(lambda: not viewer.busy and len(viewer.document.layers) == 2)
        self.assertEqual(viewer.document.count, 700)
        np.testing.assert_array_equal(viewer.document.layer_points(1)[0], original_moving)
        viewer.reference_layer.set(viewer.layer_names[0])
        viewer.moving_layer.set(viewer.layer_names[1])
        viewer.landmark_pair = (0, 1)
        # Simulate an imprecise manual pick, leaving a small offset for ICP.
        offset = np.array([.012, -.009, .006])
        viewer.landmarks = {"reference": [p.copy() + offset for p in reference[:3]],
                            "moving": [p.copy() for p in original_moving[:3]]}
        viewer.align_landmarks()
        landmark_pose = viewer.document.layers[1].transform.copy()
        np.testing.assert_allclose(viewer.document.layer_points(1)[0], reference + offset, atol=1e-12)
        viewer.undo()
        np.testing.assert_allclose(viewer.document.layer_points(1)[0], original_moving, atol=1e-12)
        viewer.redo()
        np.testing.assert_array_equal(viewer.document.layers[1].transform, landmark_pose)
        viewer.align_icp()
        self.wait_for(lambda: not viewer.busy)
        np.testing.assert_allclose(viewer.document.layer_points(1)[0], reference, atol=1e-7)
        viewer.undo()
        np.testing.assert_array_equal(viewer.document.layers[1].transform, landmark_pose)
        viewer.redo()
        np.testing.assert_allclose(viewer.document.layer_points(1)[0], reference, atol=1e-7)
        np.testing.assert_array_equal(viewer.document.layers[1].points, original_moving)
        np.testing.assert_array_equal(viewer.document.layers[0].points, reference)
        output = self.directory / "общая_карта.ply"
        with patch("stereo.viewer.filedialog.asksaveasfilename", return_value=str(output)):
            viewer.save_ply()
        self.wait_for(lambda: not viewer.busy and output.exists())
        merged, rgb, units, _ = load_cloud(output)
        self.assertEqual(len(merged), 700)
        self.assertEqual(units, "metres")
        np.testing.assert_allclose(merged[:350], reference, atol=1e-12)
        np.testing.assert_allclose(merged[350:], reference, atol=1e-7)
        np.testing.assert_array_equal(rgb[350:], colors)

    def test_selected_cloud_scope_never_edits_other_layers_with_stale_selections(self):
        for action in ("delete_selection", "keep_selection"):
            with self.subTest(action=action):
                viewer = self.viewer()
                other = self.points + [.2, .3, .4]
                viewer.document.add_cloud(other, self.colors, "metres")
                viewer.refresh(fit=True)
                # The old selection includes every point in the other layer.
                viewer.document.select(np.array([True] * 4 + [False, True, False, False]))
                viewer.selection_scope.set("Выбранное облако")
                viewer.active_layer.set(viewer.layer_names[1])
                np.testing.assert_array_equal(viewer.edit_scope(), [False] * 4 + [True] * 4)
                getattr(viewer, action)()
                np.testing.assert_array_equal(viewer.document.layer_points(0)[0], self.points)
                expected = other[[0, 2, 3]] if action == "delete_selection" else other[[1]]
                np.testing.assert_array_equal(viewer.document.layer_points(1)[0], expected)
                viewer.undo()
                np.testing.assert_array_equal(viewer.document.layer_points(0)[0], self.points)
                np.testing.assert_array_equal(viewer.document.layer_points(1)[0], other)

    def test_failed_full_source_load_stays_incomplete_and_cannot_edit_or_export_preview(self):
        source = self.directory / "повреждённый_источник.ply"
        original = b"not a valid PLY file\n"
        source.write_bytes(original)
        # Even matching declared and preview counts must not bypass load failure.
        viewer = self.viewer(source_path=source, total_count=len(self.points))
        self.wait_for(lambda: not viewer.busy)
        self.assertFalse(viewer.complete)
        self.assertIn("Ошибка", viewer.message.get())
        viewer.document.select(np.ones(len(self.points), bool))
        viewer.delete_selection()
        viewer.keep_selection()
        np.testing.assert_array_equal(viewer.points, self.points)
        self.assertFalse(viewer.document.can_undo)
        with patch("stereo.viewer.filedialog.asksaveasfilename", return_value=str(source)) as dialog:
            viewer.save_ply()
            viewer.save_project()
        dialog.assert_not_called()
        self.assertEqual(source.read_bytes(), original)

    def test_save_project_close_after_closes_only_after_successful_save(self):
        viewer = self.viewer()
        viewer.apply_selection(np.array([True, False, False, False]))
        viewer.delete_selection()
        expected = viewer.points.copy()
        path = self.directory / "сохранить_и_закрыть.cloudproj"
        with patch("stereo.viewer.filedialog.asksaveasfilename", return_value=str(path)):
            viewer.save_project(close_after=True)
        self.wait_for(lambda: viewer.closed and not viewer.window.winfo_exists())
        self.assertFalse(viewer.changed)
        reopened = CloudDocument.load_project(path)
        np.testing.assert_array_equal(reopened.arrays()[0], expected)
        self.assertEqual(reopened.total_count, len(self.points))

    def test_landmark_clicks_use_each_layers_current_camera_and_pair_change_clears_them(self):
        reference = np.array([[-.3, -.2, 0.], [.3, -.2, 0.], [0., .3, 0.], [.1, .1, .4]])
        pose = np.eye(4)
        pose[:3, :3] = axis_rotation(2, .4)
        pose[:3, 3] = [.15, .1, .2]
        moving = transform_points(reference, pose)
        viewer = self.viewer(reference)
        viewer.document.add_cloud(moving, self.colors, "metres")
        viewer.refresh(fit=True)
        viewer.reference_layer.set(viewer.layer_names[0])
        viewer.moving_layer.set(viewer.layer_names[1])
        viewer.start_landmarks()
        for index in range(3):
            for layer in (0, 1):
                self.assertEqual(viewer.view_layer, layer,
                                 f"landmark {index}, layer {layer}: {viewer.message.get()}")
                # Each frame has its own fitted camera; leave room at borders
                # so this test clicks visible landmarks rather than clipping.
                viewer.zoom = .65
                displayed = viewer.points[viewer.displayed()]
                xy = project_points(displayed, *viewer.viewport(), **viewer.camera())[0]
                self.drag(viewer, [xy[index]])
        self.assertIsNone(viewer.landmark_task)
        self.assertEqual(viewer.landmark_pair, (0, 1))
        np.testing.assert_allclose(viewer.landmarks["reference"], reference[:3], atol=1e-14)
        np.testing.assert_allclose(viewer.landmarks["moving"], moving[:3], atol=1e-14)
        viewer.reference_layer.set(viewer.layer_names[1])
        viewer.reference_combo.event_generate("<<ComboboxSelected>>")
        self.root.update()
        self.assertIsNone(viewer.landmark_pair)
        self.assertEqual(viewer.landmarks, {"reference": [], "moving": []})
        viewer.align_landmarks()
        np.testing.assert_array_equal(viewer.document.layers[1].transform, np.eye(4))


if __name__ == "__main__":
    unittest.main()
