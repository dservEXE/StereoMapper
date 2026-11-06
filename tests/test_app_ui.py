"""First-run workflow, state gates and small-window checks without USB devices."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from queue import Queue
from threading import Event
import time
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch
import numpy as np
from stereo.app import StereoApp,ImagePanel
from stereo.scan_window import ObjectScanWindow
from stereo.ui import ScrolledFrame,Foldout,quick_guide
from stereo.storage_window import StorageWindow,session_cloud


class AppUXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host=tk.Tk(); cls.host.withdraw()

    @classmethod
    def tearDownClass(cls):cls.host.destroy()

    def setUp(self):
        self.temp=TemporaryDirectory(); self.errors=[]
        self.previous=self.host.report_callback_exception
        self.host.report_callback_exception=lambda *e:self.errors.append(e)
        self.window=tk.Toplevel(self.host)
        self.app=StereoApp(self.window,Path(self.temp.name))
        self.window.geometry('1000x680+10+10'); self.host.update()

    def tearDown(self):
        self.app.worker=None
        self.window.destroy(); self.host.update_idletasks()
        self.host.report_callback_exception=self.previous
        self.temp.cleanup()
        self.assertFalse(self.errors,self.errors)

    def worker(self,engine=True,frame=False):
        worker=SimpleNamespace(engine=SimpleNamespace() if engine else None,
            frame=object() if frame else None,scan=None,commands=Queue(),events=Queue(),
            frames=Queue(),previews=Queue(),stop_event=Event(),is_alive=lambda:True,
            max_depth=1.2,near_minimum=.1,ghost_filter='normal',raw=None)
        self.app.worker=worker; self.app.connected=True; self.app.sync_ui(force=True)
        return worker

    def pump(self,predicate,timeout=2):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            self.host.update()
            if predicate():return
            time.sleep(.01)
        self.fail('UI state did not update')

    def enabled(self,button):return str(button.cget('state'))!='disabled'

    def visible_inside_window(self,widget):
        self.assertTrue(widget.winfo_ismapped())
        top=widget.winfo_rooty()-self.window.winfo_rooty()
        left=widget.winfo_rootx()-self.window.winfo_rootx()
        self.assertGreaterEqual(top,0); self.assertLessEqual(top+widget.winfo_height(),self.window.winfo_height())
        self.assertGreaterEqual(left,0); self.assertLessEqual(left+widget.winfo_width(),self.window.winfo_width())

    def test_first_run_has_clear_pages_and_precision_is_only_in_advanced(self):
        self.assertIsNone(self.app.worker)
        self.assertEqual(list(self.app.workflow_pages),['cameras','calibration','scan','weighing','advanced'])
        self.assertEqual(self.app.workflow.select(),str(self.app.workflow_pages['cameras']))
        self.assertFalse(self.app.precision_button.winfo_viewable())
        self.assertTrue(self.enabled(self.app.connect_button))
        for button in (self.app.scan_button,self.app.depth_button,self.app.capture_button,self.app.calibrate_button):
            self.assertFalse(self.enabled(button))
        self.assertTrue(self.enabled(self.app.open_model_button))
        self.app.show_page('advanced'); self.host.update()
        self.assertTrue(self.app.precision_button.winfo_ismapped())
        self.assertNotEqual(self.app.precision_button.cget('style'),'Accent.TButton')

    def test_primary_actions_remain_visible_at_minimum_size_even_with_scrolled_settings(self):
        for page,buttons in (('cameras',[self.app.connect_button]),
                             ('calibration',[self.app.capture_button,self.app.calibrate_button]),
                             ('scan',[self.app.depth_button,self.app.scan_button])):
            self.app.show_page(page); self.host.update()
            for button in buttons:self.visible_inside_window(button)
            pane=self.app.ui_panes[list(self.app.workflow_pages).index(page)]
            pane.canvas.yview_moveto(1); self.host.update()
            for button in buttons:self.visible_inside_window(button)

    def test_calibration_unlocks_capture_and_calculation_only_after_enough_pairs(self):
        worker=self.worker(engine=False)
        self.assertTrue(self.enabled(self.app.capture_button))
        self.assertFalse(self.enabled(self.app.calibrate_button))
        worker.events.put(('count',14)); self.pump(lambda:self.app.calibration_pairs==14)
        self.assertFalse(self.enabled(self.app.calibrate_button))
        worker.events.put(('count',15)); self.pump(lambda:self.enabled(self.app.calibrate_button))
        self.assertEqual(float(self.app.calibration_progress.cget('value')),15)
        worker.events.put(('calibration_busy',True)); self.pump(lambda:self.app.calibration_busy)
        self.assertFalse(self.enabled(self.app.capture_button)); self.assertFalse(self.enabled(self.app.calibrate_button))

    def test_depth_toggle_sends_real_command_and_scan_requires_calibration(self):
        worker=self.worker(engine=False)
        self.assertFalse(self.enabled(self.app.scan_button))
        worker.engine=SimpleNamespace(); self.app.sync_ui()
        self.assertTrue(self.enabled(self.app.scan_button))
        self.app.depth_button.invoke()
        self.assertEqual(worker.commands.get_nowait(),('depth',True))
        self.assertTrue(self.app.depth_enabled.get())
        self.app.depth_button.invoke()
        self.assertEqual(worker.commands.get_nowait(),('depth',False))

    def test_disconnect_disables_live_tasks_but_retains_last_frame_export(self):
        worker=self.worker(frame=True)
        worker.events.put(('disconnected',None)); self.pump(lambda:not self.app.connected)
        self.assertFalse(self.enabled(self.app.depth_button)); self.assertFalse(self.enabled(self.app.scan_button))
        frame_actions=[button for button,kind in self.app.ui_actions if kind=='frame']
        self.assertTrue(all(self.enabled(button) for button in frame_actions))
        self.assertTrue(all(str(widget.cget('state'))==state for widget,state in self.app.settings_controls))

    def test_connecting_and_active_scan_prevent_conflicting_actions(self):
        worker=self.worker()
        self.app.connection_pending=True; self.app.sync_ui()
        self.assertFalse(self.enabled(self.app.connect_button)); self.assertFalse(self.enabled(self.app.scan_button))
        self.app.connection_pending=False; worker.scan=object(); self.app.sync_ui()
        self.assertFalse(self.enabled(self.app.connect_button)); self.assertFalse(self.enabled(self.app.depth_button))
        worker.scan=None; self.app.sync_ui()
        self.assertTrue(self.enabled(self.app.scan_button))

    def test_navigation_and_disclosures_do_not_change_processing_or_capture_settings(self):
        before={name:variable.get() for name,variable in self.app.vars.items()}
        for page in self.app.workflow_pages:self.app.show_page(page)
        def visit(widget):
            if isinstance(widget,Foldout):widget.toggle(); widget.toggle()
            for child in widget.winfo_children():visit(child)
        visit(self.window); self.host.update()
        self.assertEqual(before,{name:variable.get() for name,variable in self.app.vars.items()})
        self.app.parse(); self.app.depth_options()
        self.assertFalse(self.window.bind_all('<MouseWheel>'))

    def test_placeholder_is_explanatory_and_clears_stale_measurement_geometry(self):
        panel=self.app.panels[0]
        self.assertGreater(len(panel.canvas.find_all()),2)
        panel.set(np.zeros((100,160,3),np.uint8)); self.assertIsNotNone(panel.geometry)
        self.app.show_page('scan'); self.host.update()
        panel.set(np.full((100,160,3),100,np.uint8))
        self.app.show_page('cameras'); self.host.update()
        self.assertIsNotNone(panel.geometry)
        panel.clear(); self.assertIsNone(panel.geometry); self.assertIsNone(panel.source)
        self.assertGreater(len(panel.canvas.find_all()),2)

    def test_scan_window_shows_preparation_progress_result_and_keeps_controls_visible(self):
        self.worker()
        scan=ObjectScanWindow(self.app); self.app.scan_window=scan
        scan.window.geometry('1000x680+20+20'); self.host.update()
        try:
            self.assertEqual(scan.phase.get(),'Готовы к съёмке')
            for button in (scan.start_button,scan.stop_button,scan.view_button):
                bottom=button.winfo_rooty()+button.winfo_height()
                self.assertLessEqual(bottom,scan.window.winfo_rooty()+scan.window.winfo_height())
            with patch.object(self.app,'send') as send:
                scan.start_button.invoke(); send.assert_called_once()
                self.assertEqual(send.call_args.args[0],'scan_start')
            self.assertTrue(scan.active); self.assertFalse(self.enabled(scan.start_button))
            self.assertTrue(self.enabled(scan.stop_button)); self.assertFalse(self.enabled(self.app.scan_button))
            scan.update('scan_finished',dict(directory=str(Path(self.temp.name)/'result'),points=1500,coverage=.6,reason='Готово'))
            self.assertEqual(scan.phase.get(),'Скан сохранён'); self.assertTrue(self.enabled(scan.view_button))
            self.assertIn('1,500',scan.summary.get())
            self.assertEqual(str(scan.view_button.cget('style')),'Accent.TButton')
            self.assertEqual(str(scan.start_button.cget('style')),'TButton','one main button at a time')
        finally:
            scan.active=False; scan.window.destroy()
            self.assertIsNone(scan.poll_id)

    def test_getting_started_guide_opens_without_changing_workflow_or_starting_cameras(self):
        quick_guide(self.window); self.host.update()
        guides=[w for w in self.window.winfo_children() if isinstance(w,tk.Toplevel)]
        self.assertEqual(len(guides),1); self.assertIsNone(self.app.worker)
        guides[0].destroy()

    def test_session_library_prefers_project_then_measured_data_and_opens_selected_only(self):
        directory=Path(self.temp.name)/'output'/'scan_test'; directory.mkdir(parents=True)
        (directory/'scan.ply').write_text('test',encoding='utf-8')
        measured=directory/'measured.sqlite'; measured.write_text('test',encoding='utf-8')
        self.assertEqual(session_cloud(directory),measured)
        project=directory/'edited.cloudproj'; project.write_text('test',encoding='utf-8')
        self.assertEqual(session_cloud(directory),project)
        opened=[]
        library=StorageWindow(self.window,self.temp.name,on_open_cloud=opened.append)
        try:
            self.pump(lambda:not library.working and len(library.items)==1)
            key=library.table.get_children()[0]; library.table.selection_set(key); library.selected()
            self.assertTrue(self.enabled(library.open_button))
            library.open_button.invoke(); self.assertEqual(opened,[project])
            self.assertTrue(project.exists()); self.assertTrue(measured.exists())
        finally:
            library.window.destroy()
            self.assertIsNone(library.poll_id)


if __name__=='__main__':unittest.main()
