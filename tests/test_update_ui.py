"""Updater UI tests use real Tk but never import OpenCV or open cameras."""
import importlib.util
import gc
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
import urllib.error
import tkinter as tk
from tkinter import ttk

_package = "_sm_updater_ui_tests"
_package_module = types.ModuleType(_package)
_package_module.__path__ = [str(Path(__file__).parents[1] / "stereo")]
sys.modules[_package] = _package_module
for _name in ("version", "updates", "update_ui"):
    _spec = importlib.util.spec_from_file_location(_package+"."+_name, Path(__file__).parents[1]/"stereo"/(_name+".py"))
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
ui = sys.modules[_package+".update_ui"]


def release(tag="v0.13.1"):
    return {"tag":tag,"version":tag[1:],"requires_setup":False,
            "url":"https://github.com/dservEXE/StereoMapper/releases/download/"+tag+"/StereoMapper-Update.exe"}


class UpdateUI(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest("Tk desktop unavailable: " + str(error))
        self.root.withdraw()
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.env = patch.dict(os.environ, {"LOCALAPPDATA":str(self.directory / "local")})
        self.env.start()
        parent = ttk.Frame(self.root); parent.pack()
        workflow = ttk.Notebook(parent); workflow.pack()
        self.app = types.SimpleNamespace(root=self.root,workflow=workflow,directory=self.directory,
            connected=False, connection_pending=False, calibration_busy=False, worker=None, closing=False)
        self.app.close = self.close_app
        self.controller = ui.UpdateController(self.app,parent)
        self.root.after_cancel(self.controller.startup_id)
        self.controller.startup_id = None

    def close_app(self):
        self.app.closing = True
        self.controller.close()

    def tearDown(self):
        if hasattr(self,"controller"):
            self.controller.close()
            if self.controller.thread:
                self.controller.thread.join(2)
        if hasattr(self,"root"):
            self.root.update_idletasks()
            self.root.destroy()
        if hasattr(self,"env"):
            self.env.stop()
        if hasattr(self,"temporary"):
            self.temporary.cleanup()
        # Dispose Tk object cycles here on the main thread, before the next test
        # starts a worker whose allocations could otherwise trigger cyclic GC.
        self.controller = self.app = self.root = None
        gc.collect()

    def spin(self, condition, timeout=3.):
        end = time.perf_counter()+timeout
        while not condition() and time.perf_counter()<end:
            self.root.update()
            time.sleep(.005)
        self.assertTrue(condition(), "UI did not settle")

    def test_automatic_offer_later_and_manual_reopen(self):
        threads=[]
        def check(_):
            threads.append(threading.get_ident())
            return release()
        with patch.object(ui,"check_release",side_effect=check) as network:
            self.controller.check(False)
            self.spin(lambda:not self.controller.working)
            self.assertEqual(self.controller.banner.winfo_manager(),"pack")
            self.assertIn("0.13.1",self.controller.text.get())
            self.assertNotEqual(threads[0],threading.get_ident())
            self.controller.later()
            self.assertEqual(self.controller.banner.winfo_manager(),"")
            self.controller.settings["last"]=0
            self.controller.check(False)
            self.spin(lambda:not self.controller.working)
            self.assertEqual(self.controller.banner.winfo_manager(),"")
            self.controller.check(True)
            self.spin(lambda:not self.controller.working)
            self.assertEqual(self.controller.banner.winfo_manager(),"pack")
            self.assertEqual(network.call_count,3)

    def test_auto_setting_and_interval_do_not_block_manual_check(self):
        with patch.object(ui,"check_release",return_value=None) as network:
            self.controller.auto.set(False); self.controller.configure()
            self.controller.check(False)
            self.assertFalse(self.controller.working); network.assert_not_called()
            self.controller.check(True)
            self.spin(lambda:not self.controller.working)
            network.assert_called_once()
            self.controller.auto.set(True); self.controller.configure()
            self.controller.check(False)
            self.assertFalse(self.controller.working)
            network.assert_called_once()

    def test_offline_check_keeps_mainloop_responsive(self):
        gate=threading.Event(); heartbeat=[]
        def offline(_):
            gate.wait(2)
            raise urllib.error.URLError("no internet")
        with patch.object(ui,"check_release",side_effect=offline):
            self.controller.check(True)
            self.root.after(1,lambda:heartbeat.append(True))
            self.spin(lambda:bool(heartbeat))
            self.assertTrue(self.controller.working)
            gate.set()
            self.spin(lambda:not self.controller.working)
            self.assertIn("Нет связи",self.controller.text.get())
            self.assertFalse(self.app.closing)

    def test_active_capture_or_export_blocks_installation(self):
        self.controller.available=release()
        with patch.object(ui,"download_release") as download, patch.object(ui.subprocess,"Popen") as run:
            for field in ("connected","connection_pending","calibration_busy"):
                setattr(self.app,field,True)
                self.controller.install()
                setattr(self.app,field,False)
            self.app.worker=types.SimpleNamespace(scan=None,calibration_busy=False,
                                                 export_thread=types.SimpleNamespace(is_alive=lambda:True))
            self.controller.install()
            download.assert_not_called(); run.assert_not_called()
            self.assertIn("дождитесь",self.controller.text.get())

    def test_new_environment_offers_full_setup_instead_of_applying_code(self):
        self.controller.available=dict(release(),requires_setup=True,page="https://github.com/dservEXE/StereoMapper/releases/tag/v0.13.1")
        with patch.object(ui.webbrowser,"open") as browser, patch.object(ui,"download_release") as download:
            self.controller.install()
            browser.assert_called_once_with(self.controller.available["page"])
            download.assert_not_called()
            self.assertFalse(self.app.closing)

    def test_close_cancels_worker_and_removes_queued_or_late_download(self):
        pending=self.directory/"pending"; pending.mkdir()
        installer=pending/"StereoMapper-Update.exe"; installer.write_bytes(b"fake")
        gate=threading.Event()
        with patch.object(ui,"cleanup") as cleanup:
            self.controller.job("downloaded",lambda:(gate.wait(2) and installer,self.directory))
            self.controller.close()
            gate.set()
            self.controller.thread.join(2)
            self.assertTrue(self.controller.closed)
            self.assertTrue(self.controller.cancelled.is_set())
            cleanup.assert_called_once_with(pending)
            self.assertIsNone(self.controller.poll_id)
            self.assertIsNone(self.controller.schedule_id)

    def test_verified_download_launches_expected_protocol_and_closes_app(self):
        installer=self.directory/"StereoMapper-Update.exe"
        installer.write_bytes(b"test")
        self.controller.prepared=(installer,self.directory)
        with patch.object(ui.subprocess,"Popen") as run, patch.object(ui,"cleanup") as cleanup:
            self.controller.launch(installer,self.directory)
            args=run.call_args[0][0]
            self.assertEqual(args[0:2],[str(installer),"--update"])
            self.assertEqual(args[-2:],[str(self.directory),str(installer.parent)])
            self.assertTrue(self.app.closing)
            cleanup.assert_not_called()
            self.assertIsNone(self.controller.prepared)

    def test_preference_write_error_does_not_break_update_check(self):
        with patch.object(Path,"write_text",side_effect=PermissionError("read only")), patch.object(ui,"check_release",return_value=None):
            self.controller.configure()
            self.controller.check(True)
            self.spin(lambda:not self.controller.working)
            self.assertIn("Обновлений пока нет",self.controller.text.get())


if __name__ == "__main__": unittest.main()
