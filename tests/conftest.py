"""Keeps Tk usable under pytest's output capture.

Tcl opens the standard streams once, with the first interpreter, and keeps their OS handles until
the last interpreter is gone. pytest's capture redirects those streams and later frees the handles,
so an interpreter first created during a test could hold a handle that Windows has since reused for
a script file: starting Tk then failed with "couldn't read file init.tcl". One interpreter made
while capture is suspended holds the real streams for the whole run. The retry covers what that
cannot, such as an antivirus scanner briefly holding a script file.
"""
import time
import tkinter as tk

ATTEMPTS, RETRY_DELAY = 5, .05
_create_root = tk.Tk.__init__
_stream_holder = None


def _create_root_patiently(self, *args, **kwargs) -> None:
    for attempt in range(ATTEMPTS):
        try:
            _create_root(self, *args, **kwargs)
            return
        except tk.TclError as error:
            if attempt == ATTEMPTS - 1 or ".tcl" not in str(error):
                raise
            time.sleep(RETRY_DELAY)


def pytest_configure(config) -> None:
    """Runs between loading conftests and collecting, when pytest's capture is suspended."""
    global _stream_holder
    _stream_holder = tk.Tcl()
    tk.Tk.__init__ = _create_root_patiently
