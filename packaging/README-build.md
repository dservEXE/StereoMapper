# Windows installer and optional updates

Build with a Windows x64 Python 3.12 environment containing requirements.txt,
requirements-photo.txt, requirements_gpu.txt, packaging/requirements-build.txt
and the CPU PyTorch wheel. Download the pinned model through
stereo.mono.model_directory before building. Builds must never include user data.

1. Run packaging/build_portable.py --output dist/app from the build venv.
   For a verified existing private runtime, use --runtime-source PATH and
   --models-source PATH. Neither option copies studies or settings.
2. Run packaging/portable_selftest.py with dist/app/runtime/python.exe and
   --root dist/app --report REPORT.json. This never opens USB devices.
3. Run packaging/build-payloads.py --application dist/app --output dist.
4. Run packaging/build-setup.ps1 -Payload dist/payload.zip -SkipVerify,
   then run it with -Payload dist/update-payload.zip -CodeOnly.
5. Verify the full executable using --verify-payload --result REPORT.txt;
   test a fresh install and preservation of existing user files.

StereoMapper-Setup.exe is a complete offline first installer. It includes a
private CPython, Tcl/Tk, OpenCV, NumPy, Pillow, local photo model and optional
CUDA runtime. No Python installation or Git client is required on the target.
StereoMapper-Update.exe contains application code only and requires an existing
compatible installation. Both installers validate the payload SHA-256 before
replacing managed components; data, output, calibration and settings are preserved.

The release workflow publishes both assets when VERSION in stereo/version.py is
increased on main. Release notes must include StereoMapper-Runtime: 1. If a code
change requires incompatible dependencies, increase RUNTIME_ABI in updates.py
and the marker so previous clients offer the full installer instead of a code update.

The installer ZIP is streamed from an overlay after the PE image rather than
loaded into a .NET resource array. Assets must remain below GitHub's 2 GiB limit.
Dependency licenses and local MSVC provenance remain in packaging/licenses.
