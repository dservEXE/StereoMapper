"""USB stereo depth and experimental visual mapping."""

import os

# Disable the runtime before OpenCV initializes it in any worker thread.
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import sys
from pathlib import Path

# Optional NVIDIA PyTorch from gpu_pack.py: must come before the bundled one on the import path.
_gpu_runtime = Path(__file__).resolve().parent.parent / "gpu_runtime"
if (_gpu_runtime / "torch" / "__init__.py").is_file():
    sys.path.insert(0, str(_gpu_runtime))

import cv2

# OpenCL is independent of the optional NVIDIA CUDA stereo engine. Keep the
# unreliable OpenCL cache disabled for board detection and calibration.
cv2.ocl.setUseOpenCL(False)
# Leave CPU time for USB decoding and the GUI instead of oversubscribing cores.
cv2.setNumThreads(min(8, max(2,(os.cpu_count() or 4)//2)))
