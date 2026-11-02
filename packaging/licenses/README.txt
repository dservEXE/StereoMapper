THIRD-PARTY COMPONENTS — StereoMapper 0.9 Windows x64

The private CPython interpreter and third-party libraries retain their original
licenses. Python-LICENSE.txt contains the license copied from the included Python
3.12.14 runtime. The packages/ directory contains available license and notice
files copied from the installed distributions; full distribution metadata also
remains with the runtime packages. THIRD_PARTY_COMPONENTS.json lists the versions
and license identifiers found in their metadata.

Single-photo relative depth model:
  Depth Anything V2 Small — Transformers Version
  Authors: Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu,
           Jiashi Feng, Hengshuang Zhao.
  Repository: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
  Revision: 5426e4f0f36572d16453bbda7a8389317b1bef99
  License: Apache License, Version 2.0 (model repository metadata).
  Model weights/configuration are bundled unmodified.
  Model card source (saved as Depth-Anything-V2-Small-MODEL-CARD.md):
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/blob/5426e4f0f36572d16453bbda7a8389317b1bef99/README.md
  Apache license text source (repository has no separate LICENSE file):
  https://www.apache.org/licenses/LICENSE-2.0.txt

Optional NVIDIA CUDA runtime components retain their NVIDIA licenses, included
under packages/nvidia_*/ or packages/nvidia-*/. These components are used by the
CuPy stereo engine. This package does not install an NVIDIA display driver.

Application-specific code is separate from these third-party components.

Microsoft Visual C++ Runtime 14.51.36247.0 x64:
App-local release CRT DLLs were extracted unchanged from the official signed
vc_redist.x64.exe. See Microsoft-VC-Runtime-PROVENANCE.json and the included
Microsoft-VC-Runtime-LICENSE.html for source, integrity hashes, and terms.
No Microsoft installer is executed by this package.
