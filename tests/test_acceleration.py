import unittest
from unittest.mock import patch
import numpy as np
from test_core import fixture
from stereo.depth import DepthEngine
from stereo.gpu_stereo import gpu_info,runtime


class AccelerationTests(unittest.TestCase):
    @unittest.skipUnless(gpu_info(),"Optional NVIDIA CUDA runtime not available")
    def test_optimized_cuda_cost_window_equals_exact_reference_at_image_edges(self):
        cp,kernels=runtime();h,w,d=7,11,16
        values=np.random.default_rng(23).integers(0,128,(h,w,d),dtype=np.uint8)
        padded=np.pad(values,((2,2),(2,2),(0,0)),mode='edge')
        expected=np.zeros(values.shape,np.uint16)
        for y in range(5):
            for x in range(5):expected+=padded[y:y+h,x:x+w]
        expected=np.uint8((expected+12)//25)
        raw=cp.asarray(values);horizontal=cp.empty(values.shape,cp.uint16);out=cp.empty(values.shape,cp.uint8)
        args=(np.int32(h),np.int32(w),np.int32(d));grid=((h*w*d+255)//256,)
        kernels['cost_horizontal'](grid,(256,),(raw,horizontal,*args))
        kernels['cost_vertical'](grid,(256,),(horizontal,out,*args))
        np.testing.assert_array_equal(cp.asnumpy(out),expected)

    def test_parallel_cpu_preserves_geometry_and_validity(self):
        c,a,b=fixture()
        first=DepthEngine(c,disparities=64,estimate_surfaces=False)
        second=DepthEngine(c,disparities=64,estimate_surfaces=False,parallel=True)
        try:
            x,y=first.compute(a,b),second.compute(a,b)
            np.testing.assert_array_equal(x.valid,y.valid)
            np.testing.assert_allclose(x.points,y.points,equal_nan=True)
        finally:
            first.close();second.close()

    @unittest.skipUnless(gpu_info(),"Optional NVIDIA CUDA runtime not available")
    def test_gpu_metric_depth_photometric_change_and_safe_fallback(self):
        c,a,b=fixture()
        b=np.uint8(np.clip(b.astype(np.int16)+12,0,255))
        engine=DepthEngine(c,disparities=64,estimate_surfaces=False,acceleration="cuda",parallel=True)
        try:
            frame=engine.compute(a,b)
            self.assertTrue(engine.acceleration.startswith("NVIDIA"))
            self.assertGreater(frame.valid.mean(),.5)
            np.testing.assert_allclose(np.nanmedian(frame.points[:,:,2]),.9,atol=.01)
            self.assertTrue(np.isnan(frame.points[~frame.valid]).all())
            with patch.object(engine.left_matcher,"compute",side_effect=RuntimeError("test GPU failure")):
                recovered=engine.compute(a,b)
            self.assertTrue(engine.acceleration.startswith("CPU"))
            self.assertIn("test GPU failure",engine.acceleration_warning)
            self.assertGreater(recovered.valid.mean(),.4)
        finally:
            engine.close()
