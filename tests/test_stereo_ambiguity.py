import unittest
import numpy as np
from stereo.gpu_stereo import CudaSGM,gpu_info,runtime


@unittest.skipUnless(gpu_info(),'Optional CUDA runtime unavailable')
class StereoAmbiguityTests(unittest.TestCase):
    def guard(self,costs,winner=16,minimum=0):
        cp,kernels=runtime()
        costs=np.asarray(costs,np.uint8)
        output=cp.asarray(np.full(len(costs),(winner+minimum)*16,np.int16))
        kernels['reject_ambiguous'](((len(costs)+255)//256,),(256,),
            (cp.asarray(costs),output,np.int32(len(costs)),np.int32(costs.shape[1]),np.int32(minimum)))
        return cp.asnumpy(output)

    def test_flat_broad_and_shallow_cost_valleys_are_not_periodic_aliases(self):
        d=np.arange(48)
        costs=np.full((4,48),30,np.uint8)
        costs[0]=8
        costs[1]=np.minimum(30,2+np.maximum(abs(d-16)-7,0))
        costs[2,10:28]=8;costs[2,16]=6;costs[2,24]=6
        costs[3]=np.minimum(30,2+abs(d-16)//4)
        np.testing.assert_array_equal(self.guard(costs),np.full(4,256))

    def test_only_separated_prominent_competing_minima_are_rejected(self):
        costs=np.full((4,48),30,np.uint8)
        costs[:,16]=2
        costs[0,32]=2
        costs[1,32]=3
        costs[2,32]=10  # Clearly inferior alternative.
        costs[3,20:28]=2  # A flat valley has no prominent competing minimum.
        np.testing.assert_array_equal(self.guard(costs),[-16,-16,256,256])
        np.testing.assert_array_equal(self.guard(costs,minimum=-47),[-768,-768,-496,-496])

    def test_invalid_disparity_remains_invalid(self):
        cp,kernels=runtime();costs=cp.asarray(np.zeros((1,32),np.uint8))
        output=cp.asarray(np.array([-16],np.int16))
        kernels['reject_ambiguous']((1,),(32,),(costs,output,np.int32(1),np.int32(32),np.int32(0)))
        self.assertEqual(int(cp.asnumpy(output)[0]),-16)

    def test_context_selected_cost_is_not_rejected_as_repetition_by_better_raw_candidate(self):
        costs=np.full((2,48),40,np.uint8)
        costs[:,16]=12;costs[:,32]=12
        costs[0,6]=1  # SGM chose context over a very different photometric match.
        costs[1,32]=1  # Two unequal valleys are not indistinguishable repetitions.
        np.testing.assert_array_equal(self.guard(costs),[256,256])

    def test_unique_thin_foreground_keeps_its_measured_disparity(self):
        rng=np.random.default_rng(61)
        left=rng.integers(0,256,(96,256),dtype=np.uint8)
        right=np.zeros_like(left);right[:,:-4]=left[:,4:]
        # A narrow real object in front of a different textured background.
        detail=rng.integers(0,256,(64,10),dtype=np.uint8)
        left[16:80,126:136]=detail;right[16:80,106:116]=detail
        ordinary=CudaSGM(64).compute(left,right)/16
        guarded=CudaSGM(64,ambiguity_guard=True).compute(left,right)/16
        region=np.s_[28:68,129:133]
        baseline=np.abs(ordinary[region]-20)<1
        kept=np.abs(guarded[region]-20)<1
        self.assertGreater(baseline.mean(),.8)
        self.assertGreater(kept[baseline].mean(),.95)
        np.testing.assert_array_equal(guarded[region][kept],ordinary[region][kept])


if __name__=='__main__':unittest.main()
