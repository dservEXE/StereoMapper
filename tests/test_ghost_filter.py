import unittest
import numpy as np
from stereo.depth import DepthFrame
from stereo.ghost_filter import supported_disparity,MultiViewConfirmation


def plane(camera_x=0.,slope=0.,phantom=False):
    h,w=64,96;fx=200.;K=np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1.]])
    yy,xx=np.indices((h,w));rays=(xx-w/2)/fx
    z=(.3+slope*camera_x)/(1-slope*rays)
    if phantom:z[20:28,40:48]=.12
    points=np.stack((rays*z,(yy-h/2)/fx*z,z),axis=2).astype(np.float32)
    image=np.zeros((h,w,3),np.uint8)
    return DepthFrame(image,image,fx*.06/z,points,np.ones((h,w),bool),image),K


class GhostTests(unittest.TestCase):
    def test_isolated_spike_removed_without_averaging_surface_or_two_pixel_leaf(self):
        d=np.full((24,30),16.,np.float32);valid=np.ones(d.shape,bool)
        d[8,8]=40.
        mask=supported_disparity(d,valid)
        self.assertFalse(mask[8,8]);self.assertTrue(mask[7,8]);self.assertTrue(mask[0,0])
        d[:,16:18]=60.
        mask=supported_disparity(d,valid)
        self.assertTrue(mask[5:20,16:18].all())
        self.assertEqual(d[8,8],40.)

    def test_single_frame_phantom_is_not_added_to_confirmed_map(self):
        old,K=plane();current,_=plane(phantom=True)
        check=MultiViewConfirmation(K,.06)
        first,_=check.filter(old,np.eye(4));self.assertFalse(first.valid.any())
        result,rejected=check.filter(current,np.eye(4))
        self.assertFalse(result.valid[20:28,40:48].any())
        self.assertTrue(result.valid[:16].all())
        self.assertEqual(rejected,64)
        self.assertTrue(np.isnan(result.points[~result.valid]).all())
        self.assertTrue(current.valid.all())

    def test_motion_compensation_confirms_tilted_surface_that_pixel_comparison_rejects(self):
        old,K=plane(slope=2.);current,_=plane(camera_x=.02,slope=2.)
        check=MultiViewConfirmation(K,.06);check.filter(old,np.eye(4))
        pose=np.eye(4);pose[0,3]=.02
        moved,_=check.filter(current,pose)
        self.assertGreater(moved.valid.mean(),.7)
        incorrect=MultiViewConfirmation(K,.06);incorrect.filter(old,np.eye(4))
        wrong,_=incorrect.filter(current,np.eye(4))
        self.assertLess(wrong.valid.mean(),.05)

    def test_strict_requires_three_observations_and_rejects_nan_history(self):
        frame,K=plane();check=MultiViewConfirmation(K,.06,observations=3)
        self.assertFalse(check.filter(frame,np.eye(4))[0].valid.any())
        self.assertFalse(check.filter(frame,np.eye(4))[0].valid.any())
        self.assertTrue(check.filter(frame,np.eye(4))[0].valid.all())
        broken,_=plane();broken.points[:]=np.nan;broken.valid[:]=False
        check=MultiViewConfirmation(K,.06);check.filter(broken,np.eye(4))
        self.assertFalse(check.filter(frame,np.eye(4))[0].valid.any())
