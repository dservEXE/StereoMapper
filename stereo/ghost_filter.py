"""Conservative spatial and motion-compensated multi-view depth checks."""
from collections import deque
from dataclasses import replace
import cv2
import numpy as np
import cv2


def supported_disparity(disparity,valid,minimum=3,tolerance=1.25):
    """Reject isolated depth spikes; never fill an unknown pixel or average edges."""
    count=np.zeros(valid.shape,np.uint8)
    h,w=valid.shape
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if not dx and not dy:continue
            ys=slice(max(0,-dy),min(h,h-dy));xs=slice(max(0,-dx),min(w,w-dx))
            yn=slice(max(0,dy),min(h,h+dy));xn=slice(max(0,dx),min(w,w+dx))
            same=valid[yn,xn] & (np.abs(disparity[ys,xs]-disparity[yn,xn])<=tolerance)
            count[ys,xs]+=same
    return valid & (count>=minimum)


def tracking_gray(frame):
    """Image used to follow motion: ToF infrared when present, since colour views hold a seam."""
    tracking=getattr(frame,"tracking",None)
    return tracking if tracking is not None else cv2.cvtColor(frame.left,cv2.COLOR_BGR2GRAY)


class MultiViewConfirmation:
    def __init__(self,K,baseline,observations=2,image_motion=False,depth_tolerance=None):
        self.K=np.asarray(K,np.float64)
        # A direct depth sensor has no stereo base: its noise model replaces the disparity one.
        self.depth_tolerance=depth_tolerance
        self.baseline=None if depth_tolerance else float(baseline)
        self.required=observations-1
        self.history=deque(maxlen=self.required)
        self.image_motion=image_motion

    def motion_agreement(self,current,previous,xx,yy,u,v):
        """Check geometry against independent 2D image tracks, including reverse flow."""
        h,w=current.left.shape[:2]
        scale=min(1.,640/w)
        size=(round(w*scale),round(h*scale))
        a,b=[cv2.resize(tracking_gray(f),size) for f in (current,previous)]
        forward=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a,b,None)
        backward=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(b,a,None)
        x=np.clip(np.rint(xx*scale).astype(int),0,size[0]-1)
        y=np.clip(np.rint(yy*scale).astype(int),0,size[1]-1)
        flow=forward[y,x]
        target=np.column_stack((x,y))+flow
        tx,ty=np.rint(target).astype(int).T
        inside=(tx>=0)&(tx<size[0])&(ty>=0)&(ty<size[1])
        tx=np.clip(tx,0,size[0]-1);ty=np.clip(ty,0,size[1]-1)
        geometric=np.linalg.norm(target-np.column_stack((u,v))*scale,axis=1)<=1.2
        reciprocal=np.linalg.norm(flow+backward[ty,tx],axis=1)<=.8
        return inside&geometric&reciprocal

    def filter(self,frame,pose):
        valid=np.zeros(frame.valid.shape,bool)
        if self.history:
            yy,xx=np.where(frame.valid)
            points=frame.points[yy,xx].astype(np.float64)
            votes=np.zeros(len(points),np.uint8)
            for previous,old_pose in self.history:
                # Current camera -> previous camera, compensating translation/rotation.
                R=old_pose[:3,:3].T@pose[:3,:3]
                t=old_pose[:3,:3].T@(pose[:3,3]-old_pose[:3,3])
                transformed=np.einsum('ij,kj->ik',points,R,optimize=False)+t
                z=transformed[:,2]
                positive=z>.01
                safe_z=np.maximum(z,.01)
                u=np.rint(self.K[0,0]*transformed[:,0]/safe_z+self.K[0,2]).astype(np.int32)
                v=np.rint(self.K[1,1]*transformed[:,1]/safe_z+self.K[1,2]).astype(np.int32)
                h,w=previous.valid.shape
                inside=positive&(u>=0)&(u<w)&(v>=0)&(v<h)
                old_v,old_u=np.clip(v,0,h-1),np.clip(u,0,w-1)
                old_z=previous.points[old_v,old_u,2]
                if self.depth_tolerance:
                    tolerance=self.depth_tolerance(safe_z)
                else:
                    # Two disparity pixels plus a small floor for pose/pixel rounding.
                    tolerance=np.maximum(.002 if self.required>1 else .003,
                        (1 if self.required>1 else 2)*safe_z**2/(self.K[0,0]*self.baseline))
                    tolerance=np.minimum(tolerance,np.maximum(.002 if self.required>1 else .003,
                        (.01 if self.required>1 else .02)*safe_z))
                consistent=inside&previous.valid[old_v,old_u]&np.isfinite(old_z)&(np.abs(old_z-z)<=tolerance)
                if self.image_motion:
                    consistent &= self.motion_agreement(frame,previous,xx,yy,u,v)
                votes+=consistent
            valid[yy,xx]=votes>=self.required
        self.history.append((frame,pose.copy()))
        points=frame.points.copy();points[~valid]=np.nan
        # Keep original observations in history; this result alone enters the map.
        return replace(frame,points=points,valid=valid),int(np.count_nonzero(frame.valid&~valid))
