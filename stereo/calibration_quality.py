"""Image-only pose diversity checks, before intrinsics can become degenerate."""
import cv2
import numpy as np


def view_geometry(corners,pattern,size):
    columns,rows=pattern
    grid=np.mgrid[:columns,:rows].T.reshape(-1,2).astype(np.float32)
    points=np.asarray(corners,np.float32).reshape(-1,2)
    if points.shape!=grid.shape or not np.isfinite(points).all():
        raise ValueError("Неверный набор углов доски в сохранённой серии.")
    H,_=cv2.findHomography(grid,points,0)
    if H is None or not np.isfinite(H).all():
        raise ValueError("Не удалось определить геометрию ракурса доски.")
    denominator=grid@H[2,:2]+H[2,2]
    mean=max(abs(float(denominator.mean())),1e-12)
    perspective=float(np.ptp(denominator)/mean)
    direction=H[2,:2]*np.array([columns-1,rows-1])/mean
    normalized=points/np.asarray(size)
    cells=np.clip((normalized*np.array([4,3])).astype(int),[0,0],[3,2])
    return dict(perspective=perspective,perspective_direction=direction.tolist(),
        center=normalized.mean(axis=0).tolist(),
        frame_area=float(np.prod(np.ptp(normalized,axis=0))),
        cells=sorted(set((cells[:,1]*4+cells[:,0]).tolist())))


def pose_diversity(pairs,pattern,size):
    if not pairs or size is None:
        return dict(pairs=0,tilted=0,required_tilted=5,covered_cells=0,total_cells=12,
            center_span=[0.,0.],ready=False,views=[])
    views=[]
    for left,right in pairs:
        a,b=[view_geometry(c,pattern,size) for c in (left,right)]
        # A projective denominator varies on an inclined plane. This is not an
        # angle estimate: avoid trusting intrinsics from a frontal-only series.
        views.append(dict(perspective=min(a['perspective'],b['perspective']),
            center=a['center'],frame_area=a['frame_area'],cells=a['cells']))
    tilted=sum(v['perspective']>=.06 for v in views)
    cells=set(c for view in views for c in view['cells'])
    centers=np.asarray([v['center'] for v in views])
    return dict(pairs=len(pairs),tilted=tilted,required_tilted=5,covered_cells=len(cells),total_cells=12,
        center_span=np.ptp(centers,axis=0).tolist(),ready=len(pairs)>=15 and tilted>=5,views=views)
