"""Measured multi-view accumulation with conservative free-space correction."""
import cv2
import numpy as np
from .map_store import VoxelMap


class RefiningMap(VoxelMap):
    def __init__(self,*args,K,baseline,depth_tolerance=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.K=np.asarray(K,np.float64)
        self.depth_tolerance=depth_tolerance
        self.baseline=None if depth_tolerance else float(baseline)
        self.cursor=0
        self.removed=0
        self.pending=0
        self.connection.execute("CREATE TABLE IF NOT EXISTS contradictions (key INTEGER PRIMARY KEY, misses INTEGER NOT NULL)")
        self.connection.commit()

    def add_points(self,*args,**kwargs):
        super().add_points(*args,**kwargs)
        # This map can delete corrected samples; the largest ID is not its size.
        with self.lock:
            self.count=int(self.connection.execute("SELECT COUNT(*) FROM voxels").fetchone()[0])

    def revise(self,frame,pose,budget=50000):
        """Remove a point only after three measured rays show it lies in empty space.

        Missing depth and a nearer occluding surface never count as contradictions.
        Boundary pixels are excluded to protect thin leaves and parallax edges.
        """
        with self.lock:
            rows=self.connection.execute("""SELECT v.id,v.key,v.x/v.n,v.y/v.n,v.z/v.n,COALESCE(c.misses,0)
                FROM voxels v LEFT JOIN contradictions c ON c.key=v.key
                WHERE v.id>? ORDER BY v.id LIMIT ?""",(self.cursor,budget)).fetchall()
            if not rows:
                self.cursor=0
                return 0
            self.cursor=rows[-1][0] if len(rows)==budget else 0
        xyz=np.asarray([r[2:5] for r in rows],np.float64)
        camera=np.einsum('ij,jk->ik',xyz-pose[:3,3],pose[:3,:3],optimize=False)
        z=camera[:,2]; safe=np.maximum(z,.001)
        u=np.rint(self.K[0,0]*camera[:,0]/safe+self.K[0,2]).astype(int)
        v=np.rint(self.K[1,1]*camera[:,1]/safe+self.K[1,2]).astype(int)
        h,w=frame.valid.shape
        inside=(z>.01)&(u>=1)&(u<w-1)&(v>=1)&(v<h-1)
        x,y=np.clip(u,0,w-1),np.clip(v,0,h-1)
        depth=frame.points[:,:,2]
        kernel=np.ones((5,5),np.uint8)
        interior=cv2.erode(frame.valid.astype(np.uint8),kernel).astype(bool)
        lo=cv2.erode(np.where(frame.valid,depth,np.inf).astype(np.float32),kernel)
        hi=cv2.dilate(np.where(frame.valid,depth,-np.inf).astype(np.float32),kernel)
        current=depth[y,x]
        sensor=(3*self.depth_tolerance(safe) if self.depth_tolerance
                else 3*safe**2/(self.K[0,0]*self.baseline))
        tolerance=np.maximum(max(.003,3*self.voxel),sensor)
        span=hi[y,x].astype(np.float64)-lo[y,x].astype(np.float64)
        reliable=inside&interior[y,x]&np.isfinite(current)&(span>=0)&(span<=tolerance)
        free_space=reliable & (current>z+tolerance)
        agrees=reliable & (np.abs(current-z)<=tolerance)
        conflicts=[(rows[i][1],) for i in np.flatnonzero(free_space)]
        pending=np.asarray([r[5]>0 for r in rows],bool)
        resets=[(rows[i][1],) for i in np.flatnonzero(agrees&pending)]
        with self.lock,self.connection:
            self.connection.executemany("DELETE FROM contradictions WHERE key=?",resets)
            self.connection.executemany("""INSERT INTO contradictions VALUES (?,1)
                ON CONFLICT(key) DO UPDATE SET misses=misses+1""",conflicts)
            removed=self.connection.execute("""DELETE FROM voxels WHERE key IN
                (SELECT key FROM contradictions WHERE misses>=3)""").rowcount
            self.connection.execute("DELETE FROM contradictions WHERE misses>=3")
            self.count=int(self.connection.execute("SELECT COUNT(*) FROM voxels").fetchone()[0])
            self.pending=int(self.connection.execute("SELECT COUNT(*) FROM contradictions").fetchone()[0])
        self.removed+=removed
        return removed


class ScanCoverage:
    """Coverage of a selected area in the first rectified view, not all hidden faces."""
    def __init__(self,K,domain,reference_pose=None):
        self.K=np.asarray(K)
        self.reference_pose=np.eye(4) if reference_pose is None else np.array(reference_pose,float,copy=True)
        self.domain=np.asarray(domain,bool)
        self.known=np.zeros(self.domain.shape,bool)

    def update(self,frame,pose):
        points=frame.points[frame.valid]
        world=np.einsum('ij,kj->ik',points,pose[:3,:3],optimize=False)+pose[:3,3]
        return self.add_world(world)

    def add_world(self,world):
        world=(np.asarray(world)-self.reference_pose[:3,3])@self.reference_pose[:3,:3]
        z=world[:,2]; safe=np.maximum(z,.001)
        u=np.rint(self.K[0,0]*world[:,0]/safe+self.K[0,2]).astype(int)
        v=np.rint(self.K[1,1]*world[:,1]/safe+self.K[1,2]).astype(int)
        h,w=self.domain.shape
        inside=(z>.01)&(u>=0)&(u<w)&(v>=0)&(v<h)
        self.known[v[inside],u[inside]]=True
        return self.fraction

    @property
    def fraction(self):
        return float(np.count_nonzero(self.known&self.domain)/max(1,self.domain.sum()))
