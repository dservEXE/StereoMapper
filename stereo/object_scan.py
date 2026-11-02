"""A bounded, refinable multi-view photo scan; all accepted points are measured."""
from pathlib import Path
from collections import deque
import json
import time
import numpy as np

from .depth import DepthEngine
from .mapping import VisualOdometry
from .ghost_filter import MultiViewConfirmation
from .refining_map import RefiningMap,ScanCoverage
from .precision import verify_stationary
from .cloud_orientation import initial_camera_pose


class ObjectScan:
    def __init__(self,directory,calibration,*,wide_view=True,acceleration="auto",width=None,
                 max_depth=1.2,near_minimum=.10,voxel_mm=1.,minimum_seconds=5.,
                 maximum_seconds=60.,automatic=True,region=None,ghost_filter="normal"):
        if ghost_filter not in ('normal','strict'):
            raise ValueError("Для объёмного снимка выберите обычную или строгую защиту.")
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.engine=None
        self.cloud=None
        try:
            self.engine=DepthEngine(calibration,max_depth=max_depth,disparities=192,quality="high",
                processing_width=width,temporal=False,near_priority=True,wide_view=wide_view,
                detail=True,estimate_surfaces=False,near_minimum=near_minimum,
                acceleration=acceleration,parallel=True,ghost_filter=ghost_filter)
            self.settings=dict(max_depth=max_depth,near_minimum=near_minimum,region=region,
                ghost_filter=ghost_filter,wide_view=wide_view,quality="high",map_stride=1)
            K=self.engine.calibration.P1[:,:3]
            self.cloud=RefiningMap(path=self.directory/"measured.sqlite",voxel=voxel_mm/1000,
                stride=1,K=K,baseline=calibration.baseline)
            self.initial_pose=initial_camera_pose(self.engine.calibration)
            self.odometry=VisualOdometry(K,initial_pose=self.initial_pose)
            self.confirmation=MultiViewConfirmation(K,calibration.baseline,3,image_motion=True)
            domain=self.engine.roi_mask.copy()
            if region:
                # Region is normalized to the rectified view, so resolution can change.
                h,w=domain.shape
                x0,y0,x1,y1=region
                mask=np.zeros_like(domain)
                mask[int(y0*h):int(y1*h),int(x0*w):int(x1*w)]=True
                domain &= mask
            if not domain.any(): raise ValueError("Выбранная область вне изображения камеры.")
            self.coverage=ScanCoverage(K,domain,reference_pose=self.initial_pose)
            self.engine.calibration.save(self.directory/"calibration.npz")
            calibration.save(self.directory/"raw_calibration.npz")
        except Exception:
            self.close()
            raise
        self.minimum=max(5.,float(minimum_seconds))
        self.maximum=float(maximum_seconds)
        self.automatic=automatic
        self.started=time.perf_counter()
        self.frames=self.integrated=self.rejected=self.unconfirmed=0
        self.history=deque()
        self.reason=None
        self.last_frame=self.raw=None
        self.tracking_anchor=self.tracking_pose=None
        self.stationary=False
        self.report={}

    def estimate_pose(self,frame):
        self.stationary=False
        if self.tracking_anchor is not None:
            try:
                for old,new in zip(self.tracking_anchor,(frame.left,frame.right)):
                    verify_stationary(old,new)
            except ValueError:
                pass
            else:
                # Compare to a fixed anchor, so slow movement accumulates into
                # a detectable shift instead of being ignored frame after frame.
                self.stationary=True
                return self.tracking_pose.copy(),"Камера неподвижна: положение зафиксировано, поверхность уточняется"
        pose,status=self.odometry.update(frame)
        if pose is not None:
            self.tracking_anchor=(frame.left,frame.right)
            self.tracking_pose=pose.copy()
        return pose,status

    def process(self,raw):
        frame=self.engine.compute(*raw)
        self.last_frame,self.raw=frame,raw
        self.frames+=1
        integrated_now=False
        self.stationary=False
        pose,status=self.estimate_pose(frame) if not frame.quality_warning else (None,frame.quality_warning)
        if not frame.valid.any() and not frame.quality_warning:
            status=f"Нет принятой глубины до {self.engine.maximum:g} м. Проверьте диапазон, свет, резкость и калибровку."
        if pose is not None:
            accepted,self.unconfirmed=self.confirmation.filter(frame,pose)
            if accepted.valid.any():
                # Correct older free-space outliers before adding the next verified surface.
                removed=self.cloud.revise(accepted,pose)
                self.cloud.add(accepted,pose)
                if removed:
                    self.coverage.known[:]=False
                    points,_=self.cloud.arrays()
                    self.coverage.add_world(points)
                else:
                    self.coverage.update(accepted,pose)
                self.integrated+=1
                integrated_now=True
        else:
            self.rejected+=1
            self.unconfirmed=int(frame.valid.sum())
        elapsed=time.perf_counter()-self.started
        self.history.append((elapsed,self.coverage.fraction,self.integrated))
        while len(self.history)>1 and self.history[1][0]<elapsed-4:
            self.history.popleft()
        enough=(elapsed>=self.minimum and self.integrated>=12 and self.coverage.fraction>=.9
            and integrated_now and self.cloud.pending==0)
        plateau=(elapsed-self.history[0][0]>=3 and self.integrated-self.history[0][2]>=5
            and self.coverage.fraction-self.history[0][1]<.002)
        if self.automatic and enough and plateau:
            self.reason="Достигнуто покрытие ≥90% выбранной области; прирост замедлился."
        elif self.maximum>0 and elapsed>=self.maximum:
            self.reason="Достигнут предел времени. Неподтверждённые участки остались неизвестными."
        self.report=dict(seconds=elapsed,frames=self.frames,integrated=self.integrated,
            points=len(self.cloud),coverage=self.coverage.fraction,removed=self.cloud.removed,
            pending_corrections=self.cloud.pending,
            unconfirmed=self.unconfirmed,tracking=status,reason=self.reason,
            valid_points=int(frame.valid.sum()),
            stationary=self.stationary,
            acceleration=self.engine.acceleration,alignment_warning=frame.quality_warning,
            archive=str(self.cloud.path),processing_size=self.engine.calibration.size,
            preview=self.coverage.known.copy(),image=frame.left,depth_image=frame.color)
        return frame,self.report

    def finish(self,reason="Сбор остановлен пользователем"):
        self.reason=self.reason or reason
        if len(self.cloud):
            self.cloud.export(self.directory/"scan.ply")
        if self.raw:
            from .calibration import write_image
            write_image(self.directory/"raw_left.png",self.raw[0])
            write_image(self.directory/"raw_right.png",self.raw[1])
        metadata={k:v for k,v in self.report.items() if k not in ('preview','image','depth_image')}
        metadata.update(kind="multi_view_object_scan",units="metres",reason=self.reason,
            axes="initial raw left camera: X right, Y down, Z forward",
            start_orientation="camera pair held level by user; gravity not measured",
            initial_rectified_camera_to_world=self.initial_pose.tolist(),
            points=len(self.cloud),frames=self.frames,integrated=self.integrated,
            coverage=self.coverage.fraction,removed=self.cloud.removed,
            voxel_mm=self.cloud.voxel*1000,confirmation_observations=3,
            coverage_definition="measured pixels projected into selected area of first rectified view; not completeness of hidden faces",
            guaranteed_accuracy_mm=None,estimated_points=False)
        metadata.update(self.settings)
        (self.directory/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
        self.close()
        return dict(metadata,directory=str(self.directory))

    def close(self):
        if self.engine is not None: self.engine.close()
        if self.cloud is not None: self.cloud.close()
