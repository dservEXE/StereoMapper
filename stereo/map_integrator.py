"""Track and archive maps without holding up live depth or USB preview."""
from queue import Queue, Empty, Full
from threading import Thread, Event
from types import SimpleNamespace
import time
import numpy as np


class MapIntegrator(Thread):
    def __init__(self,odometry,cloud,estimates,confirmation=None):
        super().__init__(daemon=True,name="map-tracking-storage")
        self.odometry,self.cloud,self.estimates=odometry,cloud,estimates
        self.queue=Queue(maxsize=1)
        self.stop_event=Event()
        self.attempts=self.integrated=self.skipped=0
        self.tracking_ms=self.insert_ms=0.
        self.status="Карта ожидает кадр"
        self.error=None
        self.confirmation=confirmation
        self.waiting_confirmation=self.unconfirmed_points=0

    def submit(self,frame):
        try:
            self.queue.put_nowait(frame)
        except Full:
            try:
                self.queue.get_nowait()
                self.skipped+=1
            except Empty:
                pass
            self.queue.put_nowait(frame)

    def run(self):
        while True:
            try:
                frame=self.queue.get(timeout=.1)
            except Empty:
                if self.stop_event.is_set():return
                continue
            self.attempts+=1
            try:
                started=time.perf_counter()
                if frame.quality_warning:
                    pose,self.status=None,"Накопление приостановлено: стереопара не выровнена"
                else:
                    pose,self.status=self.odometry.update(frame)
                self.tracking_ms=(time.perf_counter()-started)*1000
                self.insert_ms=0.
                if pose is not None and frame.valid.any():
                    started=time.perf_counter()
                    accepted=frame
                    if self.confirmation:
                        accepted,self.unconfirmed_points=self.confirmation.filter(frame,pose)
                        if not accepted.valid.any():
                            self.waiting_confirmation+=1
                            self.status="Положение найдено; ожидаю повторное подтверждение поверхности"
                            continue
                    if getattr(self.odometry,"still",False) and self.integrated:
                        continue
                    self.cloud.add(accepted,pose)
                    if frame.estimated_points is not None:
                        estimated=SimpleNamespace(points=frame.estimated_points,
                            left=np.uint8(frame.left.astype(np.float32)*.5+64))
                        self.estimates.add(estimated,pose)
                    self.insert_ms=(time.perf_counter()-started)*1000
                    self.integrated+=1
            except Exception as error:
                self.error=str(error)
                self.status=f"Запись карты остановлена: {error}"
                return
