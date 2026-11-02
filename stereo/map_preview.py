from queue import Queue, Empty, Full
from threading import Thread, Event
import time

from .mapping import top_view


class MapPreview(Thread):
    """Render a bounded sample; the complete map stays in its persistent store."""
    def __init__(self):
        super().__init__(daemon=True)
        self.requests, self.results = Queue(maxsize=1), Queue(maxsize=1)
        self.stop_event = Event()
        self.error = None

    @staticmethod
    def latest(queue, value):
        try:
            queue.put_nowait(value)
        except Full:
            try:
                queue.get_nowait()
            except Empty:
                pass
            queue.put_nowait(value)

    def request(self, generation, cloud, estimates, pose, extent):
        self.latest(self.requests, (generation,cloud,estimates,pose.copy(),extent,list(cloud.trajectory)))

    def run(self):
        while not self.stop_event.is_set():
            try:
                generation,cloud,estimates,pose,extent,trajectory = self.requests.get(timeout=.2)
            except Empty:
                continue
            try:
                started = time.perf_counter()
                points = cloud.preview(30000)[0]
                predicted = estimates.preview(15000)[0]
                if cloud.bounds is not None:
                    extent = max(extent, float(max(abs(cloud.bounds[0][[0,2]]).max(), abs(cloud.bounds[1][[0,2]]).max()))*1.1)
                image = top_view(points,trajectory,extent=extent,surround=True,pose=pose,estimated=predicted)
                self.latest(self.results,(generation,image,(time.perf_counter()-started)*1000))
            except Exception as error:
                self.error = str(error)
