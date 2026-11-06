from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread, Event, Lock
from collections import deque
from types import SimpleNamespace
import json
import time

import numpy as np
import cv2

from .camera import CapturePump
from .calibration import Calibration, CalibrationSession, write_image
from .depth import DepthEngine, measure
from .mapping import VisualOdometry, VoxelMap, top_view, export_ply
from .model3d import surface_from_frame, export_surface
from .map_preview import MapPreview
from .map_integrator import MapIntegrator
from .ghost_filter import MultiViewConfirmation
from .cloud_orientation import initial_camera_pose


def stamp():
    return datetime.now(timezone(timedelta(hours=3))).strftime("%Y%m%d_%H%M%S_%f")


class CameraWorker(Thread):
    def __init__(self, root, config, pattern, square_mm, max_depth, disparities,
                 quality="balanced", processing_width=640, temporal=True, near_priority=True, voxel_mm=5., wide_view=True,
                 map_stride=2, preview_points=250000, estimate_surfaces=True, near_minimum=.10,acceleration="auto",ghost_filter="normal"):
        super().__init__(daemon=True)
        self.root, self.config = Path(root), config
        self.log_path=self.root/"data"/"logs"/f"events_{stamp()}.jsonl"
        self.log_lock=Lock()
        self.pattern, self.square_mm = pattern, square_mm
        self.max_depth, self.disparities = max_depth, disparities
        self.quality, self.processing_width = quality, processing_width
        self.temporal, self.near_priority = temporal, near_priority
        self.voxel_mm = voxel_mm
        self.wide_view = wide_view
        self.map_stride, self.preview_points = int(map_stride),int(preview_points)
        self.estimate_surfaces, self.near_minimum = estimate_surfaces,near_minimum
        self.acceleration = acceleration
        self.ghost_filter=ghost_filter
        self.map_renderer = None
        self.map_integrator = None
        self.map_generation = 0
        self.map_render_ms = 0.
        self.commands, self.events, self.frames = Queue(), Queue(), Queue(maxsize=1)
        self.previews = Queue(maxsize=1)
        self.stop_event = Event()
        self.calibration = self.engine = self.odometry = self.frame = None
        self.mapping = False
        self.depth_enabled = False
        self.snapshot_engine = None
        self.capture_pump = None
        self.last_capture_generation = 0
        self.next_map_render = 0.
        self.cloud = VoxelMap(voxel=voxel_mm / 1000.)
        self.estimated_cloud = VoxelMap(voxel=max(.005, voxel_mm / 1000.), maximum=200000)
        self.collector = None
        self.raw = None
        self.frame_raw = None
        self.capture_info = {}
        self.pending_capture = None
        self.export_thread = None
        self.calibration_job=None
        self.calibration_busy=False
        self.scan=None
        self.received_frames = self.depth_frames = 0
        self.processed_frames = 0
        self.integrated_frames = self.mapping_attempts = 0
        self.frame_times, self.map_times = deque(), deque()
        self.map_image = top_view(np.empty((0, 3)))

    def emit(self, kind, value):
        if kind in ('error','calibration_error'):
            record=dict(time=datetime.now(timezone(timedelta(hours=3))).isoformat(),kind=kind,message=str(value),
                camera=self.config.__dict__,capture=self.capture_info,
                calibration_series=str(self.collector.directory) if self.collector else None)
            try:
                with self.log_lock:
                    self.log_path.parent.mkdir(parents=True,exist_ok=True)
                    with self.log_path.open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
            except OSError:
                pass
        self.events.put((kind, value))

    def close_maps(self):
        self.cloud.close()
        self.estimated_cloud.close()

    def new_series(self):
        self.collector = CalibrationSession(self.root / "data" / f"calibration_{stamp()}",
                                            *self.pattern, self.square_mm)
        self.collector.source_info = self.capture_info
        self.emit("status", "Новая серия: 0 пар. Доска должна быть неподвижна при съёмке каждой пары.")
        self.emit("count", 0)
        self.emit("calibration_guidance",self.collector.guidance())

    def apply_calibration(self, calibration):
        if calibration.signature != self.config.signature():
            raise ValueError("Калибровка другого подключения или порядка камер. "
                             "Выберите прежние индексы и порядок либо снимите новую серию.")
        if self.raw and calibration.size != (self.raw[0].shape[1], self.raw[0].shape[0]):
            raise ValueError("Размер кадров отличается от калибровки. Верните исходное разрешение.")
        engine = DepthEngine(calibration, self.max_depth, self.disparities, self.quality,
                             self.processing_width, self.temporal, self.near_priority, self.wide_view,
                             estimate_surfaces=self.estimate_surfaces,near_minimum=self.near_minimum,
                             acceleration=self.acceleration,parallel=True,ghost_filter=self.ghost_filter)
        if self.engine:
            self.engine.close()
        self.calibration, self.engine = calibration, engine
        self.snapshot_engine = None
        self.frame = None
        self.reset_map()
        self.emit("calibration", calibration)

    def reset_map(self):
        if self.map_integrator:
            self.map_integrator.stop_event.set()
            self.map_integrator.join()
            self.map_integrator=None
        if hasattr(self,"cloud"):
            self.cloud.close()
            self.estimated_cloud.close()
        self.map_generation += 1
        directory = self.root/"data"/"maps"/stamp()
        self.cloud = VoxelMap(voxel=self.voxel_mm/1000., maximum=self.preview_points,
                              path=directory/"measured.sqlite",stride=self.map_stride)
        self.estimated_cloud = VoxelMap(voxel=max(.005,self.voxel_mm/1000.), maximum=self.preview_points//2,
                                       path=directory/"estimated.sqlite",stride=max(2,self.map_stride))
        if self.engine:
            self.engine.calibration.save(directory/"calibration.npz")
            self.calibration.save(directory/"raw_calibration.npz")
        (directory/"session.json").write_text(json.dumps({"units":"metres","voxel_mm":self.voxel_mm,
            "ghost_filter":self.ghost_filter,"confirmation_observations":3 if self.ghost_filter=='strict' else 2 if self.ghost_filter!='off' else 1,
            "axes":"initial raw left camera: X right, Y down, Z forward",
            "start_orientation":"camera pair held level by user; gravity not measured",
            "initial_rectified_camera_to_world":initial_camera_pose(self.engine.calibration).tolist() if self.engine else None,
            "map_stride":self.map_stride,
            "measured_store":"measured.sqlite","estimated_store":"estimated.sqlite"},indent=2),encoding="utf-8")
        self.odometry = VisualOdometry(self.engine.calibration.P1[:, :3],
            initial_pose=initial_camera_pose(self.engine.calibration)) if self.engine else None
        if self.odometry and self.is_alive():
            confirmation=MultiViewConfirmation(self.engine.calibration.P1[:,:3],self.calibration.baseline,
                observations=3 if self.ghost_filter=='strict' else 2) if self.ghost_filter!='off' else None
            self.map_integrator=MapIntegrator(self.odometry,self.cloud,self.estimated_cloud,confirmation)
            self.map_integrator.start()
        self.integrated_frames = self.mapping_attempts = 0
        self.map_times.clear()
        self.emit("map_count", 0)

    def command(self, name, value=None):
        if self.calibration_busy and name in ('calibrate','capture','new_series','board_settings','load','restore_series','precision','scan_start'):
            raise ValueError("Расчёт калибровки ещё идёт; дождитесь результата. Видео продолжает работать.")
        if self.scan and name not in ('scan_stop','scan_extend','distance_report'):
            raise ValueError("Идёт объёмный снимок. Завершите его в окне накопления перед этой операцией.")
        if name == "depth":
            if value and self.engine is None:
                self.emit("depth_enabled", False)
                raise ValueError("Для расчёта глубины сначала нужна калибровка.")
            self.depth_enabled = bool(value)
            self.emit("depth_enabled", self.depth_enabled)
            if not self.depth_enabled and self.mapping:
                self.mapping = False
                self.emit("mapping", False)
            self.emit("status", "Расчёт глубины включён" if value else "Расчёт глубины приостановлен; видео продолжает работать")
        elif name == "surfaces":
            self.estimate_surfaces = bool(value)
            if self.engine:
                self.engine.estimate_surfaces = self.estimate_surfaces
            self.emit("status", "Оценка поверхностей включена" if value else "Оценка поверхностей выключена; нагрузка уменьшена")
        elif name == "board_settings":
            columns, rows, square_mm = value
            if (columns, rows) != self.pattern or abs(square_mm - self.square_mm) > 1e-5:
                self.pattern, self.square_mm = (columns, rows), square_mm
                self.new_series()
                self.emit("status", "Размер доски изменён. Начата новая серия; прежние снимки сохранены.")
        elif name == "measure_snapshot":
            if self.engine is None or value["calibration_token"] != id(self.engine):
                raise ValueError("Калибровка изменилась или отсутствует. Повторите измерение.")
            if self.snapshot_engine is None:
                self.snapshot_engine = DepthEngine(self.calibration, self.max_depth, self.disparities,
                    self.quality, self.processing_width, temporal=False, near_priority=self.near_priority,
                    wide_view=self.wide_view,estimate_surfaces=False,near_minimum=self.near_minimum)
            snapshot = self.snapshot_engine.compute(*value["raw"])
            self.emit("snapshot_measurement", measure(snapshot, *value["point"]))
        elif name == "distance_report":
            if self.engine is None:
                raise ValueError("Для оценки дистанции сначала нужна калибровка.")
            self.emit("distance_report", self.engine.distance_report())
        elif name == "precision":
            if self.engine is None or self.raw is None:
                raise ValueError("Для измерения листа нужна калибровка и видеопара.")
            from .precision import make_precision_snapshot
            self.mapping = False
            self.emit("mapping", False)
            if self.capture_pump is None:
                raise ValueError("Для нового точного снимка требуется подключённая камера.")
            self.emit("status", "Снимок для листа: остановите камеру и растение. Собираю 5 пар…")
            pairs = []
            for _ in range(5):
                capture = self.capture_pump.take(self.stop_event)
                if capture is None:
                    return
                pairs.append(capture[0])
            result = make_precision_snapshot(self.calibration, pairs, self.wide_view,
                                            lambda text: self.emit("status", text))
            self.emit("precision", result)
            self.emit("status", "Подробный снимок готов. Измерьте размер и проверьте эталон.")
        elif name == "capture_after":
            self.pending_capture = float(value)
        elif name == "cancel_capture":
            self.pending_capture = None
        elif name == "capture":
            if self.raw is None:
                raise ValueError("Сначала дождитесь изображения.")
            count = self.collector.add(*self.raw, self.config.signature())
            self.emit("count", count)
            guidance=self.collector.guidance()
            self.emit("calibration_guidance",guidance)
            self.emit("status", f"Сохранено {count} пар; с наклоном {guidance['tilted']}. "
                "Добавьте наклоны доски влево/вправо и вверх/вниз; затем снимки по краям кадра.")
            self.emit("capture_finished", None)
        elif name == "new_series":
            self.new_series()
        elif name == "calibrate":
            if len(self.collector.points)<15:
                raise ValueError("Для калибровки нужно минимум 15 принятых пар. FPS на это количество не влияет.")
            self.calibration_busy=True
            self.emit("calibration_busy",True)
            self.depth_enabled=self.mapping=False
            self.emit("depth_enabled",False)
            self.emit("mapping",False)
            collector=self.collector
            def solve():
                def progress(text):
                    if self.stop_event.is_set():raise RuntimeError("Расчёт остановлен после отключения камер.")
                    self.emit("status",text)
                try:
                    result=collector.solve(progress)
                    self.commands.put(("_calibration_result",(collector,result,None)))
                except Exception as error:
                    self.commands.put(("_calibration_result",(collector,None,str(error))))
            self.calibration_job=Thread(target=solve,daemon=True,name="camera-calibration")
            self.calibration_job.start()
            self.emit("status", "Расчёт калибровки начат; видео работает. FPS не входит в расчёт геометрии.")
        elif name == "_calibration_result":
            self.calibration_busy=False
            self.emit("calibration_busy",False)
            collector,calibration,error=value
            if collector is not self.collector:
                raise ValueError("Серия изменилась во время расчёта. Повторите калибровку.")
            if error:
                self.emit("calibration_error",error)
                return
            self.apply_calibration(calibration)
            self.depth_enabled = True
            self.emit("depth_enabled", True)
            calibration.save(self.collector.directory / "calibration.npz")
            calibration.save(self.root / "data" / "calibration.npz")
            self.emit("status", f"Калибровка сохранена. База {calibration.baseline * 1000:.1f} мм; "
                      f"RMS {calibration.rms:.2f} px; использовано {self.collector.last_report['used_pairs']} "
                      f"из {self.collector.last_report['total_pairs']} пар. Теперь проверьте известное расстояние.")
        elif name == "scan_start":
            if not self.engine or self.raw is None:
                raise ValueError("Для объёмного снимка нужна калибровка этого разрешения.")
            from .object_scan import ObjectScan
            self.mapping=self.depth_enabled=False
            self.emit("mapping",False)
            self.emit("depth_enabled",False)
            self.emit("status","Подготавливаю подробное накопление; двигайте камеру медленно…")
            options=dict(value)
            options.setdefault('ghost_filter',self.ghost_filter if self.ghost_filter in ('normal','strict') else 'normal')
            self.scan=ObjectScan(self.root/"output"/f"scan_{stamp()}",self.calibration,
                wide_view=self.wide_view,acceleration=self.acceleration,max_depth=options.pop('max_depth',self.max_depth),
                near_minimum=options.pop('near_minimum',self.near_minimum),**options)
            self.emit("scan_started",str(self.scan.directory))
        elif name == "scan_extend":
            if self.scan:
                self.scan.maximum=0
                self.scan.automatic=False
                self.emit("status","Накопление продолжится до нажатия «Завершить и сохранить».")
        elif name == "scan_stop":
            self.finish_scan()
        elif name == "load":
            self.apply_calibration(Calibration.load(value))
            self.depth_enabled = True
            self.emit("depth_enabled", True)
            self.emit("status", "Калибровка загружена; прежняя карта сброшена.")
        elif name == "restore_series":
            self.emit("status", "Загружаю сохранённые пары и ищу углы доски…")
            collector = CalibrationSession.restore(value, lambda message: self.emit("status", message))
            if collector.signature != self.config.signature() or collector.size != self.calibration_size():
                raise ValueError("Серия снята при другом разрешении или порядке камер.")
            self.collector = collector
            self.emit("count", len(collector.points))
            self.emit("calibration_guidance",collector.guidance())
            self.emit("status", "Серия загружена. Можно повторно рассчитать калибровку.")
        elif name == "mapping":
            if value and self.engine is None:
                self.emit("mapping", False)
                raise ValueError("Для накопления карты сначала нужна калибровка.")
            self.mapping = bool(value)
            if value and not self.depth_enabled:
                self.depth_enabled = True
                self.emit("depth_enabled", True)
            self.emit("mapping", self.mapping)
            self.emit("status", "Накопление новой карты включено" if value else "Накопление карты остановлено")
        elif name == "reset":
            self.reset_map()
            self.emit("status", "Карта сброшена. Новая система координат начинается здесь.")
        elif name == "export":
            if self.export_thread and self.export_thread.is_alive():
                self.emit("status","Сохранение предыдущего снимка ещё идёт…")
                return
            if self.frame is None:
                raise ValueError("Сначала нужна калибровка и карта глубины.")
            state=SimpleNamespace(**self.__dict__)
            state.emit=self.emit
            def save():
                try:
                    CameraWorker.export(state)
                except Exception as error:
                    self.emit("error",f"Сохранение карты: {error}")
            self.export_thread=Thread(target=save,daemon=True)
            self.export_thread.start()
            self.emit("status","Сохраняю полный архив в PLY; видео и глубина продолжают работать…")
        elif name == "view":
            # An archive-backed measured cloud is reloaded in full by the viewer.
            # In-memory/composite clouds need every point here: there is no source
            # file from which the editor could recover a discarded preview sample.
            preview_limit = self.preview_points if self.cloud.path and not value else None
            points, colors = self.cloud.arrays(preview_limit)
            if not len(points) and self.mapping:
                raise ValueError("Карта ещё ждёт повторного подтверждения. Подержите камеру неподвижно около секунды.")
            if not len(points) and self.frame is not None:
                points = self.frame.points[self.frame.valid]
                colors = self.frame.left[self.frame.valid][:, ::-1]
            if value:
                estimates, estimate_colors = self.estimated_cloud.arrays()
                if not len(estimates) and self.frame is not None and self.frame.estimated_points is not None:
                    mask = np.isfinite(self.frame.estimated_points).all(axis=2)
                    estimates = self.frame.estimated_points[mask]
                    estimate_colors = np.uint8(self.frame.left[mask][:, ::-1].astype(np.float32) * .5 + 64)
                if len(estimates):
                    points = np.vstack((points, estimates))
                    colors = np.vstack((colors, estimate_colors))
            if not len(points):
                raise ValueError("Сначала нужна карта глубины с достоверными точками.")
            self.emit("map_model",dict(points=points.copy(),colors=colors.copy(),
                title="Карта с оценкой поверхностей" if value else "Измеренная 3D-карта",
                total=max(len(self.cloud)+ (len(self.estimated_cloud) if value else 0),len(points)),
                preview_limit=self.preview_points,archive=str(self.cloud.path) if self.cloud.path and len(self.cloud) and not value else None))

    def calibration_size(self):
        return (self.raw[0].shape[1], self.raw[0].shape[0]) if self.raw else None

    def finish_scan(self,reason="Сбор остановлен пользователем"):
        if self.scan is None: return
        scan,self.scan=self.scan,None
        self.emit("status","Сохраняю объёмный снимок и полную 3D-модель…")
        try:
            result=scan.finish(reason)
            self.emit("scan_finished",result)
            self.emit("status",f"Объёмный снимок сохранён: {result['directory']}")
        finally:
            scan.close()

    def export(self):
        if self.frame is None:
            raise ValueError("Сначала нужна калибровка и карта глубины.")
        directory = self.root / "output" / stamp()
        directory.mkdir(parents=True)
        f = self.frame
        if self.frame_raw is not None:
            write_image(directory / "raw_left.png", self.frame_raw[0])
            write_image(directory / "raw_right.png", self.frame_raw[1])
        write_image(directory / "left.png", f.left)
        write_image(directory / "right.png", f.right)
        write_image(directory / "depth_color.png", f.color)
        if f.preview_color is not None:
            write_image(directory / "depth_preview.png", f.preview_color)
            write_image(directory / "depth_interpolated_mask.png", np.uint8(f.preview_filled) * 255)
        write_image(directory / "map.png", self.map_image)
        np.save(directory / "depth_metres.npy", f.points[:, :, 2])
        if f.valid.any():
            export_ply(directory / "visible_cloud.ply", f.points[f.valid], f.left[f.valid][:, ::-1])
            vertices, rgb, faces = surface_from_frame(f)
            export_surface(directory / "visible_surface.ply", vertices, rgb, faces)
        if f.estimated_points is not None:
            estimated = np.isfinite(f.estimated_points).all(axis=2)
            if estimated.any():
                export_ply(directory / "estimated_surfaces.ply", f.estimated_points[estimated],
                           f.left[estimated][:, ::-1])
        map_count = len(self.cloud)
        if map_count:
            map_count = self.cloud.export(directory / "accumulated_cloud.ply")
            np.savetxt(directory / "trajectory.csv", self.cloud.trajectory, delimiter=",",
                       header="x_m,y_m,z_m", comments="")
        estimate_count = len(self.estimated_cloud)
        if estimate_count:
            estimate_count = self.estimated_cloud.export(directory / "accumulated_estimates.ply")
        self.engine.calibration.save(directory / "calibration.npz")
        self.calibration.save(directory / "raw_calibration.npz")
        metadata = {"units": "metres", "axes": "X right, Y down, Z forward",
                    "map_origin": "initial raw left-camera frame after map reset",
                    "start_orientation": "camera pair held level by user; gravity not measured",
                    "initial_rectified_camera_to_world": initial_camera_pose(self.engine.calibration).tolist(),
                    "visible_cloud_axes": "current rectified left camera: X right, Y down, Z forward",
                    "method": self.engine.acceleration+" + ORB/PnP odometry; no loop closure",
                    "max_depth_m": self.max_depth, "valid_fraction": float(f.valid.mean()),
                    "displayed_map": "accumulated" if self.mapping else "current_view",
                    "source_size": self.calibration.size, "processing_size": self.engine.calibration.size,
                    "quality": self.quality, "near_priority": self.near_priority, "wide_view": self.wide_view,
                    "ghost_filter":self.ghost_filter,"map_confirmation_observations":3 if self.ghost_filter=='strict' else 2 if self.ghost_filter!='off' else 1,
                    "voxel_size_m": self.cloud.voxel,
                    "capture": self.capture_info, "received_stereo_pairs": self.received_frames,
                    "depth_frames": self.depth_frames, "map_frames": self.integrated_frames,
                    "mapping_rejected_frames": self.mapping_attempts - self.integrated_frames,
                    "baseline_m": self.calibration.baseline, "camera": self.config.__dict__,
                    "map_points": map_count, "map_limit": None,"preview_point_limit":self.preview_points,
                    "map_archive":str(self.cloud.path),"map_stride":self.map_stride,
                    "estimated_map_points": estimate_count,
                "hardware_synchronization": "not established by this application",
                    "depth_color_range_m": f.color_limits,
                    "vertical_alignment_error_px": f.alignment_error,
                    "depth_quality_warning": f.quality_warning,
                    "preview_interpolation": "small gaps and bounded planar estimates; excluded from measured depth and accumulated map; estimated_surfaces.ply is separate",
                    "preview_interpolated_fraction": float(f.preview_filled.mean()) if f.preview_filled is not None else 0.}
        (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
        self.emit("status", f"Сохранено: {directory}")
        self.emit("exported", str(directory))

    def run(self):
        camera = None
        cv2.ocl.setUseOpenCL(False)
        try:
            self.map_renderer = MapPreview()
            self.map_renderer.start()
            camera = CapturePump(self.config, lambda message: self.emit("status", message))
            camera.previews = self.previews
            self.capture_pump = camera
            camera.start()
            first = camera.take(self.stop_event)
            if first is None:
                return
            self.raw, captured_at, sequence = first
            self.capture_info = camera.info
            self.last_capture_generation = camera.stream_generation
            log = self.root / "data" / "logs" / f"capture_{stamp()}.json"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps(self.capture_info, ensure_ascii=False, indent=2), encoding="utf-8")
            self.new_series()
            default = self.root / "data" / "calibration.npz"
            if default.exists():
                try:
                    self.apply_calibration(Calibration.load(default))
                except Exception as error:
                    self.emit("status", f"Просмотр включён; калибровка не применена: {error}")
            self.emit("connected", (self.raw[0].shape[1], self.raw[0].shape[0]))
            fps = 0.0
            while not self.stop_event.is_set():
                capture = camera.take(self.stop_event)
                if capture is None:
                    break
                old_size = self.calibration_size()
                self.raw, captured_at, sequence = capture
                self.capture_info = camera.info
                if camera.stream_generation != self.last_capture_generation:
                    if self.scan:
                        self.finish_scan("Видеопоток прервался. Сохранены ранее подтверждённые точки.")
                    self.last_capture_generation = camera.stream_generation
                    self.frame_times.clear()
                    if self.engine:
                        self.engine.previous = None
                    if self.mapping:
                        self.mapping = False
                        self.emit("mapping", False)
                    if old_size != self.calibration_size():
                        self.engine = self.calibration = self.snapshot_engine = self.frame = None
                        self.depth_enabled = False
                        self.emit("depth_enabled", False)
                        self.emit("calibration_invalidated", self.calibration_size())
                        self.new_series()
                    self.emit("status", "Поток восстановлен. Проверьте изображение; накопление карты приостановлено.")
                self.received_frames = camera.count
                self.processed_frames += 1
                try:
                    while True:
                        name, value = self.commands.get_nowait()
                        try:
                            self.command(name, value)
                        except Exception as error:
                            self.emit("error", str(error))
                except Empty:
                    pass
                if self.pending_capture is not None and captured_at >= self.pending_capture:
                    self.pending_capture = None
                    self.emit("capture_taken", None)
                    try:
                        self.command("capture")
                    except Exception as error:
                        self.emit("error", str(error))
                tracking = "Накопление выключено"
                depth_ms = 0
                tracking_ms = insert_ms = 0.
                did_compute = False
                if self.scan:
                    try:
                        _,report=self.scan.process(self.raw)
                        self.emit("scan_progress",report)
                        if report['reason']:
                            self.finish_scan()
                    except Exception as error:
                        self.emit("error",f"Объёмный снимок: {error}")
                        self.finish_scan(f"Сбор прерван: {error}")
                elif self.engine and self.depth_enabled and not self.calibration_busy:
                    depth_started = time.perf_counter()
                    self.frame = self.engine.compute(*self.raw)
                    self.frame_raw = self.raw
                    self.depth_frames += 1
                    did_compute = True
                    depth_ms = (time.perf_counter() - depth_started) * 1000
                    local = self.frame.points[::4, ::4].reshape(-1, 3)
                    if self.mapping:
                        mapper=self.map_integrator
                        if mapper.error:
                            self.mapping=False
                            self.emit("mapping",False)
                            self.emit("error",mapper.status)
                        else:
                            mapper.submit(self.frame)
                        tracking=mapper.status
                        tracking_ms,insert_ms=mapper.tracking_ms,mapper.insert_ms
                        self.mapping_attempts=mapper.attempts
                        if mapper.integrated>self.integrated_frames:
                            self.map_times.append(time.perf_counter())
                        self.integrated_frames=mapper.integrated
                    if time.perf_counter() >= self.next_map_render:
                        if self.mapping:
                            self.map_renderer.request(self.map_generation,self.cloud,self.estimated_cloud,
                                                      self.odometry.world_pose,self.max_depth)
                        else:
                            self.map_image = top_view(local,extent=self.max_depth,
                                estimated=self.frame.estimated_points[::4,::4].reshape(-1,3)
                                if self.frame.estimated_points is not None else None)
                        self.next_map_render = time.perf_counter() + .5
                    try:
                        generation,image,render_ms = self.map_renderer.results.get_nowait()
                        if generation==self.map_generation and self.mapping:
                            self.map_image,self.map_render_ms = image,render_ms
                    except Empty:
                        pass
                now = time.perf_counter()
                self.received_frames = camera.count
                if did_compute:
                    self.frame_times.append(now)
                for history in (self.frame_times, self.map_times):
                    while history and history[0] < now - 2:
                        history.popleft()
                rate = lambda history: ((len(history) - 1) / max(history[-1] - history[0], .001)
                                        if len(history) > 1 else 0.)
                fps = rate(self.frame_times) if self.depth_enabled else 0.
                packet = {"raw": self.raw, "depth": self.frame, "map": self.map_image,
                          "fps": fps, "tracking": tracking, "map_count": len(self.cloud.cells),
                          "mapping": self.mapping, "received_frames": self.received_frames,
                          "depth_frames": self.depth_frames, "map_frames": self.integrated_frames,
                          "map_rejected": max(0,self.mapping_attempts-self.integrated_frames-(self.map_integrator.waiting_confirmation if self.map_integrator else 0)),
                          "map_confirmation_waits":self.map_integrator.waiting_confirmation if self.map_integrator else 0,
                          "unconfirmed_points":self.map_integrator.unconfirmed_points if self.map_integrator else 0,
                          "map_skipped":self.map_integrator.skipped if self.map_integrator else 0,
                          "map_fps": rate(self.map_times), "depth_ms": depth_ms,
                          "stage_ms":dict(self.engine.last_timings) if self.engine else {},
                          "acceleration":self.engine.acceleration if self.engine else None,
                          "acceleration_warning":self.engine.acceleration_warning if self.engine else None,
                          "tracking_ms":tracking_ms,"insert_ms":insert_ms,"map_render_ms":self.map_render_ms,
                          "map_archive":str(self.cloud.path) if self.cloud.path else None,
                          "capture": self.capture_info, "capture_fps": camera.fps,
                          "depth_enabled": self.depth_enabled, "restart_count": camera.restart_count,
                          "estimated_map_count": len(self.estimated_cloud.cells),
                          "discarded_stereo_pairs": max(0, sequence - self.processed_frames - 1),
                          "latency_ms": (now - captured_at) * 1000}
                try:
                    self.frames.put_nowait(packet)
                except Full:
                    try:
                        self.frames.get_nowait()
                    except Empty:
                        pass
                    self.frames.put_nowait(packet)
        except Exception as error:
            if not self.stop_event.is_set():
                self.emit("error", str(error))
        finally:
            if self.scan:
                try:
                    self.finish_scan("Камеры отключены. Сохранены ранее подтверждённые точки.")
                except Exception as error:
                    self.emit("error",f"Сохранение объёмного снимка: {error}")
            if self.map_integrator:
                self.map_integrator.stop_event.set()
                self.map_integrator.join()
            if self.map_renderer:
                self.map_renderer.stop_event.set()
                self.map_renderer.join(timeout=2)
            if camera:
                camera.stop_event.set()
                camera.join(timeout=5)
            self.close_maps()
            if self.engine:
                self.engine.close()
            if self.snapshot_engine:
                self.snapshot_engine.close()
            if self.export_thread:
                self.export_thread.join()
            self.emit("disconnected", None)
