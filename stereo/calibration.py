from dataclasses import dataclass, replace
from pathlib import Path
import json

import cv2
import numpy as np
from .calibration_quality import pose_diversity,view_geometry


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Не удалось сохранить {path}")
    encoded.tofile(str(path))


@dataclass
class Calibration:
    size: tuple
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    roi1: np.ndarray
    roi2: np.ndarray
    rms: float
    epipolar_error: float
    signature: str

    @property
    def baseline(self):
        return float(np.linalg.norm(self.T))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            values = {k: data[k] for k in cls.__dataclass_fields__}
        values["size"] = tuple(int(x) for x in values["size"])
        for key in ("rms", "epipolar_error"):
            values[key] = float(values[key])
        values["signature"] = str(values["signature"])
        calibration = cls(**values)
        if len(calibration.size) != 2 or min(calibration.size) <= 0:
            raise ValueError("Некорректный размер калибровки.")
        expected = {"K1": (3, 3), "K2": (3, 3), "R": (3, 3),
                    "R1": (3, 3), "R2": (3, 3), "P1": (3, 4),
                    "P2": (3, 4), "Q": (4, 4)}
        for key, shape in expected.items():
            if values[key].shape != shape or not np.isfinite(values[key]).all():
                raise ValueError(f"Некорректная матрица {key}.")
        if not np.isfinite(calibration.T).all() or calibration.T.size != 3:
            raise ValueError("Некорректная база стереокамеры.")
        if calibration.baseline <= 0 or calibration.T.reshape(-1)[0] >= 0:
            raise ValueError("Ожидается горизонтальная стереопара: левая камера физически слева.")
        return calibration

    def rectification_maps(self):
        maps = []
        for K, D, R, P in ((self.K1, self.D1, self.R1, self.P1),
                           (self.K2, self.D2, self.R2, self.P2)):
            maps.append(cv2.initUndistortRectifyMap(K, D, R, P, self.size, cv2.CV_32FC1))
        return maps

    def scaled_to(self, size, alpha=None):
        """Scale calibrated pixel coordinates, retaining the metric extrinsics."""
        if tuple(size) == tuple(self.size) and alpha is None:
            return self
        sx, sy = size[0] / self.size[0], size[1] / self.size[1]
        K1, K2 = self.K1.copy(), self.K2.copy()
        for K in (K1, K2):
            K[0] *= sx
            K[1] *= sy
        if alpha is None:
            P1, P2 = self.P1.copy(), self.P2.copy()
            for P in (P1, P2):
                P[0] *= sx
                P[1] *= sy
            Q = sx * self.Q @ np.diag([1 / sx, 1 / sy, 1 / sx, 1.])
            def scale_roi(roi):
                return np.rint(np.asarray(roi) * [sx, sy, sx, sy]).astype(int)
            return replace(self, size=tuple(size), K1=K1, K2=K2, P1=P1, P2=P2, Q=Q,
                           roi1=scale_roi(self.roi1), roi2=scale_roi(self.roi2))
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, self.D1, K2, self.D2, tuple(size), self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0 if alpha is None else alpha)
        return replace(self, size=tuple(size), K1=K1, K2=K2, R1=R1, R2=R2, P1=P1,
                       P2=P2, Q=Q, roi1=np.array(roi1), roi2=np.array(roi2))


class CalibrationSession:
    def __init__(self, directory, columns=9, rows=6, square_mm=25.0):
        if columns < 3 or rows < 3 or columns == rows or square_mm <= 0:
            raise ValueError("Нужна прямоугольная доска, например 9 × 6 внутренних углов; клетка > 0 мм.")
        self.directory = Path(directory)
        self.pattern = (columns, rows)
        self.square_mm = square_mm
        self.points = []
        self.size = None
        self.signature = None
        self.source_info = {}
        self.last_report = None

    def add(self, left, right, signature):
        size = (left.shape[1], left.shape[0])
        if left.shape != right.shape:
            raise ValueError("Размеры изображений должны совпадать.")
        if self.size is not None and (size != self.size or signature != self.signature):
            raise ValueError("Подключение изменилось. Начните новую серию калибровки.")
        corners = []
        for frame in (left, right):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, pts = cv2.findChessboardCornersSB(gray, self.pattern,
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not found:
                raise ValueError("Все внутренние углы доски должны быть видны в ОБЕИХ камерах.")
            corners.append(pts.astype(np.float32))
        if self.points:
            differences=[float(np.mean(np.linalg.norm(corners[0]-p[0],axis=2))) for p in self.points]
            nearest=int(np.argmin(differences))
            current=view_geometry(corners[0],self.pattern,size)
            old=view_geometry(self.points[nearest][0],self.pattern,size)
            perspective_change=np.linalg.norm(np.asarray(current['perspective_direction'])-old['perspective_direction'])
            if differences[nearest] < 12 and perspective_change < .025:
                raise ValueError("Этот ракурс уже есть. Передвиньте или наклоните доску.")
        self.size, self.signature = size, signature
        index = len(self.points) + 1
        write_image(self.directory / f"left_{index:03d}.png", left)
        write_image(self.directory / f"right_{index:03d}.png", right)
        self.points.append(corners)
        np.savez_compressed(self.directory / "detections.npz",
                            left=np.asarray([p[0] for p in self.points]),
                            right=np.asarray([p[1] for p in self.points]))
        metadata = {"pattern_inner_corners": self.pattern, "square_mm": self.square_mm,
                    "size": self.size, "signature": signature, "pairs": len(self.points)}
        metadata["capture"] = self.source_info
        metadata['pose_quality']=self.guidance()
        (self.directory / "session.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
        return len(self.points)

    def guidance(self):
        return pose_diversity(self.points,self.pattern,self.size)

    def write_report(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/"quality_report.json").write_text(
            json.dumps(self.last_report,ensure_ascii=False,indent=2),encoding="utf-8")

    @classmethod
    def restore(cls, directory, status=lambda message: None):
        directory = Path(directory)
        metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        session = cls(directory, *metadata["pattern_inner_corners"], metadata["square_mm"])
        session.size, session.signature = tuple(metadata["size"]), metadata["signature"]
        session.source_info = metadata.get("capture", {})
        cache = directory / "detections.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                left, right = data["left"], data["right"]
            if len(left) == len(right) == metadata["pairs"] and np.isfinite(left).all() and np.isfinite(right).all():
                session.points = list(zip(left, right))
                return session
        for index in range(1, metadata["pairs"] + 1):
            status(f"Проверка сохранённой серии: пара {index} из {metadata['pairs']}…")
            corners = []
            for side in ("left", "right"):
                image = read_image(directory / f"{side}_{index:03d}.png")
                if image is None or (image.shape[1], image.shape[0]) != session.size:
                    raise ValueError(f"Не удалось прочитать {side}_{index:03d}.png с исходным размером.")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                scale = min(1., 960 / gray.shape[1])
                small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
                found, pts = cv2.findChessboardCornersSB(small, session.pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
                if not found:
                    raise ValueError(f"Не найдена доска в {side}_{index:03d}.png.")
                pts = (pts / scale).astype(np.float32)
                pts = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .001))
                corners.append(pts)
            session.points.append(corners)
        np.savez_compressed(cache, left=np.asarray([p[0] for p in session.points]),
                            right=np.asarray([p[1] for p in session.points]))
        return session

    def solve(self, progress=None):
        if len(self.points) < 15:
            raise ValueError("Нужно минимум 15 разных ракурсов; рекомендуется 20–30.")
        diversity=self.guidance()
        if diversity['tilted']<diversity['required_tilted']:
            message=(f"Недостаточно наклонов доски: {diversity['tilted']} из {len(self.points)} ракурсов "
                "имеют заметную перспективу; нужно минимум 5. Приближение и отдаление фронтальной доски "
                "не заменяют наклоны. Добавьте положения с наклоном на 20–35° влево/вправо и вверх/вниз, "
                "замирая перед каждой парой. Пары уже сохранены; можно дополнить эту серию.")
            self.last_report=dict(total_pairs=len(self.points),used_pairs=0,excluded_pairs=[],accepted=False,
                rejection_reason="insufficient_pose_diversity",message=message,pose_quality=diversity)
            self.write_report()
            raise ValueError(message)
        indices = list(range(len(self.points)))
        excluded = []
        for attempt in range(3):
            if progress:
                progress(f"Калибровка: проход {attempt + 1}, {len(indices)} пар…")
            try:
                calibration, views = self._fit([self.points[i] for i in indices], progress)
            except Exception as error:
                self.last_report=dict(total_pairs=len(self.points),used_pairs=len(indices),accepted=False,
                    rejection_reason="calibration_fit_failed",message=str(error),pose_quality=diversity)
                self.write_report()
                raise
            for index, view in zip(indices, views):
                view["pair"] = index + 1
            self.last_report = {"total_pairs": len(self.points), "used_pairs": len(indices),
                                "excluded_pairs": [i + 1 for i in excluded], "views": views,
                                "rms_px": calibration.rms, "vertical_error_px": calibration.epipolar_error,
                                "pose_quality":diversity,
                                "accepted": bool(calibration.rms <= 2 and calibration.epipolar_error <= 1)}
            self.write_report()
            if self.last_report["accepted"]:
                return calibration
            if attempt == 2:
                break
            scores = np.array([v["mono_rms_px"] + v["vertical_median_px"] for v in views])
            median = np.median(scores)
            mad = np.median(np.abs(scores - median))
            outliers = np.where(scores > max(3., median + 3.5 * max(mad, .15)))[0]
            budget = min(len(self.points) // 4 - len(excluded), len(indices) - 15)
            if not len(outliers) or budget <= 0:
                break
            remove = set(outliers[np.argsort(scores[outliers])[::-1]][:budget])
            excluded.extend(index for position, index in enumerate(indices) if position in remove)
            indices = [index for position, index in enumerate(indices) if position not in remove]
        raise ValueError(f"Калибровка отклонена: RMS {calibration.rms:.2f} px, "
                         f"ошибка по строкам {calibration.epipolar_error:.2f} px. "
                         f"Использовано {len(indices)} из {len(self.points)} пар. "
                         "Держите доску неподвижно при съёмке каждой пары; проверьте резкость, плоскость доски "
                         "и разные наклоны. Подробности сохранены в quality_report.json внутри серии.")

    def _fit(self, pairs, progress=None):
        obj = np.zeros((self.pattern[0] * self.pattern[1], 3), np.float32)
        obj[:, :2] = np.mgrid[:self.pattern[0], :self.pattern[1]].T.reshape(-1, 2)
        obj *= self.square_mm / 1000.0  # All reconstructed coordinates are metres.
        objects = [obj.copy() for _ in pairs]
        left = [p[0] for p in pairs]
        right = [p[1] for p in pairs]
        if progress: progress("Калибровка: рассчитываю левый объектив…")
        _, K1, D1, rv1, tv1 = cv2.calibrateCamera(objects, left, self.size, None, None)
        if progress: progress("Калибровка: рассчитываю правый объектив…")
        _, K2, D2, rv2, tv2 = cv2.calibrateCamera(objects, right, self.size, None, None)
        if progress: progress("Калибровка: взаимное положение камер и проверка выравнивания…")
        rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
            objects, left, right, K1, D1, K2, D2, self.size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            flags=cv2.CALIB_FIX_INTRINSIC)
        if T[0, 0] >= 0:
            raise ValueError("Левая и правая камеры, вероятно, перепутаны. Поменяйте их и снимите серию заново.")
        if abs(T[1, 0]) > abs(T[0, 0]):
            raise ValueError("Эта версия рассчитана на горизонтальную стереопару.")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, self.size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        errors, views = [], []
        for i, (a, b) in enumerate(pairs):
            pa, _ = cv2.projectPoints(obj, rv1[i], tv1[i], K1, D1)
            pb, _ = cv2.projectPoints(obj, rv2[i], tv2[i], K2, D2)
            mono_error = float(np.sqrt((np.mean((pa - a) ** 2) + np.mean((pb - b) ** 2))))
            a = cv2.undistortPoints(a, K1, D1, R=R1, P=P1)
            b = cv2.undistortPoints(b, K2, D2, R=R2, P=P2)
            vertical = np.abs(a[:, 0, 1] - b[:, 0, 1])
            errors.extend(vertical)
            views.append({"mono_rms_px": mono_error, "vertical_median_px": float(np.median(vertical)),
                          "vertical_p95_px": float(np.percentile(vertical, 95))})
        error = float(np.median(errors))
        if not np.isfinite(rms) or not np.isfinite(error):
            raise ValueError("Вырожденная калибровка. Нужны разные наклоны и расстояния до доски.")
        calibration = Calibration(self.size, K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q,
                                  np.array(roi1), np.array(roi2), float(rms), error, self.signature)
        return calibration, views
