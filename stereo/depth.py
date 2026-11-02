from dataclasses import dataclass
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from .surfaces import plane_estimates
from .ghost_filter import supported_disparity


@dataclass
class DepthFrame:
    left: np.ndarray
    right: np.ndarray
    disparity: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    color: np.ndarray
    units: str = "metres"
    method: str = "stereo"
    preview_color: np.ndarray | None = None
    preview_filled: np.ndarray | None = None
    color_limits: tuple | None = None
    alignment_error: float | None = None
    quality_warning: str | None = None
    estimated_points: np.ndarray | None = None


@lru_cache(maxsize=12)
def coordinates(shape):
    yy,xx = np.indices(shape,dtype=np.int32)
    yy.setflags(write=False)
    xx.setflags(write=False)
    return yy,xx


def consistency_mask(left, right, tolerance=1.5):
    """Reject occlusions and left/right disagreements; disparities are pixels."""
    height, width = left.shape
    yy, xx = coordinates((height, width))
    xr = np.rint(xx - left).astype(np.int32)
    inside = (xr >= 0) & (xr < width)
    sampled = right[yy, np.clip(xr, 0, width - 1)]
    return inside & (left > 0.5) & (sampled < 0) & (np.abs(left + sampled) <= tolerance)


def depth_colors(z, valid, minimum=0.05, maximum=8.0):
    scaled = np.zeros(z.shape, np.float32)
    scaled[valid] = (np.log(np.clip(z[valid], minimum, maximum)) - np.log(minimum)) / (
        np.log(maximum) - np.log(minimum))
    color = cv2.applyColorMap(np.uint8(np.clip(scaled, 0, 1) * 255), cv2.COLORMAP_TURBO)
    color[~valid] = (18, 22, 30)
    return color


def clean_disparity(disparity, valid, minimum_component=50):
    filtered = cv2.medianBlur(disparity.astype(np.float32), 3)
    # Median filtering may improve a valid sample, but never creates validity.
    stable = valid & (np.abs(filtered - disparity) <= 1.5)
    disparity = np.where(stable, filtered, disparity)
    # Separate small depth speckles even when they touch a large valid surface.
    encoded = np.where(valid, np.rint(disparity * 16), -16).astype(np.int16)
    cv2.filterSpeckles(encoded, -16, minimum_component, 24)
    return disparity, valid & (encoded != -16)


def texture_mask(gray, threshold=12.):
    # Blur first so sensor grain is not mistaken for useful scene texture.
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.)
    def variance(window):
        mean = cv2.boxFilter(smooth, cv2.CV_32F, (window, window), normalize=True)
        return cv2.boxFilter(smooth * smooth, cv2.CV_32F, (window, window), normalize=True) - mean * mean
    # Broader structure can support SGM on softly shaded skin or fabric.
    # Its higher threshold still rejects independent sensor grain.
    return (variance(9) >= threshold) | (variance(21) >= 3 * threshold)


def smooth_preview(disparity, valid, gray, iterations=3):
    """Fill tiny enclosed gaps for display only; never alter measured validity."""
    values = np.where(valid, disparity, 0).astype(np.float32)
    known = valid.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~valid).astype(np.uint8), 8)
    small = np.zeros(count, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= 256
    small[np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))] = False
    eligible = small[labels]
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(iterations):
        weight = known.astype(np.float32)
        neighbours = cv2.boxFilter(weight, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        mean = cv2.boxFilter(values * weight, -1, (3, 3), normalize=False,
                             borderType=cv2.BORDER_CONSTANT) / np.maximum(neighbours, 1)
        lo = cv2.erode(np.where(known, values, 1e6), kernel)
        hi = cv2.dilate(np.where(known, values, -1e6), kernel)
        luminance = cv2.boxFilter(gray.astype(np.float32) * weight, -1, (3, 3), normalize=False,
                                  borderType=cv2.BORDER_CONSTANT) / np.maximum(neighbours, 1)
        fill = (~known) & eligible & (neighbours >= 5)
        fill &= (hi - lo <= np.maximum(.75, .03 * mean)) & (np.abs(gray.astype(np.float32) - luminance) <= 12)
        if not fill.any():
            break
        values[fill] = mean[fill]
        known |= fill
    return values, known & ~valid


def scene_color_limits(z, valid, maximum):
    if valid.sum() < 50:
        return .05, maximum
    low, high = map(float, np.percentile(z[valid], (2, 98)))
    middle = (low + high) / 2
    # Avoid amplifying millimetre noise on a nearly flat surface.
    span = max(high - low, .15, middle * .25)
    return max(.05, middle - span / 2), min(maximum, middle + span / 2)


def alignment_error(left, right, detector):
    """Robust vertical residual on real correspondences, in processing pixels."""
    ka, da = detector.detectAndCompute(left, None)
    kb, db = detector.detectAndCompute(right, None)
    if da is None or db is None or min(len(ka), len(kb)) < 20:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(da, db, k=2)
    backwards = {m.queryIdx: m.trainIdx for m in matcher.match(db, da)}
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < .7 * n.distance and backwards.get(m.trainIdx) == m.queryIdx:
                good.append(m)
    if len(good) < 20:
        return None
    a = np.float32([ka[m.queryIdx].pt for m in good])
    b = np.float32([kb[m.trainIdx].pt for m in good])
    # Fundamental-matrix RANSAC works even when the pair is not rectified well.
    _, inliers = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1., .995, 1000)
    if inliers is None:
        return None
    selected = inliers.ravel().astype(bool) & (a[:, 0] - b[:, 0] > .5)
    if selected.sum() < 20:
        return None
    spread = np.ptp(a[selected], axis=0)
    if spread[0] < left.shape[1] * .2 or spread[1] < left.shape[0] * .2:
        return None
    return float(np.median(np.abs(a[selected, 1] - b[selected, 1])))


class DepthEngine:
    def __init__(self, calibration, max_depth=8.0, disparities=128, quality="balanced",
                 processing_width=None, temporal=True, near_priority=False, wide_view=False, detail=False,
                 estimate_surfaces=True, near_minimum=None, acceleration="cpu", parallel=False,ghost_filter="normal"):
        self.source_calibration = calibration
        size = calibration.size
        if processing_width and size[0] > processing_width:
            size = (processing_width, round(size[1] * processing_width / size[0]))
        self.calibration = effective = calibration.scaled_to(size, alpha=1 if wide_view else None)
        self.maps = [cv2.initUndistortRectifyMap(K, D, R, P, size, cv2.CV_32FC1)
                     for K, D, R, P in ((calibration.K1, calibration.D1, effective.R1, effective.P1),
                                        (calibration.K2, calibration.D2, effective.R2, effective.P2))]
        self.maximum = max_depth
        self.wide_view, self.detail = wide_view, detail
        self.estimate_surfaces = estimate_surfaces
        self.last_timings = {}
        self.ghost_filter=ghost_filter
        self.lr_tolerance=.75 if ghost_filter=='strict' else 1.0 if ghost_filter=='normal' else 1.5
        self.acceleration_requested = acceleration
        self.acceleration, self.acceleration_warning = "CPU SGBM", None
        self.pair_pool = ThreadPoolExecutor(max_workers=2,thread_name_prefix="stereo-match") if parallel else None
        near_minimum = near_minimum if near_minimum is not None else (.05 if quality=="high" else .10)
        if not .05<=near_minimum<=.5:
            raise ValueError("Ближняя граница поиска должна быть от 5 до 50 см.")
        self.near_minimum = near_minimum
        self.quality, self.temporal = quality, temporal
        self.previous = None
        self.last_stationary = False
        self.color_limits = None
        self.alignment_detector = cv2.ORB_create(nfeatures=1000, fastThreshold=12)
        self.segmenter = (cv2.ximgproc.segmentation.createGraphSegmentation(sigma=.8, k=200, min_size=150)
                          if hasattr(cv2.ximgproc, "segmentation") and not detail else None)
        self.num = min(disparities, ((size[0] - 16) // 16) * 16)
        if self.num < 16 or max_depth <= 0.05:
            raise ValueError("Слишком маленький кадр или некорректная дальность.")
        block = 3 if detail else 5
        params = dict(numDisparities=self.num, blockSize=block, P1=8 * block ** 2, P2=32 * block ** 2,
                      disp12MaxDiff=1, uniquenessRatio=20 if ghost_filter=='strict' else 10, speckleWindowSize=100,
                      speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left_matcher = cv2.StereoSGBM_create(minDisparity=0, **params)
        self.wls = None
        if quality != "fast" and hasattr(cv2, "ximgproc"):
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.wls.setLambda(8000 if quality == "balanced" else 12000)
            self.wls.setSigmaColor(1.2)
            self.wls.setLRCthresh(24)
            self.wls.setDepthDiscontinuityRadius(1)
        else:
            self.right_matcher = cv2.StereoSGBM_create(minDisparity=1 - self.num, **params)
        def source_mask(maps):
            mx, my = maps
            inside = (mx >= 1) & (mx < calibration.size[0] - 2) & (my >= 1) & (my < calibration.size[1] - 2)
            return cv2.erode(inside.astype(np.uint8), np.ones((block, block), np.uint8)).astype(bool)
        self.roi_mask, self.right_roi_mask = [source_mask(maps) for maps in self.maps]
        self.near_passes = []
        start = max(0, self.num - 16)
        search_limit = self.num
        if near_priority:
            # Overlapping range bands retain the normal range across most of the
            # image; one giant search would invalidate almost the entire left edge.
            target_disparity = min(size[0] - 32, int(np.ceil(effective.P1[0, 0] * effective.baseline / near_minimum)) + 16)
            for _ in range(8):
                if search_limit >= target_disparity:
                    break
                local_width = size[0] - start
                number = min(256, ((local_width - 32) // 32) * 16)
                if number < 16:
                    break
                near_params = dict(params, numDisparities=number, uniquenessRatio=20 if ghost_filter=='strict' else 15)
                matcher = cv2.StereoSGBM_create(minDisparity=0, **near_params)
                reverse_matcher = cv2.ximgproc.createRightMatcher(matcher)
                near_wls = None
                if quality != "fast":
                    near_wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
                    near_wls.setLambda(8000)
                    near_wls.setSigmaColor(1.2)
                    near_wls.setLRCthresh(16)
                    near_wls.setDepthDiscontinuityRadius(1)
                roi = self.roi_mask[:, start:]
                self.near_passes.append((start, number, matcher, reverse_matcher, near_wls, roi))
                search_limit = max(search_limit, start + number)
                if number == 16 or search_limit >= target_disparity:
                    break
                start += number - 16
        self.search_limit = search_limit
        self.cpu_matchers = (self.left_matcher,self.right_matcher,self.near_passes.copy())
        if acceleration != "cpu":
            try:
                from .gpu_stereo import gpu_info, CudaSGM
                name = gpu_info()
                if name is None:
                    raise RuntimeError("CUDA-модуль недоступен; используется процессор.")
                unique=12 if ghost_filter=='strict' else 5
                guarded=ghost_filter=='strict'
                self.left_matcher,self.right_matcher = CudaSGM(self.num,uniqueness=unique,ambiguity_guard=guarded),CudaSGM(self.num,1-self.num,unique,guarded)
                near_unique=12 if ghost_filter=='strict' else 7
                self.near_passes = [(s,n,CudaSGM(n,uniqueness=near_unique,ambiguity_guard=guarded),CudaSGM(n,1-n,near_unique,guarded),w,r)
                                    for s,n,_,__,w,r in self.near_passes]
                self.acceleration = f"NVIDIA SGM · {name}"
            except Exception as error:
                self.acceleration_warning = str(error)
        yy, xx = np.indices(size[::-1])
        largest_left = np.max(np.where(self.roi_mask, xx, -1), axis=1)
        smallest_right = np.min(np.where(self.right_roi_mask, xx, size[0]), axis=1)
        maximum_overlap_disparity = max(1, int(np.max(largest_left - smallest_right)))
        self.geometric_minimum = effective.P1[0, 0] * effective.baseline / maximum_overlap_disparity
        self.search_minimum = effective.P1[0, 0] * effective.baseline / max(search_limit - 1, 1)

    def match_pair(self, matcher, reverse, a, b):
        if self.pair_pool is not None and self.acceleration.startswith("CPU"):
            first=self.pair_pool.submit(matcher.compute,a,b)
            second=self.pair_pool.submit(reverse.compute,b,a)
            return first.result(),second.result()
        return matcher.compute(a,b),reverse.compute(b,a)

    def close(self):
        if getattr(self,"pair_pool",None):
            self.pair_pool.shutdown(wait=False)
            self.pair_pool=None

    def __del__(self):
        self.close()

    def distance_report(self, distances=(.10, .15, .20, .30)):
        height, width = self.roi_mask.shape
        yy, xx = np.indices((height, width))
        fx, baseline = self.calibration.P1[0, 0], self.calibration.baseline
        result = []
        for distance in distances:
            d = fx * baseline / distance
            xr = np.rint(xx - d).astype(int)
            common = self.roi_mask & (xr >= 0) & (xr < width)
            common &= self.right_roi_mask[yy, np.clip(xr, 0, width - 1)]
            result.append({"distance_m": distance, "disparity_px": float(d),
                           "overlap_fraction": float(common.mean()), "search_available": bool(d < self.search_limit - 1),
                           "lateral_mm_per_pixel": float(distance / fx * 1000),
                           "depth_mm_per_disparity_pixel": float(distance ** 2 / (fx * baseline) * 1000)})
        return result

    def compute(self,left,right,rectified=False):
        try:
            return self._compute(left,right,rectified)
        except Exception as error:
            if not self.acceleration.startswith("NVIDIA"):
                raise
            self.left_matcher,self.right_matcher,self.near_passes=self.cpu_matchers
            self.acceleration,self.acceleration_warning="CPU SGBM",f"CUDA отключена после ошибки: {error}"
            self.previous=None
            return self._compute(left,right,rectified)

    def _compute(self, left, right, rectified=False):
        started = time.perf_counter()
        self.last_timings = {}
        self.last_stationary = False
        if (left.shape[1], left.shape[0]) != self.source_calibration.size or left.shape != right.shape:
            raise ValueError("Размер кадров изменился. Выберите разрешение калибровки.")
        if rectified:
            if (left.shape[1], left.shape[0]) != self.calibration.size:
                left = cv2.resize(left, self.calibration.size, interpolation=cv2.INTER_AREA)
                right = cv2.resize(right, self.calibration.size, interpolation=cv2.INTER_AREA)
        else:
            left = cv2.remap(left, *self.maps[0], cv2.INTER_LINEAR)
            right = cv2.remap(right, *self.maps[1], cv2.INTER_LINEAR)
        a = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        self.last_timings["rectify_ms"] = (time.perf_counter()-started)*1000
        stage = time.perf_counter()
        vertical = alignment_error(a, b, self.alignment_detector)
        self.last_timings["alignment_ms"] = (time.perf_counter()-stage)*1000
        if vertical is not None and vertical > max(.9, a.shape[1] / 640 * 1.2):
            self.previous = None
            valid = np.zeros(a.shape, bool)
            disparity = np.full(a.shape, np.nan, np.float32)
            points = np.full((*a.shape, 3), np.nan, np.float32)
            color = depth_colors(disparity, valid)
            return DepthFrame(left, right, disparity, points, valid, color,
                preview_color=color.copy(), preview_filled=valid.copy(),
                color_limits=self.color_limits or (.05, self.maximum), alignment_error=vertical,
                quality_warning=f"Стереопара расходится по вертикали: {vertical:.1f} px. "
                                "Остановите камеру; если ошибка остаётся, повторите калибровку.")
        stage = time.perf_counter()
        try:
            raw_left,raw_right = self.match_pair(self.left_matcher,self.right_matcher,a,b)
        except Exception as error:
            if not self.acceleration.startswith("NVIDIA"):
                raise
            self.left_matcher,self.right_matcher,self.near_passes = self.cpu_matchers
            self.acceleration,self.acceleration_warning = "CPU SGBM",f"CUDA отключена после ошибки: {error}"
            raw_left,raw_right = self.match_pair(self.left_matcher,self.right_matcher,a,b)
        disparity = raw_left.astype(np.float32) / 16.0
        reverse = raw_right.astype(np.float32) / 16.0
        reverse[reverse <= -self.num] = np.inf
        valid = consistency_mask(disparity, reverse,self.lr_tolerance) & self.roi_mask
        if self.wls is not None:
            options = {"ROI":(2,2,a.shape[1]-4,a.shape[0]-4)} if self.acceleration.startswith("NVIDIA") else {}
            disparity = self.wls.filter(raw_left, left, disparity_map_right=raw_right,**options).astype(np.float32) / 16.0
            valid &= self.wls.getConfidenceMap() >= (180 if self.ghost_filter=='strict' else 128)
            valid &= consistency_mask(disparity, reverse,self.lr_tolerance)
        for start, number, matcher, reverse_matcher, wls, roi in self.near_passes:
            near_a, near_b = a[:, start:], b[:, :a.shape[1] - start]
            candidate_raw,reverse_raw = self.match_pair(matcher,reverse_matcher,near_a,near_b)
            candidate = candidate_raw.astype(np.float32) / 16.0
            backwards = reverse_raw.astype(np.float32) / 16.0
            backwards[backwards <= -number] = np.inf
            reliable = consistency_mask(candidate, backwards, .5 if self.ghost_filter=='strict' else .75) & roi
            if wls is not None:
                options = {"ROI":(2,2,near_a.shape[1]-4,near_a.shape[0]-4)} if self.acceleration.startswith("NVIDIA") else {}
                candidate = wls.filter(candidate_raw, left[:, start:], disparity_map_right=reverse_raw,**options).astype(np.float32) / 16.0
                reliable &= wls.getConfidenceMap() >= (200 if self.ghost_filter=='strict' else 160)
                reliable &= consistency_mask(candidate, backwards, .75 if self.ghost_filter=='strict' else 1.0)
            candidate, reliable = clean_disparity(candidate, reliable, 30 if self.detail else 80)
            candidate += start
            replace = reliable & (~valid[:, start:] | (disparity[:, start:] >= start - 8))
            disparity[:, start:][replace] = candidate[replace]
            valid[:, start:] |= replace
        self.last_timings["stereo_ms"] = (time.perf_counter()-stage)*1000
        stage = time.perf_counter()
        yy, xx = coordinates(disparity.shape)
        xr = np.rint(xx - disparity).astype(np.int32)
        valid &= self.roi_mask & (xr >= 0) & (xr < b.shape[1])
        valid &= self.right_roi_mask[yy, np.clip(xr, 0, b.shape[1] - 1)]
        if self.quality != "fast":
            xr = np.rint(xx - disparity).astype(np.int32)
            right_texture = texture_mask(b)[yy, np.clip(xr, 0, b.shape[1] - 1)]
            valid &= texture_mask(a) & right_texture & (xr >= 0) & (xr < b.shape[1])
        if self.quality != "fast":
            disparity, valid = clean_disparity(disparity, valid, 20 if self.detail else 50)
        if self.ghost_filter!='off':
            valid=supported_disparity(disparity,valid,minimum=4 if self.ghost_filter=='strict' else 3,
                                      tolerance=.75 if self.ghost_filter=='strict' else 1.25)
        if self.temporal and self.previous is not None:
            old_gray, old_disp, old_valid = self.previous
            features = cv2.goodFeaturesToTrack(old_gray, 120, .02, 12)
            stationary = False
            if features is not None and len(features) >= 15:
                current, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, a, features, None)
                if current is not None:
                    flow = np.linalg.norm(current - features, axis=2).ravel()[status.ravel() == 1]
                    stationary = len(flow) >= 15 and np.median(flow) < .25 and np.percentile(flow, 90) < .7
            if stationary:
                self.last_stationary = True
                blend = valid & old_valid & (np.abs(disparity - old_disp) < .75)
                blend &= cv2.absdiff(a, old_gray) < 10
                disparity[blend] = .65 * disparity[blend] + .35 * old_disp[blend]
        self.previous = (a.copy(), disparity.copy(), valid.copy()) if self.temporal else None
        points = cv2.reprojectImageTo3D(disparity, self.calibration.Q)
        valid &= np.isfinite(points).all(axis=2) & (points[:, :, 2] >= 0.05) & (
            points[:, :, 2] <= self.maximum)
        points[~valid] = np.nan
        target_limits = scene_color_limits(points[:, :, 2], valid, self.maximum)
        if self.color_limits is None:
            self.color_limits = target_limits
        else:
            self.color_limits = tuple(.75 * old + .25 * new for old, new in zip(self.color_limits, target_limits))
        low, high = self.color_limits
        self.last_timings["filter_ms"] = (time.perf_counter()-stage)*1000
        stage = time.perf_counter()
        measured_color = depth_colors(points[:, :, 2], valid, minimum=low, maximum=high)
        if not self.estimate_surfaces:
            disparity[~valid] = np.nan
            self.last_timings["preview_ms"] = (time.perf_counter()-stage)*1000
            return DepthFrame(left,right,disparity,points,valid,measured_color,
                              color_limits=self.color_limits,alignment_error=vertical)
        preview_disp, filled = smooth_preview(disparity, valid, a)
        planes, planar = plane_estimates(disparity, valid, left, self.roi_mask, self.segmenter)
        preview_disp[planar] = planes[planar]
        filled |= planar
        preview_points = cv2.reprojectImageTo3D(preview_disp, self.calibration.Q)
        filled &= np.isfinite(preview_points).all(axis=2)
        filled &= (preview_points[:, :, 2] >= .05) & (preview_points[:, :, 2] <= self.maximum)
        preview_xr = np.rint(xx - preview_disp).astype(np.int32)
        filled &= (preview_xr >= 0) & (preview_xr < b.shape[1])
        filled &= self.right_roi_mask[yy, np.clip(preview_xr, 0, b.shape[1] - 1)]
        preview = depth_colors(preview_points[:, :, 2], valid | filled, minimum=low, maximum=high)
        # Estimated surfaces remain visibly distinct without making small holes harsh.
        preview[filled] = np.uint8(preview[filled].astype(np.float32) * .75 + 35)
        estimated_points = np.where(filled[:, :, None], preview_points, np.nan).astype(np.float32)
        disparity[~valid] = np.nan
        self.last_timings["preview_ms"] = (time.perf_counter()-stage)*1000
        return DepthFrame(left, right, disparity, points, valid,
                          measured_color,
                          preview_color=preview, preview_filled=filled, color_limits=self.color_limits,
                          alignment_error=vertical, estimated_points=estimated_points)


def measure(frame, x, y, radius=3):
    height, width = frame.valid.shape
    if not (0 <= x < width and 0 <= y < height) or not frame.valid[y, x]:
        return None
    center = frame.points[y, x]
    if not np.isfinite(center).all() or center[2] <= 0:
        return None
    window = (slice(max(0, y - radius), min(height, y + radius + 1)),
              slice(max(0, x - radius), min(width, x + radius + 1)))
    region = frame.points[window]
    usable = region[frame.valid[window] & np.isfinite(region).all(axis=2)]
    # Estimate depth from neighbours on approximately the same surface as the click.
    z = center[2]
    usable = usable[(usable[:, 2] > 0) & (np.abs(usable[:, 2] - z) < max(0.03, z * 0.03))]
    if len(usable) < 5:
        return None
    # Robust depth belongs on the selected pixel's camera ray. Independent
    # XYZ medians move the endpoint towards denser support at edges and holes.
    point = center.astype(np.float64) * (float(np.median(usable[:, 2])) / z)
    return {"z": float(point[2]), "range": float(np.linalg.norm(point)),
            "point": point.tolist(), "samples": len(usable)}
