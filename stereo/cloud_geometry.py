"""Shared camera, point-cloud rasterization and full-resolution selection.

All point arrays are N x 3 world coordinates; colors are N x 3 RGB, 0..255.
Camera coordinates use X right, Y down, Z forward. ``rotation`` maps centered
world coordinates into camera coordinates. Normalization is fixed by ``center``
and ``extent``: ``camera = ((points-center)/extent) @ rotation.T + (0,0,2)``.
Projection uses ``min(width,height)*1.65*zoom`` pixels of focal length, with
``pan`` measured in screen pixels. ``depth`` is normalized camera Z, not metres.

``project_points`` and ``selection_mask`` always operate on every input point.
Only ``render_points`` samples its input, and it computes default bounds before
sampling. Reuse the same center/extent for every operation during editing so
that deleting points never changes the camera or the selected region.
"""

import numpy as np


def _points_array(points):
    value = np.asarray(points, dtype=np.float64)
    if value.size == 0 and value.ndim == 1:
        value = value.reshape(0, 3)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    return value


def cloud_bounds(points):
    """Return ``(center: float[3], extent: float)`` over finite input points.

    Extent is the largest axis span, floored at 1e-5. An empty/invalid-only
    cloud has center (0,0,0), extent 1. Nonfinite points never affect bounds.
    """
    points = _points_array(points)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return np.zeros(3, dtype=np.float64), 1.0
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    with np.errstate(over="ignore"):
        span = high - low
    if not np.isfinite(span).all():
        raise ValueError("point coordinate range is too large")
    return low / 2 + high / 2, max(float(np.max(span)), 1e-5)


def _camera(points, width, height, rotation, zoom, pan, center, extent):
    if (not np.isfinite(width) or not np.isfinite(height)
            or width <= 0 or height <= 0 or int(width) != width or int(height) != height):
        raise ValueError("width and height must be positive integer pixels")
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3 x 3 matrix")
    pan = np.asarray(pan, dtype=np.float64)
    if pan.shape != (2,) or not np.isfinite(pan).all():
        raise ValueError("pan must be two finite screen coordinates")
    if not np.isscalar(zoom) or not np.isfinite(zoom) or zoom <= 0:
        raise ValueError("zoom must be finite and positive")
    if center is None or extent is None:
        default_center, default_extent = cloud_bounds(points)
        center = default_center if center is None else center
        extent = default_extent if extent is None else extent
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center must be a finite 3D coordinate")
    if not np.isscalar(extent) or not np.isfinite(extent) or extent <= 0:
        raise ValueError("extent must be finite and positive")
    return rotation, float(zoom), pan, center, float(extent)


def project_points(points, width, height, rotation=None, zoom=1., pan=(0, 0),
                   center=None, extent=None, projection="perspective"):
    """Return ``(xy[N,2], depth[N], in_front[N])`` for every input point.

    ``xy`` contains unrounded screen coordinates. Invalid or behind-camera
    points have NaN xy and false ``in_front``; depth remains camera Z where
    calculable. ``in_front`` does not imply that the point is inside the canvas.
    Invalid camera arguments/array shapes raise ValueError.
    """
    if projection not in ("perspective", "orthographic"):
        raise ValueError("projection must be perspective or orthographic")
    points = _points_array(points)
    rotation, zoom, pan, center, extent = _camera(
        points, width, height, rotation, zoom, pan, center, extent)
    xy = np.full((len(points), 2), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        camera = ((points - center) / extent) @ rotation.T
        camera[:, 2] += 2.
        depth = camera[:, 2]
        in_front = np.isfinite(camera).all(axis=1) & (depth > 1e-6)
        focal = min(width, height) * 1.65 * zoom
        divisor = depth[in_front, None] if projection == "perspective" else 2.
        xy[in_front] = camera[in_front, :2] / divisor * focal
        xy[in_front] += np.array([width / 2, height / 2]) + pan
    in_front &= np.isfinite(xy).all(axis=1)
    xy[~in_front] = np.nan
    return xy, depth, in_front


def points_in_polygon(xy, polygon):
    """Even-odd polygon inclusion, including edges, for N x 2 screen points.

    The polygon may repeat its first vertex. Empty/degenerate polygons select
    nothing. Nonfinite query coordinates are excluded; nonfinite polygon
    vertices and malformed arrays raise ValueError. Self intersections use the
    even-odd rule, matching a lasso's usual behavior.
    """
    xy = np.asarray(xy, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    if xy.size == 0 and xy.ndim == 1:
        xy = xy.reshape(0, 2)
    if polygon.size == 0 and polygon.ndim == 1:
        polygon = polygon.reshape(0, 2)
    if xy.ndim != 2 or xy.shape[1] != 2 or polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("screen points and polygon must be N x 2 arrays")
    if not np.isfinite(polygon).all():
        raise ValueError("polygon vertices must be finite")
    result = np.zeros(len(xy), dtype=bool)
    if len(polygon) < 3 or len(np.unique(polygon, axis=0)) < 3:
        return result
    if np.linalg.matrix_rank(polygon - polygon[0]) < 2:
        return result
    tolerance = 1e-7
    low, high = np.min(polygon, axis=0), np.max(polygon, axis=0)
    candidates = (np.isfinite(xy).all(axis=1)
                  & (xy >= low - tolerance).all(axis=1)
                  & (xy <= high + tolerance).all(axis=1))
    indices = np.flatnonzero(candidates)
    x, y = xy[indices].T
    inside = np.zeros(len(indices), dtype=bool)
    boundary = np.zeros(len(indices), dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = previous
        x1, y1 = current
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy))
        if length:
            cross = (x - x0) * dy - (y - y0) * dx
            boundary |= ((np.abs(cross) <= tolerance * length)
                         & (x >= min(x0, x1) - tolerance) & (x <= max(x0, x1) + tolerance)
                         & (y >= min(y0, y1) - tolerance) & (y <= max(y0, y1) + tolerance))
        if dy:
            crossing = (y0 > y) != (y1 > y)
            inside ^= crossing & (x < (y - y0) * dx / dy + x0)
        previous = current
    result[indices] = inside | boundary
    return result


def _point_size(point_size):
    if not np.isscalar(point_size) or not np.isfinite(point_size) or point_size <= 0:
        raise ValueError("point_size must be positive")
    return int(np.clip(int(point_size), 1, 6))


def _raster_owners(xy, depth, in_front, width, height, point_size):
    """Nearest point index per pixel, or -1; deterministic lowest-index ties."""
    width, height = int(width), int(height)
    point_size = _point_size(point_size)
    half = point_size // 2
    # Cull before conversion: huge off-screen coordinates must not overflow int.
    good = (in_front & (xy[:, 0] >= -point_size) & (xy[:, 0] < width + point_size)
            & (xy[:, 1] >= -point_size) & (xy[:, 1] < height + point_size))
    indices = np.flatnonzero(good)
    centers = np.rint(xy[indices]).astype(np.int64)
    z = depth[indices]
    zbuffer = np.full(width * height, np.inf, dtype=np.float64)
    missing = np.iinfo(np.int64).max
    owners = np.full(width * height, missing, dtype=np.int64)

    def fragments():
        for dy in range(-half, point_size - half):
            ys = centers[:, 1] + dy
            for dx in range(-half, point_size - half):
                xs = centers[:, 0] + dx
                valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                yield ys[valid] * width + xs[valid], z[valid], indices[valid]

    for pixels, values, _ in fragments():
        np.minimum.at(zbuffer, pixels, values)
    for pixels, values, point_indices in fragments():
        nearest = values == zbuffer[pixels]
        np.minimum.at(owners, pixels[nearest], point_indices[nearest])
    owners[owners == missing] = -1
    return owners.reshape(height, width)


def _visible_mask(xy, depth, in_front, width, height, point_size):
    owners = _raster_owners(xy, depth, in_front, width, height, point_size)
    mask = np.zeros(len(xy), dtype=bool)
    mask[owners[owners >= 0]] = True
    return mask


def selection_mask(points, polygon, width, height, *, rotation=None, zoom=1.,
                   pan=(0, 0), center=None, extent=None, visible_only=False,
                   point_size=2, depth_range=None, projection="perspective"):
    """Select full-cloud points inside a screen lasso/rectangle polygon.

    Returns bool[N] in the original input order. By default selection passes
    through all depth layers. ``visible_only=True`` limits it to points owning
    at least one pixel in a zbuffer of the FULL cloud, using ``point_size``.
    ``depth_range=(near,far)`` additionally restricts normalized camera Z,
    inclusive; it does not change which other points occlude the selection.
    Invalid/behind-camera/outside-canvas points are excluded in every mode.
    """
    xy, depth, in_front = project_points(points, width, height, rotation, zoom, pan, center, extent, projection)
    selected = (in_front & (xy[:, 0] >= 0) & (xy[:, 0] < width)
                & (xy[:, 1] >= 0) & (xy[:, 1] < height)
                & points_in_polygon(xy, polygon))
    if depth_range is not None:
        interval = np.asarray(depth_range, dtype=np.float64)
        if interval.shape != (2,) or np.isnan(interval).any() or interval[0] > interval[1]:
            raise ValueError("depth_range must be (near, far), with near <= far")
        selected &= (depth >= interval[0]) & (depth <= interval[1])
    if visible_only:
        selected &= _visible_mask(xy, depth, in_front, width, height, point_size)
    return selected


def box_mask(points, minimum, maximum):
    """Return bool[N] for finite points inside an inclusive WORLD-space box.

    Opposite corners may be supplied in either order on each axis. Unlike a
    screen selection this box does not depend on the camera rotation or depth.
    """
    points = _points_array(points)
    first, second = np.asarray(minimum, dtype=np.float64), np.asarray(maximum, dtype=np.float64)
    if first.shape != (3,) or second.shape != (3,) or not np.isfinite([first, second]).all():
        raise ValueError("box corners must be finite 3D coordinates")
    low, high = np.minimum(first, second), np.maximum(first, second)
    return np.isfinite(points).all(axis=1) & (points >= low).all(axis=1) & (points <= high).all(axis=1)


def pick_point(points, x, y, width, height, *, rotation=None, zoom=1., pan=(0, 0),
               center=None, extent=None, radius=10, point_size=2, projection="perspective"):
    """Return the closest visible projected point within ``radius`` pixels.

    Returns an original input index or None. Screen distance ranks candidates;
    camera depth and then input index break ties. Occlusion uses the FULL cloud,
    including points outside the pick radius; rendering samples have no effect.
    """
    if not np.isfinite([x, y, radius]).all() or radius < 0:
        raise ValueError("pick coordinates and radius must be finite; radius >= 0")
    xy, depth, in_front = project_points(points, width, height, rotation, zoom, pan, center, extent, projection)
    with np.errstate(invalid="ignore", over="ignore"):
        distance = np.sum((xy - [x, y]) ** 2, axis=1)
    candidates = np.flatnonzero(in_front & (distance <= radius * radius))
    if not len(candidates):
        return None
    visible = _visible_mask(xy, depth, in_front, width, height, point_size)
    candidates = candidates[visible[candidates]]
    if not len(candidates):
        return None
    order = np.lexsort((candidates, depth[candidates], distance[candidates]))
    return int(candidates[order[0]])


def render_points(points, colors, width, height, *, rotation=None, zoom=1., pan=(0, 0),
                  center=None, extent=None, selection=None, max_points=250000,
                  point_size=2, axes=True, projection="perspective"):
    """Render RGB cloud colors to a uint8[height,width,3] BGR image.

    Deterministic stride sampling limits raster cost to ``max_points``; camera
    bounds always use ALL points. ``selection`` is bool[N] and highlights the
    selected sampled points in amber. Each square point splat uses a true
    nearest-depth zbuffer, including overlaps between different splat pixels.
    Nonfinite coordinates are skipped. Nonfinite colors become neutral gray.
    """
    points = _points_array(points)
    colors = np.asarray(colors)
    if colors.size == 0 and colors.ndim == 1:
        colors = colors.reshape(0, 3)
    if colors.shape != points.shape:
        raise ValueError("colors must be an N x 3 RGB array matching points")
    if np.iscomplexobj(colors):
        raise ValueError("colors must be real RGB values")
    if colors.dtype.kind not in "buif":
        colors = np.asarray(colors, dtype=np.float64)
    if not np.isscalar(max_points) or not np.isfinite(max_points) or max_points < 1:
        raise ValueError("max_points must be positive")
    rotation, zoom, pan, center, extent = _camera(
        points, width, height, rotation, zoom, pan, center, extent)
    if selection is not None:
        selection = np.asarray(selection, dtype=bool)
        if selection.shape != (len(points),):
            raise ValueError("selection must have one boolean per point")
    step = max(1, int(np.ceil(len(points) / max_points)))
    indices = np.arange(0, len(points), step)
    xy, depth, in_front = project_points(points[indices], width, height, rotation, zoom, pan, center, extent, projection)
    image = np.full((int(height), int(width), 3), (24, 20, 16), dtype=np.uint8)
    owners = _raster_owners(xy, depth, in_front, width, height, point_size)
    # Convert only the render sample; full-cloud RGB conversion would allocate
    # 24 bytes per point on every mouse movement, despite the preview budget.
    sample_colors = np.asarray(colors[indices], dtype=np.float64)
    rgb = np.uint8(np.clip(np.nan_to_num(sample_colors, nan=190, posinf=255, neginf=0), 0, 255))
    bgr = rgb[:, ::-1].copy()
    if selection is not None:
        bgr[selection[indices]] = (25, 220, 255)
    good = owners >= 0
    image[good] = bgr[owners[good]]
    if axes:
        import cv2
        origin = np.array([min(60, int(width) // 5), max(10, int(height) - 60)])
        length = min(38, max(8, min(width, height) // 6))
        for index, (name, color) in enumerate(zip("XYZ", ((80, 80, 245), (90, 215, 90), (245, 155, 80)))):
            end = origin + np.rint(rotation[:2, index] * length).astype(int)
            cv2.arrowedLine(image, tuple(origin), tuple(end), color, 2, cv2.LINE_AA, tipLength=.18)
            cv2.putText(image, name, tuple(end + [4, -4]), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    return image
