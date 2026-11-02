from pathlib import Path
from itertools import islice

import numpy as np


def surface_from_frame(frame, step=4):
    points = frame.points[::step, ::step]
    colors = frame.left[::step, ::step, ::-1]
    valid = np.isfinite(points).all(axis=2)
    indices = np.full(valid.shape, -1, np.int32)
    indices[valid] = np.arange(np.count_nonzero(valid))
    faces = []
    a, b, c, d = indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1], indices[1:, 1:]
    flat_points = points[valid]
    for triangle in (np.stack((a, b, c), axis=2), np.stack((b, d, c), axis=2)):
        tri = triangle.reshape(-1, 3)
        tri = tri[(tri >= 0).all(axis=1)]
        if not len(tri):
            continue
        vertices = flat_points[tri]
        # Do not bridge holes or join foreground and background across a depth edge.
        z = vertices[:, :, 2]
        near = np.min(z, axis=1)
        continuous = np.ptp(z, axis=1) < np.maximum(.035, near * .035)
        faces.extend(tri[continuous])
    return points[valid], colors[valid], np.asarray(faces, np.int32).reshape(-1, 3)


def export_surface(path, points, colors, faces, units="metres"):
    if not len(points):
        raise ValueError("Нет достоверных точек для 3D-модели.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"ply\nformat ascii 1.0\ncomment units {units}; X right Y down Z forward\n")
        stream.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        np.savetxt(stream, np.column_stack((points, colors)), fmt="%.6f %.6f %.6f %d %d %d")
        if len(faces):
            np.savetxt(stream, np.column_stack((np.full(len(faces), 3), faces)), fmt="%d %d %d %d")


def read_model(path,max_points=250000,return_total=False):
    path = Path(path)
    if path.suffix.lower() != ".ply":
        raise ValueError("Выберите модель PLY, сохранённую этим приложением.")
    with path.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply" or stream.readline().strip() != "format ascii 1.0":
            raise ValueError("Этот просмотрщик поддерживает текстовый PLY. Экспортируйте модель в ASCII PLY.")
        count, properties, units, in_vertex = 0, [], "unknown", False
        for line in stream:
            line = line.strip()
            if line == "end_header":
                break
            if line.startswith("comment units "):
                units = line.split("comment units ", 1)[1].split(";")[0]
            if line.startswith("element "):
                in_vertex = line.startswith("element vertex ")
                if in_vertex:
                    count = int(line.split()[-1])
            elif line.startswith("property ") and in_vertex:
                properties.append(line.split()[-1])
        if count<=0 or max_points<=0:
            raise ValueError("В модели нет вершин или неверен бюджет просмотра.")
        step=max(1,int(np.ceil(count/max_points)))
        selected=[]
        actual=0
        for index,line in enumerate(islice(stream,count)):
            actual=index+1
            if index%step==0:selected.append(line)
        if actual!=count:
            raise ValueError("Повреждённый файл: число вершин не совпадает с заголовком.")
        data = np.loadtxt(selected, ndmin=2)
    if not all(key in properties for key in ("x", "y", "z")):
        raise ValueError("Повреждённый файл 3D-модели.")
    points = data[:, [properties.index(k) for k in ("x", "y", "z")]]
    colors = (data[:, [properties.index(k) for k in ("red", "green", "blue")]]
              if all(k in properties for k in ("red", "green", "blue")) else np.full(points.shape, 190))
    good = np.isfinite(points).all(axis=1)
    result=points[good], np.uint8(np.clip(colors[good], 0, 255)), units
    return (*result,count) if return_total else result


def render_cloud(points, colors, width, height, yaw=.35, pitch=-.25, zoom=1., pan=(0, 0), max_points=80000,
                 rotation=None,point_size=2,axes=False):
    image = np.full((height, width, 3), (24, 20, 16), np.uint8)
    if not len(points):
        return image
    stride = max(1, int(np.ceil(len(points) / max_points)))
    p, c = points[::stride].astype(np.float64), colors[::stride]
    low, high = np.min(p, axis=0), np.max(p, axis=0)
    p = (p - (low + high) / 2) / max(float(np.max(high - low)), 1e-5)
    cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    if rotation is None:
        rotation = np.array([[cy, 0, sy], [sx * sy, cx, -sx * cy], [-cx * sy, sx, cx * cy]])
    p = np.einsum('ij,kj->ik',p,rotation,optimize=False)
    p[:, 2] += 2.0
    focal = min(width, height) * 1.65 * zoom
    xy = np.rint(p[:, :2] / p[:, 2:] * focal + np.array([width / 2 + pan[0], height / 2 + pan[1]])).astype(int)
    point_size=int(np.clip(point_size,1,6))
    good = (xy[:, 0] >= point_size) & (xy[:, 0] < width - point_size) & (xy[:, 1] >= point_size) & (xy[:, 1] < height - point_size)
    order = np.argsort(p[good, 2])[::-1]
    xy, c = xy[good][order], c[good][order, ::-1]
    # Far points first; nearby points overwrite them in the raster.
    for dx in range(point_size):
        for dy in range(point_size):
            image[xy[:, 1] + dy-point_size//2, xy[:, 0] + dx-point_size//2] = c
    if axes:
        import cv2
        origin=np.array([60,height-65])
        for index,(name,color) in enumerate(zip('XYZ',((80,80,245),(90,215,90),(245,155,80)))):
            end=origin+np.rint(rotation[:2,index]*38).astype(int)
            cv2.arrowedLine(image,tuple(origin),tuple(end),color,2,cv2.LINE_AA,tipLength=.18)
            cv2.putText(image,name,tuple(end+[4,-4]),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    return image
