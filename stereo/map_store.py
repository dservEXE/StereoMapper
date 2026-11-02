"""Persistent voxel accumulation with bounded previews and streaming export."""
from pathlib import Path
from threading import RLock
from contextlib import closing
import sqlite3

import numpy as np


class VoxelMap:
    def __init__(self, voxel=.005, maximum=250000, path=None, stride=2, readonly=False):
        if not .00025 <= voxel <= .1 or stride not in (1, 2, 3, 4):
            raise ValueError("Шаг карты 0,25–100 мм; выбор пикселей от 1 до 4.")
        self.voxel, self.maximum, self.stride = float(voxel), int(maximum), int(stride)
        self.path = Path(path) if path is not None else None
        self.lock, self.readonly = RLock(), readonly
        if self.path is not None and not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        uri = self.path.resolve().as_uri()+"?mode=ro" if readonly and self.path else str(self.path) if self.path else ":memory:"
        self.connection = sqlite3.connect(uri, uri=readonly, check_same_thread=False, timeout=10.)
        self.connection.execute("PRAGMA cache_size=-32768")
        if not readonly:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS voxels (
                id INTEGER PRIMARY KEY, key INTEGER NOT NULL UNIQUE,
                x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
                r REAL NOT NULL, g REAL NOT NULL, b REAL NOT NULL, n INTEGER NOT NULL)""")
            self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.connection.execute("CREATE TABLE IF NOT EXISTS trajectory (id INTEGER PRIMARY KEY, x REAL, y REAL, z REAL)")
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('format', 'stereo-map-v1')")
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('voxel', ?)", (str(self.voxel),))
            self.connection.commit()
        metadata = dict(self.connection.execute("SELECT name,value FROM metadata"))
        if metadata.get("format") != "stereo-map-v1":
            self.close()
            raise ValueError("Это не архив карты приложения.")
        stored_voxel = float(metadata["voxel"])
        if not readonly and not np.isclose(stored_voxel, self.voxel):
            self.close()
            raise ValueError("Шаг ячеек существующего архива отличается от настроек.")
        self.voxel = stored_voxel
        self.count = int(self.connection.execute("SELECT COUNT(*) FROM voxels").fetchone()[0])
        self.trajectory = [np.array(row, np.float64) for row in self.connection.execute("SELECT x,y,z FROM trajectory ORDER BY id")]
        self.bounds = None

    @property
    def cells(self):
        # Compatibility: len(map.cells) means ALL stored voxels, not a RAM cache.
        return self

    def __len__(self):
        return self.count

    def _ensure_connection(self):
        if self.connection is None:
            if self.path is None:
                raise ValueError("Карта в памяти уже закрыта.")
            uri = self.path.resolve().as_uri()+"?mode=ro" if self.readonly else str(self.path)
            self.connection = sqlite3.connect(uri,uri=self.readonly,check_same_thread=False,timeout=10.)
            self.connection.execute("PRAGMA cache_size=-32768")
            if not self.readonly:
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.execute("PRAGMA synchronous=NORMAL")
        return self.connection

    def add(self, frame, pose):
        points = frame.points[::self.stride, ::self.stride].reshape(-1, 3)
        colors = frame.left[::self.stride, ::self.stride, ::-1].reshape(-1, 3)
        good = np.isfinite(points).all(axis=1)
        transformed=np.einsum("ij,kj->ik",points[good],pose[:3,:3],optimize=False)+pose[:3,3]
        self.add_points(transformed, colors[good], pose[:3, 3])

    def add_points(self, points, colors, position=None):
        if self.readonly:
            raise ValueError("Архив открыт только для просмотра.")
        points, colors = np.asarray(points, np.float64), np.asarray(colors, np.float64)
        good = np.isfinite(points).all(axis=1)
        points, colors = points[good], colors[good]
        if not len(points):
            return
        cells = np.floor(points/self.voxel).astype(np.int64)
        offset = 1 << 20
        if (cells < -offset).any() or (cells >= offset).any():
            raise ValueError("Координаты вышли за диапазон архива при выбранном шаге ячейки.")
        unsigned = cells+offset
        packed = (unsigned[:,0] << 42) | (unsigned[:,1] << 21) | unsigned[:,2]
        keys, groups, counts = np.unique(packed, return_inverse=True, return_counts=True)
        averaged_points = np.column_stack([np.bincount(groups, weights=points[:,i]) for i in range(3)])/counts[:,None]
        averaged_colors = np.column_stack([np.bincount(groups, weights=colors[:,i]) for i in range(3)])/counts[:,None]
        # One observation per voxel per frame, regardless of pixel density.
        rows = zip(keys.tolist(), *[column.tolist() for column in averaged_points.T],
                   *[column.tolist() for column in averaged_colors.T])
        with self.lock, self._ensure_connection():
            self.connection.executemany("""INSERT INTO voxels (key,x,y,z,r,g,b,n) VALUES (?,?,?,?,?,?,?,1)
                ON CONFLICT(key) DO UPDATE SET x=x+excluded.x,y=y+excluded.y,z=z+excluded.z,
                r=r+excluded.r,g=g+excluded.g,b=b+excluded.b,n=n+1""", rows)
            # IDs are contiguous: only implicit IDs are inserted and rows are never deleted.
            self.count = int(self.connection.execute("SELECT COALESCE(MAX(id),0) FROM voxels").fetchone()[0])
            if position is not None and (not self.trajectory or np.linalg.norm(position-self.trajectory[-1])>.02):
                self.connection.execute("INSERT INTO trajectory(x,y,z) VALUES (?,?,?)", tuple(map(float, position)))
                self.trajectory.append(np.array(position, np.float64))
        low, high = points.min(axis=0), points.max(axis=0)
        self.bounds = (low,high) if self.bounds is None else (np.minimum(self.bounds[0],low),np.maximum(self.bounds[1],high))

    def arrays(self, max_points=None):
        with self.lock:
            return self._arrays(self._ensure_connection(), max_points)

    @staticmethod
    def _arrays(connection, max_points):
        largest_id = int(connection.execute("SELECT COALESCE(MAX(id),0) FROM voxels").fetchone()[0])
        step = max(1, int(np.ceil(largest_id/max_points))) if max_points else 1
        query = "SELECT x/n,y/n,z/n,r/n,g/n,b/n FROM voxels"
        if step>1:
            query += f" WHERE id%{step}=1"
        if max_points:
            query += f" LIMIT {int(max_points)}"
        rows = connection.execute(query).fetchall()
        if not rows:
            return np.empty((0,3), np.float32), np.empty((0,3), np.uint8)
        data = np.asarray(rows, np.float64)
        return data[:,:3].astype(np.float32), np.uint8(np.clip(np.rint(data[:,3:]),0,255))

    def preview(self, max_points=100000):
        # A separate WAL reader lets display run alongside map writes.
        if self.path is None:
            return self.arrays(max_points)
        with closing(sqlite3.connect(self.path.resolve().as_uri()+"?mode=ro", uri=True, timeout=10.)) as reader:
            return self._arrays(reader, max_points)

    def export(self, path):
        if self.path is not None and Path(path).resolve()==self.path.resolve():
            raise ValueError("Выберите отдельный PLY-файл: нельзя перезаписать сам архив карты.")
        if self.path is not None:
            # Freeze one WAL read snapshot; writes and capture can continue.
            with closing(sqlite3.connect(self.path.resolve().as_uri()+"?mode=ro",uri=True,timeout=10.)) as reader:
                reader.execute("BEGIN")
                return self._export(reader,path)
        with self.lock:
            return self._export(self._ensure_connection(),path)

    @staticmethod
    def _export(connection,path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii", newline="\n") as stream:
            count = int(connection.execute("SELECT COUNT(*) FROM voxels").fetchone()[0])
            if not count:
                raise ValueError("В архиве нет точек.")
            stream.write(f"ply\nformat ascii 1.0\ncomment units metres; complete stored map\nelement vertex {count}\n"
                         "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            cursor = connection.execute("SELECT x/n,y/n,z/n,r/n,g/n,b/n FROM voxels ORDER BY id")
            while True:
                rows = cursor.fetchmany(50000)
                if not rows:
                    break
                data = np.asarray(rows, np.float64)
                data[:,3:] = np.clip(np.rint(data[:,3:]),0,255)
                np.savetxt(stream,data,fmt="%.6f %.6f %.6f %d %d %d")
        return count

    def close(self):
        with self.lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None

    def __del__(self):
        if getattr(self,"connection",None) is not None:
            try:
                self.close()
            except Exception:
                pass
