from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from threading import Event
from types import SimpleNamespace
import numpy as np
from stereo.map_store import VoxelMap
from stereo.map_integrator import MapIntegrator
from stereo.model3d import read_model


class MapArchiveTests(unittest.TestCase):
    def test_mapping_queue_does_not_block_depth_and_finishes_on_stop(self):
        entered,release=Event(),Event()
        class SlowStore:
            def __init__(self):self.saved=[]
            def add(self,frame,pose):
                entered.set()
                release.wait(3)
                self.saved.append(frame.number)
        class Poses:
            def update(self,frame):return np.eye(4),"tracking"
        store=SlowStore()
        mapper=MapIntegrator(Poses(),store,SlowStore())
        def frame(n):return SimpleNamespace(number=n,quality_warning=None,valid=np.ones((1,1),bool),estimated_points=None)
        mapper.start()
        try:
            mapper.submit(frame(1))
            self.assertTrue(entered.wait(2))
            # While disk is busy, the producer can replace a waiting frame.
            mapper.submit(frame(2));mapper.submit(frame(3))
            self.assertEqual(mapper.skipped,1)
            mapper.stop_event.set();release.set();mapper.join(4)
            self.assertFalse(mapper.is_alive())
            self.assertEqual(store.saved,[1,3])
            self.assertEqual(mapper.integrated,2)
        finally:
            release.set();mapper.stop_event.set();mapper.join(4)

    def test_more_than_old_limit_survives_reopen_preview_and_full_export(self):
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/"map.sqlite"
            store=VoxelMap(voxel=.001,path=path,maximum=10000,stride=1)
            count=500123
            ids=np.arange(count)
            points=np.column_stack(((ids%1000)*.001+.00025,(ids//1000)*.001+.00025,np.ones(count)))
            colors=np.tile([20,100,230],(count,1))
            store.add_points(points[:400000],colors[:400000])
            store.add_points(points[400000:],colors[400000:])
            self.assertEqual(len(store),count)
            preview,_=store.preview(10000)
            self.assertLessEqual(len(preview),10000)
            self.assertLess(float(preview[:,1].min()),.01)
            self.assertGreater(float(preview[:,1].max()),.49)
            # Revisiting a voxel averages it, without increasing or corrupting count.
            store.add_points(points[-1:]+[.0001,0,0],colors[-1:])
            self.assertEqual(len(store),count)
            store.close()
            reopened=VoxelMap(path=path,readonly=True)
            self.assertEqual(len(reopened),count)
            self.assertEqual(reopened.voxel,.001)
            with self.assertRaises(ValueError):reopened.export(path)
            ply=Path(temporary)/"full.ply"
            self.assertEqual(reopened.export(ply),count)
            with ply.open() as stream:
                header=[]
                for line in stream:
                    header.append(line.strip())
                    if line.strip()=="end_header":break
                self.assertIn(f"element vertex {count}",header)
                self.assertEqual(sum(1 for _ in stream),count)
            loaded,rgb,units,total=read_model(ply,max_points=1000,return_total=True)
            self.assertEqual(total,count)
            self.assertLessEqual(len(loaded),1000)
            self.assertGreater(float(loaded[:,1].max()),.49)
            reopened.close()

    def test_averaging_negative_cells_and_readonly_guard(self):
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/"map.sqlite"
            store=VoxelMap(voxel=.01,path=path)
            store.add_points(np.array([[-.015,.005,1.005],[-.014,.005,1.005]]),np.array([[10,20,30],[30,40,50]]))
            store.add_points(np.array([[-.013,.005,1.005]]),np.array([[40,60,80]]))
            points,colors=store.arrays()
            self.assertEqual(len(store),1)
            np.testing.assert_allclose(points[0],[-.01375,.005,1.005],atol=1e-7)
            np.testing.assert_array_equal(colors[0],[30,45,60])
            store.close()
            reader=VoxelMap(path=path,readonly=True)
            with self.assertRaises(ValueError):reader.add_points(points,colors)
            reader.close()

    def test_out_of_range_cannot_alias_an_existing_cell(self):
        store=VoxelMap(voxel=.001)
        with self.assertRaises(ValueError):
            store.add_points(np.array([[2000.,0,1.]]),np.array([[0,0,0]]))
        self.assertEqual(len(store),0)
        store.close()
