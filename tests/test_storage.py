from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
from stereo.storage import ResearchCatalog


class StorageTests(unittest.TestCase):
    def test_inventory_and_selected_deletion_preserve_program_calibration_and_history(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            files={'output/precision_old/left.png':100,'data/maps/old/measured.sqlite':200,
                   'data/calibration_old/left.png':50,'data/calibration.npz':80,
                   'models/model.safetensors':300,'output/leaf_measurements.csv':40,'main.py':30}
            for name,size in files.items():
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x'*size)
            catalog=ResearchCatalog(root)
            found=catalog.scan(protected=[root/'data/maps/old'])
            self.assertEqual(len(found),3)
            self.assertEqual(sum(x.size for x in found),350)
            self.assertTrue(next(x for x in found if x.path.name=='old').blocked)
            self.assertEqual(catalog.delete(root/'output/precision_old'),100)
            self.assertFalse((root/'output/precision_old').exists())
            for name in ('main.py','data/calibration.npz','models/model.safetensors','output/leaf_measurements.csv'):
                self.assertTrue((root/name).exists())
            with self.assertRaises(ValueError):catalog.delete(root/'data/maps/old',[root/'data/maps/old/measured.sqlite'])

    def test_parent_external_and_nested_paths_are_rejected(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);catalog=ResearchCatalog(root)
            (root/'output/one/nested').mkdir(parents=True)
            for path in (root,root/'output',root/'data/maps',root/'models',root.parent,root/'output/one/nested'):
                with self.assertRaises(ValueError):catalog.delete(path)
            self.assertTrue((root/'output/one/nested').exists())

    def test_redirect_inside_tree_blocks_removal_before_any_file_is_deleted(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'output/one';folder.mkdir(parents=True)
            (folder/'image.png').write_bytes(b'keep')
            redirected=folder/'link';redirected.mkdir()
            with patch('stereo.storage.reparse',side_effect=lambda p:Path(p)==redirected),patch('stereo.storage.shutil.rmtree') as remove:
                with self.assertRaises(ValueError):ResearchCatalog(root).delete(folder)
                remove.assert_not_called()
            self.assertEqual((folder/'image.png').read_bytes(),b'keep')
