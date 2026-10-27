import tarfile
import tempfile
import unittest
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dataset import SparseFASERCALMapDataset
from dataset.slim_shards import write_shard

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS = REPO_ROOT / "events_v7.0_500_npz"


def _dataset(files, train, stage1=False):
    dataset = SparseFASERCALMapDataset(SimpleNamespace(
        train=train, stage1=stage1, preprocessing_input="log", preprocessing_output="log",
        label_smoothing=0.02, mixup_alpha=0.0, dataset_path=str(EVENTS),
        metadata_path=str(EVENTS / "metadata_stats.pkl")))
    dataset.data_files = list(files)
    dataset.augmentations_enabled = train
    return dataset


@unittest.skipUnless((EVENTS / "metadata_stats.pkl").is_file(), "local events are not available")
class SlimShardTests(unittest.TestCase):
    def test_slim_events_load_identically_and_keep_their_names(self):
        files = sorted(glob(str(EVENTS / "*.npz")))[:40]
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            write_shard((0, files, str(temporary)))
            with tarfile.open(temporary / "shard-0000.tar") as tar:
                names = tar.getnames()
                tar.extractall(temporary / "staged", filter="data")
            self.assertEqual(names, [f"{EVENTS.name}/{Path(f).name}" for f in files])
            slim = [str(temporary / "staged" / name) for name in names]
            self.assertLess(sum(Path(f).stat().st_size for f in slim),
                            0.5 * sum(Path(f).stat().st_size for f in files))

            for train in (True, False):
                full_set, slim_set = _dataset(files, train), _dataset(slim, train)
                for index in range(len(files)):
                    np.random.seed(index)
                    a = full_set[index]
                    np.random.seed(index)
                    b = slim_set[index]
                    self.assertEqual(a.keys(), b.keys())
                    for key in a:
                        if torch.is_tensor(a[key]):
                            self.assertTrue(torch.equal(a[key], b[key]), key)
                        else:
                            self.assertTrue(np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key)
            # The MAE's stage-1 relational labels need the truth arrays left out.
            with self.assertRaises(KeyError):
                _dataset(slim, train=True, stage1=True)[0]


if __name__ == "__main__":
    unittest.main()
