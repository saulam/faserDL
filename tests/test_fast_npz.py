import os
import tempfile
import unittest
import zipfile
from glob import glob
from pathlib import Path

import numpy as np

from dataset.fast_npz import FastNpz, load_npz

REPO_ROOT = Path(__file__).resolve().parents[1]


def _same(test, reference, fast):
    test.assertEqual(reference.dtype, fast.dtype)
    test.assertEqual(reference.shape, fast.shape)
    test.assertEqual(reference.flags.writeable, fast.flags.writeable)
    test.assertEqual(reference.flags.c_contiguous, fast.flags.c_contiguous)
    test.assertEqual(reference.flags.f_contiguous, fast.flags.f_contiguous)
    test.assertEqual(reference.tobytes("A"), fast.tobytes("A"))


class FastNpzTests(unittest.TestCase):
    ARRAYS = {
        "scalar": np.int64(7),
        "flag": np.bool_(True),
        "hits": np.random.default_rng(0).normal(size=(215, 6)).astype(np.float32),
        "fortran": np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
        "empty": np.zeros((0,), dtype=np.int32),
        "big_endian": np.arange(5, dtype=">i4"),
        "structured": np.array([(1, 2, 11)], dtype=[("g4_id", "<i4"), ("track_id", "<i4"), ("pdg", "<i4")]),
        "structured_empty": np.zeros(0, dtype=[("g4_id", "<i4"), ("pdg", "<i4")]),
        "objects": np.array([{"a": 1}, None], dtype=object),
    }

    def _check_file(self, path):
        # FastNpz itself, whatever FASER_FAST_NPZ says.
        reference, fast = np.load(path, allow_pickle=True), FastNpz(path)
        self.assertEqual(list(reference.files), list(fast.files))
        for key in reference.files:
            if reference[key].dtype.hasobject:
                self.assertEqual(list(reference[key]), list(fast[key]))
            else:
                _same(self, reference[key], fast[key])

    def test_matches_np_load_for_stored_and_deflated_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            for save in (np.savez, np.savez_compressed):
                path = os.path.join(directory, f"{save.__name__}.npz")
                save(path, **self.ARRAYS)
                self._check_file(path)
                fast = FastNpz(path)
                self.assertIn("hits", fast)
                self.assertIn("hits.npy", fast)
                self.assertTrue(fast["hits"].flags.writeable)
                with self.assertRaises(KeyError):
                    fast["missing"]

    def test_detects_corrupted_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "event.npz")
            np.savez(path, hits=self.ARRAYS["hits"])
            data = bytearray(Path(path).read_bytes())
            position = data.index(self.ARRAYS["hits"].tobytes()[:16]) + 100
            data[position] ^= 0xFF
            Path(path).write_bytes(bytes(data))
            with self.assertRaisesRegex(OSError, "CRC"):
                FastNpz(path)["hits"]

    def test_load_npz_falls_back_to_np_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "event.npz")
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_BZIP2) as archive, \
                    archive.open("hits.npy", "w") as member:
                np.lib.format.write_array(member, self.ARRAYS["hits"])   # bzip2: zipfile only
            data = load_npz(path)
            self.assertNotIsInstance(data, FastNpz)
            _same(self, np.load(path)["hits"], data["hits"])

    @unittest.skipUnless(glob(str(REPO_ROOT / "events_v7.0_*_npz" / "*.npz")), "local events are not available")
    def test_matches_np_load_on_local_events(self):
        files = sorted(glob(str(REPO_ROOT / "events_v7.0_*_npz" / "*.npz")))
        for index in np.random.default_rng(1).choice(len(files), 50, replace=False):
            with self.subTest(path=files[index]):
                self._check_file(files[index])


if __name__ == "__main__":
    unittest.main()
