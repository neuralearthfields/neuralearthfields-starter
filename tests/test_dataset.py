"""Check coordinate sampling, pixel splits, and DataLoader integration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin, rowcol
from torch.utils.data import DataLoader

from dataset import LandCoverDataset


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.source = Path(cls.temporary.name) / "land-cover.tif"
        cls.labels = np.resize(
            np.array([0, 1, 17, 21, 51, 61], dtype="uint8"), (37, 45)
        )
        cls.transform = from_origin(-123, 38, 0.01, 0.01)
        with rasterio.open(
            cls.source,
            "w",
            driver="GTiff",
            width=45,
            height=37,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=cls.transform,
            tiled=True,
            blockxsize=16,
            blockysize=16,
        ) as src:
            src.write(cls.labels, 1)
        cls.train = LandCoverDataset(cls.source, 1025, nearby_fraction=0.5, seed=0)
        cls.validation = LandCoverDataset(cls.source, 1025, split="validation", seed=1)

    def test_labels_and_split_separation(self):
        pixels = []
        for dataset in (self.train, self.validation):
            rows, cols = rowcol(self.transform, *dataset.coords.numpy().T)
            actual = dataset.class_codes[dataset.targets.numpy()]
            np.testing.assert_array_equal(actual, self.labels[rows, cols])
            self.assertTrue(np.isin(actual, [1, 21, 51, 61]).all())
            np.testing.assert_array_equal(
                dataset.held_out(rows, cols), dataset.split == "validation"
            )
            pixels.append(set(zip(rows, cols, strict=True)))
        self.assertFalse(pixels[0] & pixels[1])
        self.assertLess(len(pixels[0]), len(self.train))

    def test_reproducibility_and_local_source(self):
        with patch("dataset.hf_hub_download") as download:
            repeat = LandCoverDataset(self.source, 1025, nearby_fraction=0.5, seed=0)
        download.assert_not_called()
        torch.testing.assert_close(self.train.coords, repeat.coords)
        torch.testing.assert_close(self.train.targets, repeat.targets)

    def test_dataloader_and_partial_batch(self):
        batches = list(DataLoader(self.train, batch_size=256))
        self.assertEqual(
            [len(coords) for coords, _ in batches], [256, 256, 256, 256, 1]
        )
        torch.testing.assert_close(
            torch.cat([coords for coords, _ in batches]), self.train.coords
        )
        torch.testing.assert_close(
            torch.cat([target for _, target in batches]), self.train.targets
        )
        self.assertEqual(batches[0][0].dtype, torch.float32)
        self.assertEqual(batches[0][1].dtype, torch.int64)

    def test_invalid_settings(self):
        for options in (
            {"count": 0},
            {"split": "test"},
            {"nearby_fraction": 1.5},
            {"radius_pixels": -1},
        ):
            with self.subTest(options=options), self.assertRaises(AssertionError):
                LandCoverDataset(self.source, **({"count": 10} | options))


if __name__ == "__main__":
    unittest.main()
