"""Sample land-cover pixels as coordinate/label pairs."""

from pathlib import Path

import numpy as np
import rasterio
import torch
from huggingface_hub import hf_hub_download
from rasterio.windows import Window
from torch.utils.data import Dataset


class LandCoverDataset(Dataset):
    """A fixed sample pool with disjoint training and validation pixels."""

    class_codes = np.r_[1:17, 18:22, 51:62].astype("uint8")

    def __init__(
        self,
        source,
        count,
        split="train",
        nearby_fraction=0,
        radius_pixels=5,
        seed=0,
    ):
        """Load the source and sample coordinate/label pairs for one split."""
        assert count > 0
        assert split in ("train", "validation")
        assert 0 <= nearby_fraction <= 1
        assert radius_pixels >= 0
        self.source = Path(source)
        self.split = split
        self.nearby_fraction = nearby_fraction
        self.radius_pixels = radius_pixels
        self.rng = np.random.default_rng(seed)
        self.lookup = np.full(256, -1, dtype=np.int64)
        self.lookup[self.class_codes] = np.arange(len(self.class_codes))
        self.download()
        self.coords, self.targets = self.sample_pixels(count)

    def __len__(self):
        """Return the number of sampled pixels."""
        return len(self.targets)

    def __getitem__(self, index):
        """Return a longitude/latitude pair and its class index."""
        return self.coords[index], self.targets[index]

    def download(self):
        """Download the source COG if it is not already local."""
        if not self.source.exists():
            hf_hub_download(
                repo_id="taylor-geospatial/neural-earth-fields-hackathon",
                repo_type="dataset",
                filename=self.source.name,
                local_dir=self.source.parent,
                token=False,
            )

    @staticmethod
    def held_out(rows, cols):
        """Assign each pixel to one split, including repeated nearby samples."""
        return (
            (rows.astype(np.int64) * 73856093) ^ (cols.astype(np.int64) * 19349663)
        ) % 5 == 0

    def sample_pixels(self, count):
        """Sample coordinates and labels across randomly selected tiles."""
        coords = np.empty((count, 2), dtype="float32")
        targets = np.empty(count, dtype="int64")
        with rasterio.open(self.source) as src:
            for start in range(0, count, 512):
                stop = min(start + 512, count)
                coords[start:stop], targets[start:stop] = self.sample_tile(
                    src, stop - start
                )
        return torch.from_numpy(coords), torch.from_numpy(targets)

    def sample_tile(self, src, count):
        """Mix random pixels and nearby neighbors within one valid tile."""
        tile_height, tile_width = src.block_shapes[0]
        while True:
            tile_row = (
                self.rng.integers((src.height + tile_height - 1) // tile_height)
                * tile_height
            )
            tile_col = (
                self.rng.integers((src.width + tile_width - 1) // tile_width)
                * tile_width
            )
            window = Window(tile_col, tile_row, tile_width, tile_height)
            labels = self.lookup[src.read(1, window=window)]
            tile_rows, tile_cols = np.indices(labels.shape)
            in_split = self.held_out(tile_rows + tile_row, tile_cols + tile_col) == (
                self.split == "validation"
            )
            valid = (labels >= 0) & in_split
            candidates = np.flatnonzero(valid)
            if len(candidates):
                break

        rows, cols = np.unravel_index(self.rng.choice(candidates, count), labels.shape)
        nearby_count = int(count * self.nearby_fraction)
        if nearby_count:
            row_offsets, col_offsets = self.rng.integers(
                -self.radius_pixels, self.radius_pixels + 1, (2, nearby_count)
            )
            neighbor_rows = rows[:nearby_count] + row_offsets
            neighbor_cols = cols[:nearby_count] + col_offsets
            inside = (
                (neighbor_rows >= 0)
                & (neighbor_rows < labels.shape[0])
                & (neighbor_cols >= 0)
                & (neighbor_cols < labels.shape[1])
            )
            neighbor_rows = neighbor_rows.clip(0, labels.shape[0] - 1)
            neighbor_cols = neighbor_cols.clip(0, labels.shape[1] - 1)
            use_neighbor = inside & valid[neighbor_rows, neighbor_cols]
            rows[-nearby_count:] = np.where(
                use_neighbor, neighbor_rows, rows[:nearby_count]
            )
            cols[-nearby_count:] = np.where(
                use_neighbor, neighbor_cols, cols[:nearby_count]
            )

        lon, lat = src.transform * (cols + tile_col + 0.5, rows + tile_row + 0.5)
        coords = np.column_stack((lon, lat)).astype("float32")
        return coords, labels[rows, cols]
