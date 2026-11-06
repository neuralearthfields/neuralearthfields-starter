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
        # Map source class codes to 0..30; -1 marks ocean and invalid labels.
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
        pixel_hash = (rows.astype(np.int64) * 73856093) ^ (
            cols.astype(np.int64) * 19349663
        )
        # The same pixel always lands in the same split; roughly 20% are held out.
        return pixel_hash % 5 == 0

    def sample_pixels(self, count):
        """Sample coordinates and labels across randomly selected tiles."""
        coords = np.empty((count, 2), dtype="float32")
        targets = np.empty(count, dtype="int64")
        with rasterio.open(self.source) as src:
            # Sample 512 pixels per tile, rather than loading the full raster.
            for start in range(0, count, 512):
                stop = min(start + 512, count)
                coords[start:stop], targets[start:stop] = self.sample_tile(
                    src, stop - start
                )
        return torch.from_numpy(coords), torch.from_numpy(targets)

    def sample_tile(self, src, count):
        """Mix random pixels and nearby neighbors within one valid tile."""
        window, labels, valid = self.sample_valid_tile(src)
        valid_rows, valid_cols = np.nonzero(valid)
        selected = self.rng.choice(len(valid_rows), count)
        rows, cols = valid_rows[selected], valid_cols[selected]
        rows, cols = self.sample_neighbors(rows, cols, valid)

        # Add tile offsets and use pixel centers for longitude/latitude.
        lon, lat = src.transform * (
            cols + window.col_off + 0.5,
            rows + window.row_off + 0.5,
        )
        coords = np.column_stack((lon, lat)).astype("float32")
        return coords, labels[rows, cols]

    def sample_valid_tile(self, src):
        """Choose a tile containing labeled pixels from the requested split."""
        tile_height, tile_width = src.block_shapes[0]
        tile_rows = (src.height + tile_height - 1) // tile_height
        tile_cols = (src.width + tile_width - 1) // tile_width
        while True:
            tile_row = self.rng.integers(tile_rows) * tile_height
            tile_col = self.rng.integers(tile_cols) * tile_width
            window = Window(tile_col, tile_row, tile_width, tile_height)
            labels = self.lookup[src.read(1, window=window)]
            rows, cols = np.indices(labels.shape)
            held_out = self.held_out(rows + tile_row, cols + tile_col)
            in_split = held_out if self.split == "validation" else ~held_out
            valid = (labels >= 0) & in_split
            if valid.any():
                return window, labels, valid

    def sample_neighbors(self, rows, cols, valid):
        """Replace part of the sample with neighbors of the first few pixels."""
        nearby_count = int(len(rows) * self.nearby_fraction)
        if nearby_count == 0:
            return rows, cols

        row_offsets, col_offsets = self.rng.integers(
            -self.radius_pixels, self.radius_pixels + 1, (2, nearby_count)
        )
        neighbor_rows = rows[:nearby_count] + row_offsets
        neighbor_cols = cols[:nearby_count] + col_offsets
        inside = (
            (neighbor_rows >= 0)
            & (neighbor_rows < valid.shape[0])
            & (neighbor_cols >= 0)
            & (neighbor_cols < valid.shape[1])
        )
        neighbor_rows = neighbor_rows.clip(0, valid.shape[0] - 1)
        neighbor_cols = neighbor_cols.clip(0, valid.shape[1] - 1)
        use_neighbor = inside & valid[neighbor_rows, neighbor_cols]
        # Fall back to the original pixel if its neighbor is outside the tile or split.
        rows[-nearby_count:] = np.where(
            use_neighbor, neighbor_rows, rows[:nearby_count]
        )
        cols[-nearby_count:] = np.where(
            use_neighbor, neighbor_cols, cols[:nearby_count]
        )
        return rows, cols
