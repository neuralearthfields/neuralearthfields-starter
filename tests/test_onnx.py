"""Check the notebook's encoder export and embedding raster inference."""

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import onnx
import onnxruntime as ort
import rasterio
import torch
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def notebook_definitions(filename):
    """Load definitions without downloading data or launching training."""
    namespace = {}
    notebook = nbformat.read(ROOT / filename, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        tree.body = [
            node
            for node in tree.body
            if isinstance(
                node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
            )
        ]
        exec(compile(tree, filename, "exec"), namespace)  # noqa: S102
    return namespace


class OnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        starter = notebook_definitions("starter.ipynb")
        cls.inference = notebook_definitions("inference.ipynb")
        cls.inference["class_codes"] = np.array([1, 21, 51], dtype="uint8")
        torch.manual_seed(0)
        cls.encoder = nn.Sequential(
            starter["Wrap"](), nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU()
        ).eval()
        model = nn.Sequential(cls.encoder, nn.Linear(8, 3))
        cls.coords = torch.rand(4096, 2) * torch.tensor([360, 180]) - torch.tensor(
            [180, 90]
        )
        notebook = nbformat.read(ROOT / "starter.ipynb", as_version=4)
        cell = next(c for c in notebook.cells if c.id == "onnx-export")
        exec(  # noqa: S102
            cell.source,
            {
                "torch": torch,
                "model": model,
                "cfg": SimpleNamespace(output=cls.directory),
            },
        )
        cls.path = cls.directory / "model.onnx"
        cls.session = ort.InferenceSession(
            cls.path.read_bytes(), providers=["CPUExecutionProvider"]
        )

    def test_single_file_encoder_only(self):
        self.assertEqual(list(self.directory.iterdir()), [self.path])
        graph = onnx.load(self.path, load_external_data=False)
        onnx.checker.check_model(graph, full_check=True)
        self.assertEqual([v.name for v in graph.graph.input], ["coords", "time"])
        self.assertEqual([v.name for v in graph.graph.output], ["embeddings"])
        for initializer in graph.graph.initializer:
            self.assertNotEqual(initializer.data_location, onnx.TensorProto.EXTERNAL)
            self.assertFalse(initializer.external_data)
        weights = [
            v for v in graph.graph.initializer if v.name.endswith(("weight", "bias"))
        ]
        self.assertEqual(
            sum(np.prod(v.dims) for v in weights),
            sum(p.numel() for p in self.encoder.parameters()),
        )
        for value in [*graph.graph.input, *graph.graph.output]:
            kind = value.type
            if kind.HasField("optional_type"):
                kind = kind.optional_type.elem_type
            self.assertTrue(kind.tensor_type.shape.dim[0].dim_param)
        self.assertEqual(self.session.get_inputs()[1].type, "optional(tensor(double))")

    def test_dynamic_batches_and_optional_time(self):
        for count in (1, 17, 4096):
            with self.subTest(batch_size=count), torch.inference_mode():
                coords = self.coords[:count].clone()
                expected = self.encoder(coords).numpy()
                torch.testing.assert_close(coords, self.coords[:count])
                baseline = self.session.run(None, {"coords": coords.numpy()})[0]
                np.testing.assert_allclose(
                    np.asarray(baseline), expected, rtol=1e-4, atol=1e-4
                )
                for time in (
                    None,
                    np.empty((0, 1)),
                    np.arange(count, dtype="float64").reshape(-1, 1) + 1609459200,
                    np.full((count, 1), np.nan),
                ):
                    actual = self.session.run(
                        None, {"coords": coords.numpy(), "time": time}
                    )[0]
                    np.testing.assert_array_equal(actual, baseline)

    def test_masked_tile_and_partial_batch(self):
        coords = self.coords[:35].numpy()
        valid = np.ones((5, 7), dtype=bool)
        valid.ravel()[::3] = False
        embeddings = self.inference["predict_tile"](
            self.session, coords, valid, batch_size=7
        )
        expected = np.asarray(self.session.run(["embeddings"], {"coords": coords})[0])
        np.testing.assert_allclose(
            embeddings[:, valid].T, expected[valid.ravel()], rtol=1e-4, atol=1e-4
        )
        self.assertTrue(np.isnan(embeddings[:, ~valid]).all())
        empty = self.inference["predict_tile"](
            self.session, coords, np.zeros_like(valid), batch_size=7
        )
        self.assertTrue(np.isnan(empty).all())
        with self.assertRaises(AssertionError):
            self.inference["predict_tile"](self.session, coords, valid, batch_size=0)

    def test_cog_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.tif"
            labels = np.resize(np.array([0, 1, 17, 21, 51], dtype="uint8"), (32, 40))
            transform = from_origin(-123, 38, 0.01, 0.01)
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=40,
                height=32,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
                nodata=0,
            ) as raster:
                raster.write(labels, 1)
            window = Window(5, 4, 19, 13)
            path = self.inference["export_embeddings"](
                self.session, source, temporary, window=window, batch_size=7
            )
            valid = np.isin(labels[4:17, 5:24], self.inference["class_codes"])
            coords = self.inference["pixel_coordinates"](transform, window)
            expected = self.inference["predict_tile"](
                self.session, coords, valid, batch_size=19
            )
            with rasterio.open(path) as raster:
                self.assertEqual(raster.tags(ns="IMAGE_STRUCTURE")["LAYOUT"], "COG")
                self.assertEqual(raster.crs.to_epsg(), 4326)
                self.assertEqual(raster.transform, window_transform(window, transform))
                np.testing.assert_allclose(
                    raster.read(), expected, rtol=1e-4, atol=1e-4, equal_nan=True
                )


if __name__ == "__main__":
    unittest.main()
