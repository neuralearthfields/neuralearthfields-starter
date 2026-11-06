<p align="center">
  <a href="https://neuralearthfields.github.io/">
    <img src="assets/earth-transformation.png" alt="Earth transformed from imagery into a neural field" width="900">
  </a>
</p>

<h1 align="center">Neural Earth Fields</h1>
<p align="center">Learning continuous representations of our planet.</p>
<p align="center"><a href="https://neuralearthfields.github.io/">Benchmark &amp; Hackathon</a> · Starter notebooks</p>

Train a [Wrap encoder](https://github.com/MarcCoru/locationencoder/blob/main/locationencoder/pe/wrap.py) and MLP to predict global land cover from longitude and latitude.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralearthfields/neuralearthfields-starter/blob/main/starter.ipynb)

## Get started

```sh
uv sync
uv run jupyter lab starter.ipynb
```

[starter.ipynb](starter.ipynb) downloads the full [land-cover COG](https://huggingface.co/datasets/taylor-geospatial/neural-earth-fields-hackathon), samples land and lakes globally, and trains the model. It includes learning curves, a linear probe, and PCA RGB plots. Ocean and invalid labels are excluded.

![Global land cover and zoomed views of Western Europe, California, and the Sierra Nevada](assets/land-cover.png)

*The source land-cover map at four scales.*

## Bottleneck embeddings

![Land cover, bottleneck PCA RGB, MLP predictions, and linear-probe predictions](assets/embeddings.png)

*One trained model: 128-dimensional bottleneck embeddings shown as PCA RGB, with land-cover reconstructions from the MLP and a fresh linear probe.*

Defaults are `Wrap (4) → 256 → 128 → 256 → 31`, with batch size 4,096. The middle 128 features are the embeddings. Edit the `Config` class to change capacity, training length, or sampling. Wrap learns broad patterns but misses local boundaries.

[dataset.py](dataset.py) contains `LandCoverDataset`, including downloading, tile sampling, and coordinate conversion. Training and validation use disjoint pixel hashes: nearby sampling can repeat pixels, so splitting sampled rows with `random_split` could put the same pixel in both sets.

## Export the trained model

The export cell in `starter.ipynb` writes `outputs/model.onnx`: the embedding encoder with weights embedded in one file. Classifier heads stay in the PyTorch checkpoint.

Input `coords` is float32 `[batch, 2]` longitude/latitude in EPSG:4326 degrees. Optional `time` is float64 `[batch, 1]` UTC Unix seconds; omit it or pass `None` when empty. This model ignores time. Output `embeddings` is `[batch, 128]` by default, with a dynamic batch dimension. No custom metadata is added.

## Export rasters

[inference.ipynb](inference.ipynb) loads the ONNX encoder for coordinate inference and float32 embedding COGs. Change `batch_size` to control memory use. The source COG supplies the output grid and ocean/invalid-pixel mask.

Set the output directory and enable `write_global_cogs` to export the full grid, or pass a `Window` to `export_embeddings` for a subset. The 128-band embeddings need about 31.5 TB before compression, plus temporary files and overviews. The source grid is nominally 100 m in EPSG:4326; ground spacing varies with latitude.

Python 3.12–3.14. Use `.venv/bin/python` in your editor and `uv add PACKAGE` for dependencies. Plotting helpers live in `plot.py`.

Check formatting and ONNX/raster inference with `uv run ruff check .`, `uv run ruff format --check .`, and `uv run python -m unittest discover -s tests`.

Dataset by Demuzere, He, Martilli, and Zonato (2023), [Zenodo](https://zenodo.org/records/7670653), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
