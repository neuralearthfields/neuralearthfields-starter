# neuralearthfields-starter

A minimal notebook for training a [Wrap encoder](https://github.com/MarcCoru/locationencoder/blob/main/locationencoder/pe/wrap.py) and MLP to predict land cover from longitude and latitude.

```sh
uv sync
uv run jupyter lab starter.ipynb
```

Run [starter.ipynb](starter.ipynb) from top to bottom.
It downloads the full [CGLC-MODIS-LCZ map from Hugging Face](https://huggingface.co/datasets/taylor-geospatial/neural-earth-fields-hackathon) and trains on land and lake pixels sampled across the entire map.
Lakes (21) are included. Ocean (17) and invalid labels are excluded from training and evaluation, and masked in exported rasters.
The notebook opens an interactive COG map with pan and zoom, then uses PyTorch Lightning to train and plot cross-entropy, accuracy, macro F1, and mean IoU, then fits a linear classifier on frozen embeddings and compares land cover, predictions, and PCA RGB.
The current Wrap baseline learns broad patterns but misses local land-cover boundaries.

Edit the first code cell to change model dimensions, training steps, batch size, or sampling settings.
The hourglass MLP uses `input_dim → hidden_dim → output_dim`, with equal input/output feature widths and a smaller middle layer. A separate linear head maps the output features to `num_classes` logits.
Defaults: `Wrap (4) → 256 → 128 → 256 → 31`. Raster embeddings come from the 128-dimensional middle layer.
By default, each batch contains 50% random anchors and 50% nearby pixels.
Parallel tile reads and prefetching keep the next batch ready.
Lightning handles the training loop and logs metrics to CSV in `outputs/`. The notebook includes rendered figures from a default run. Run the download/map cell with a live kernel to use the interactive viewer.

Checkpoints are saved in `outputs/`.
Open [inference.ipynb](inference.ipynb) to export the saved checkpoint as float32 embedding and uint8 class COGs. It defaults to the full source grid. Set the output directory and enable `write_global_cogs` there when ready.
The default 128-band global embeddings contain about 31.5 TB before compression, plus temporary files and overviews. The source grid is nominally 100 m in EPSG:4326; ground spacing varies with latitude.

The [source dataset](https://zenodo.org/records/7670653) is by Demuzere, He, Martilli, and Zonato (2023), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The interactive map uses Development Seed’s Lonboard to read COG tiles and overviews through the notebook kernel. No tile server is needed. Keep the kernel running while exploring the map.

Python 3.12–3.14 is supported. `uv sync` creates `.venv`; select `.venv/bin/python` in your editor. Dependencies live in `pyproject.toml`, with resolved versions in `uv.lock`. Use `uv add PACKAGE` to add a dependency.

Plotting and interactive COG display live in `plot.py`, keeping the notebook focused on data, training, and embeddings.
