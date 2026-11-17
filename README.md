<h1 align="center">Neural Earth Fields</h1>
<p align="center">Learning continuous representations of our planet.</p>
<p align="center"><a href="https://neuralearthfields.github.io/">Benchmark &amp; Hackathon</a> · Starter notebook</p>

Train a model that maps longitude/latitude to one of 11 [ESA WorldCover](https://esa-worldcover.org) land-cover classes. Two scores:

- **Most Efficient Overfitter:** memorize all 5,504,940 points of a 5 km land grid with as few stored values as possible. Efficiency score S = P + E², with P the stored values in your ONNX file and E the misclassified points; lower is better, exact fits (E = 0) rank first.
- **Overparameterized Interpolator:** accuracy at held-out 1 km locations, with no penalty on model size; higher is better. The starter notebook estimates it on a small set of random test points; the final evaluation runs separately on the 1 km grid.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralearthfields/neuralearthfields-starter/blob/main/starter.ipynb)

## Get started

```sh
uv sync
uv run jupyter lab starter.ipynb
```

[starter.ipynb](starter.ipynb) downloads the data (34 MB, once), trains a Wrap encoder + MLP baseline (~1M parameters), exports it to `outputs/model.onnx`, packs `outputs/submission.zip` (the file you upload to the competition) and computes both scores. For very large models set `external_weights = True` in `Config`: the weights then go into a separate `model.onnx.data` file that is packed into the same ZIP. Change the model and the `Config` class to try your own ideas. Trained models are cached in `checkpoints/`; set `retrain = True` to train again.

## Submission format

One ONNX file with input `coords` (float32 `[batch, 2]`, longitude/latitude in degrees) and output `logits` (`[batch, 11]`, WorldCover classes in code order).

Data: ESA WorldCover 10 m 2021 v200, © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), sampled at HEALPix points by [data-preprocessing](https://github.com/neuralearthfields/data-preprocessing).
