import csv
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from async_geotiff import GeoTIFF
from lonboard import Map, RasterLayer
from lonboard.raster import EncodedImage
from matplotlib.colors import ListedColormap
from obstore.store import LocalStore
from PIL import Image


def read_palette(path, class_codes):
    colors = np.zeros((256, 4), dtype="uint8")
    with Path(path).open() as stream:
        for row in csv.reader(stream):
            if int(row[0]) in class_codes:
                colors[int(row[0])] = list(map(int, row[1:5]))
    return colors


async def plot_cog(source, colors, height=600):
    source = Path(source)
    geotiff = await GeoTIFF.open(
        source.name, store=LocalStore(str(source.parent.resolve()))
    )

    def render_tile(tile):
        buffer = BytesIO()
        Image.fromarray(colors[tile.array.data[0]]).save(buffer, format="PNG")
        return EncodedImage(data=buffer.getvalue(), media_type="image/png")

    layer = RasterLayer.from_geotiff(geotiff, render_tile=render_tile)
    return Map(layer, height=height)


def plot_metrics(path, output_path):
    with Path(path).open() as stream:
        history = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), layout="constrained")
    for ax, key, label in zip(
        axes.ravel(),
        ["cross_entropy", "accuracy", "macro_f1", "miou"],
        ["Cross-entropy", "Accuracy", "Macro F1", "Mean IoU"],
    ):
        keys = [(key, "Validation")]
        if key in ("cross_entropy", "accuracy"):
            keys.insert(
                0,
                ("train_loss" if key == "cross_entropy" else "train_accuracy", "Train"),
            )
        for metric, split in keys:
            rows = [row for row in history if row.get(metric)]
            ax.plot(
                [int(row["step"]) for row in rows],
                [float(row[metric]) for row in rows],
                label=split,
            )
        ax.set(xlabel="Training step", ylabel=label)
        if key != "cross_entropy":
            ax.set_ylim(0, 1)
    axes[0, 0].legend()
    fig.savefig(output_path, dpi=160)
    plt.show()


def plot_comparison(
    truth, predictions, rgb, colors, class_codes, hidden_dim, output_path
):
    valid = truth >= 0
    cmap = ListedColormap(colors[class_codes, :3] / 255)
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), layout="constrained")
    for ax, (title, values) in zip(
        [axes[0, 0], axes[1, 0], axes[1, 1]],
        [("Land cover", truth), *predictions.items()],
    ):
        ax.imshow(
            np.ma.array(values, mask=~valid),
            cmap=cmap,
            vmin=-0.5,
            vmax=len(class_codes) - 0.5,
            interpolation="nearest",
        )
        ax.set_title(title)
    axes[0, 1].imshow(np.where(valid[..., None], rgb, 0), interpolation="nearest")
    axes[0, 1].set_title(f"{hidden_dim}-D bottleneck → PCA RGB")
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()
