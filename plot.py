import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sns
from IPython.display import Image, display
from matplotlib.colors import ListedColormap
from rasterio.windows import from_bounds
from sklearn.decomposition import PCA

sns.set_theme(
    context="notebook",
    style="whitegrid",
    font="serif",
    rc={"font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"]},
)


def show_figure(fig, output_path):
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(output_path), width=round(fig.get_figwidth() * 100)))


def read_palette(source, class_codes):
    colors = np.zeros((256, 4), dtype="uint8")
    with rasterio.open(source) as src:
        palette = src.colormap(1)
    for code in class_codes:
        colors[code] = palette[int(code)]
    return colors


def plot_land_cover(source, colors, output_path):
    views = {
        "Global": None,
        "Western Europe": (-12, 35, 20, 60),
        "California": (-124, 32, -114, 42),
        "Sierra Nevada": (-121, 37, -119, 39),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    with rasterio.open(source) as src:
        for ax, (title, bounds) in zip(axes.ravel(), views.items()):
            bounds = bounds or tuple(src.bounds)
            window = from_bounds(*bounds, transform=src.transform)
            aspect = (bounds[2] - bounds[0]) / (bounds[3] - bounds[1])
            labels = src.read(1, window=window, out_shape=(1200, round(1200 * aspect)))
            ax.imshow(
                colors[labels],
                extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
                interpolation="nearest",
            )
            ax.grid(False)
            ax.set(
                title=title, xlabel="Longitude", ylabel="Latitude", facecolor="#eeeeee"
            )
    show_figure(fig, output_path)


def pca_rgb(fit_features, features, shape, seed=0):
    pca = PCA(3, random_state=seed).fit(fit_features)
    lo, hi = np.percentile(pca.transform(fit_features), [2, 98], axis=0)
    rgb = np.clip((pca.transform(features) - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.round(3)}")
    return rgb.reshape(*shape, 3)


def preview_grid(source, height=256):
    with rasterio.open(source) as src:
        width = round(height * src.width / src.height)
        labels = src.read(1, out_shape=(height, width))
        rows, cols = np.indices(labels.shape)
        rows = np.minimum(
            ((rows + 0.5) * src.height / height).astype(int), src.height - 1
        )
        cols = np.minimum(((cols + 0.5) * src.width / width).astype(int), src.width - 1)
        lon, lat = src.transform * (cols.ravel() + 0.5, rows.ravel() + 0.5)
    return labels, np.column_stack((lon, lat)).astype("float32")


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
    show_figure(fig, output_path)


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
        ax.grid(False)
        ax.set_axis_off()
    show_figure(fig, output_path)
