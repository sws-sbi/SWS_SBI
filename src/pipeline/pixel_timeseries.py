"""
Save pixel timeseries plots for each pipeline step to disk.

Usage:
    python save_pixel_timeseries.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mouse_dataset import MouseDataset

# --------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------
ROOT = Path("./")
DATA_PATH = ROOT / "data" / "pipeline_in_steps"
REGION_MAP_PATH = ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse1.npy"
SAVE_FOLDER = ROOT / "figures" / "pixel_timeseries"

TITLES = [
    "Masked Data",
    r"Normalized Fluorescence ($F / F_0$)",
    "Denoised",
    "Bandpass Filtered (0.5 Hz - 4.0 Hz)",
    "Normalized",
]


# --------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------
def load_data_in_regions(step, region_map):
    """Load the data for a given pipeline step, masking to valid brain regions."""
    data = np.load(DATA_PATH / f"{step}.npy")[0, 0]
    if step != 0:
        mask = region_map != 0
        data = data[:, mask]
    return data


def get_reference_pixel(contour_mask, region_map):
    """Pick a stable reference pixel that is valid in both the contour and region masks."""
    valid_pixels = np.argwhere(contour_mask & (region_map != 0))
    if len(valid_pixels) == 0:
        raise ValueError("No pixels are valid in both the contour mask and region map.")
    mid = valid_pixels[len(valid_pixels) // 2]
    return mid


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_pixel_timeseries(
    x,
    y,
    mask,
    step,
    region_map,
    xlabel_ts=None,
    ylabel_ts=None,
    title_ts=None,
    savepath="./plot.png",
):
    data = load_data_in_regions(step, region_map)

    if step == 6:
        ts_data = data[6, 100:300]
    elif step == 0:
        ts_data = data[100:300, x, y]
    else:
        pixel_idx = np.flatnonzero(mask.ravel())
        target_flat = x * mask.shape[1] + y
        idx_in_mask = np.searchsorted(pixel_idx, target_flat)
        ts_data = data[100:300, idx_in_mask]

    fig, ax_ts = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))

    ax_ts.plot(ts_data, color="#23008D", linewidth=1.2)
    ax_ts.spines[["top", "right"]].set_visible(False)

    if xlabel_ts:
        ax_ts.set_xlabel(xlabel_ts, fontsize=12, labelpad=8)
    if ylabel_ts:
        ax_ts.set_ylabel(ylabel_ts, fontsize=12, labelpad=8)
    if title_ts:
        ax_ts.set_title(
            f"{title_ts} - Pixel[{x}, {y}]",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

    plt.tight_layout()

    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to {savepath}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    region_map = np.load(REGION_MAP_PATH)

    ds = MouseDataset()
    ds.load()
    contour_mask = ds._mask(1)  # mask for mouse 1

    mid = get_reference_pixel(contour_mask, region_map)
    print("Reference pixel (row, col):", mid)

    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    for step, title in enumerate(TITLES):
        plot_pixel_timeseries(
            x=mid[0],
            y=mid[1],
            mask=contour_mask,
            step=step,
            region_map=region_map,
            xlabel_ts="Frame",
            ylabel_ts="Pixel Value",
            title_ts=title,
            savepath=SAVE_FOLDER / f"{step}.png",
        )


if __name__ == "__main__":
    main()
