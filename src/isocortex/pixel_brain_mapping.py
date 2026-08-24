import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.transform import resize
from skimage.segmentation import expand_labels

from utils.allensdk import (
    save_masks,
    get_reference_space,
    create_reference_space_from_regions,
)
from utils.plotting import plot_flat_hover_map


# --------------------------------------------------------------------------
# Colors
# --------------------------------------------------------------------------
def make_unique_colors(df, step=100):
    """Nudge duplicate RGB triplets apart so every region gets its own color."""

    def clamp(x):
        return max(0, min(255, int(x)))

    groups = defaultdict(list)
    for j in range(df.shape[1]):
        rgb = (int(df.iat[0, j]), int(df.iat[1, j]), int(df.iat[2, j]))
        groups[rgb].append(j)

    for (r, g, b), idxs in groups.items():
        for i, j in enumerate(idxs[1:], start=1):
            df.iat[0, j] = clamp(r + i * step)
            df.iat[1, j] = clamp(g + i * step)
            df.iat[2, j] = clamp(b + i * step)
    return df


def df_to_color_map(df):
    """[R,G,B] rows, region id as column label -> {id: (r, g, b)}."""
    color_map = {
        int(col): (int(df.iat[0, j]), int(df.iat[1, j]), int(df.iat[2, j]))
        for j, col in enumerate(df.columns)
    }
    color_map[0] = (0, 0, 0)  # background
    return color_map


def region_id_frame(structures, regions):
    """Two rows: ids, then acronyms - in the order given by `regions`."""
    acr_to_id = {node["acronym"]: int(node["id"]) for node in structures}
    missing = [a for a in regions if a not in acr_to_id]
    if missing:
        print(f"warning: not in structure tree, skipped: {missing}")
    ordered = [a for a in regions if a in acr_to_id]
    return pd.DataFrame([[acr_to_id[a] for a in ordered], ordered])


# --------------------------------------------------------------------------
# Flat map construction
# --------------------------------------------------------------------------
def flatten_bottom_to_top(annotation):
    """Project the volume onto the coronal/sagittal plane, top-most label wins."""
    flat = np.zeros((annotation.shape[0], annotation.shape[2]))
    for i in range(annotation.shape[1]):
        sl = annotation[:, -i, :]
        mask = sl != 0
        flat[mask] = sl[mask]
    return flat


def apply_sobel_and_gauss_filter(array, sigma):
    smoothed = gaussian_filter(sobel(array.astype(float)), sigma=sigma)
    edge_mask = smoothed > 0

    filtered = array.copy()
    filtered[edge_mask] = 0
    return filtered, edge_mask


def shift_up(array, n):
    """Shift content up by n rows, zero-filling the bottom."""
    out = np.zeros_like(array)
    if n < array.shape[0]:
        out[: array.shape[0] - n, :] = array[n:, :]
    return out


def downsample(array, target_h, target_w):
    return resize(
        array, (target_h, target_w), order=0, preserve_range=True, anti_aliasing=False
    ).astype(array.dtype)


def write(path, writer, save=True):
    if not save:
        print(f"[save=False] would write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    writer(path)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
def main(
    # --- inputs ---
    project_root=Path("./"),
    regions=[
        "MOp",
        "MOs",
        "SSp-n",
        "SSp-bfd",
        "SSp-ll",
        "SSp-m",
        "SSp-ul",
        "SSp-tr",
        "VISam",
        "RSPagl",
        "RSPd",
        "PTLp",
    ],
    allen_dir=None,  # default: project_root/data/allen_data/{res}_micron_resolution
    raw_dataset=None,  # default: project_root/data/raw_dataset
    save=True,
    mask_dir=None,  # structure_{id}.nrrd cache; default allen_dir/ccf_2017
    colormap_csv=None,  # default: <mappings>/unique_colormap.csv
    mouse1_map=None,  # default: <mappings>/pixel_brain_map_mouse1.npy
    mouse2_map=None,  # default: <mappings>/pixel_brain_map_mouse2.npy
    # --- pipeline ---
    color_step=100,  # RGB offset applied to duplicate region colors
    crop_coronal=(110, 325),
    crop_sagittal=(185, 400),
    sigma=0.0,  # gaussian sigma on the sobel edges
    expand_distance=0,  # pixels to grow labels back into the edge gaps
    crop_factor=7,  # rows mouse 2's map is shifted up by
    mouse2_shift="both",  # "full" | "downsampled" | "both" (notebook = both)
    target_size=(100, 100),
    # --- display ---
    show_2d=False,
    plot_width=800,
    plot_height=800,
):
    mappings = project_root / "data" / "pixel_brain_mappings"
    allen_dir = allen_dir or (
        project_root / "data" / "allen_data" / f"{25}_micron_resolution"
    )
    raw_dataset = raw_dataset or project_root / "data" / "raw_dataset"
    mask_dir = mask_dir or allen_dir / "ccf_2017"
    colormap_csv = colormap_csv or mappings / "unique_colormap.csv"
    mouse1_map = mouse1_map or mappings / "pixel_brain_map_mouse1.npy"
    mouse2_map = mouse2_map or mappings / "pixel_brain_map_mouse2.npy"

    sys.path.insert(0, str(project_root.resolve()))

    def show(flat, boundaries=True):
        if show_2d:
            plot_flat_hover_map(
                flat=flat,
                structure_tree=structure_tree,
                show_boundaries=boundaries,
                width=plot_width,
                height=plot_height,
                cmap=color_map,
            )

    # --- reference space + masks -----------------------------------------
    rsp = get_reference_space(allen_dir)
    structure_tree = rsp.structure_tree

    structures = structure_tree.get_structures_by_acronym(list(regions))
    region_ids = [node["id"] for node in structures]
    acronyms = [node["acronym"] for node in structures]  # tree order, not input order
    print(f"{len(region_ids)} regions: {dict(zip(acronyms, region_ids))}")

    region_csv = mappings / "id_acronym_lookup.csv"
    write(
        region_csv,
        lambda p: region_id_frame(structures, regions).to_csv(
            p, index=False, header=False
        ),
        save,
    )

    save_masks(rsp, region_ids, mask_dir)
    rsp_from_masks = create_reference_space_from_regions(
        rsp, acronyms, region_ids, mask_dir
    )

    # --- colors ----------------------------------------------------------
    color_map_df = pd.DataFrame(structure_tree.get_colormap())[region_ids]
    unique_color_map_df = make_unique_colors(color_map_df.copy(), step=color_step)
    color_map = df_to_color_map(unique_color_map_df)

    write(
        colormap_csv,
        lambda p: unique_color_map_df.to_csv(p, index=False, header=False),
        save,
    )

    # --- flatten + crop --------------------------------------------------
    flat = flatten_bottom_to_top(rsp_from_masks.annotation)
    show(flat)

    areas = flat[crop_coronal[0] : crop_coronal[1], crop_sagittal[0] : crop_sagittal[1]]
    show(flat)

    # --- edge filter, then grow labels back over the gaps ----------------
    filtered, edge_mask = apply_sobel_and_gauss_filter(areas, sigma=0.2)

    labels_clean = filtered.copy()
    labels_clean[edge_mask] = 0
    areas = expand_labels(labels_clean, distance=5)

    filtered, edge_mask = apply_sobel_and_gauss_filter(areas, sigma=sigma)
    show(filtered)

    labels_clean = filtered.copy()
    labels_clean[edge_mask] = 0
    merged = expand_labels(labels_clean, distance=expand_distance)
    rotated = np.rot90(merged, axes=(1, 0))
    show(rotated)

    # --- per-mouse alignment + downsample --------------------------------
    target_h, target_w = target_size
    mouse1 = downsample(rotated, target_h, target_w)

    if mouse2_shift in ("full", "both"):
        mouse2 = shift_up(rotated, crop_factor)
    else:
        mouse2 = rotated.copy()
    show(mouse2)

    mouse2 = downsample(mouse2, target_h, target_w)
    if mouse2_shift in ("downsampled", "both"):
        mouse2 = shift_up(mouse2, crop_factor)

    show(mouse1)
    show(mouse2)

    write(mouse1_map, lambda p: np.save(p, mouse1), save)
    write(mouse2_map, lambda p: np.save(p, mouse2), save)

    return mouse1, mouse2, color_map


if __name__ == "__main__":
    main(
        sigma=0.1,
        expand_distance=0,
        show_2d=False,
    )
