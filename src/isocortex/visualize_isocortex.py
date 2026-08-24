#!/usr/bin/env python3
"""Render the isocortex regions as 3D surfaces inside a translucent brain outline,
with regions missing from the connectivity matrix highlighted in red.
"""

import numpy as np
from pathlib import Path

from utils.plotting import plot_surfaces_and_injections
from utils.allensdk import (
    create_reference_space_from_regions,
    get_reference_space,
    save_masks,
)

ROOT_ID = 997
ROOT_ACRONYM = "root"

ISOCORTEX = [
    "FRP",
    "MOp",
    "MOs",
    "SSp-n",
    "SSp-bfd",
    "SSp-ll",
    "SSp-m",
    "SSp-ul",
    "SSp-tr",
    "SSp-un",
    "SSs",
    "GU",
    "VISC",
    "AUDd",
    "AUDp",
    "AUDpo",
    "AUDv",
    "VISal",
    "VISam",
    "VISl",
    "VISp",
    "VISpl",
    "VISpm",
    "VISli",
    "VISpor",
    "ACAd",
    "ACAv",
    "PL",
    "ILA",
    "ORBl",
    "ORBm",
    "ORBvl",
    "AId",
    "AIp",
    "AIv",
    "RSPagl",
    "RSPd",
    "RSPv",
    "PTLp",
    "TEa",
    "PERI",
    "ECT",
]

MISSING_REGIONS = [
    "SSp-un",
    "AUDpo",
    "VISli",
    "VISpor",
]  # Regions not in connectome matrix


def build_color_maps(
    structure_tree,
    region_ids,
    acronyms,
    missing_regions,
    missing_color,
    root_opacity,
    region_opacity,
):
    """id -> acronym, id -> RGB and id -> opacity, root included."""
    base_colormap = structure_tree.get_colormap()

    id_acronym_lookup = dict(zip(region_ids, acronyms))
    id_acronym_lookup[ROOT_ID] = ROOT_ACRONYM

    region_color_map = {sid: base_colormap[sid] for sid in id_acronym_lookup}
    id_to_opacity = {sid: region_opacity for sid in id_acronym_lookup}
    id_to_opacity[ROOT_ID] = root_opacity

    missing_ids = [
        node["id"]
        for node in structure_tree.get_structures_by_acronym(list(missing_regions))
    ]
    for sid in missing_ids:
        if sid not in region_color_map:
            print(f"  warning: missing region {sid} is not in the plotted set")
        region_color_map[sid] = list(missing_color)

    return id_acronym_lookup, region_color_map, id_to_opacity


def separate_regions(annotation, root_annotation, region_ids, dtype=np.int32):
    """(n_regions + 1, *volume) stack: root first, then one volume per region."""
    stack = np.zeros((len(region_ids) + 1, *annotation.shape), dtype=dtype)
    stack[0] = root_annotation

    for i, sid in enumerate(region_ids):
        region_data = annotation.copy()
        region_data[region_data != sid] = 0
        stack[i + 1] = region_data

    return stack


def main(
    # --- inputs ---
    regions=ISOCORTEX,
    missing_regions=MISSING_REGIONS,
    # --- appearance ---
    missing_color=(255, 0, 0),
    region_opacity=1,
    root_opacity=0.05,
    default_opacity=0.8,
    top_outline=True,
    top_outline_width=1,
    plot_width=800,
    plot_height=800,
    dtype=np.int32,
):
    allen_dir = Path("./") / "data" / "allen_data" / "25_micron_resolution"
    mask_dir = allen_dir / "ccf_2017"
    # --- reference space ---------------------------------------------------
    rsp = get_reference_space(allen_dir)
    structure_tree = rsp.structure_tree

    structures = structure_tree.get_structures_by_acronym(list(regions))
    region_ids = [node["id"] for node in structures]
    acronyms = [node["acronym"] for node in structures]  # tree order, not input order
    print(f"{len(region_ids)} regions requested at {25} um")
    save_masks(rsp, region_ids, mask_dir)
    rsp_from_masks = create_reference_space_from_regions(
        rsp=rsp,
        acronyms=acronyms,
        ids=region_ids,
        mask_dir=mask_dir,
    )

    save_masks(rsp, ids_to_save=[997], mask_dir=mask_dir)
    # --- brain outline -----------------------------------------------------
    root_annotation = create_reference_space_from_regions(
        rsp=rsp,
        acronyms=[ROOT_ACRONYM],
        ids=[ROOT_ID],
        mask_dir=mask_dir,
    ).annotation

    # --- colors / opacities ------------------------------------------------
    id_acronym_lookup, region_color_map, id_to_opacity = build_color_maps(
        structure_tree,
        region_ids,
        acronyms,
        missing_regions,
        missing_color,
        root_opacity,
        region_opacity,
    )

    # --- one volume per region, root first ---------------------------------
    surfaces = separate_regions(
        rsp_from_masks.annotation, root_annotation, region_ids, dtype=dtype
    )
    print(f"surface stack: {surfaces.shape} ({surfaces.nbytes / 1e9:.1f} GB)")

    # --- plot --------------------------------------------------------------
    fig = plot_surfaces_and_injections(
        surfaces=surfaces,
        id_to_color=region_color_map,
        id_to_acronym=id_acronym_lookup,
        width=plot_width,
        height=plot_height,
        id_to_opacity=id_to_opacity,
        default_opacity=default_opacity,
        top_outline=top_outline,
        top_outline_width=top_outline_width,
    )

    fig.show()

    return fig


if __name__ == "__main__":
    main()
