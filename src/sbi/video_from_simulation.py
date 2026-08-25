from ..pipeline.video import save_trials_grid_video
import numpy as np
from pathlib import Path
import pandas as pd
from ..wc_model import run

PROJECT_ROOT = Path("./")



def remove_0_from_regions(mapping):
    regions = np.unique(mapping)
    if regions[0] == 0:
        regions = regions[1:]
    else:
        print("that can't be right, first region should be 0!")
    return regions


def activity_per_region_to_activity_per_pixel_single(
    region_trace, mapping, region_idxs, contour=None
):
    """
    region_trace: (n_regions, n_frames)
    mapping:      (H, W) int region ids
    contour:      (H, W) bool, optional — pixels outside stay 0
    returns:      (n_frames, H, W)
    """
    n_regions, n_frames = region_trace.shape
    H, W = mapping.shape
    out = np.zeros((n_frames, H, W), dtype=float)

    if contour is None:
        contour = np.ones_like(mapping, dtype=bool)
    else:
        contour = contour.astype(bool)

    region_pixels = [np.where((mapping == rid) & contour) for rid in region_idxs]

    for r in range(n_regions):
        ys, xs = region_pixels[r]
        if ys.size == 0:
            continue
        out[:, ys, xs] = region_trace[r, :][:, None]

    return out


def save_median_pipeline_video(label, data):
    PIXEL_BRAIN_MAPPING_MOUSE1_SAVE_PATH = (
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse1.npy"
    )
    contour1 = np.load(
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "contour_mask1.npy"
    )

    mouse1_mapping = np.load(PIXEL_BRAIN_MAPPING_MOUSE1_SAVE_PATH)
    region_idxs = remove_0_from_regions(mouse1_mapping)
    n_mice, n_trials, n_regions, n_frames = data.shape

    # Reconstruct pixel-space for every mouse × trial
    H, W = mouse1_mapping.shape
    pixel_data = np.zeros((n_mice, n_trials, n_frames, H, W), dtype=float)

    mappings = [mouse1_mapping]
    contours = [contour1]

    for m_idx, (mapping, contour) in enumerate(zip(mappings, contours)):
        for t_idx in range(n_trials):
            pixel_data[m_idx, t_idx] = activity_per_region_to_activity_per_pixel_single(
                region_trace=data[m_idx, t_idx],
                mapping=mapping,
                region_idxs=region_idxs,
                contour=contour,
            )

    _save_video(
        data=pixel_data,
        i=label,
        mice=[1],
        trials=[[1]],
    )


def _save_video(data, i, fps=25, mice=[1], trials=[[1]]):
    mp4_savefolder = PROJECT_ROOT / "videos"
    mp4_savefolder.mkdir(parents=True, exist_ok=True)
    fname = f"{i}.mp4"
    save_path = mp4_savefolder / fname
    save_trials_grid_video(
        data,
        mice=mice,
        trials=trials,
        save_path=save_path,
        fps=fps,
        global_scale=True,
        label=i,
    )
    print(f"Saved video to: {save_path}")


def main():
    table_indices = get_table_indices()

    theta_map = {
        "I_e": -0.399,
        "I_i": 0.4353,
        "ou": 0.1534,
        "g_A": 0.8128,
        "g_L": 0.1649,
        "ei_scaling": 0.4191,
        'B': -3.759,
        "tau_e": 93.2935,
        "tau_i": 61.9805,
        'tau_m': 711.9818,
        "w_ee": 4.2067,
        "w_ii": 4.6362,
        "w_ei": 12.5907,
        "w_ie": 1.2809,
    }
    theta_tilde = {
        "I_e": -0.399,
        "I_i": 0.4353,
        "ou": 0.1534,
        "g_A": 0.8128,
        "g_L": 0.1649,
        "ei_scaling": 0.4191,
        "B": -2.5,
        "tau_e": 93.2935,
        "tau_i": 61.9805,
        "tau_m": 800.1443,
        "w_ee": 4.2067,
        "w_ii": 4.6362,
        "w_ei": 12.5907,
        "w_ie": 1.2809,
    }



    save_median_pipeline_video("MAP", _simulate(theta_map, table_indices))
    save_median_pipeline_video("Perturbed", _simulate(theta_tilde, table_indices))


def get_table_indices():
    mouse_mapping = np.load(
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse1.npy"
    )
    id_acronym_df = pd.read_csv(
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "id_acronym_lookup.csv",
        header=0,
        index_col=False,
    )
    id_acronym_dict = {
        int(k): v for k, v in id_acronym_df.to_dict(orient="records")[0].items()
    }
    region_ids = np.unique(mouse_mapping)[1:]
    region_acronyms = [id_acronym_dict[rid] for rid in region_ids]
    local_dist_df = pd.read_excel(
        PROJECT_ROOT / "data" / "tables" / "local.xlsx",
        sheet_name="distances",
        index_col=0,
    )
    return [local_dist_df.columns.get_loc(a) for a in region_acronyms]


def _simulate(theta_dict, table_indices):
    sim = np.asarray(run(**theta_dict), dtype=np.float32).T[table_indices]

    if not np.isfinite(sim).all():
        raise ValueError("simulation produced NaN or infinite activity values")

    lo, hi = float(sim.min()), float(sim.max())
    sim = np.zeros_like(sim) if (hi - lo) <= 1e-12 else (sim - lo) / (hi - lo)
    return sim[None, None]

if __name__ == "__main__":
    main()
