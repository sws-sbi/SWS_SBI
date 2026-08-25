from mouse_dataset import MouseDataset
from video import save_trials_grid_video
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("./")
PIXEL_BRAIN_MAPPING_MOUSE1_SAVE_PATH = (
    PROJECT_ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse1.npy"
)
PIXEL_BRAIN_MAPPING_MOUSE2_SAVE_PATH = (
    PROJECT_ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse2.npy"
)
from typing import Optional, Sequence


def apply_processing_steps(
    ds: MouseDataset,
    processing_steps: Optional[Sequence[str]] = None,
    *,
    fs: float = 25.0,
    band_low: float = 0.5,
    band_high: float = 4.0,
    band_order: int = 6,
    detrend_order: int = 1,
    zscore_eps: float = 1e-8,
    dff_no_zero_denom: float = 1e-6,
    denoise_K: int = 50,
    fft_lam: float = 1e-2,
    spikes: bool = False,
    save_videos: bool = True,
    step_labels: Sequence[str] = None,
) -> None:

    if processing_steps is None:
        processing_steps = []

    only_in_mask = "mask" in processing_steps

    def mask():
        print("applying mouse masks")
        ds.mask(only_in_mask=only_in_mask)

    def subtract_background():
        print("subtracting background")
        ds.subtract_background(only_in_mask=only_in_mask)

    def detrend():
        print(f"removing trend order={detrend_order}")
        ds.detrend(order=detrend_order, only_in_mask=only_in_mask)

    def bandpass():
        print(
            f"applying bandpass (fs={fs}, low={band_low}, high={band_high}, order={band_order})"
        )
        ds.bandpass(
            fs=fs,
            low=band_low,
            high=band_high,
            order=band_order,
            only_in_mask=only_in_mask,
        )

    def minmax_norm():
        print("normalizing (minmax)")
        ds.minmax_normalize(only_in_mask=only_in_mask)

    def zscore_norm():
        print("normalizing (z-score)")
        ds.zscore_normalize(eps=zscore_eps, only_in_mask=only_in_mask)

    def dFF_norm():
        print("converting to ΔF/F (airPLS baseline)")
        ds.deltaFF(no_zero_denom=dff_no_zero_denom, only_in_mask=only_in_mask)

    def denoise():
        print("denoising (SVD/PCA, K={denoise_K})")
        ds.denoise(K=denoise_K, only_in_mask=only_in_mask)

    def fft_deconv():
        print("FFT deconvolve (lam={fft_lam})")
        ds.fft_deconvolve(lam=fft_lam, only_in_mask=only_in_mask)

    def oasis():
        print("denoising + deconvolving using OASIS")
        ds.oasis(only_in_mask=only_in_mask)
        if spikes:
            ds.convert_to_spikes()

    step_fns = {
        "mask": mask,
        "sub_background": subtract_background,
        "detrend": detrend,
        "bandpass": bandpass,
        "minmax_norm": minmax_norm,
        "zscore_norm": zscore_norm,
        "dFF": dFF_norm,
        "denoise": denoise,
        "fft_deconv": fft_deconv,
        "oasis": oasis,
    }

    valid = set(step_fns.keys())
    for i, (step, label) in enumerate(zip(processing_steps, step_labels)):
        if step not in valid:
            raise ValueError(f"Unknown step '{step}'. Valid steps: {sorted(valid)}")
        step_fns[step]()  # run
        if save_videos:
            _save_video(data=ds.data, i=i, step=label)
        print(ds.data.shape)


def run_pipeline(
    *,
    pipeline_number: int,
    processing_steps: Sequence[str],
    save_spikes_or_calcium="calcium" or "spikes",
    save_dataset=True,
    step_labels=[],
    save_videos=True,
):
    """
    valid step strings: "mask", "sub_background", "detrend", "bandpass", "minmax_norm", "zscore_norm", "dFF", "denoise", "fft_deconv", "oasis"
    """
    out_dir = PROJECT_ROOT / "data" / "pipelines"

    ds = MouseDataset(save_contours=True)
    print("loading dataset")
    ds.load()
    if save_spikes_or_calcium == "spikes":
        apply_processing_steps(
            ds=ds,
            processing_steps=processing_steps,
            spikes=True,
            save_videos=save_videos,
            step_labels=step_labels,
        )
    else:
        apply_processing_steps(
            ds=ds,
            processing_steps=processing_steps,
            spikes=False,
            save_videos=save_videos,
            step_labels=step_labels,
        )
    if save_dataset:
        fname = f"{pipeline_number}.npy"
        npy_path = out_dir / fname
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, ds.data)
        print(f"Saved dataset to: {npy_path}")

    return ds.data


def remove_0_from_regions(mapping):
    regions = np.unique(mapping)
    if regions[0] == 0:
        regions = regions[1:]
    else:
        print("that can't be right, first region should be 0!")
    return regions


def activity_per_pixel_to_activity_per_region(
    data,
    region_idxs,
    mouse1_mapping,
    mouse2_mapping,
    contour1,
    contour2,
):
    n_mice, n_trials, n_frames, _, _ = data.shape
    out = np.zeros((n_mice, n_trials, len(region_idxs), n_frames), dtype=float)
    print(len(region_idxs))
    print(region_idxs)
    mappings = [mouse1_mapping, mouse2_mapping]
    contours = [contour1, contour2]

    for m_idx in range(n_mice):
        mapping = mappings[m_idx]
        contour = contours[m_idx]
        for t_idx in range(n_trials):
            for i, r_idx in enumerate(region_idxs):
                sel = (mapping == r_idx) & contour
                region_pixels = data[
                    m_idx, t_idx, :, mapping == r_idx
                ]  # (frames, n_pix)

                if not sel.any():
                    continue  # trace stays zeros

                region_pixels = data[m_idx, t_idx, :, sel]  # (n_pix, frames)
                out[m_idx, t_idx, i, :] = np.median(region_pixels, axis=0)
    return out


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


def save_median_pipeline_data(pipeline):
    contour1 = np.load(
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "contour_mask1.npy"
    )
    contour2 = np.load(
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "contour_mask2.npy"
    )
    mouse1 = np.load(
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse1.npy"
    )
    mouse2 = np.load(
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse2.npy"
    )

    m1_region_idxs = remove_0_from_regions(mouse1)
    m2_region_idxs = remove_0_from_regions(mouse2)

    region_idxs = (
        m1_region_idxs
        if all(m1_region_idxs == m2_region_idxs)
        else print("that can't be right, first region should be 0!")
    )
    data = np.load(PROJECT_ROOT / "data" / "pipelines" / f"{pipeline}.npy")
    median_activity_per_region = activity_per_pixel_to_activity_per_region(
        data=data,
        region_idxs=region_idxs,
        mouse1_mapping=mouse1,
        mouse2_mapping=mouse2,
        contour1=contour1,
        contour2=contour2,
    )
    save_dir = PROJECT_ROOT / "data" / "median_pipelines"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        PROJECT_ROOT / "data" / "median_pipelines" / f"{pipeline}.npy",
        median_activity_per_region,
    )


def save_median_pipeline_video(
    pipeline,
    mice=(1, 2),
    trials=((1, 2, 3), (1, 2, 3)),
    fps=25,
    i=5,
):
    """
    Loads the saved median-region pipeline data, reconstructs pixel-space videos
    for each mouse, and saves a grid video identical in layout to the raw pipeline video.

    median data shape: (n_mice, n_trials, n_regions, n_frames)
    reconstructed pixel data shape fed to save_trials_grid_video: (M, Trials, T, H, W)
    """

    PIXEL_BRAIN_MAPPING_MOUSE1_SAVE_PATH = (
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse1.npy"
    )
    PIXEL_BRAIN_MAPPING_MOUSE2_SAVE_PATH = (
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse2.npy"
    )
    contour1 = np.load(
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "contour_mask1.npy"
    )
    contour2 = np.load(
        PROJECT_ROOT / "data" / "pixel_brain_mappings" / "contour_mask2.npy"
    )

    mouse1_mapping = np.load(PIXEL_BRAIN_MAPPING_MOUSE1_SAVE_PATH)
    mouse2_mapping = np.load(PIXEL_BRAIN_MAPPING_MOUSE2_SAVE_PATH)

    m1_region_idxs = remove_0_from_regions(mouse1_mapping)
    m2_region_idxs = remove_0_from_regions(mouse2_mapping)

    region_idxs = (
        m1_region_idxs
        if all(m1_region_idxs == m2_region_idxs)
        else print("that can't be right, first region should be 0!")
    )

    median_data = np.load(
        PROJECT_ROOT / "data" / "median_pipelines" / f"{pipeline}.npy"
    )
    n_mice, n_trials, n_regions, n_frames = median_data.shape
    mappings = [mouse1_mapping, mouse2_mapping]

    # Reconstruct pixel-space for every mouse × trial
    H, W = mouse1_mapping.shape
    pixel_data = np.zeros((n_mice, n_trials, n_frames, H, W), dtype=float)

    mappings = [mouse1_mapping, mouse2_mapping]
    contours = [contour1, contour2]

    for m_idx, (mapping, contour) in enumerate(zip(mappings, contours)):
        for t_idx in range(n_trials):
            pixel_data[m_idx, t_idx] = activity_per_region_to_activity_per_pixel_single(
                region_trace=median_data[m_idx, t_idx],
                mapping=mapping,
                region_idxs=region_idxs,
                contour=contour,
            )

    _save_video(
        data=pixel_data,
        i=i,
        step="Median",
        mice=mice,
        trials=trials,
    )
    save_path = PROJECT_ROOT / "videos" / f"{i}_Median.mp4"
    print(f"Saved median pipeline video to: {save_path}")


def _save_video(data, i, fps=25, step="", mice=[1], trials=[[1]]):
    mp4_savefolder = PROJECT_ROOT / "videos"
    mp4_savefolder.mkdir(parents=True, exist_ok=True)
    fname = f"{i}_{step}.mp4"
    save_path = mp4_savefolder / fname
    save_trials_grid_video(
        data,
        mice=mice,
        trials=trials,
        save_path=save_path,
        fps=fps,
        global_scale=True,
        label=step,
    )
    print(f"Saved video to: {save_path}")


def main(
    generate_pipeline=False,
    save_video=False,
    generate_median_pipeline=True,
    save_median_video=False,
    mice=[1],
    trials=[[1]],
):
    pipelines = [
        (
            2,
            ["mask", "dFF", "oasis", "bandpass", "minmax_norm"],
            "calcium",
            ["Masked", r"Fluorescence", "Denoised", "Filtered", "Normalized"],
        )
    ]
    for pipeline in pipelines:
        if generate_pipeline:
            run_pipeline(
                pipeline_number=pipeline[0],
                processing_steps=pipeline[1],
                save_spikes_or_calcium=pipeline[2],
                save_dataset=True,
                save_videos=save_video,
                step_labels=pipeline[3],
            )
        if generate_median_pipeline:
            save_median_pipeline_data(pipeline=pipeline[0])
            if save_median_video:
                save_median_pipeline_video(
                    pipeline=pipeline[0],
                    mice=mice,
                    trials=trials,
                    fps=25,
                )
        else:
            if save_median_video:
                save_median_pipeline_video(
                    pipeline=pipeline[0],
                    mice=mice,
                    trials=trials,
                    fps=25,
                    i=len(pipeline[1]),
                )


if __name__ == "__main__":
    main(
        generate_pipeline=True,
        save_video=True,
        generate_median_pipeline=True,
        save_median_video=True,
        mice=[1],
        trials=[[1]],
    )
