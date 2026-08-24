from mouse_dataset import MouseDataset
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
    for step in processing_steps:
        if step not in valid:
            raise ValueError(f"Unknown step '{step}'. Valid steps: {sorted(valid)}")
        step_fns[step]()  # run
        print(ds.data.shape)


def run_pipeline(
    *,
    pipeline_number: int,
    processing_steps: Sequence[str],
    save_spikes_or_calcium="calcium" or "spikes",
    save_dataset=True,
):
    """
    valid step strings: "mask", "sub_background", "detrend", "bandpass", "minmax_norm", "zscore_norm", "dFF", "denoise", "fft_deconv", "oasis"
    """
    out_dir = PROJECT_ROOT / "data" / "pipeline_in_steps"

    ds = MouseDataset(save_contours=True)
    print("loading dataset")
    ds.load()
    if save_spikes_or_calcium == "spikes":
        apply_processing_steps(
            ds=ds,
            processing_steps=processing_steps,
            spikes=True,
        )
    else:
        apply_processing_steps(
            ds=ds,
            processing_steps=processing_steps,
            spikes=False,
        )
    if save_dataset:
        fname = f"{pipeline_number}.npy"
        npy_path = out_dir / fname
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, ds.data)
        print(f"Saved dataset to: {npy_path}")
    return ds.data






def main():
    pipelines = [
    (0, ["mask"], "calcium"),
    (1, ["mask", "dFF"], "calcium"),
    (2, ["mask", "dFF", "oasis"], "calcium"),
    (3, ["mask", "dFF", "oasis", "bandpass"], "calcium"),
    (4, ["mask", "dFF", "oasis", "bandpass", "minmax_norm"], "calcium"),
]
    for pipeline in pipelines:
        run_pipeline(
            pipeline_number=pipeline[0],
            processing_steps=pipeline[1],
            save_spikes_or_calcium=pipeline[2],
            save_dataset=True,
        )


if __name__ == "__main__":
    main()
