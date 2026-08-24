from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
from oasis.functions import deconvolve
from scipy.signal import butter, filtfilt
from skimage import measure
from skimage.draw import polygon

from airPLS import airPLS

PROJECT_ROOT = Path("./")
DATASET_DIR = PROJECT_ROOT / "data" / "raw_dataset"
PIXEL_BRAIN_MAPPING_MOUSE1_SAVE_PATH = (
    PROJECT_ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse1.npy"
)
PIXEL_BRAIN_MAPPING_MOUSE2_SAVE_PATH = (
    PROJECT_ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse2.npy"
)

Array3D = np.ndarray  # (T,H,W)
Array5D = np.ndarray  # (M,N,T,H,W)
Array2D = np.ndarray  # (H,W)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)


def _largest_contour_mask(
    img2d: Array2D, shape=(100, 100), levels=np.linspace(0.2, 0.8, 1000)
) -> Array2D:
    best, best_area = None, -1.0
    for lvl in levels:
        cs = measure.find_contours(img2d, level=float(lvl))
        if not cs:
            continue
        c = max(cs, key=len)
        x, y = c[:, 1], c[:, 0]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area > best_area:
            best, best_area = c, area
    if best is None:
        raise RuntimeError("No contour found.")
    rr, cc = polygon(best[:, 0], best[:, 1], shape)
    m = np.zeros(shape, bool)
    m[rr, cc] = True
    return m


def get_masks(
    data: Array5D,
    save: bool = True,
    save_dir: Path = PROJECT_ROOT / "data" / "pixel_brain_mappings",
) -> tuple[Array2D, Array2D]:
    # mean frame per mouse: average over trials, then time
    mean1 = data[0].mean(axis=0).mean(axis=0)
    mean2 = data[1].mean(axis=0).mean(axis=0)

    m1 = _largest_contour_mask(_minmax(mean1))
    m2 = _largest_contour_mask(_minmax(mean2))

    reg1 = np.load(PIXEL_BRAIN_MAPPING_MOUSE1_SAVE_PATH)
    reg2 = np.load(PIXEL_BRAIN_MAPPING_MOUSE2_SAVE_PATH)

    # keep only pixels that are (a) in brain contour and (b) mapped to a region
    m1 &= np.where(m1, reg1, 0) != 0
    m2 &= np.where(m2, reg2, 0) != 0

    if save:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "contour_mask1.npy", m1)
        np.save(save_dir / "contour_mask2.npy", m2)
        print(f"wrote contour masks to {save_dir}")

    return m1, m2


# ---------------------------
# Per-trial operations
# ---------------------------
def mask_trial(Y: Array3D, mask: Array2D) -> Array3D:
    binary_mask = mask != 0
    return Y * binary_mask


def apply_background_subtraction(Y: Array3D) -> Array3D:
    return Y - Y.mean(axis=0)


def apply_detrend(Y: Array3D, order: int = 1) -> Array3D:
    T, H, W = Y.shape
    t = np.linspace(-1.0, 1.0, T)
    X = np.vstack([t**k for k in range(order + 1)]).T  # (T, p)
    coeffs, *_ = np.linalg.lstsq(X, Y.reshape(T, -1), rcond=None)
    return Y - (X @ coeffs).reshape(T, H, W)


def apply_bandpass(Y: Array3D, fs: float, low=0.5, high=4.0, order=6) -> Array3D:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    T, H, W = Y.shape
    return filtfilt(b, a, Y.reshape(T, -1), axis=0).reshape(T, H, W)


def normalize_minmax(Y: Array3D) -> Array3D:
    lo = Y.min(axis=0, keepdims=True)
    hi = Y.max(axis=0, keepdims=True)
    den = hi - lo
    den[den == 0] = 1
    return (Y - lo) / den


def normalize_zscore(Y: Array3D, eps: float = 1e-8) -> Array3D:
    mu = Y.mean(axis=0, keepdims=True)
    sd = np.maximum(Y.std(axis=0, keepdims=True), eps)
    return (Y - mu) / sd


def _get_F0(Y: Array3D, mask: Array2D = None) -> Array3D:
    F0 = np.zeros_like(Y)
    T, H, W = Y.shape

    if mask is not None:
        pixel_indices = np.flatnonzero(mask)
    else:
        pixel_indices = np.arange(H * W)  # all pixels

    for i in pixel_indices:
        y, x = np.unravel_index(i, (H, W))
        tr = Y[:, y, x]
        F0[:, y, x] = tr if np.nanstd(tr) == 0 else airPLS(x=tr)

    return F0


def deltaF_over_F(
    Y: Array3D, mask: Array2D = None, no_zero_denom: float = 1e-6
) -> Array3D:

    F0 = _get_F0(Y, mask)
    return (Y - F0) / (F0 + no_zero_denom)


def apply_denoise(Y: Array3D, mask: Array2D, K: int) -> Array3D:
    T, H, W = Y.shape
    X = Y[:, mask].T  # (n_pix, T)
    C = (X.T @ X) / max(X.shape[0] - 1, 1)
    w, V = np.linalg.eigh(C)
    Vk = V[:, np.argsort(w)[::-1][: min(K, T)]]
    Xhat = (X @ Vk) @ Vk.T
    out = np.zeros((T, H, W), float)
    out[:, mask] = Xhat.T
    return out


def LogNormalKernel(T: int, mu: float = 2.2, sigma: float = 0.91) -> np.ndarray:
    t = np.linspace(1, T, T)
    k = (1 / (t * sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(t) - mu) ** 2) / (2 * sigma)
    )
    k[0] = 0.0
    return k.astype(float)


def apply_fft_deconvolution(
    Y: Array3D, kernel: np.ndarray, lam: float = 1e-2
) -> Array3D:
    Y = np.asarray(Y, float)
    T, H, W = Y.shape
    k = np.asarray(kernel, float)
    if k.shape[0] != T:
        k = k[:T] if k.shape[0] > T else np.pad(k, (0, T - k.shape[0]))

    V = np.fft.rfft(Y, n=T, axis=0)
    K = np.fft.rfft(k, n=T)
    den = (np.abs(K) ** 2 + lam)[:, None, None]
    return np.fft.irfft(V * np.conj(K)[:, None, None] / den, n=T, axis=0)


def apply_oasis_denoise_deconv(Y: Array3D, mask: Array2D = None):
    def sn_mad_diff(y: np.ndarray) -> float:
        dy = np.diff(y)
        mad = np.median(np.abs(dy - np.median(dy)))
        return mad / (0.6745 * np.sqrt(2))

    g = (0.94,)
    T, H, W = Y.shape
    calcium = np.zeros_like(Y, float)
    spikes = np.zeros_like(Y, float)

    pixel_indices = np.flatnonzero(mask) if mask is not None else np.arange(H * W)

    for i in pixel_indices:
        y, x = np.unravel_index(i, (H, W))
        tr = Y[:, y, x]
        if not np.isfinite(tr).all() or np.std(tr) == 0:
            continue
        sn = max(float(sn_mad_diff(tr)), 1e-6)
        c, s, *_ = deconvolve(tr, g=g, sn=sn, b=0.0, b_nonneg=False, penalty=1)
        calcium[:, y, x] = c
        spikes[:, y, x] = s

    return calcium, spikes


@dataclass
class MouseDataset:
    dataset_dir: Path = DATASET_DIR
    mice: tuple[int, ...] = (1, 2)
    trials: tuple[int, ...] = (1, 2, 3)
    save_contours: bool = True
    contour_dir: Path = PROJECT_ROOT / "data" / "pixel_brain_mappings"

    _data: Optional[Array5D] = field(default=None, init=False, repr=False)
    _masks: Optional[tuple[Array2D, Array2D]] = field(
        default=None, init=False, repr=False
    )
    _spikes: Optional[Array5D] = field(default=None, init=False, repr=False)

    def load(self) -> Array5D:
        Y = np.zeros((len(self.mice), len(self.trials), 1000, 100, 100), float)
        for mi, m in enumerate(self.mice):
            for ti, t in enumerate(self.trials):
                fpath = self.dataset_dir / str(m) / f"{t}.h5"
                with h5py.File(fpath, "r") as f:
                    Y[mi, ti] = f[f"video_mouse{m - 1}_trial{t - 1}"][...]
        self._data, self._spikes = Y, None
        return Y

    def convert_to_spikes(self) -> None:
        self._data = self._spikes

    @property
    def data(self) -> Array5D:
        return self._data if self._data is not None else self.load()

    @property
    def spikes(self) -> Array5D:
        if self._spikes is None:
            raise RuntimeError("Spikes not available – run oasis() first.")
        return self._spikes

    def masks(self) -> tuple[Array2D, Array2D]:
        if self._masks is None:
            self._masks = get_masks(self.data)
        return self._masks

    def _mask(self, mouse: int) -> Array2D:
        return self.masks()[self.mice.index(mouse)]

    def apply(
        self,
        fn: Callable,
        *,
        use_mask: bool = False,
        only_in_mask: bool = False,
        keep_outside: bool = True,
        **kwargs,
    ) -> None:
        is_oasis = fn is apply_oasis_denoise_deconv
        spikes_out = np.zeros_like(self.data) if is_oasis else None

        out = (
            self.data.copy()
            if (only_in_mask and keep_outside)
            else np.zeros_like(self.data)
        )

        for mi, m in enumerate(self.mice):
            mask2d = self._mask(m) if (use_mask or only_in_mask or is_oasis) else None
            for ti, t in enumerate(self.trials):
                Yin = self.data[mi, ti]
                res = fn(Yin, mask2d, **kwargs) if use_mask else fn(Yin, **kwargs)

                if is_oasis:
                    res, sp = res
                    spikes_out[mi, ti] = sp

                if only_in_mask:
                    out[mi, ti][:, mask2d] = res[:, mask2d]
                else:
                    out[mi, ti] = res

        self._data = out
        if is_oasis:
            self._spikes = spikes_out

    # ---- pipeline steps (thin wrappers) ----
    def mask(self, only_in_mask: bool = True, **sel):
        self.apply(mask_trial, use_mask=only_in_mask, **sel)

    def subtract_background(self, only_in_mask: bool = True, **sel):
        self.apply(apply_background_subtraction, only_in_mask=only_in_mask, **sel)

    def detrend(self, order: int = 1, only_in_mask: bool = True, **sel):
        self.apply(apply_detrend, order=order, only_in_mask=only_in_mask, **sel)

    def bandpass(
        self, fs: float, low=0.5, high=4.0, order=6, only_in_mask: bool = True, **sel
    ):
        self.apply(
            apply_bandpass,
            fs=fs,
            low=low,
            high=high,
            order=order,
            only_in_mask=only_in_mask,
            **sel,
        )

    def minmax_normalize(self, only_in_mask: bool = True, **sel):
        self.apply(normalize_minmax, only_in_mask=only_in_mask, **sel)

    def zscore_normalize(self, eps: float = 1e-8, only_in_mask: bool = True, **sel):
        self.apply(normalize_zscore, eps=eps, only_in_mask=only_in_mask, **sel)

    def deltaFF(self, no_zero_denom: float = 1e-6, only_in_mask: bool = True, **sel):
        self.apply(
            deltaF_over_F,
            use_mask=only_in_mask,
            only_in_mask=only_in_mask,
            no_zero_denom=no_zero_denom,
            **sel,
        )

    def denoise(self, K: int, only_in_mask: bool = True, **sel):
        self.apply(apply_denoise, use_mask=only_in_mask, K=K, **sel)

    def fft_deconvolve(self, lam: float = 1e-2, only_in_mask: bool = True, **sel):
        self.apply(
            apply_fft_deconvolution,
            kernel=LogNormalKernel(1000),
            lam=lam,
            only_in_mask=only_in_mask,
            **sel,
        )

    def oasis(self, only_in_mask: bool = True, **sel):
        self.apply(
            apply_oasis_denoise_deconv,
            use_mask=only_in_mask,
            only_in_mask=only_in_mask,
            **sel,
        )
