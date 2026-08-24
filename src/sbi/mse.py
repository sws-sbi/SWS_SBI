import pickle
import re

import numpy as np
import pandas as pd
import torch

from .sbi_utils import config
from .sbi_utils.summary import extract_summary, extract_summary_in_parts
from ..wc_model import run

METRICS = ("fc", "dfc", "psds", "avg_psd", "hist")
N_SAMPLES = 100


# -- Data helpers -------------------------------------------------------------


def get_table_indices():
    mouse_mapping = np.load(
        config.ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse1.npy"
    )
    id_acronym_df = pd.read_csv(
        config.ROOT / "data" / "pixel_brain_mappings" / "id_acronym_lookup.csv",
        header=0,
        index_col=False,
    )
    id_acronym_dict = {
        int(k): v for k, v in id_acronym_df.to_dict(orient="records")[0].items()
    }
    region_ids = np.unique(mouse_mapping)[1:]
    region_acronyms = [id_acronym_dict[rid] for rid in region_ids]
    local_dist_df = pd.read_excel(
        config.ROOT / "data" / "tables" / "local.xlsx",
        sheet_name="distances",
        index_col=0,
    )
    return [local_dist_df.columns.get_loc(a) for a in region_acronyms]


def _load_empirical(mouse_idx, trial_idx):
    emp_data = np.load(
        config.ROOT / "data" / "median_pipelines" / f"{config.PIPELINE}.npy"
    )
    return torch.tensor(emp_data[mouse_idx, trial_idx], dtype=torch.float32)


def _get_x_obs(emp_tensor):
    return extract_summary(
        ue=emp_tensor,
        freqs=config.FREQS.cpu(),
        target_freqs=config.TARGET_FREQS.cpu(),
        bin_edges=config.BIN_EDGES.cpu(),
        dfc_delays=config.DFC_DELAYS,
        n_bins=config.N_BINS,
    ).cpu()


def _summary_in_parts(data_tensor):
    MIN_FREQ = 0.5
    MAX_FREQ = 4.0
    FREQ_STEP = 0.25

    FREQS = torch.tensor(
        np.fft.rfftfreq(1000, d=1 / 25),
        dtype=torch.float32,
        device="cpu",
    )
    TARGET_FREQS = torch.arange(
        MIN_FREQ,
        MAX_FREQ + FREQ_STEP,
        FREQ_STEP,
        dtype=torch.float32,
        device="cpu",
    )

    fc, dfc, psds, avg_psd, hist = extract_summary_in_parts(
        ue=data_tensor,
        freqs=FREQS,
        target_freqs=TARGET_FREQS,
        bin_edges=config.BIN_EDGES.cpu(),
        dfc_delays=config.DFC_DELAYS,
        n_bins=config.N_BINS,
    )
    return fc.cpu(), dfc.cpu(), psds.cpu(), avg_psd.cpu(), hist.cpu()


def _mse(a, b):
    a = np.array(a, dtype=np.float32).flatten()
    b = np.array(b, dtype=np.float32).flatten()
    return np.sum((a - b) ** 2) / len(a)


def _simulate(theta_dict, table_indices):
    sim = np.asarray(run(**theta_dict), dtype=np.float32).T[table_indices]

    if not np.isfinite(sim).all():
        raise ValueError("simulation produced NaN or infinite activity values")

    lo, hi = float(sim.min()), float(sim.max())
    sim = np.zeros_like(sim) if (hi - lo) <= 1e-12 else (sim - lo) / (hi - lo)
    return torch.tensor(sim, dtype=torch.float32)


# -- Per-round sampling -------------------------------------------------------


def _mses_from_posterior_samples(posterior, x_obs, table_indices, emp_parts, n_samples):
    """Draw n_samples thetas, simulate each, return {metric: (n_samples,) array}.
    """
    samples = posterior.sample((n_samples,), x=x_obs)  # (n_samples, n_params)

    per_sample = {k: np.full(n_samples, np.nan, dtype=np.float32) for k in METRICS}

    for i, theta in enumerate(samples):
        print(f"  Sample {i + 1}/{n_samples}", end="\r")
        theta_dict = dict(zip(config.PARAM_NAMES, theta.tolist()))
        try:
            sim_parts = _summary_in_parts(_simulate(theta_dict, table_indices))
            for key, sim_part, emp_part in zip(METRICS, sim_parts, emp_parts):
                per_sample[key][i] = _mse(sim_part, emp_part)
        except Exception as e:
            print(f"\n  Sample {i + 1} failed: {e} — skipping")

    n_ok = int(np.isfinite(per_sample[METRICS[0]]).sum())
    print(f"  {n_ok}/{n_samples} samples succeeded")
    return per_sample


# -- Manual theta -------------------------------------------------------------


def mse_for_theta(theta, mouse_idx: int = 0, trial_idx: int = 0):
    """Simulate one parameter vector; return {metric: mse} against the recording.

    theta is either a sequence in config.PARAM_NAMES order, or a dict.
    """
    if isinstance(theta, dict):
        missing = [name for name in config.PARAM_NAMES if name not in theta]
        extra = [name for name in theta if name not in config.PARAM_NAMES]
        if missing:
            raise ValueError(f"Theta dictionary is missing parameters: {missing}")
        if extra:
            raise ValueError(f"Theta dictionary contains unknown parameters: {extra}")
        theta_dict = {name: float(theta[name]) for name in config.PARAM_NAMES}
    else:
        theta_tensor = torch.as_tensor(theta, dtype=torch.float32).flatten()
        if theta_tensor.numel() != len(config.PARAM_NAMES):
            raise ValueError(
                f"Expected {len(config.PARAM_NAMES)} theta values in the order "
                f"{config.PARAM_NAMES}, but received {theta_tensor.numel()}."
            )
        theta_dict = dict(zip(config.PARAM_NAMES, theta_tensor.tolist()))

    emp_parts = _summary_in_parts(_load_empirical(mouse_idx, trial_idx))
    sim_parts = _summary_in_parts(_simulate(theta_dict, get_table_indices()))

    result = {
        name: float(_mse(sim_part, emp_part))
        for name, sim_part, emp_part in zip(METRICS, sim_parts, emp_parts)
    }
    for name, value in result.items():
        print(f"{name}: {value}")
    return result


# -- Main ---------------------------------------------------------------------


def main(
    mouse_idx: int = 0,
    trial_idx: int = 0,
    n_samples: int = N_SAMPLES,
    manual_theta: dict | None = None,
    pkl_dir=None,
    results_path=None,
):
    pkl_dir = pkl_dir or config.ROOT / "data" / "sbi" / "pkls"
    results_path = results_path or (
        config.ROOT / "data" / "mse" / f"mse_mouse{mouse_idx}_trial{trial_idx}.npz"
    )

    emp_tensor = _load_empirical(mouse_idx, trial_idx)
    emp_parts = _summary_in_parts(emp_tensor)
    x_obs = _get_x_obs(emp_tensor)
    table_indices = get_table_indices()

    manual = {}
    if manual_theta is not None:
        print("Evaluating manual theta ...")
        manual = mse_for_theta(manual_theta, mouse_idx, trial_idx)

    # --- find posterior pkls ---
    pattern = re.compile(
        rf"posterior_round_(\d+)_mouse{mouse_idx}_trial{trial_idx}\.pkl$"
    )
    candidates = sorted(
        (int(pattern.match(p.name).group(1)), p)
        for p in pkl_dir.glob("*.pkl")
        if pattern.match(p.name)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No posterior pkls in {pkl_dir} for mouse{mouse_idx}_trial{trial_idx}."
        )

    # --- one row of n_samples MSEs per round ---
    n_rounds = len(candidates)
    results = {k: np.full((n_rounds, n_samples), np.nan, np.float32) for k in METRICS}
    rounds = np.array([r for r, _ in candidates], dtype=int)

    results_path.parent.mkdir(parents=True, exist_ok=True)

    for row, (round_num, pkl_path) in enumerate(candidates):
        print(f"\nRound {round_num} — drawing {n_samples} posterior samples ...")
        with open(pkl_path, "rb") as f:
            posterior = pickle.load(f)

        per_sample = _mses_from_posterior_samples(
            posterior, x_obs, table_indices, emp_parts, n_samples
        )
        for key in METRICS:
            results[key][row] = per_sample[key]

        np.savez(
            results_path,
            rounds=rounds[: row + 1],
            **{k: v[: row + 1] for k, v in results.items()},
            **{f"manual_{k}": v for k, v in manual.items()},
        )

    print(f"\nSaved {results_path}")
    return results_path


if __name__ == "__main__":
    theta = {
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

    main(mouse_idx=0, trial_idx=0, manual_theta=theta)
