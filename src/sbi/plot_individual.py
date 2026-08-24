from __future__ import annotations

import pathlib

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from ..wc_model import run
from ..fitness import _fc, _dfc, _psd, _avg_psd
from .sbi_utils.data import load_region_mapping
from .sbi_utils import config

BACKGROUND = "#FFFFFF"
TEXT_PRIMARY = "#000000"
TEXT_MUTED = "#333333"
COLOR_EMP = "#B25400"
COLOR_SIM = "#0072B2"
COLOR_DIFF_CMAP = "RdBu_r"
COLOR_MAIN_CMAP = "magma"

CMAP_MAIN = matplotlib.colormaps.get_cmap(COLOR_MAIN_CMAP).copy()
CMAP_DIFF = matplotlib.colormaps.get_cmap(COLOR_DIFF_CMAP).copy()
INTERPOLATION = "none"

_FIG_KW = dict(facecolor=BACKGROUND)


def _to_tensor(arr):
    arr = np.asarray(arr)
    if not arr.flags.writeable:
        arr = arr.copy()
    return torch.as_tensor(arr, dtype=torch.float32, device=config.DEVICE)


def _style_ax(ax):
    ax.set_facecolor(BACKGROUND)
    ax.tick_params(
        colors=TEXT_PRIMARY, labelsize=10, direction="out", length=4, width=0.9
    )
    ax.xaxis.label.set_color(TEXT_PRIMARY)
    ax.yaxis.label.set_color(TEXT_PRIMARY)
    ax.title.set_color(TEXT_PRIMARY)
    ax.xaxis.label.set_fontsize(13)
    ax.yaxis.label.set_fontsize(13)
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color(TEXT_PRIMARY)
        spine.set_linewidth(0.9)


def _clean_spines(ax, keep=("left", "bottom")):
    for name, spine in ax.spines.items():
        spine.set_visible(name in keep)
        if name in keep:
            spine.set_color(TEXT_PRIMARY)
            spine.set_linewidth(0.9)


def _save(fig, out_dir: pathlib.Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=BACKGROUND,
        edgecolor=BACKGROUND,
    )
    plt.close(fig)
    print(f"Saved {png_path.name}")


def _styled_colorbar(mappable, ax, label=""):
    cb = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.set_facecolor(BACKGROUND)
    cb.ax.yaxis.set_tick_params(
        color=TEXT_PRIMARY, labelcolor=TEXT_PRIMARY, labelsize=9
    )
    cb.outline.set_edgecolor(TEXT_PRIMARY)
    cb.outline.set_linewidth(0.8)
    if label:
        cb.set_label(label, color=TEXT_PRIMARY, fontsize=10)
    return cb


def _matrix_ticks(ax, n, region_labels=None):
    if region_labels is not None:
        ax.set_xticks(range(len(region_labels)))
        ax.set_xticklabels(region_labels, rotation=90, fontsize=7, color=TEXT_PRIMARY)
        ax.set_yticks(range(len(region_labels)))
        ax.set_yticklabels(region_labels, fontsize=7, color=TEXT_PRIMARY)
    else:
        ticks = range(n)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(i + 1) for i in ticks], fontsize=8, color=TEXT_PRIMARY)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(i + 1) for i in ticks], fontsize=8, color=TEXT_PRIMARY)


def save_timeseries_figure(data, label, color, out_dir, name, region_labels=None):
    n_regions = data.shape[0]
    fig, ax = plt.subplots(figsize=(10, 5), **_FIG_KW)
    _style_ax(ax)

    im = ax.imshow(data, aspect="auto", cmap=CMAP_MAIN, interpolation=INTERPOLATION)
    ax.set_title(f"{label} Timeseries", pad=10, color=TEXT_PRIMARY)
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Region")

    if region_labels is not None:
        ax.set_yticks(range(n_regions))
        ax.set_yticklabels(region_labels, fontsize=7, color=TEXT_PRIMARY)

    _clean_spines(ax)
    _styled_colorbar(im, ax, label="Normalized Activity")
    fig.tight_layout()
    _save(fig, out_dir, name)


def save_fc_figure(emp_t, sim_t, out_dir, region_labels=None):
    emp_fc = _fc(emp_t).cpu().numpy()
    sim_fc = _fc(sim_t).cpu().numpy()
    diff = emp_fc - sim_fc
    v = np.max(np.abs(diff))
    n = emp_fc.shape[0]

    panels = [
        (
            "fc_empirical",
            emp_fc,
            "Empirical FC",
            dict(cmap=CMAP_MAIN, interpolation=INTERPOLATION),
        ),
        (
            "fc_simulated",
            sim_fc,
            "Simulated FC",
            dict(cmap=CMAP_MAIN, interpolation=INTERPOLATION),
        ),
        (
            "fc_delta",
            diff,
            r"$\Delta$FC (Empirical $-$ Simulated)",
            dict(cmap=CMAP_DIFF, interpolation=INTERPOLATION, vmin=-v, vmax=v),
        ),
    ]

    for fname, mat, title, kwargs in panels:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), **_FIG_KW)
        _style_ax(ax)
        im = ax.imshow(mat, **kwargs)
        ax.set_title(title, pad=10, color=TEXT_PRIMARY)
        ax.set_xlabel("Region")
        ax.set_ylabel("Region")
        _matrix_ticks(ax, n, region_labels)
        _clean_spines(ax)
        _styled_colorbar(im, ax, label="Pearson r")
        fig.tight_layout()
        _save(fig, out_dir, fname)


def save_dfc_figure(emp_t, sim_t, delay, out_dir, region_labels=None):
    emp_dfc = _dfc(emp_t, delay).cpu().numpy()
    sim_dfc = _dfc(sim_t, delay).cpu().numpy()
    diff = emp_dfc - sim_dfc
    v = np.max(np.abs(diff))
    n = emp_dfc.shape[0]

    panels = [
        (
            f"dfc_tau{delay}_empirical",
            emp_dfc,
            rf"Empirical DFC ($\tau$={delay})",
            dict(cmap=CMAP_MAIN, interpolation=INTERPOLATION),
        ),
        (
            f"dfc_tau{delay}_simulated",
            sim_dfc,
            rf"Simulated DFC ($\tau$={delay})",
            dict(cmap=CMAP_MAIN, interpolation=INTERPOLATION),
        ),
        (
            f"dfc_tau{delay}_delta",
            diff,
            rf"$\Delta$DFC ($\tau$={delay})",
            dict(cmap=CMAP_DIFF, interpolation=INTERPOLATION, vmin=-v, vmax=v),
        ),
    ]

    for fname, mat, title, kwargs in panels:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), **_FIG_KW)
        _style_ax(ax)
        im = ax.imshow(mat, **kwargs)
        ax.set_title(title, pad=10, color=TEXT_PRIMARY)
        ax.set_xlabel("Region")
        ax.set_ylabel("Region")
        _matrix_ticks(ax, n, region_labels)
        _clean_spines(ax)
        _styled_colorbar(im, ax, label="Pearson r")
        fig.tight_layout()
        _save(fig, out_dir, fname)


def save_psd_figure(emp_t, sim_t, out_dir, region_labels=None, normalized=False):
    emp_psd = _psd(emp_t, config.FREQS, config.TARGET_FREQS).cpu().numpy()
    sim_psd = _psd(sim_t, config.FREQS, config.TARGET_FREQS).cpu().numpy()
    diff = emp_psd - sim_psd
    v = np.max(np.abs(diff))
    tf = config.TARGET_FREQS.cpu().numpy()
    extent = [tf[0], tf[-1], emp_psd.shape[0], 0]

    if normalized:
        v_min = min(emp_psd.min(), sim_psd.min())
        v_max = max(emp_psd.max(), sim_psd.max())
        main_kwargs = dict(
            cmap=CMAP_MAIN, interpolation=INTERPOLATION, vmin=v_min, vmax=v_max
        )
    else:
        main_kwargs = dict(cmap=CMAP_MAIN, interpolation=INTERPOLATION)

    panels = [
        ("psd_empirical", emp_psd, "Empirical PSD", main_kwargs),
        ("psd_simulated", sim_psd, "Simulated PSD", main_kwargs),
        (
            "psd_delta",
            diff,
            r"$\Delta$PSD (Empirical $-$ Simulated)",
            dict(cmap=CMAP_DIFF, interpolation=INTERPOLATION, vmin=-v, vmax=v),
        ),
    ]

    for fname, mat, title, kwargs in panels:
        fig, ax = plt.subplots(figsize=(8, 5.5), **_FIG_KW)
        _style_ax(ax)
        im = ax.imshow(mat, aspect="auto", extent=extent, **kwargs)
        ax.set_title(title, pad=10, color=TEXT_PRIMARY)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Region")
        if region_labels is not None:
            y_ticks = np.arange(len(region_labels)) + 0.5
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(region_labels, fontsize=7, color=TEXT_PRIMARY)
        _clean_spines(ax)
        _styled_colorbar(im, ax, label="Power")
        fig.tight_layout()
        _save(fig, out_dir, fname)


def save_avg_psd_figure(emp_t, sim_t, out_dir):
    """Single line-plot figure: region-averaged PSD, empirical vs simulated."""
    emp_psd = _psd(emp_t, config.FREQS, config.TARGET_FREQS)
    sim_psd = _psd(sim_t, config.FREQS, config.TARGET_FREQS)
    emp_avg = _avg_psd(emp_psd).cpu().numpy()
    sim_avg = _avg_psd(sim_psd).cpu().numpy()
    tf = config.TARGET_FREQS.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5), **_FIG_KW)
    _style_ax(ax)
    ax.fill_between(tf, emp_avg, alpha=0.12, color=COLOR_EMP)
    ax.fill_between(tf, sim_avg, alpha=0.12, color=COLOR_SIM)
    ax.plot(tf, emp_avg, color=COLOR_EMP, linewidth=2.2, label="Empirical")
    ax.plot(
        tf, sim_avg, color=COLOR_SIM, linewidth=2.2, label="Simulated", linestyle="--"
    )

    ax.set_title("Average Power Spectral Density", pad=10, color=TEXT_PRIMARY)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    _clean_spines(ax)
    legend = ax.legend(loc="best", frameon=False, fontsize=11)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.tight_layout()
    _save(fig, out_dir, "psd_average")


def save_histogram_figure(emp, sim, out_dir):
    """Single bar-chart figure: value distribution, empirical vs simulated."""
    bin_edges = np.linspace(0, 1, config.N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]

    emp_counts, _ = np.histogram(emp.flatten(), bins=bin_edges)
    sim_counts, _ = np.histogram(sim.flatten(), bins=bin_edges)
    ymax = max(emp_counts.max(), sim_counts.max())

    fig, ax = plt.subplots(figsize=(8, 5), **_FIG_KW)
    _style_ax(ax)
    ax.bar(
        bin_centers,
        emp_counts,
        width=width * 0.9,
        color=COLOR_EMP,
        alpha=0.75,
        label="Empirical",
        linewidth=0.5,
        edgecolor=BACKGROUND,
    )
    ax.bar(
        bin_centers,
        sim_counts,
        width=width * 0.9,
        color=COLOR_SIM,
        alpha=0.75,
        label="Simulated",
        linewidth=0.5,
        edgecolor=BACKGROUND,
    )

    ax.set_title("Value Distribution", pad=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_ylim(0, ymax * 1.15)
    _clean_spines(ax)
    legend = ax.legend(loc="best", frameon=False, fontsize=11)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.tight_layout()
    _save(fig, out_dir, "value_distribution")


def save_all_publication_figures(
    emp,
    sim,
    out_dir: pathlib.Path,
    normalized: bool = True,
    region_labels=None,
):
    """
    Generate and save every comparison as its own figure
    inside `out_dir`.
    """
    out_dir = pathlib.Path(out_dir)
    emp_t = _to_tensor(emp)
    sim_t = _to_tensor(sim)

    save_timeseries_figure(
        emp, "Empirical", COLOR_EMP, out_dir, "timeseries_empirical", region_labels
    )
    save_timeseries_figure(
        sim,
        r"Simulated $\tilde{\theta}$",
        COLOR_SIM,
        out_dir,
        "timeseries_simulated",
        region_labels,
    )
    save_fc_figure(emp_t, sim_t, out_dir, region_labels)
    for delay in config.DFC_DELAYS:
        save_dfc_figure(emp_t, sim_t, delay, out_dir, region_labels)
    save_psd_figure(emp_t, sim_t, out_dir, region_labels, normalized)
    save_avg_psd_figure(emp_t, sim_t, out_dir)
    save_histogram_figure(emp, sim, out_dir)


def main(
    theta: list[float],
    normalize: bool = True,
    out_dir: str | None = None,
):
    if len(theta) != len(config.PARAM_NAMES):
        raise ValueError(
            f"Expected {len(config.PARAM_NAMES)} parameters, got {len(theta)}."
        )

    tag = f"mouse{config.MOUSE_IDX}_trial{config.TRIAL_IDX}"
    if out_dir is None:
        out_dir = config.ROOT / "data" / "compare" / f"figures_{tag}"
    out_dir = pathlib.Path(out_dir)

    region_table_ids, region_acronyms = load_region_mapping(
        mouse_mapping_path=config.MOUSE_MAP_P,
        id_acronym_path=config.ID_ACRONYM_P,
        local_table_path=config.LOCAL_TABLE_P,
    )

    emp_raw = np.load(
        config.ROOT / "data" / "median_pipelines" / f"{config.PIPELINE}.npy"
    )
    emp = emp_raw[config.MOUSE_IDX, config.TRIAL_IDX]  # (N, T)

    theta_dict = dict(zip(config.PARAM_NAMES, theta))
    sim_full = np.array(run(**theta_dict).T)  # (N_all, T)
    sim = sim_full[region_table_ids]  # (N_regions, T)

    if normalize:
        sim_min, sim_max = sim.min(), sim.max()
        sim = (sim - sim_min) / (sim_max - sim_min)

    save_all_publication_figures(
        emp=emp,
        sim=sim,
        out_dir=out_dir,
        normalized=normalize,
        region_labels=region_acronyms,
    )
    print(f"All figures saved to: {out_dir}")


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
    main(
        theta=list(theta.values()),
        normalize=True,
        out_dir="./figures/summary_statistics_and_timeseries/",
    )
