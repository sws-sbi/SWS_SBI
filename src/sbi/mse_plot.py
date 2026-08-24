import numpy as np
import matplotlib.pyplot as plt

from .sbi_utils import config

# -- Style (explicit, no rcParams) -------------------------------------------

BACKGROUND = "#FFFFFF"
TEXT_PRIMARY = "#000000"
COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#D55E00",  # vermillion
]

_FIG_KW = dict(facecolor=BACKGROUND)

PLOT_SPECS = [
    (
        "fc",
        "Functional Connectivity",
        "Functional Connectivity MSE per Round",
        "fc_mse",
    ),
    ("dfc", "Delayed FC", "Delayed Functional Connectivity MSE per Round", "dfc_mse"),
    (
        "psds",
        "Power Spectral Density",
        "Power Spectral Density MSE per Round",
        "psds_mse",
    ),
    ("avg_psd", "Average PSD", "Average PSD MSE per Round", "avg_psd_mse"),
    ("hist", "Value Distribution", "Value Distribution MSE per Round", "hist_mse"),
]
METRICS = [key for key, *_ in PLOT_SPECS]


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BACKGROUND)
    ax.set_title(title, fontsize=20, fontweight="bold", color=TEXT_PRIMARY, pad=14)
    ax.set_xlabel(xlabel, fontsize=17, color=TEXT_PRIMARY)
    ax.set_ylabel(ylabel, fontsize=17, color=TEXT_PRIMARY)
    ax.tick_params(colors=TEXT_PRIMARY, labelsize=15, direction="out")
    for spine in ax.spines.values():
        spine.set_color(TEXT_PRIMARY)
        spine.set_linewidth(0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(
        path, dpi=300, bbox_inches="tight", facecolor=BACKGROUND, edgecolor=BACKGROUND
    )
    plt.close(fig)
    print(f"Saved {path}")


# -- Loading / aggregation ----------------------------------------------------


def load_results(results_path):
    """npz -> (rounds, {metric: (n_rounds, n_samples)}, {metric: manual value})."""
    with np.load(results_path) as f:
        rounds = f["rounds"]
        per_sample = {k: f[k] for k in METRICS if k in f}
        manual = {k: float(f[f"manual_{k}"]) for k in METRICS if f"manual_{k}" in f}

    missing = [k for k in METRICS if k not in per_sample]
    if missing:
        print(f"note: {results_path.name} has no data for {missing}")

    n_failed = {
        k: int(np.isnan(v).sum()) for k, v in per_sample.items() if np.isnan(v).any()
    }
    if n_failed:
        print(f"note: failed samples excluded from the statistics: {n_failed}")

    return rounds, per_sample, manual


def aggregate(per_sample):
    """Mean and std across samples, ignoring failed (NaN) samples."""
    means = {k: np.nanmean(v, axis=1) for k, v in per_sample.items()}
    stds = {k: np.nanstd(v, axis=1) for k, v in per_sample.items()}
    return means, stds


# -- Figures ------------------------------------------------------------------

SAMPLE_STYLES = ("band", "scatter")


def _check_style(sample_style, samples):
    if sample_style not in SAMPLE_STYLES:
        raise ValueError(f"sample_style must be one of {SAMPLE_STYLES}")
    if sample_style == "scatter" and samples is None:
        raise ValueError("sample_style='scatter' needs the per-sample array")


def _scatter_samples(ax, rounds, samples, color, alpha, size, jitter, rng):
    samples = np.asarray(samples, dtype=float)
    x = np.repeat(np.asarray(rounds, dtype=float), samples.shape[1])
    if jitter:
        x = x + (rng or np.random.default_rng(0)).uniform(-jitter, jitter, x.shape)
    ax.scatter(
        x, samples.ravel(), color=color, alpha=alpha, s=size, linewidths=0, zorder=1
    )


def _plot_individual(
    rounds,
    means,
    stds,
    title,
    name,
    color,
    out_dir,
    samples=None,
    sample_style="band",
    scatter_alpha=0.15,
    scatter_size=14,
    scatter_jitter=0.0,
    log_y=True,
    rng=None,
    manual_value=None,
    manual_round=None,
    xticks=None,
):
    """
    sample_style="band"    -> ±1 std band around the mean
    sample_style="scatter" -> all per-sample MSEs as low-alpha dots
    log_y=True             -> logarithmic y-axis (non-positive values cannot be
                              drawn on a log axis and are dropped)
    """
    means, stds = np.asarray(means), np.asarray(stds)
    _check_style(sample_style, samples)

    fig, ax = plt.subplots(figsize=(8, 5), **_FIG_KW)
    if sample_style == "scatter":
        _scatter_samples(
            ax, rounds, samples, color, scatter_alpha, scatter_size, scatter_jitter, rng
        )
    else:
        lower = means - stds
        if log_y:  # a log axis cannot show <= 0; clip the band instead
            positive = means[means > 0]
            floor = 0.1 * positive.min() if positive.size else 1e-12
            lower = np.maximum(lower, floor)
        ax.fill_between(rounds, lower, means + stds, color=color, alpha=0.2)
    ax.plot(
        rounds,
        means,
        color="black",
        linewidth=2.2,
        marker="o",
        markersize=6,
        markerfacecolor=color,
    )

    if manual_value is not None:
        star_x = manual_round if manual_round is not None else rounds[-1]
        ax.scatter(
            [star_x],
            [manual_value],
            marker="*",
            s=150,
            color=color,
            edgecolors=TEXT_PRIMARY,
            linewidths=0.8,
            zorder=5,
        )
        ax.annotate(
            r"$\tilde{\theta}$",
            xy=(star_x, manual_value),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=17,
            color=TEXT_PRIMARY,
            annotation_clip=False,
            zorder=5,
        )

    _style_ax(ax, title=title, xlabel="Round", ylabel="MSE")
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(xticks if xticks is not None else rounds)
    fig.tight_layout()
    _save(fig, name, out_dir)


def print_latex_table(rounds, means, stds, n_samples, manual=None):
    keys = [k for k in METRICS if k in means]
    headers = [label for key, label, *_ in PLOT_SPECS if key in means]
    manual = manual or {}
    has_manual = any(k in manual for k in keys)

    print("\n")
    print(r"\begin{table*}")
    print(r"\centering")
    print(
        r"\caption{Posterior Predictive MSE per Round (mean $\pm$ std over "
        rf"{n_samples} posterior samples)"
        + (
            r"; last row is the manual simulation $\tilde{\theta}$"
            if has_manual
            else ""
        )
        + "}"
    )
    print(r"\label{tab:absolute_mse}")
    print(r"\begin{tabular}{c" + "c" * len(keys) + "}")
    print(r"\toprule")
    print(
        r"\textbf{Round} & " + " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\"
    )
    print(r"\midrule")
    for i, r in enumerate(rounds):
        cells = " & ".join(f"${means[k][i]:.3f} \\pm {stds[k][i]:.3f}$" for k in keys)
        print(f"{r} & {cells} \\\\")
    if has_manual:
        print(r"\midrule")
        cells = " & ".join(f"${manual[k]:.3f}$" if k in manual else "--" for k in keys)
        print(r"$\tilde{\theta}$ & " + cells + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


# -- Main ---------------------------------------------------------------------


def main(
    mouse_idx: int = 0,
    trial_idx: int = 0,
    results_path=None,
    out_dir=None,
    normalize_combined: bool = True,
    manual_round=None,  # x position of the manual-theta star; default = last round
    xticks=(0, 5, 10, 15, 20, 25),
    latex_table: bool = True,
    sample_style: str = "band",  # "band" = mean +/- std, "scatter" = all samples
    scatter_alpha: float = 0.15,
    scatter_size: float = 14,
    scatter_jitter: float = 0.0,  # +/- x-jitter in round units, e.g. 0.15
    log_y: bool = True,  # logarithmic y-axis
    seed: int = 0,
):
    results_path = results_path or (
        config.ROOT / "data" / "mse" / f"mse_mouse{mouse_idx}_trial{trial_idx}.npz"
    )
    out_dir = out_dir or config.ROOT / "figures" / "predictive_checks"

    rounds, per_sample, manual = load_results(results_path)
    means, stds = aggregate(per_sample)
    n_samples = next(iter(per_sample.values())).shape[1]
    print(f"{len(rounds)} rounds x {n_samples} samples from {results_path.name}")

    rng = np.random.default_rng(seed)

    for (key, _, title, name), color in zip(PLOT_SPECS, COLORS):
        if key not in means:
            continue
        _plot_individual(
            rounds,
            means[key],
            stds[key],
            title,
            name,
            color,
            out_dir,
            samples=per_sample[key],
            sample_style=sample_style,
            scatter_alpha=scatter_alpha,
            scatter_size=scatter_size,
            scatter_jitter=scatter_jitter,
            log_y=log_y,
            rng=rng,
            manual_value=manual.get(key),
            manual_round=manual_round,
            xticks=xticks,
        )

    if latex_table:
        print_latex_table(rounds, means, stds, n_samples, manual=manual)


if __name__ == "__main__":
    main(
        mouse_idx=0,
        trial_idx=0,
        sample_style="scatter",
        scatter_jitter=0.1,
        latex_table=False,
        scatter_alpha=0.3,
        log_y=True,
    )
