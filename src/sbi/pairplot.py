from .sbi_utils.summary import extract_summary
import torch
import pickle
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior
from sbi.analysis import pairplot
import matplotlib.pyplot as plt
import numpy as np

from .sbi_utils import config

# ==========================================================================================


def main(
    mouse_idx,
    trial_idx,
    bin_edges,
    n_bins,
    freqs,
    target_freqs,
    dfc_delays,
    project_root,
    num_samples,
    pipeline,
    trial: int = 0,
    round: int = 25,
):
    # -- Paths -----------------------------------------------------------------
    pairplot_dir = project_root / "figures" / "pairplots"
    pairplot_dir.mkdir(parents=True, exist_ok=True)
    pairplot_path = project_root / "figures" / "pairplots" / f"round_{round}.png"

    # -- Load posterior --------------------------------------------------------
    pkl_dir = project_root / "data" / "sbi" / "pkls"
    pkl_path = pkl_dir / f"posterior_round_{round}_mouse{mouse_idx}_trial{trial}.pkl"
    with open(pkl_path, "rb") as f:
        posterior = pickle.load(f)

    # -- Observed summary ------------------------------------------------------
    emp_data = np.load(project_root / "data" / "median_pipelines" / f"{pipeline}.npy")
    emp_data = torch.tensor(emp_data[mouse_idx, trial_idx])
    x_obs = extract_summary(
        ue=emp_data,
        freqs=freqs.cpu(),
        target_freqs=target_freqs.cpu(),
        bin_edges=bin_edges.cpu(),
        dfc_delays=dfc_delays,
        n_bins=n_bins,
    ).cpu()

    # -- Prior samples ---------------------------------------------------------
    lows = torch.tensor([v["low"] for _, v in config.PARAMETER_SPACE.items()])
    highs = torch.tensor([v["high"] for _, v in config.PARAMETER_SPACE.items()])
    prior_sbi, _, _ = process_prior(BoxUniform(low=lows, high=highs))

    prior_samples = prior_sbi.sample((num_samples,))
    posterior_samples = posterior.sample((num_samples,), x=x_obs)

    # -- Pairplot --------------------------------------------------------------
    limits = [[v["low"], v["high"]] for v in config.PARAMETER_SPACE.values()]
    labels = [
        r"$I_e$",
        r"$I_i$",
        r"$\sigma_{ou}$",
        r"$g_A$",
        r"$g_L$",
        r"$ei_{scaling}$",
        r"$B$",
        r"$\tau_{e}$",
        r"$\tau_{i}$",
        r"$\tau_{m}$",
        r"$w_{ee}$",
        r"$w_{ii}$",
        r"$w_{ei}$",
        r"$w_{ie}$",
    ]

    fig, ax = pairplot(
        samples=[prior_samples, posterior_samples],
        limits=limits,
        labels=labels,
        diag=["kde", "kde"],
    )
    plt.savefig(pairplot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved pairplot -> {pairplot_path}")

    # # -- MAP estimate ----------------------------------------------------------
    map_estimate = posterior.set_default_x(x_obs).map()
    theta_map = map_estimate.squeeze(0).tolist()
    print("\nMAP estimate:")
    for p, e in zip(config.PARAM_NAMES, theta_map):
        print(f"  '{p}': {e},")


if __name__ == "__main__":
    main(
        mouse_idx=config.MOUSE_IDX,
        trial_idx=config.TRIAL_IDX,
        bin_edges=config.BIN_EDGES,
        n_bins=config.N_BINS,
        freqs=config.FREQS,
        target_freqs=config.TARGET_FREQS,
        dfc_delays=config.DFC_DELAYS,
        project_root=config.ROOT,
        num_samples=100_000,
        pipeline=config.PIPELINE,
        round=25,
    )
