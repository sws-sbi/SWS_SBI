from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .summary import extract_summary


def load_region_mapping(
    mouse_mapping_path: Path,
    id_acronym_path: Path,
    local_table_path: Path,
) -> np.ndarray:
    """Return the column indices into the distance table for each brain region."""
    id_acronym_lookup_df = pd.read_csv(id_acronym_path, header=0, index_col=False)
    id_acronym_lookup_dict = {
        int(k): v for k, v in id_acronym_lookup_df.to_dict(orient="records")[0].items()
    }

    mouse_mapping = np.load(mouse_mapping_path)
    region_ids = np.unique(mouse_mapping)[1:]
    region_acronyms = [id_acronym_lookup_dict[rid] for rid in region_ids]

    local_dist_df = pd.read_excel(local_table_path, sheet_name="distances", index_col=0)

    missing = [a for a in region_acronyms if a not in local_dist_df.columns]
    if missing:
        raise ValueError(
            f"The following regions from the mouse mapping are not present in "
            f"the distance table: {missing}"
        )

    return (
        np.array([local_dist_df.columns.get_loc(a) for a in region_acronyms]),
        region_acronyms,
    )


def load_target_data(
    pipeline: int,
    mouse_idx: int,
    trial_idx: int,
    project_root: Path,
    device: torch.device,
    freqs: torch.Tensor,
    target_freqs: torch.Tensor,
    bin_edges: torch.Tensor,
    dfc_delays: list,
    n_bins: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Load the observed pipeline data, compute summary statistics for each trial,
    and return them as a list and a stacked tensor.

    Returns
    -------
    target_summaries : list of (D,) tensors, one per trial
    x_obs_all        : (n_trials, D) tensor
    """
    target_path = project_root / "data" / "median_pipelines" / f"{pipeline}.npy"
    if not target_path.exists():
        raise FileNotFoundError(f"Target data not found: {target_path}")

    data = np.load(target_path)

    def _to_tensor(arr):
        return torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=device)

    target_data = _to_tensor(data[mouse_idx, trial_idx])

    return extract_summary(
        target_data,
        freqs,
        target_freqs,
        bin_edges,
        dfc_delays,
        n_bins,
    ).cpu()
