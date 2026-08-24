import torch

from ...fitness import _dfc_features, _psd, _avg_psd, _hist, _zscore, _fc


def fc(data: torch.Tensor) -> torch.Tensor:
    z = _zscore(data)
    fc = (z @ z.T) / (data.shape[1] - 1)
    n = fc.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    return fc[idx[0], idx[1]]


def extract_summary(
    ue: torch.Tensor,
    freqs: torch.Tensor,
    target_freqs: torch.Tensor,
    bin_edges: torch.Tensor,
    dfc_delays: list,
    n_bins: int,
) -> torch.Tensor:
    """
    Compute the full summary statistic vector for a single (N, T) activity tensor.

    Concatenates: dominant PSD bin, max power, FC, dFC, PSD, avg PSD, activity histogram.
    """
    parts = []
    sim_psds = _psd(ue, freqs, target_freqs)

    parts.append(_fc(ue).flatten())
    parts.append(_dfc_features(ue, dfc_delays).flatten())
    parts.append(sim_psds.flatten())
    parts.append(_avg_psd(sim_psds).flatten())
    parts.append(_hist(ue, n_bins, bin_edges).flatten())

    return torch.cat(parts)


def extract_summary_in_parts(
    ue: torch.Tensor,
    freqs: torch.Tensor,
    target_freqs: torch.Tensor,
    bin_edges: torch.Tensor,
    dfc_delays: list,
    n_bins: int,
) -> torch.Tensor:
    """
    Compute the full summary statistic vector for a single (N, T) activity tensor.

    Concatenates: dominant PSD bin, max power, FC, dFC, PSD, avg PSD, activity histogram.
    """
    sim_psds = _psd(ue, freqs, target_freqs)

    return fc(ue).flatten(), _dfc_features(ue, dfc_delays).flatten(), sim_psds.flatten(), _avg_psd(sim_psds).flatten(), _hist(ue, n_bins, bin_edges).flatten()