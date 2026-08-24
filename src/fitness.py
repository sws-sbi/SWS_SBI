import torch


def _zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)


def _fc(data: torch.Tensor) -> torch.Tensor:
    z = _zscore(data)
    T = data.shape[1]
    return (z @ z.T) / (T - 1)


def _dfc(data: torch.Tensor, delay_frames: int) -> torch.Tensor:
    data1, data2 = data[:, :-delay_frames], data[:, delay_frames:]
    T = data1.shape[1]
    data1_norm, data2_norm = _zscore(data1), _zscore(data2)
    return (data1_norm @ data2_norm.T) / (T - 1)


def _dfc_features(data: torch.Tensor, delay_frames: list) -> torch.Tensor:
    parts = [_dfc(data, d).reshape(-1) for d in delay_frames]
    return (
        torch.cat(parts)
        if parts
        else torch.tensor([], dtype=torch.float32, device=data.device)
    )


def _psd(
    data: torch.Tensor, freqs: torch.Tensor, target_freqs: torch.Tensor
) -> torch.Tensor:
    n_regions, n_frames = data.shape
    fft_vals = torch.fft.rfft(data, dim=1)
    power = fft_vals.abs().pow(2) / n_frames

    f0 = freqs[:-1]
    f1 = freqs[1:]

    idx = torch.searchsorted(freqs.contiguous(), target_freqs.contiguous()) - 1
    idx = idx.clamp(0, len(freqs) - 2)

    t = (target_freqs - f0[idx]) / (f1[idx] - f0[idx] + 1e-12)
    out = power[:, idx] * (1 - t) + power[:, idx + 1] * t
    return out


def _avg_psd(psd_per_region: torch.Tensor) -> torch.Tensor:
    return psd_per_region.mean(dim=0)


def _hist(data: torch.Tensor, n_bins: int, bin_edges: torch.Tensor) -> torch.Tensor:
    n_regions, T = data.shape
    b = torch.bucketize(data, bin_edges, right=False) - 1
    b = b.clamp(0, n_bins - 1)
    one_hot = torch.zeros(n_regions, n_bins, dtype=torch.float32, device=data.device)
    one_hot.scatter_add_(1, b, torch.ones_like(b, dtype=torch.float32))
    probs = one_hot / T
    return probs.reshape(-1)
