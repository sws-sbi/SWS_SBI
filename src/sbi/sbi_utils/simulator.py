import jax
import jax.numpy as jnp
import numpy as np
import torch

from ...wc_model import run
from .summary import extract_summary


def _torch_to_jax(x: torch.Tensor):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous().clone()))


def _jax_to_torch(x) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x)).contiguous()


# (B, 14) -> (B, T, N)
_batch_wc_jit = jax.jit(
    jax.vmap(
        lambda th: run(
            I_e=th[0],
            I_i=th[1],
            ou=th[2],
            g_A=th[3],
            g_L=th[4],
            ei_scaling=th[5],
            B=th[6],
            tau_e=th[7],
            tau_i=th[8],
            tau_m=th[9],
            w_ee=th[10],
            w_ii=th[11],
            w_ei=th[12],
            w_ie=th[13],
        ),
        in_axes=0,
    )
)


def simulate_batch(
    theta: torch.Tensor,
    region_table_ids: np.ndarray,
    device: torch.device,
    freqs: torch.Tensor,
    target_freqs: torch.Tensor,
    bin_edges: torch.Tensor,
    dfc_delays: list,
    n_bins: int,
) -> torch.Tensor:
    """
    Run the WC model for a batch of parameters and return summary statistics.

    Parameters
    ----------
    theta           : (B, 14) parameter tensor
    region_table_ids: indices selecting the relevant brain regions from model output

    Returns
    -------
    summaries : (B, D) tensor of summary statistics, NaN/Inf-safe
    """
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)

    theta = theta.to(device=device, dtype=torch.float32)
    theta_jax = _torch_to_jax(theta).astype(jnp.float32)
    ue_jax = _batch_wc_jit(theta_jax)

    # (B, T, N) -> (B, N, T)
    ue_batch = _jax_to_torch(ue_jax).to(dtype=torch.float32)
    ue_batch = ue_batch.permute(0, 2, 1)[:, region_table_ids, :].to(device)

    summaries = torch.stack(
        [
            extract_summary(
                ue_batch[i], freqs, target_freqs, bin_edges, dfc_delays, n_bins
            ).cpu()
            for i in range(ue_batch.shape[0])
        ]
    )
    summaries = torch.nan_to_num(summaries, nan=0.0, posinf=1e6, neginf=-1e6)
    valid = torch.isfinite(summaries).all(dim=1)
    if (~valid).any():
        print(f"Warning: {(~valid).sum()} bad simulations replaced with zeros")
    return summaries


def build_simulator(
    region_table_ids, device, freqs, target_freqs, bin_edges, dfc_delays, n_bins
):
    """Return a plain callable suitable for sbi's simulate_for_sbi."""

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        return simulate_batch(
            theta,
            region_table_ids,
            device,
            freqs,
            target_freqs,
            bin_edges,
            dfc_delays,
            n_bins,
        )

    return simulator
