import os
import warnings

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="jax|sbi")

import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

import numpy as np

from .sbi_utils.data import load_region_mapping, load_target_data
from .sbi_utils.simulator import build_simulator
from .sbi_utils.inference import run_sequential
from .sbi_utils.prior import build_prior

from .sbi_utils import config

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Region mapping
    region_table_ids, region_acronyms = load_region_mapping(
        mouse_mapping_path=config.MOUSE_MAP_P,
        id_acronym_path=config.ID_ACRONYM_P,
        local_table_path=config.LOCAL_TABLE_P,
    )

    # Observed summaries
    target_summary = load_target_data(
        pipeline=config.PIPELINE,
        mouse_idx=config.MOUSE_IDX,
        trial_idx=config.TRIAL_IDX,
        project_root=config.ROOT,
        device=config.DEVICE,
        freqs=config.FREQS,
        target_freqs=config.TARGET_FREQS,
        bin_edges=config.BIN_EDGES,
        dfc_delays=config.DFC_DELAYS,
        n_bins=config.N_BINS,
    )

    # Prior and simulator
    prior = build_prior()

    simulator = build_simulator(
        region_table_ids=region_table_ids,
        device=config.DEVICE,
        freqs=config.FREQS,
        target_freqs=config.TARGET_FREQS,
        bin_edges=config.BIN_EDGES,
        dfc_delays=config.DFC_DELAYS,
        n_bins=config.N_BINS,
    )

    run_sequential(
        simulator=simulator,
        prior=prior,
        target_summary=target_summary,
        num_rounds=config.NUM_ROUNDS,
        samples_per_round=config.SAMPLES_PER_ROUND,
        training_batch_size=config.TRAIN_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
    )


if __name__ == "__main__":
    main()
