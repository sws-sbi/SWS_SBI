import torch
from pathlib import Path
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = Path("./")
SEED = 42


PARAMETER_SPACE = {
    "I_e": {"low": -1.0, "high": 0.0},
    "I_i": {"low": 0.0, "high": 1.0},
    "ou": {"low": 0.05, "high": 0.35},
    "g_A": {"low": 0.0, "high": 2.0},
    "g_L": {"low": 0.0, "high": 2.0},
    "ei_scaling": {"low": 0.25, "high": 1.5},
    "B": {"low": -6.0, "high": -0.5},
    "tau_e": {"low": 20.0, "high": 100.0},
    "tau_i": {"low": 20.0, "high": 100.0},
    "tau_m": {"low": 200.0, "high": 800.0},
    "w_ee": {"low": 0.1, "high": 20.0},
    "w_ii": {"low": 0.1, "high": 20.0},
    "w_ei": {"low": 0.1, "high": 20.0},
    "w_ie": {"low": 0.1, "high": 20.0},
}


PARAM_NAMES = list(PARAMETER_SPACE.keys())

PIPELINE = 2

SAMPLES_PER_ROUND = 10_000
NUM_ROUNDS = 25
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 5e-4

MIN_FREQ = 0.5
MAX_FREQ = 4.0
FREQ_STEP = 0.25
N_BINS = 10
DFC_DELAYS = [2, 3, 5]

FREQS = torch.tensor(
    np.fft.rfftfreq(1000, d=1 / 25),
    dtype=torch.float32,
    device=DEVICE,
)

TARGET_FREQS = torch.arange(
    MIN_FREQ,
    MAX_FREQ + FREQ_STEP,
    FREQ_STEP,
    dtype=torch.float32,
    device=DEVICE,
)

BIN_EDGES = torch.linspace(
    start=0,
    end=1,
    steps=N_BINS + 1,
    device=DEVICE,
)


MOUSE_MAP_P = ROOT / "data" / "pixel_brain_mappings" / "pixel_brain_map_mouse1.npy"
ID_ACRONYM_P = ROOT / "data" / "pixel_brain_mappings" / "id_acronym_lookup.csv"
LOCAL_TABLE_P = ROOT / "data" / "tables" / "local.xlsx"

MOUSE_IDX = 0
TRIAL_IDX = 0
