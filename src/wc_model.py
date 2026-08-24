import jax
import jax.numpy as jnp
from jax import random, jit
import pandas as pd
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("./")
DATA_PATH = PROJECT_ROOT / "data"
ATLAS_TABLE_PATH = DATA_PATH / "tables" / "atlas.xlsx"
LOCAL_TABLE_PATH = DATA_PATH / "tables" / "local.xlsx"
MOUSE_MAPPING = DATA_PATH / "pixel_brain_mappings" / "pixel_brain_map_mouse1.npy"


# ##################################################################################################
# #########                                                                                #########
# ######      FUNCTION FOR LOADING TABLES AND FILTERING OUT ALL BUT THE 12 TARGET REGIONS    #######
# #########                                                                                #########
# ##################################################################################################


def get_table_data(filtered=False):
    atlas_conn_df = pd.read_excel(
        ATLAS_TABLE_PATH,
        sheet_name="connectivity",
        index_col=0,
    )

    atlas_dist_df = pd.read_excel(
        ATLAS_TABLE_PATH,
        sheet_name="distances",
        index_col=0,
    )

    local_conn_df = pd.read_excel(
        LOCAL_TABLE_PATH,
        sheet_name="connectivity",
        index_col=0,
    )

    local_dist_df = pd.read_excel(
        LOCAL_TABLE_PATH,
        sheet_name="distances",
        index_col=0,
    )

    if filtered:
        id_acronym_lookup_df = pd.read_csv(
            DATA_PATH / "pixel_brain_mappings" / "id_acronym_lookup.csv",
            header=0,
            index_col=False,
        )
        id_acronym_lookup_dict = {
            int(k): v
            for k, v in id_acronym_lookup_df.to_dict(orient="records")[0].items()
        }

        mouse1_mapping = np.load(MOUSE_MAPPING)
        region_ids = np.unique(mouse1_mapping)[1:]

        region_acronyms = [
            id_acronym_lookup_dict[region_id] for region_id in region_ids
        ]
        atlas_conn_df_filtered = atlas_conn_df.loc[region_acronyms, region_acronyms]
        atlas_dist_df_filtered = atlas_dist_df.loc[region_acronyms, region_acronyms]
        local_conn_df_filtered = local_conn_df.loc[region_acronyms, region_acronyms]
        local_dist_df_filtered = local_dist_df.loc[region_acronyms, region_acronyms]

        return (
            atlas_conn_df_filtered,
            atlas_dist_df_filtered,
            local_conn_df_filtered,
            local_dist_df_filtered,
        )

    else:
        return (
            atlas_conn_df,
            atlas_dist_df,
            local_conn_df,
            local_dist_df,
        )


# ##################################################################################################
# ################################                                ##################################
# ###########################                CONSTANTS                  ############################
# ################################                                ##################################
# ##################################################################################################

C = 1000 * 30  # 1000mm * 30m/s
DT = 0.1
N = 40

IND_START = int(20_000 / DT)
IND_END = int(60_000 / DT)
PIC_NMB = 1_000

CMAT_A_DF, DMAT_A_DF, CMAT_L_DF, DMAT_L_DF = get_table_data(filtered=(N == 12))


CMAT_L = jax.device_put(jnp.asarray(CMAT_L_DF.to_numpy(), dtype=jnp.float32))
DMAT_L = jax.device_put(jnp.asarray(DMAT_L_DF.to_numpy(), dtype=jnp.float32))
CMAT_A = jax.device_put(jnp.asarray(CMAT_A_DF.to_numpy(), dtype=jnp.float32))
DMAT_A = jax.device_put(jnp.asarray(DMAT_A_DF.to_numpy(), dtype=jnp.float32))


DMAT_L = ((DMAT_L / C) / DT).round().astype(int)
DMAT_A = ((DMAT_A / C) / DT).round().astype(int)

MAX_DELAY = int(jnp.floor(jnp.max(jnp.stack([DMAT_L, DMAT_A]))))  # 3

DMAT_L = DMAT_L.at[jnp.diag_indices(N)].set(0)
DMAT_A = DMAT_A.at[jnp.diag_indices(N)].set(0)

CMAT_L = CMAT_L.at[jnp.diag_indices(N)].set(0)
CMAT_A = CMAT_A.at[jnp.diag_indices(N)].set(0)

SEED = random.PRNGKey(0)
MAXAMP = 0.01


BETA_E = 2.5
BETA_I = 2.5
BETA_M = -10

MU_E = 0.0
MU_I = 0.0
MU_M = 0.5

TAU_OU = 20.0


# ##################################################################################################
# #######################                                                   ########################
# ###################      COMPUTATIONS THAT CAN BE DONE OUTSIDE OF RUN       ######################
# ######################                                                   #########################
# ##################################################################################################
SQRT_2_TAU_OU = jnp.sqrt(2 / TAU_OU)
INV_TAU_OU = 1 / TAU_OU

NODE_IDX = jnp.arange(N)[None, :]
SAMPLING_IND = (IND_END - IND_START) // PIC_NMB
DELAY_STEPS = MAX_DELAY + 1

SHAPE = (N,)

SEED = random.PRNGKey(0)
KEY1, KEY2, KEY3 = jax.random.split(SEED, 3)
UE_INIT = jax.random.uniform(KEY1, SHAPE, minval=0.0, maxval=MAXAMP)
UI_INIT = jax.random.uniform(KEY2, SHAPE, minval=0.0, maxval=MAXAMP)
MECH_INIT = jax.random.uniform(KEY3, SHAPE, minval=0.0, maxval=MAXAMP)

NOISE_UE_INIT = jnp.zeros_like(UE_INIT)
NOISE_UI_INIT = jnp.zeros_like(UI_INIT)

UE_HIST_INIT = jnp.tile(UE_INIT[None, :], (DELAY_STEPS, 1))
UI_HIST_INIT = jnp.tile(UI_INIT[None, :], (DELAY_STEPS, 1))

INIT_CARRY = (UE_HIST_INIT, UI_HIST_INIT, MECH_INIT, NOISE_UE_INIT, NOISE_UI_INIT, SEED)

DMAT_L_RANGE = -DMAT_L - 1
DMAT_A_RANGE = -DMAT_A - 1
OUTPUT_XS_RANGE = jnp.arange(1, PIC_NMB)
EULER_XS_RANGE = jnp.arange(0, SAMPLING_IND)


# ##################################################################################################
# ###############################                                ###################################
# ##########################      PRE-DEFINED TRANSFER FUNCTIONS       #############################
# ###############################                                ###################################
# ##################################################################################################


@jit
def Fe(x):
    return jax.nn.sigmoid(BETA_E * (x - MU_E))


@jit
def Fi(x):
    return jax.nn.sigmoid(BETA_I * (x - MU_I))


@jit
def Fm(x):
    return jax.nn.sigmoid(BETA_M * (x - MU_M))


# ##################################################################################################
# ###############################                                ###################################
# ##########################            MAIN RUN FUNCTION              #############################
# ###############################                                ###################################
# ##################################################################################################


@jit
def run(
    I_e: float,
    I_i: float,
    ou: float,
    g_A: float,
    g_L: float,
    ei_scaling: float,
    B: float,
    tau_e: float,
    tau_i: float,
    tau_m: float,
    w_ee: float,
    w_ii: float,
    w_ei: float,
    w_ie: float,
):
    g_L *= 5396.61865234375
    noise_scale_dt = ou * SQRT_2_TAU_OU * DT

    def euler_step(carry, ind):
        """Single Euler time step - optimized"""
        ue_hist, ui_hist, mech, noise_ue, noise_ui, eta = carry
        ue = ue_hist[-1]
        ui = ui_hist[-1]

        delayed_ue_local = ue_hist[DMAT_L_RANGE, NODE_IDX]
        delayed_ue_atlas = ue_hist[DMAT_A_RANGE, NODE_IDX]

        exc_input_d = g_L * jnp.sum(CMAT_L * delayed_ue_local, axis=1) + g_A * jnp.sum(
            CMAT_A * delayed_ue_atlas, axis=1
        )

        inh_input_d = ei_scaling * exc_input_d

        noise_ue_new = noise_ue + (
            DT * -noise_ue * INV_TAU_OU + noise_scale_dt * eta[0, ind]
        )
        noise_ui_new = (
            noise_ui + DT * -noise_ui * INV_TAU_OU + noise_scale_dt * eta[1, ind]
        )

        fe_arg = w_ee * ue - w_ei * ui - B * mech + exc_input_d + I_e + noise_ue_new
        fi_arg = w_ie * ue - w_ii * ui + inh_input_d + I_i + noise_ui_new

        rhs_e = (1 / tau_e) * (-ue + Fe(fe_arg))
        rhs_i = (1 / tau_i) * (-ui + Fi(fi_arg))
        rhs_mech = (1 / tau_m) * (-mech + Fm(ue))

        ue_new = ue + DT * rhs_e
        ui_new = ui + DT * rhs_i
        mech_new = mech + DT * rhs_mech

        ue_hist_new = jnp.concatenate([ue_hist[1:], ue_new[None, :]], axis=0)
        ui_hist_new = jnp.concatenate([ui_hist[1:], ui_new[None, :]], axis=0)

        return (
            ue_hist_new,
            ui_hist_new,
            mech_new,
            noise_ue_new,
            noise_ui_new,
            eta,
        ), None

    def output_step(carry, ind):
        ue_hist, ui_hist, mech_old, noise_ue_old, noise_ui_old, key = carry

        key, subkey = random.split(key)
        eta_old = random.normal(subkey, (2, SAMPLING_IND, *noise_ue_old.shape))

        (ue_hist_new, ui_hist_new, mech_new, noise_ue_new, noise_ui_new, _), _ = (
            jax.lax.scan(
                euler_step,
                (ue_hist, ui_hist, mech_old, noise_ue_old, noise_ui_old, eta_old),
                xs=EULER_XS_RANGE,
            )
        )

        ue_new = ue_hist_new[-1]

        return (
            ue_hist_new,
            ui_hist_new,
            mech_new,
            noise_ue_new,
            noise_ui_new,
            key,
        ), ue_new

    _, Ue = jax.lax.scan(output_step, INIT_CARRY, xs=OUTPUT_XS_RANGE)

    return jnp.concatenate([jnp.expand_dims(UE_INIT, 0), Ue])
