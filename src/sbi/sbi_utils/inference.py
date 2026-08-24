import pickle
from pathlib import Path

from .config import ROOT, MOUSE_IDX, TRIAL_IDX

from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils.user_input_checks import check_sbi_inputs
from sbi.utils import RestrictedPrior, get_density_thresholder

# ---------------------------------------------------------------------------
# Sequential rounds
# ---------------------------------------------------------------------------


def run_sequential(
    simulator,
    prior,
    target_summary,
    num_rounds: int,
    samples_per_round: int,
    training_batch_size: int,
    learning_rate: float,
):

    inference = SNPE(prior=prior)
    proposal = prior

    for round_idx in range(0, num_rounds):
        print(f"\nRound {round_idx + 1}/{num_rounds}")

        check_sbi_inputs(simulator, prior)
        theta_r, x_r = simulate_for_sbi(
            simulator=simulator,
            proposal=proposal,
            num_simulations=samples_per_round,
            num_workers=1,
            simulation_batch_size=250,
        )

        inference.append_simulations(theta_r, x_r, proposal=proposal)
        inference.train(
            show_train_summary=True,
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            force_first_round_loss=True,
        )

        posterior = inference.build_posterior().set_default_x(x=target_summary)

        save_results(
            project_root=ROOT,
            posterior=posterior,
            round=round_idx + 1,
            mouse_idx=MOUSE_IDX,
            trial_idx=TRIAL_IDX,
        )

        accept_reject_fn = get_density_thresholder(
            posterior,
            quantile=1e-4,
            num_samples_to_estimate_support=5_000,
        )

        proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")

    best_validation_loss = inference._best_val_loss
    print("\nBest validation loss:", best_validation_loss)
    print("\nFinish")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(
    project_root: Path,
    posterior,
    round,
    mouse_idx,
    trial_idx,
):
    out_dir = project_root / "data" / "sbi" / "pkls"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"posterior_round_{round}_mouse{mouse_idx}_trial{trial_idx}.pkl"

    print(f"\nsaving posterior from round {round} at:")
    print(f"{out_dir / fname}\n")

    with open(out_dir / fname, "wb") as f:
        pickle.dump(posterior, f)
