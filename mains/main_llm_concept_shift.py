import multiprocessing as mp
import pickle
from pathlib import Path

import simulations


def main():
    config = {
        "n_training": 10,
        "n_test": 90,
        "num_of_repititions": 500,
        "alpha": 0.05,
        "nsteps": 500,
        "M": 128,
        "p_y_null": 0.55,
        "p_y_alt": 0.58,
        "p_y_x_0_null": 0.8,
        "p_y_x_1_null": 0.5,
        "p_y_x_0_alt": 0.8,
        "p_y_x_1_alt": 0.535,
        "settings": "llm_concept_shift",
    }

    with mp.Pool(processes=180) as pool:
        powers, null_corr, alt_corr, rejection_by_test_results = (
            simulations.run_simulation(config, pool)
        )

    data = {
        "rejection_lists": rejection_by_test_results,
        "config": config,
        "null_corr": null_corr,
        "alt_corr": alt_corr,
        "resulting_powers": powers,
    }

    output_file = "results/real_data/llm_concept_shift/n_train_10_n_test_90.pkl"
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
