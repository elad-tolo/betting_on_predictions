import multiprocessing as mp
import pickle
from pathlib import Path

import simulations


def main():
    with mp.Pool(processes=150) as pool:
        for n_test in [10, 70]:
            config = {
                "n_training": 10,
                "n_test": n_test,
                "num_of_repititions": 500,
                "alpha": 0.05,
                "nsteps": 800,
                "M": 128,
                "p_y_null": 0.58,
                "p_y_alt": None,
                "p_x_null": 0.55,
                "settings": "llm_judge_label_shift",
            }
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

            output_file = (
                f"results/real_data/judge_label_shift/n_train_10_n_test_{n_test}.pkl"
            )
            path = Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
