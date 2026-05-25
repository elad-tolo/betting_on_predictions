import os
import pickle
from pathlib import Path
import multiprocessing as mp

from simulations import run_simulation

if __name__ == "__main__":

    p_x = 0.5
    delta = 0.02
    n_test = 135
    correlations = {
        "low_correlation": {"p_y_x_0": 0.4, "p_y_x_1": 0.7},
        "high_correlation": {"p_y_x_0": 0.2, "p_y_x_1": 0.85},
    }
    config = {
        "n_training": 15,
        "n_test": n_test,
        "num_of_repititions": 500,
        "alpha": 0.05,
        "nsteps": 500,
        "M": 128,
        "p_y_x_0_null": 0,
        "p_y_x_1_null": 0,
        "p_y_x_0_alt": 0,
        "p_y_x_1_alt": 0,
        "p_x": p_x,
        "settings": "concept_shift",
        "p_y_null": 0,
    }

    with mp.Pool(processes=130) as pool:
        try:
            for corr in correlations:
                config["p_y_x_0_null"] = correlations[corr]["p_y_x_0"]
                config["p_y_x_1_null"] = correlations[corr]["p_y_x_1"]

                config["p_y_x_0_alt"] = config["p_y_x_0_null"] + delta
                config["p_y_x_1_alt"] = config["p_y_x_1_null"] + delta
                config["p_y_null"] = (
                    p_x * config["p_y_x_1_null"] + (1 - p_x) * config["p_y_x_0_null"]
                )
                config["p_y_alt"] = (
                    p_x * config["p_y_x_1_alt"] + (1 - p_x) * config["p_y_x_0_alt"]
                )

                powers, null_corr, alt_corr, rejection_by_test_results = run_simulation(
                    config, pool
                )

                data = {
                    "rejection_lists": rejection_by_test_results,
                    "config": config,
                    "null_corr": null_corr,
                    "alt_corr": alt_corr,
                    "powers": powers,
                }

                output_file = (
                    f"results/simulations/concept_shift/{corr}_{n_test}_n_15.pkl"
                )
                path = Path(output_file)
                os.makedirs(path.parent, exist_ok=True)
                with path.open("wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down executor...")
            pool.terminate()
            pool.join()
            print("Pool terminated.")
