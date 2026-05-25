import os
import pickle
from pathlib import Path
import multiprocessing as mp

from simulations import run_simulation

if __name__ == "__main__":

    correlations = {
        "low_correlation": {"p_x_y_0": 0.35, "p_x_y_1": 0.65},
    }
    corr = "low_correlation"
    n_test = 30
    p_null = 0.5
    p_alt = p_null + 0.02
    nsteps = 500

    Ms = [2, 5, 32, 128]

    with mp.Pool(processes=130) as pool:
        try:
            for M in Ms:
                config_LS = {
                    "n_training": 15,
                    "n_test": n_test,
                    "num_of_repititions": 500,
                    "alpha": 0.05,
                    "nsteps": nsteps,
                    "M": M,
                    "p_x_y_0": correlations[corr]["p_x_y_0"],
                    "p_x_y_1": correlations[corr]["p_x_y_1"],
                    "p_y_null": p_null,
                    "p_y_alt": p_alt,
                    "settings": "label_shift",
                }
                (
                    powers,
                    null_corr,
                    alt_corr,
                    rejection_by_test_results,
                ) = run_simulation(config_LS, pool)

                data = {
                    "rejection_lists": rejection_by_test_results,
                    "config": config_LS,
                    "null_corr": null_corr,
                    "alt_corr": alt_corr,
                    "powers": powers,
                }
                output_file = f"results/simulations/label_shift/tuning_M/{corr}_{n_test}_n_15_M_{M}.pkl"
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
