import argparse
from pathlib import Path
import pickle
import simulations
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--validity", action="store_true")
    args = parser.parse_args()

    n_test = args.n_test
    n_train = args.n_train
    num_steps = args.steps

    config = {
        "n_training": n_train,
        "n_test": n_test,
        "num_of_repititions": args.repetitions,
        "alpha": 0.05,
        "nsteps": num_steps,
        "M": args.M,
        "settings": "multivariate_health_care",
        "initial_seed": args.seed,
        "validity": args.validity,
    }

    powers, null_corr, alt_corr, rejection_by_test_results = simulations.run_simulation(
        config, None
    )
    data = {
        "rejection_lists": rejection_by_test_results,
        "config": config,
        "null_corr": null_corr,
        "alt_corr": alt_corr,
        "resulting_powers": powers,
    }

    output_file = f"results/real_data/multivariate_health_care/M_{config['M']}_n_test_{n_test}_n_train_{n_train}_seed_{args.seed}_validity_{args.validity}.pkl"
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        logger.info(f"saving file: {output_file}")
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
