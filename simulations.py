from collections import defaultdict
import numpy as np
import pandas as pd
import logging
import os
import time
from tqdm.auto import tqdm

from algs import BayesClassifier, NaiveBayesClassifier, TabPFN
from sequential_tests.imputed import Imputed
from sequential_tests.lrt import LRT
from sequential_tests.ppi import PPI
from utils.data_sampler import (
    HealthCareMultivariateDataSampler,
    ConceptShiftDataSampler,
    LabelShiftDataSampler,
    LlmConceptShift,
    LlmLabelShift,
)

CONCEPT_SHIFT_SETTINGS = {
    "concept_shift",
    "llm_concept_shift",
}


def get_alg_name(config):
    if config["settings"] in CONCEPT_SHIFT_SETTINGS:
        return "NB"
    elif config["settings"] == "multivariate_health_care":
        return "TabPFN"
    else:
        return "NB_max"


class StatisticCalcualtor:
    def __init__(
        self,
        config,
        seed,
    ):
        self.n_training = config["n_training"]
        self.n_test = config["n_test"]
        self.M = config["M"]
        self.alg_name = get_alg_name(config)

        self.null_sampler, self.alt_sampler = self.get_samplers(config, seed)
        self.alg = self.get_alg(seed, self.null_sampler)
        self.imputed = Imputed(
            self.M,
            self.alg,
            self.null_sampler,
            self.n_test,
            self.n_training,
            seed,
            agg="sum",
        )

        null_sampler, _ = self.get_samplers(config, seed)
        self.imputed_prod = Imputed(
            self.M,
            self.alg,
            null_sampler,
            self.n_test,
            self.n_training,
            seed,
            agg="prod",
        )
        self.lrt_y = LRT(null_sampler, "y")
        self.lrt_x = LRT(null_sampler, "x")
        self.lrt_y_given_x = LRT(null_sampler, "y_given_x")
        self.ppi = PPI(self.alg, no_unlabeled=False, positive_lambda=False)
        self.ppi_no_unlabeled = PPI(self.alg, no_unlabeled=True, positive_lambda=False)
        self.ppi_positive_lambda = PPI(
            self.alg, no_unlabeled=False, positive_lambda=True
        )

    def get_alg(self, seed, null_sampler):
        if self.alg_name == "TabPFN":
            return TabPFN()
        elif self.alg_name == "NB":
            return BayesClassifier(seed)
        elif self.alg_name == "NB_max":
            return NaiveBayesClassifier(null_sampler.T_1, null_sampler.T_0)

    def get_samplers(self, config, seed):
        if config["settings"] == "label_shift":
            null_sampler = LabelShiftDataSampler(
                p_y=config["p_y_null"],
                p_x_y_0=config["p_x_y_0"],
                p_x_y_1=config["p_x_y_1"],
                seed=seed,
            )
            alt_sampler = LabelShiftDataSampler(
                p_y=config["p_y_alt"],
                p_x_y_0=config["p_x_y_0"],
                p_x_y_1=config["p_x_y_1"],
                seed=seed,
            )
            alt_sampler.T_0 = null_sampler.T_0 = null_sampler.tau_0
            alt_sampler.T_1 = null_sampler.T_1 = null_sampler.tau_1
        elif config["settings"] == "concept_shift":
            null_sampler = ConceptShiftDataSampler(
                p_x=config["p_x"],
                p_y_x_0=config["p_y_x_0_null"],
                p_y_x_1=config["p_y_x_1_null"],
                seed=seed,
            )
            alt_sampler = ConceptShiftDataSampler(
                p_x=config["p_x"],
                p_y_x_0=config["p_y_x_0_alt"],
                p_y_x_1=config["p_y_x_1_alt"],
                seed=seed,
            )
        elif config["settings"] == "llm_judge_label_shift":
            null_sampler = LlmLabelShift(
                model_name="deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
                seed=seed,
                target_mean=config.get("p_y_null"),
                target_mean_x=config.get("p_x_null"),
                data_parent_dir="data",
            )
            alt_sampler = LlmLabelShift(
                "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
                hypothesis="alt",
                seed=seed,
                target_mean=config.get("p_y_alt"),
                target_mean_x=config.get("p_x_alt"),
                data_parent_dir="data",
            )
            if config.get("validity"):
                null_sampler = LlmLabelShift(
                    model_name="deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
                    seed=seed,
                )
                alt_sampler = LlmLabelShift(
                    model_name="deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
                    hypothesis="alt",
                    seed=seed,
                )
            alt_sampler.T_0 = null_sampler.T_0 = null_sampler.tau_0
            alt_sampler.T_1 = null_sampler.T_1 = null_sampler.tau_1
            logging.info(f"{null_sampler.data['y'].mean()=}")
            logging.info(f"{alt_sampler.data['y'].mean()=}")

        elif config["settings"] == "multivariate_health_care":
            null_sampler = HealthCareMultivariateDataSampler(
                year=2017,
                seed=seed,
                data_dir="data",
            )
            alt_sampler = HealthCareMultivariateDataSampler(
                year=2019,
                seed=seed,
                data_dir="data",
            )
            if config.get("validity"):
                null_sampler = HealthCareMultivariateDataSampler(
                    year=2017,
                    seed=seed,
                    data_dir="data",
                )
                alt_sampler = HealthCareMultivariateDataSampler(
                    year=2017,
                    seed=seed,
                    data_dir="data",
                )

            config["p_y_null"] = null_sampler.data["y"].mean()

        elif config["settings"] == "llm_concept_shift":
            null_sampler = LlmConceptShift(
                "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
                seed=seed,
                hypothesis="alt" if config.get("use_permutations", False) else "null",
                target_mean=config["p_y_null"],
                target_p_y_x_0=config["p_y_x_0_null"],
                target_p_y_x_1=config["p_y_x_1_null"],
                data_parent_dir="data",
            )

            alt_sampler = LlmConceptShift(
                "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B",
                seed=seed,
                hypothesis="alt",
                target_mean=config["p_y_alt"],
                target_p_y_x_0=config["p_y_x_0_alt"],
                target_p_y_x_1=config["p_y_x_1_alt"],
                data_parent_dir="data",
            )
        return null_sampler, alt_sampler


def first_exceeds(arr, alpha):
    mask = arr > alpha
    if np.any(mask):
        return np.argmax(mask)
    else:
        return None


def get_power(bReject, num_trials):
    return len(np.where(np.array(bReject) != None)[0]) / num_trials


def get_e_sequences(config, seed, settings, nsteps, n_training):
    sequences = defaultdict(list)

    if settings in CONCEPT_SHIFT_SETTINGS:
        alg = "NB"
    elif settings == "multivariate_health_care":
        alg = "TabPFN"
    else:
        alg = "NB_max"

    statistics_calculator = StatisticCalcualtor(config, seed)
    alt_sampler = statistics_calculator.alt_sampler
    for step in range(nsteps):
        logging.info(f"step {step}")
        data_new = alt_sampler.sample(n_training)
        data_new_unlabeled = alt_sampler.sample_x(statistics_calculator.n_test)

        statistics = statistics_calculator.imputed.calc(data_new, data_new_unlabeled)
        statistic_prod = statistics_calculator.imputed_prod.calc(
            data_new, data_new_unlabeled
        )

        sequences[f"{alg}_predictions_only"].append(statistics)

        lrt_y = statistics_calculator.lrt_y.calc(data_new)
        sequences["lrt_y"].append(lrt_y)

        if settings in CONCEPT_SHIFT_SETTINGS:
            sequences["NB_prod_predictions_only"].append(statistic_prod)
            cond_lrt = statistics_calculator.lrt_y_given_x.calc(data_new)
            logging.info(f"lrt_y: {lrt_y}, cond_lrt: {cond_lrt}")
            sequences["cond_lrt"].append(cond_lrt)
        elif settings != "multivariate_health_care":
            sequences["NB_max_prod_predictions_only"].append(statistic_prod)
            data_new_x = pd.DataFrame(
                {"x": np.concatenate([data_new["x"].to_numpy(), data_new_unlabeled])}
            )
            lrt_x = statistics_calculator.lrt_x.calc(data_new_x)
            sequences["lrt_x"].append(lrt_x)

        ppi = statistics_calculator.ppi.calc(config, data_new, data_new_unlabeled)
        sequences["PPI"].append(ppi)
        ppi_no_unlabeled = statistics_calculator.ppi_no_unlabeled.calc(
            config, data_new, data_new_unlabeled
        )
        sequences["PPI_no_unlabeled"].append(ppi_no_unlabeled)
        ppi_positive_lambda = statistics_calculator.ppi_positive_lambda.calc(
            config,
            data_new,
            data_new_unlabeled,
        )
        sequences["PPI_positive_lambda"].append(ppi_positive_lambda)

    output_dic = {"sequences": sequences}

    return output_dic


def combine_martingales_growth_rate_eg(
    *martingales, eta=0.7, eps=1e-12, clip_grad=None
):
    """Combine martingales by maximizing log-wealth via Exponentiated Gradient.

    Maintains a portfolio over the K input martingales on the probability
    simplex, updating weights w_{t+1} ∝ w_t * exp(eta * ∇_w log(w^T x_t)).
    Returns the per-step portfolio returns r_t = w_t^T x_t.

    References:
        Kivinen & Warmuth (1997), "Exponentiated Gradient versus Gradient
            Descent for Linear Predictors".
        Helmbold, Schapire, Singer, Warmuth (1998), "On-Line Portfolio
            Selection Using Multiplicative Updates".
        Cover (1991), "Universal Portfolios".
    """
    K = len(martingales)
    if K == 0:
        raise ValueError("Need at least one martingale.")
    if eta <= 0:
        raise ValueError("eta must be positive.")

    weights = np.ones(K, dtype=float) / K
    combined = []

    for steps in zip(*martingales):
        x_t = np.asarray(steps, dtype=float)
        if x_t.shape != (K,):
            raise ValueError(f"Expected steps of length {K}, got {x_t.shape}.")
        if np.any(x_t < 0):
            raise ValueError("This EG implementation assumes x_t >= 0.")

        step_return = max(float(np.dot(weights, x_t)), eps)
        grad = x_t / step_return
        if clip_grad is not None:
            grad = np.minimum(grad, clip_grad)
        combined.append(step_return)

        # Numerically stable EG update: w_{t+1,k} ∝ w_{t,k} * exp(eta * grad_k).
        logw = np.log(weights + eps) + eta * grad
        logw -= logw.max()
        weights = np.exp(logw)
        weights /= weights.sum()

    return combined


# Use log-space cumsum to avoid overflow, then convert back
def safe_cumprod(seq):
    """Compute cumulative product safely by working in log-space."""
    seq = np.array(seq)
    # Clip values to avoid log(0) or log(negative)
    seq = np.clip(seq, 1e-300, None)
    log_cumsum = np.cumsum(np.log(seq))
    # Clip the log values to prevent overflow when exponentiating
    log_cumsum = np.clip(log_cumsum, -700, 700)  # exp(709) ~ 1e308 (float64 max)
    return np.exp(log_cumsum)


def simulate_tests(config, seed):
    pid = os.getpid()
    logging.basicConfig(
        filename=f"logs/process_{pid}.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("===========starting new sequence=========")
    start_time = time.time()

    out_dic = get_e_sequences(
        config,
        seed,
        config["settings"],
        config["nsteps"],
        config["n_training"],
    )

    logging.info(f"sequence simulation time: {time.time() - start_time}")

    sequences = out_dic["sequences"]

    if config["settings"] in CONCEPT_SHIFT_SETTINGS:
        ml_martingale = "NB_predictions_only"
    elif config["settings"] == "multivariate_health_care":
        ml_martingale = "TabPFN_predictions_only"
    else:
        ml_martingale = "NB_max_predictions_only"

    if config["settings"] in CONCEPT_SHIFT_SETTINGS:
        sequences["baselines"] = combine_martingales_growth_rate_eg(
            sequences["cond_lrt"],
            sequences["PPI"],
            sequences["lrt_y"],
        )

        sequences["imputed_with_baselines"] = combine_martingales_growth_rate_eg(
            sequences[ml_martingale],
            sequences["baselines"],
        )

        sequences["baselines_ppi_positive_lambda"] = combine_martingales_growth_rate_eg(
            sequences["cond_lrt"],
            sequences["PPI_positive_lambda"],
            sequences["lrt_y"],
        )

        sequences["imputed_with_baselines_ppi_positive_lambda"] = (
            combine_martingales_growth_rate_eg(
                sequences["cond_lrt"],
                sequences[ml_martingale],
                sequences["lrt_y"],
                sequences["PPI_positive_lambda"],
            )
        )

    elif config["settings"] == "multivariate_health_care":
        sequences["baselines"] = combine_martingales_growth_rate_eg(
            sequences["lrt_y"],
            sequences["PPI"],
        )

        sequences["imputed_with_baselines"] = combine_martingales_growth_rate_eg(
            sequences[ml_martingale],
            sequences["baselines"],
        )

    else:
        sequences["baselines"] = combine_martingales_growth_rate_eg(
            sequences["lrt_x"],
            sequences["lrt_y"],
            sequences["PPI"],
        )

        sequences["imputed_with_baselines"] = combine_martingales_growth_rate_eg(
            sequences[ml_martingale],
            sequences["baselines"],
        )

        sequences["baselines_positive_lambda"] = combine_martingales_growth_rate_eg(
            sequences["lrt_x"],
            sequences["lrt_y"],
            sequences["PPI_positive_lambda"],
        )

        sequences["imputed_with_baselines_positive_lambda"] = (
            combine_martingales_growth_rate_eg(
                sequences[ml_martingale],
                sequences["lrt_x"],
                sequences["lrt_y"],
                sequences["PPI_positive_lambda"],
            )
        )

    cummulative_e_values = {name: safe_cumprod(seq) for name, seq in sequences.items()}

    rejects = {}
    for name, cumprod in cummulative_e_values.items():
        rejects[name] = first_exceeds(cumprod, 1 / config["alpha"])

    logging.info(f"total simulation time: {time.time() - start_time}")

    output_dic = {"rejects": rejects}

    return output_dic


def compare_tests(config, pool=None):
    SEED = config.get("initial_seed", 0)
    res = []
    futures = []

    for i in range(config["num_of_repititions"]):
        args = (config, SEED + i)

        if pool:
            futures.append(pool.apply_async(simulate_tests, args=args))
        else:
            logging.info(f"Running simulation seed {SEED + i}")
            res.append(simulate_tests(*args))

    if pool:
        res = [f.get() for f in tqdm(futures)]

    rejection_res = [tmp["rejects"] for tmp in res]

    rejects_lists = defaultdict(list)
    for rejects in rejection_res:
        for name, r in rejects.items():
            rejects_lists[name].append(r)

    output_dic = {"rejects_lists": rejects_lists}
    return output_dic


def run_simulation(config, pool):
    corr_null = None
    corr_alt = None
    if config["settings"] != "multivariate_health_care":
        sc = StatisticCalcualtor(config, seed=0)
        null_sampler = sc.null_sampler
        alt_sampler = sc.alt_sampler

        df_0 = null_sampler.sample(20_000)
        df_1 = alt_sampler.sample(20_000)
        corr_null = df_0.corr()["x"]["y"]
        corr_alt = df_1.corr()["x"]["y"]

    output_dic = compare_tests(config, pool)
    res = output_dic["rejects_lists"]
    statistic_to_power = {
        name: get_power(rejects_list, config["num_of_repititions"])
        for name, rejects_list in res.items()
    }

    return (statistic_to_power, corr_null, corr_alt, res)
