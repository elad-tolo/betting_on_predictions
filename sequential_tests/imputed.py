import logging
import traceback

import numpy as np
import pandas as pd

from utils.ada_grad import AdaGrad


class Imputed:

    def __init__(self, M, alg, null_sampler, n_test, n_training, seed, agg="sum"):
        self.M = M
        self.null_sampler = null_sampler
        self.n_test = n_test
        self.n_training = n_training
        self.agg = agg

        self.rng_np = np.random.default_rng(seed=seed)
        self.data_old = pd.DataFrame({"x": [], "y": []})
        self.gamma = 1
        self.alg = alg
        self.ada_grad = AdaGrad(weight=self.gamma)

    def calc(self, labeled_data, unlabeled_data):
        null_datasets_output = self.get_score_of_null_datasets()

        calc_statistic_output = self.calc_statistic_finite_sample(
            labeled_data,
            null_datasets_output["scores"],
            unlabeled_data,
        )
        statistic = calc_statistic_output["statistic"]

        n1_total = null_datasets_output["n_1_total_null"]
        n0_total = null_datasets_output["n_0_total_null"]
        n_1 = calc_statistic_output["n_1_total_numerator"]
        n_0 = calc_statistic_output["n_0_total_numerator"]
        self.update_gamma(n_1, n_0, n1_total, n0_total)

        return statistic

    def get_score_of_null_datasets(self):
        n1_total_null = []
        n0_total_null = []

        scores = 0
        for _ in range(self.M):
            df_train = self.null_sampler.sample(self.n_training)
            unlabeled_data = self.null_sampler.sample_x(self.n_test)
            out_dic = self.get_score(df_train, unlabeled_data)

            n1_total_null.append(out_dic["n1_total"])
            n0_total_null.append(out_dic["n0_total"])
            scores += out_dic["score"]

        return {
            "scores": scores,
            "n_1_total_null": n1_total_null,
            "n_0_total_null": n0_total_null,
        }

    def calc_statistic_finite_sample(
        self, labeled_data, null_datasets_scores, unlabeled_data
    ):

        out_dic = self.get_score(labeled_data, unlabeled_data)

        score = out_dic["score"]
        n_1_total_numerator = out_dic["n1_total"]
        n_0_total_numerator = out_dic["n0_total"]

        statistic = (self.M + 1) * score / (score + null_datasets_scores)
        return {
            "statistic": statistic,
            "n_1_total_numerator": n_1_total_numerator,
            "n_0_total_numerator": n_0_total_numerator,
        }

    def get_score(self, labeled_data, unlabeled_data):
        predictions = self.alg.fit_predict(labeled_data, unlabeled_data)
        n1_total_predicted = predictions.sum()
        n0_total_predicted = len(predictions) - n1_total_predicted

        agg_func = np.sum if self.agg == "sum" else np.prod
        try:
            score = agg_func(self.calc_psi(predictions))
        except Exception:
            logging.warning(f"Error in calc_psi: {traceback.format_exc()}")
            score = 0

        return {
            "score": score,
            "n1_total": n1_total_predicted,
            "n0_total": n0_total_predicted,
        }

    def calc_psi(self, preds):
        return list(np.exp(self.gamma * preds))

    def update_gamma(self, n_1, n_0, n1_total, n0_total):
        alpha_t = self.gamma

        if self.agg == "sum":
            denom1 = n_1 * np.exp(alpha_t) + n_0
            denom2 = (n_1 + sum(n1_total)) * np.exp(alpha_t) + (n_0 + sum(n0_total))
            # Avoid division by zero when denominators are zero
            if denom1 > 0 and denom2 > 0:
                grad_L = (
                    n_1 * np.exp(alpha_t) / denom1
                    - (n_1 + sum(n1_total)) * np.exp(alpha_t) / denom2
                )
            else:
                grad_L = 0.0  # No gradient update when data is degenerate
        elif self.agg == "prod":
            sum_exp = sum([np.exp(alpha_t * n_j) for n_j in n1_total])
            sum_exp_derivative = sum([n_j * np.exp(alpha_t * n_j) for n_j in n1_total])
            grad_L = n_1 - (
                (n_1 * np.exp(alpha_t * n_1) + sum_exp_derivative)
                / (np.exp(alpha_t * n_1) + sum_exp)
            )
        alpha_t_plus_1 = self.ada_grad.step(grad_L, clip_max=10**6, clip_min=0.01)
        self.gamma = alpha_t_plus_1
