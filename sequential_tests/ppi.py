import numpy as np
import pandas as pd


class PPI:
    def __init__(self, alg, no_unlabeled=False, positive_lambda=False):
        self.alg = alg
        self.no_unlabeled = no_unlabeled
        self.positive_lambda = positive_lambda

        self.lambda_t_minus_1 = 0
        self.a_t_minus_1 = 1
        self.data_old = pd.DataFrame({"x": [], "y": []})
        self.data_old_unlabeled = None

    def calc(
        self,
        config,
        data_new,
        data_new_unlabeled,
    ):
        if len(self.data_old) < len(data_new):
            w_i = data_new["y"] - config["p_y_null"]
        else:
            predictions_history_labeled = self.alg.fit_predict(
                self.data_old, self.data_old["x"].to_numpy()
            )
            predictions_history_unlabeled = self.alg.fit_predict(
                self.data_old, self.data_old_unlabeled
            )
            predictions_new_data_labeled = self.alg.fit_predict(
                self.data_old, data_new["x"].to_numpy()
            )
            predictions_new_data_unlabeled = self.alg.fit_predict(
                self.data_old, data_new_unlabeled
            )
            N = len(predictions_history_unlabeled)
            n = len(predictions_history_labeled)
            cov_Yf_X = np.cov(
                predictions_history_labeled, np.array(self.data_old["y"])
            )[0, 1]

            f_x = np.concatenate(
                [predictions_history_labeled, predictions_history_unlabeled]
            )
            var_f_X = np.var(f_x)
            if self.no_unlabeled:
                eps = 0
            elif var_f_X == 0:
                eps = 0.99
            else:
                eps = cov_Yf_X / ((1 + n / N) * var_f_X)
                eps = max(0, min(1, eps))

            N = len(predictions_new_data_unlabeled)
            n = len(predictions_new_data_labeled)
            k = int(N / n)
            b = predictions_new_data_unlabeled.reshape(N // k, k).mean(axis=1)
            w_i = (
                np.array(data_new["y"])
                - eps * predictions_new_data_labeled
                + eps * b
                - config["p_y_null"]
            )
            w_i = w_i / max(1 + eps - config["p_y_null"], eps + config["p_y_null"])

        bett = 1
        lambda_t = self.lambda_t_minus_1
        a_t = self.a_t_minus_1
        for w in w_i:
            bett = bett * (1 + lambda_t * (w))
            v_t = w
            z_t = v_t / (1 + v_t * lambda_t)
            a_t = a_t + z_t**2
            lambda_t = min(0.5, max(-0.5, lambda_t + 2 / (2 - np.log(3)) * z_t / a_t))
            if self.positive_lambda:
                lambda_t = max(0, lambda_t)

        self.lambda_t_minus_1 = lambda_t
        self.a_t_minus_1 = a_t
        self.data_old = pd.concat([self.data_old, data_new])
        if self.data_old_unlabeled is None:
            self.data_old_unlabeled = data_new_unlabeled
        else:
            self.data_old_unlabeled = np.concatenate(
                [self.data_old_unlabeled, data_new_unlabeled]
            )
        return bett
