import logging
import os
import warnings

import numpy as np
import pandas as pd


class LRT:
    def __init__(self, null_sampler, var):
        self.null_sampler = null_sampler
        self.var = var

        self.data_old = pd.DataFrame({"x": [], "y": []})

    def calc(self, data_new):
        if self.var == "x":
            lrt = self.compute_E_value_LR(
                data_new["x"].sum(),
                len(data_new),
                self.data_old["x"].sum(),
                len(self.data_old),
                self.null_sampler.p_x,
            )
        elif self.var == "y":
            lrt = self.compute_E_value_LR(
                data_new["y"].sum(),
                len(data_new),
                self.data_old["y"].sum(),
                len(self.data_old),
                self.null_sampler.p_y,
            )
        elif self.var == "y_given_x":
            data_new_x_0 = data_new[data_new.x == 0]
            data_old_x_0 = self.data_old[self.data_old.x == 0]
            lrt_y_given_x_0 = self.compute_E_value_LR(
                data_new_x_0["y"].sum(),
                len(data_new_x_0),
                data_old_x_0["y"].sum(),
                len(data_old_x_0),
                self.null_sampler.p_y_x_0,
            )
            data_new_x_1 = data_new[data_new.x == 1]
            data_old_x_1 = self.data_old[self.data_old.x == 1]
            lrt_y_given_x_1 = self.compute_E_value_LR(
                data_new_x_1["y"].sum(),
                len(data_new_x_1),
                data_old_x_1["y"].sum(),
                len(data_old_x_1),
                self.null_sampler.p_y_x_1,
            )
            lrt = lrt_y_given_x_0 * lrt_y_given_x_1

        self.data_old = pd.concat([self.data_old, data_new])
        return lrt

    def compute_E_value_LR(self, x_new, n_new, x_old, n_old, p0):
        """LR E-statistic: estimates the alternative from prior data only."""
        if n_old == 0:
            return 1
        phat = x_old / n_old
        if phat == 0 or phat == 1:
            return 1
        if phat <= p0:
            return 1
        E_value = 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                logL_p0 = x_new * np.log(p0) + (n_new - x_new) * np.log(1 - p0)
                logL_phat = x_new * np.log(phat) + (n_new - x_new) * np.log(1 - phat)
                # log-space + clip to avoid overflow when exponentiating
                log_E_value = np.clip(logL_phat - logL_p0, -700, 700)
                E_value = np.exp(log_E_value)
        except RuntimeWarning as e:
            logging.warning(
                f"LRT runtime warning {e} (pid={os.getpid()}): "
                f"{x_new=}, {n_new=}, {x_old=}, {n_old=}, {p0=}"
            )
            E_value = 1
        return E_value
