import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from tabpfn import TabPFNClassifier


class NaiveBayesClassifier:
    def __init__(self, T1, T0):
        self.T1 = T1
        self.T0 = T0

    def fit_predict(self, labeled_data, unlabeled_data):
        y_hat = labeled_data["y"].mean()
        return np.where(
            unlabeled_data == 1,
            1 if y_hat > self.T1 else 0,
            1 if y_hat > self.T0 else 0,
        )


class BayesClassifier:
    def __init__(self, seed):
        self.rng_np = np.random.default_rng(seed=seed)

    def fit_predict(self, labeled_data, unlabeled_data):
        unique_vals_x = labeled_data["x"].unique()
        unique_vals_y = labeled_data["y"].unique()
        if len(unique_vals_y) == 1:
            # Only one class in target y
            p_y_given_x_NB = np.full(len(labeled_data), float(unique_vals_y[0]))
        elif len(unique_vals_x) == 1:
            # Only one feature value in x
            # We cannot learn a relationship. Predict the mean of y observed.
            p_y_mean = labeled_data["y"].mean()
            p_y_given_x_NB = np.full(len(unlabeled_data), p_y_mean)
        else:
            model_NB = CategoricalNB()
            model_NB.fit(labeled_data[["x"]], labeled_data["y"])

            if isinstance(unlabeled_data, pd.DataFrame):
                X_pred = unlabeled_data[["x"]]
            else:
                X_pred = pd.DataFrame({"x": np.asarray(unlabeled_data)})
            prob_predictions = model_NB.predict_proba(X_pred)
            p_y_given_x_NB = prob_predictions[:, 1]

        predictions = self.rng_np.binomial(n=1, p=p_y_given_x_NB)
        return predictions


class TabPFN:
    def fit_predict(self, labeled_data, unlabeled_data):
        X = np.stack(labeled_data["x"].to_numpy())
        y = labeled_data["y"].to_numpy()
        clf = TabPFNClassifier(device="cuda", n_estimators=1)
        clf.fit(X, y)
        if isinstance(unlabeled_data, pd.DataFrame):
            unlabeled_data = np.stack(unlabeled_data["x"].to_numpy())
        elif unlabeled_data.ndim == 1 and unlabeled_data.dtype == object:
            unlabeled_data = np.stack(unlabeled_data)
        predictions = clf.predict(unlabeled_data)
        return predictions
