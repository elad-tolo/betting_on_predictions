import pandas as pd
import numpy as np


class LabelShiftDataSampler:
    def __init__(self, p_y, p_x_y_0, p_x_y_1, seed):
        self.p_y = p_y
        self.p_x_y_0 = p_x_y_0
        self.p_x_y_1 = p_x_y_1
        self.p_x = p_y * p_x_y_1 + (1 - p_y) * p_x_y_0
        p_x_0_y_1 = 1 - self.p_x_y_1
        p_x_0_y_0 = 1 - self.p_x_y_0
        self.tau_0 = p_x_0_y_0 / (p_x_0_y_0 + p_x_0_y_1)
        self.tau_1 = self.p_x_y_0 / (self.p_x_y_0 + self.p_x_y_1)
        self.p_y_given_x_0 = (self.p_y * (1 - self.p_x_y_1)) / (1 - self.p_x)
        self.p_y_given_x_1 = (self.p_y * self.p_x_y_1) / self.p_x
        self.th_0 = self.p_y_given_x_0
        self.th_1 = self.p_y_given_x_1
        self.T_0 = (self.th_0 * self.tau_0) / (
            (1 - self.th_0) * (1 - self.tau_0) + (self.th_0 * self.tau_0)
        )
        self.T_1 = (self.th_1 * self.tau_1) / (
            (1 - self.th_1) * (1 - self.tau_1) + (self.th_1 * self.tau_1)
        )

        self.rng = np.random.default_rng(seed=seed)

    def sample(self, n):
        y = self.rng.binomial(1, self.p_y, size=n)
        x = [
            (
                self.rng.binomial(1, self.p_x_y_1, size=1).tolist()[0]
                if y_value
                else self.rng.binomial(1, self.p_x_y_0, size=1).tolist()[0]
            )
            for y_value in y
        ]
        return pd.DataFrame({"x": x, "y": y})

    def sample_x(self, n):
        return self.rng.binomial(1, self.p_x, size=n)


class ConceptShiftDataSampler:
    def __init__(self, p_x, p_y_x_0, p_y_x_1, seed):
        self.p_x = p_x
        self.p_y_x_0 = p_y_x_0
        self.p_y_x_1 = p_y_x_1
        self.p_y = p_x * p_y_x_1 + (1 - p_x) * p_y_x_0
        self.rng = np.random.default_rng(seed=seed)

    def sample(self, n):
        x = self.rng.binomial(1, self.p_x, size=n)
        y = [
            (
                self.rng.binomial(1, self.p_y_x_1, size=1).tolist()[0]
                if x_value
                else self.rng.binomial(1, self.p_y_x_0, size=1).tolist()[0]
            )
            for x_value in x
        ]
        return pd.DataFrame({"x": x, "y": y})

    def sample_x(self, n):
        return self.rng.binomial(1, self.p_x, size=n)


class LlmLabelShift:

    def __init__(
        self,
        model_name,
        hypothesis="null",
        target_mean=None,
        target_mean_x=None,
        seed=None,
        data_parent_dir="data",
    ):
        self.model_name = model_name
        self.hypothesis = hypothesis
        splits = ["train", "test"]
        datasets = ["gsm8k", "aqua_rat", "math"]
        data_frames = []
        data_dir = f"{data_parent_dir}/math_judge"
        for dataset in datasets:
            for split in splits:
                df = pd.read_parquet(
                    f"{data_dir}/{model_name}/{dataset}/{split}/results_xy.parquet"
                )
                data_frames.append(df)
        self.data = pd.concat(data_frames).reset_index(drop=True)
        rng = np.random.default_rng(seed=seed)
        self.rng = rng
        shuffled_indices = rng.permutation(len(self.data))
        self.data = self.data.iloc[shuffled_indices].reset_index(drop=True)

        if target_mean is not None and target_mean_x is not None:
            n00 = ((self.data["x"] == 0) & (self.data["y"] == 0)).sum()
            n01 = ((self.data["x"] == 0) & (self.data["y"] == 1)).sum()
            n10 = ((self.data["x"] == 1) & (self.data["y"] == 0)).sum()
            n11 = ((self.data["x"] == 1) & (self.data["y"] == 1)).sum()

            best_K = 0
            best_counts = (0, 0, 0, 0)

            for K in range(len(self.data), -1, -1):
                if K == 0:
                    break

                Ny = int(round(target_mean * K))
                Nx = int(round(target_mean_x * K))

                z_min = max(0, Nx - n10, Ny - n01, Nx + Ny - K)
                z_max = min(n11, Nx, Ny, n00 + Nx + Ny - K)

                if z_min <= z_max:
                    best_K = K
                    k11 = int(z_min)
                    k10 = Nx - k11
                    k01 = Ny - k11
                    k00 = K - Nx - Ny + k11
                    best_counts = (k00, k01, k10, k11)
                    break

            k00, k01, k10, k11 = best_counts
            idx00 = self.data[(self.data["x"] == 0) & (self.data["y"] == 0)].index[:k00]
            idx01 = self.data[(self.data["x"] == 0) & (self.data["y"] == 1)].index[:k01]
            idx10 = self.data[(self.data["x"] == 1) & (self.data["y"] == 0)].index[:k10]
            idx11 = self.data[(self.data["x"] == 1) & (self.data["y"] == 1)].index[:k11]

            indices_to_keep = np.concatenate([idx00, idx01, idx10, idx11])
            self.data = self.data.loc[indices_to_keep].reset_index(drop=True)

        else:
            if target_mean is not None:
                current_mean = self.data["y"].mean()
                if target_mean < current_mean:
                    # drop ones to decrease mean
                    num_zeros = (self.data["y"] == 0).sum()
                    num_ones_needed = int(target_mean * num_zeros / (1 - target_mean))
                    num_ones_to_drop = (self.data["y"] == 1).sum() - num_ones_needed
                    indices_to_remove = self.data[self.data["y"] == 1].index[
                        :num_ones_to_drop
                    ]
                    self.data.drop(indices_to_remove, inplace=True)
                elif target_mean > current_mean:
                    # drop zeros to increase mean
                    num_ones = (self.data["y"] == 1).sum()
                    num_zeros_needed = int(num_ones * (1 - target_mean) / target_mean)
                    num_zeros_to_drop = (self.data["y"] == 0).sum() - num_zeros_needed
                    indices_to_remove = self.data[self.data["y"] == 0].index[
                        :num_zeros_to_drop
                    ]
                    self.data.drop(indices_to_remove, inplace=True)

            if target_mean_x is not None:
                current_mean_x = self.data["x"].mean()
                if target_mean_x < current_mean_x:
                    # drop ones to decrease mean
                    num_zeros = (self.data["x"] == 0).sum()
                    num_ones_needed = int(
                        target_mean_x * num_zeros / (1 - target_mean_x)
                    )
                    num_ones_to_drop = (self.data["x"] == 1).sum() - num_ones_needed
                    indices_to_remove = self.data[self.data["x"] == 1].index[
                        :num_ones_to_drop
                    ]
                    self.data.drop(indices_to_remove, inplace=True)
                elif target_mean_x > current_mean_x:
                    # drop zeros to increase mean
                    num_ones = (self.data["x"] == 1).sum()
                    num_zeros_needed = int(
                        num_ones * (1 - target_mean_x) / target_mean_x
                    )
                    num_zeros_to_drop = (self.data["x"] == 0).sum() - num_zeros_needed
                    indices_to_remove = self.data[self.data["x"] == 0].index[
                        :num_zeros_to_drop
                    ]
                    self.data.drop(indices_to_remove, inplace=True)

        self.current_index = 0
        self.p_y_x_0 = self.data[self.data["x"] == 0]["y"].mean()
        self.p_y_x_1 = self.data[self.data["x"] == 1]["y"].mean()
        self.p_x_y_0 = self.data[self.data["y"] == 0]["x"].mean()
        self.p_x_y_1 = self.data[self.data["y"] == 1]["x"].mean()
        if hypothesis == "null":
            self.p_x = self.data["x"].mean()
            self.p_y = self.data["y"].mean()
            p_x_0_y_1 = 1 - self.p_x_y_1
            p_x_0_y_0 = 1 - self.p_x_y_0
            self.tau_0 = p_x_0_y_0 / (p_x_0_y_0 + p_x_0_y_1)
            self.tau_1 = self.p_x_y_0 / (self.p_x_y_0 + self.p_x_y_1)
            self.th_0 = self.p_y_x_0
            self.th_1 = self.p_y_x_1
            self.T_0 = (self.th_0 * self.tau_0) / (
                (1 - self.th_0) * (1 - self.tau_0) + (self.th_0 * self.tau_0)
            )
            self.T_1 = (self.th_1 * self.tau_1) / (
                (1 - self.th_1) * (1 - self.tau_1) + (self.th_1 * self.tau_1)
            )

    def sample(self, n):
        if self.hypothesis == "null":
            x = self.rng.binomial(1, self.p_x, size=n)
            p_y_given_x = np.where(x == 1, self.p_y_x_1, self.p_y_x_0)
            y = self.rng.binomial(1, p_y_given_x, size=n)
            return pd.DataFrame({"x": x, "y": y})
        if self.current_index + n > len(self.data):
            raise ValueError("No more unique rows available to sample.")
        ret = self.data.iloc[self.current_index : self.current_index + n].reset_index(
            drop=True
        )
        self.current_index += n
        return ret

    def sample_x(self, n):
        if self.hypothesis == "null":
            return self.rng.binomial(1, self.p_x, size=n)
        if self.current_index + n > len(self.data):
            raise ValueError("No more unique rows available to sample.")
        ret = self.data.iloc[self.current_index : self.current_index + n].reset_index(
            drop=True
        )
        self.current_index += n
        return ret["x"].to_numpy()


class LlmConceptShift:
    def __init__(
        self,
        model_name,
        seed,
        hypothesis="null",
        target_mean=None,
        target_p_y_x_0=None,
        target_p_y_x_1=None,
        data_parent_dir="data",
    ):
        self.model_name = model_name
        self.hypothesis = hypothesis
        splits = ["train", "test"]
        datasets = ["gsm8k", "aqua_rat"]
        data_frames = []
        for dataset in datasets:
            for split in splits:
                df = pd.read_parquet(
                    f"{data_parent_dir}/math/{model_name}/{dataset}/{split}/results.parquet"
                )
                df["x"] = dataset
                data_frames.append(df)
        self.data = pd.concat(data_frames).reset_index(drop=True)
        self.data["x"] = self.data["x"].map({"gsm8k": 0, "aqua_rat": 1, "math": 2})
        rng = np.random.default_rng(seed=seed)
        self.rng = rng
        shuffled_indices = rng.permutation(len(self.data))
        self.data = self.data.iloc[shuffled_indices].reset_index(drop=True)
        # Calculate available counts
        mask0 = self.data["x"] == 0
        mask1 = self.data["x"] == 1

        A00 = ((self.data["y"] == 0) & mask0).sum()
        A01 = ((self.data["y"] == 1) & mask0).sum()
        A10 = ((self.data["y"] == 0) & mask1).sum()
        A11 = ((self.data["y"] == 1) & mask1).sum()

        # Determine target p0, p1
        p0 = target_p_y_x_0
        if p0 is None:
            p0 = A01 / (A00 + A01) if (A00 + A01) > 0 else 0

        p1 = target_p_y_x_1
        if p1 is None:
            p1 = A11 / (A10 + A11) if (A10 + A11) > 0 else 0

        # Calculate max group sizes
        def get_max_n(A0, A1, p):
            if p == 0:
                return A0
            if p == 1:
                return A1
            # N * p <= A1  -> N <= A1/p
            # N * (1-p) <= A0 -> N <= A0/(1-p)
            return min(int(A1 / p), int(A0 / (1 - p)))

        N0_max = get_max_n(A00, A01, p0)
        N1_max = get_max_n(A10, A11, p1)

        N0_final = N0_max
        N1_final = N1_max

        if target_mean is not None:
            if abs(p1 - p0) > 1e-9:
                target_px = (target_mean - p0) / (p1 - p0)
                # Clip to [0, 1]
                target_px = max(0.0, min(1.0, target_px))

                if target_px == 0:
                    N1_final = 0
                    N0_final = N0_max
                elif target_px == 1:
                    N0_final = 0
                    N1_final = N1_max
                else:
                    R = target_px / (1 - target_px)
                    # N1 = R * N0
                    # N0 <= N0_max
                    # R * N0 <= N1_max -> N0 <= N1_max / R

                    N0_final = min(N0_max, int(N1_max / R))
                    N1_final = int(N0_final * R)
            else:
                # p1 == p0. If target_mean != p0, impossible to adjust via P(X).
                # We just keep max data.
                pass

        # Calculate final counts for each cell
        k01 = int(N0_final * p0)
        k00 = N0_final - k01

        k11 = int(N1_final * p1)
        k10 = N1_final - k11

        # Select indices
        idx00 = self.data[(self.data["x"] == 0) & (self.data["y"] == 0)].index[:k00]
        idx01 = self.data[(self.data["x"] == 0) & (self.data["y"] == 1)].index[:k01]
        idx10 = self.data[(self.data["x"] == 1) & (self.data["y"] == 0)].index[:k10]
        idx11 = self.data[(self.data["x"] == 1) & (self.data["y"] == 1)].index[:k11]

        indices_to_keep = np.concatenate([idx00, idx01, idx10, idx11])
        self.data = self.data.loc[indices_to_keep].reset_index(drop=True)

        self.current_index = 0
        self.p_y_x_0 = (
            self.data[self.data["x"] == 0]["y"].mean() if N0_final > 0 else p0
        )
        self.p_y_x_1 = (
            self.data[self.data["x"] == 1]["y"].mean() if N1_final > 0 else p1
        )
        if hypothesis == "null":
            self.p_x = self.data["x"].mean()
            self.p_y = self.data["y"].mean()

        shuffled_indices = rng.permutation(len(self.data))
        self.data = self.data.iloc[shuffled_indices].reset_index(drop=True)

    def sample(self, n):
        if self.hypothesis == "null":
            x = self.rng.binomial(1, self.p_x, size=n)
            p_y_given_x = np.where(x == 1, self.p_y_x_1, self.p_y_x_0)
            y = self.rng.binomial(1, p_y_given_x, size=n)
            return pd.DataFrame({"x": x, "y": y})

        if self.current_index + n > len(self.data):
            raise ValueError("No more unique rows available to sample.")
        ret = self.data.iloc[self.current_index : self.current_index + n].reset_index(
            drop=True
        )
        self.current_index += n
        return ret

    def sample_x(self, n):
        if self.hypothesis == "null":
            return self.rng.binomial(1, self.p_x, size=n)
        if self.current_index + n > len(self.data):
            raise ValueError("No more unique rows available to sample.")
        ret = self.data.iloc[self.current_index : self.current_index + n].reset_index(
            drop=True
        )
        self.current_index += n
        return ret["x"].to_numpy()


class HealthCareMultivariateDataSampler:

    def __init__(
        self,
        year,
        seed=None,
        data_dir="data",
    ):
        self.year = year
        rng = np.random.default_rng(seed=seed)
        self.rng = rng
        self.data = pd.read_parquet(f"{data_dir}/health_care/features_{year}.parquet")
        self.data = self.data.rename(columns={"generated_label_Y": "y"})
        y = self.data["y"]
        x = self.data.drop(columns=["y"]).to_numpy()
        self.data = pd.DataFrame({"x": list(x), "y": y})
        self.p_y = self.data["y"].mean()

        shuffled_indices = rng.permutation(len(self.data))
        self.data = self.data.iloc[shuffled_indices].reset_index(drop=True)
        self.current_index = 0

    def sample(self, n):
        if self.current_index + n > len(self.data):
            raise ValueError("No more unique rows available to sample.")
        ret = self.data.iloc[self.current_index : self.current_index + n].reset_index(
            drop=True
        )
        self.current_index += n
        return ret

    def sample_x(self, n):
        if self.current_index + n > len(self.data):
            raise ValueError("No more unique rows available to sample.")
        ret = self.data.iloc[self.current_index : self.current_index + n].reset_index(
            drop=True
        )
        self.current_index += n
        return np.stack(ret["x"].to_numpy())
