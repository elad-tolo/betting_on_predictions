import numpy as np


def project_to_simplex(v):
    """
    Projects a vector v onto the probability simplex.
    (i.e., finds closest w such that sum(w)=1 and w >= 0)
    Based on Duchi et al. (2008).
    """
    n = len(v)
    # Sort v in descending order
    u = np.sort(v)[::-1]

    # Calculate cumulative sums
    cssv = np.cumsum(u)

    # Find the index rho
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]

    # Calculate theta
    theta = (cssv[rho] - 1) / (rho + 1)

    # Calculate w
    w = np.maximum(v - theta, 0)
    return w


class AdaGrad:
    def __init__(self, eta0=0.0001, eps=1e-8, weight=0.5):
        """
        eta0 : base learning rate (you can leave at 1.0) 0.01 0.0001
        eps  : small constant to avoid division by zero
        """
        self.weight = weight  # initial weight in [0,1]
        self.G2 = 0.0
        self.eta0 = eta0
        self.eps = eps

    def step(self, grad, clip_max=1.0, clip_min=0.0, project=False):
        """
        Perform one AdaGrad update given observed S_t, S'_t.
        Returns: chosen a_t, reward at_t
        """
        # Clip gradient to prevent overflow when squaring
        max_grad = 1e150  # sqrt of this is ~3e75, well within float64 range
        grad = np.clip(grad, -max_grad, max_grad)

        # Accumulate squared gradients
        self.G2 += grad**2

        # Adaptive step size
        eta_t = self.eta0 / (np.sqrt(self.G2) + self.eps)

        # Update with projection onto [0,1]
        if project:
            self.weight = self.weight + eta_t * grad
            self.weight = project_to_simplex(self.weight)
        else:
            self.weight = np.clip(self.weight + eta_t * grad, clip_min, clip_max)

        return self.weight
