import numpy as np


def denoise_l2(ns, K):
    """
    minimize l2 regularized loss using a closed form solution
    """
    n = len(ns)
    L = np.array(
        [
            [1.0 if i == j else -1.0 if j == i + 1 else 0.0 for j in range(n)]
            for i in range(n - 1)
        ]
    )
    return np.linalg.inv(np.eye(n) + K * L.T @ L) @ ns


def denoise_l1(ns, K, n_epochs=1000):
    """
    minimize l1 regularized loss using gradient projection
    """
    n = len(ns)
    L = np.array(
        [
            [1.0 if i == j else -1.0 if j == i + 1 else 0.0 for j in range(n)]
            for i in range(n - 1)
        ]
    )
    mu = np.zeros(n - 1)
    for _ in range(1, n_epochs + 1):
        mu = mu - 0.25 * L @ L.T @ mu + 0.5 * L @ ns
        mu = mu / max(np.max(np.abs(mu)), K) * K
    return ns - 0.5 * L.T @ mu
