from typing import NamedTuple
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from utils import snr_db_to_np


def make_Pi(i, P, N):
    """
    make_Pi(i, P, N) @ z extracts [z_i, z_i+1, ..., z_i+P-1] where len(z) = N
    """
    mat = np.zeros((P, N))
    for j in range(P):
        mat[j, (i + j) % N] = 1
    return mat


def get_patches(sig, P):
    """
    return list of all overlapping patches of length P each from sig
    """
    N = len(sig)
    patches = []
    for i in range(N):
        patches.append(make_Pi(i, P, N) @ sig)
    return patches


def _B(j, u, alphas, Nos, K):
    return (alphas[j] * Nos[j](u)) / sum((alphas[l] * Nos[l](u)) for l in range(K))


def _G(u, alphas, Nos, Cs, K):
    """
    denoise a single patch
    """
    return (
        sum(((alphas[j] * Nos[j](u)) * Cs[j]) for j in range(K))
        @ u
        / sum((alphas[l] * Nos[l](u)) for l in range(K))
    )


def _D(z, P, K, N, Pis, Cs, alphas, Nos):
    """
    gmm-denoise the signal
    """
    return 1 / P * sum(Pis[i].T @ _G(Pis[i] @ z, alphas, Nos, Cs, K) for i in range(N))


class _DenoiseCtx(NamedTuple):
    P: int  # size of each patch
    K: int  # no. of gmm components
    N: int  # size of signal
    sigma2: float  # noise power
    Pis: list[np.ndarray]  # Pis[i] extracts [i:i+P] slice from vector of size N
    Cov1: list[np.ndarray]  # gmm.covariances[j] + sigma2 * I
    Cs: list[np.ndarray]  # gmm.covariances[j] @ Cov1[j]^-1
    alphas: list[np.ndarray]  # gmm component weights
    Nos: list[np.ndarray]  # gmm component pdfs


class GMMDenoiser:
    def __init__(self, gmm_n_components, patch_size, train_signals):
        self.K = gmm_n_components
        self.P = patch_size
        self.gmm = GaussianMixture(gmm_n_components)
        self.train_signals = train_signals

        self._trained = False

    def fit(self):
        self.gmm.fit(
            [
                patch
                for train_signal in self.train_signals
                for patch in get_patches(train_signal, self.P)
            ]
        )
        self._trained = True

    def compute_ctx(self, signal, snr_db_est):
        N = len(signal)
        sigma2 = snr_db_to_np(signal, snr_db_est)
        Pis = [make_Pi(i, self.P, N) for i in range(N)]
        Cov1 = [
            self.gmm.covariances_[j] + sigma2 * np.eye(self.P) for j in range(self.K)
        ]
        Cs = [self.gmm.covariances_[j] @ np.linalg.inv(Cov1[j]) for j in range(self.K)]
        alphas = self.gmm.weights_
        Nos = [
            multivariate_normal(mean=mu, cov=sigma).pdf
            for mu, sigma in zip(self.gmm.means_, Cov1)
        ]
        return _DenoiseCtx(self.P, self.K, N, sigma2, Pis, Cov1, Cs, alphas, Nos)

    def assert_trained(self):
        if not self._trained:
            raise RuntimeError("Model has not been fit")

    def denoise(self, signal, *, snr_db_est):
        self.assert_trained()
        ctx = self.compute_ctx(signal, snr_db_est)
        return _D(signal, self.P, self.K, ctx.N, ctx.Pis, ctx.Cs, ctx.alphas, ctx.Nos)


class GMMRecoverer:
    def __init__(self, gmm_n_components, patch_size, train_signals):
        self.denoiser = GMMDenoiser(gmm_n_components, patch_size, train_signals)

    def fit(self):
        self.denoiser.fit()

    def recover(self, signal, *, phi, snr_db_est, lr, iters, switch_t):
        self.denoiser.assert_trained()
        M, N = phi.shape  # M: measurement length, N: target length

        P, K, _, sigma2, _, Cov1, Cs, alphas, Nos = self.denoiser.compute_ctx(
            signal, snr_db_est
        )
        Pis = [make_Pi(i, P, N) for i in range(N)]
        x = np.zeros(N)

        A = np.eye(N) - lr * phi.T @ phi
        B = lr * phi.T @ signal
        denoiser = lambda z: _D(z, P, K, N, Pis, Cs, alphas, Nos)

        # PnP - PGD
        for T in range(iters):
            x = denoiser(A @ x + B)
            if T == switch_t:
                W = (
                    1
                    / P
                    * sum(
                        Pis[i].T
                        @ (
                            sum(
                                _B(j, Pis[i] @ x, alphas, Nos, K) * Cs[j]
                                for j in range(K)
                            )
                        )
                        @ Pis[i]
                        for i in range(N)
                    )
                )
                denoiser = lambda z: W @ z
        return x
