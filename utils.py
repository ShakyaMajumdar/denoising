import numpy as np


def awgn(sig, snr_db):
    """
    add white gaussian noise to the signal
    snr_db: signal to noise ratio (in dB)
    """
    sig_p = np.mean(sig**2)
    snr = 10 ** (snr_db / 10)  # dB = 10 log_10 (x)
    noise_p = sig_p / snr
    noise = np.random.normal(0, np.sqrt(noise_p), sig.shape)
    return sig + noise


def calc_snr(clean, noisy):
    """
    compute signal to noise ratio of `noisy` wrt `clean`
    """
    return 10 * np.log10(np.sum(clean**2) / np.sum((noisy - clean) ** 2))


def snr_db_to_np(ns, snr_db):
    """
    convert an estimation of snr to an estimation of noise power
    """
    p_sn = np.mean(ns**2)
    snr = 10 ** (snr_db / 10)
    p_n = p_sn / (1 + snr)
    return p_n
