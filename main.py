import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime

K = 5
N_EPOCHS = 1000
SNR_DB = 20


def awgn(sig, snr_db):
    """
    add white gaussian noise to the signal
    snr_db: signal to noise ratio (in dB)
    """
    sig_p = torch.mean(sig**2)
    snr = 10 ** (snr_db / 10)  # dB = 10 log_10 (x)
    noise_p = sig_p / snr
    noise = torch.normal(0, noise_p, sig.shape)
    return sig + noise


def loss_fn_l2(X_true, X_pred):
    """
    weighted sum factoring in squared error of the model from signal, and jaggedness of the model (l2 norm)
    """
    return torch.sum((X_true - X_pred) ** 2) + K * torch.sum(torch.diff(X_pred, 1) ** 2)

def loss_fn_l1(X_true, X_pred):
    """
    weighted sum factoring in squared error of the model from signal, and jaggedness of the model (l1 norm)
    """
    return torch.sum((X_true - X_pred) ** 2) + K * torch.sum(torch.abs(torch.diff(X_pred, 1)))

def model1(T, Xn):
    model = nn.Sequential(
        nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, N_EPOCHS + 1):
        loss = loss_fn_l2(Xn, model(T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % (N_EPOCHS // 100) == 0:
            print(f"Finished epoch {epoch}, latest loss {loss}")
    return model(T)


def model2(T, Xn):
    """
    gradient descent on l2 regularized loss
    """
    model = nn.Linear(len(T), len(Xn))
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(1, N_EPOCHS + 1):
        loss = loss_fn_l2(Xn, model(T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % (N_EPOCHS // 100) == 0:
            print(f"Finished epoch {epoch}, latest loss {loss}")
    return model(T)


def model3(T, Xn):
    """
    closed form solution for l2 regularized loss
    """
    n = len(T)
    L = torch.tensor([[1. if i==j else -1. if j==i+1 else 0. for j in range(n)] for i in range(n-1)])
    return torch.inverse(torch.eye(n) + 2*K*L.T @ L) @ Xn

def model4(T, Xn):
    """
    gradient projection, l1 regularized loss
    """
    n = len(T)
    L = torch.tensor([[1. if i==j else -1. if j==i+1 else 0. for j in range(n)] for i in range(n-1)])
    mu = torch.zeros(n-1)
    for _ in range(1, N_EPOCHS+1):
        mu = mu - 0.25 * L @ L.T @ mu + 0.5 * L @ Xn
        mu = mu/max(torch.max(torch.abs(mu)), K) * K
    return Xn - 0.5 * L.T @ mu

def get_sine_sig(T):
    return torch.sin(2 * torch.pi * T)

def get_step_sig(T):
    X = torch.zeros_like(T)
    X[:250] = 1
    X[250:500] = 3
    X[500:750] = 0
    X[750:] = 2
    return X

def main():
    T = torch.linspace(-1, 1, 1000)  #.reshape(-1, 1)
    X = get_step_sig(T)
    Xn = awgn(X, SNR_DB)
    
    fig = plt.figure()
    fig.set_size_inches(10, 6)

    plt.subplot(2, 2, 1)
    # plt.plot(T, X)
    plt.scatter(T, X, s=2)
    plt.title("original signal")

    plt.subplot(2, 2, 2)
    # plt.plot(T, Xn)
    plt.scatter(T, Xn, s=2)
    plt.title("signal with noise")
    plt.text(0.5, 1, f"{SNR_DB =}")

    X_pred = model3(T, Xn)
    L = loss_fn_l2(Xn, X_pred)
    X_pred = X_pred.detach().numpy()
    plt.subplot(2, 2, 3)
    # plt.plot(T, X_pred)
    plt.scatter(T, X_pred, s=2)
    plt.title("denoised signal (l2 reg)")
    plt.text(0.5, 0.8, f"K = {K}")
    plt.text(0.5, 0.6, f"L = {float(L):.2f}")

    X_pred = model4(T, Xn)
    L = loss_fn_l2(Xn, X_pred)
    X_pred = X_pred.detach().numpy()
    plt.subplot(2, 2, 4)
    # plt.plot(T, X_pred)
    plt.scatter(T, X_pred, s=2)
    plt.title("denoised signal (l1 reg)")
    plt.text(0.5, 0.8, f"K = {K}")
    plt.text(0.5, 0.6, f"L = {float(L):.2f}")

    plt.savefig(f"out/{datetime.now().strftime('%d-%m-%y_%H-%M-%S')}.png", dpi=100)
    plt.show()


main()
