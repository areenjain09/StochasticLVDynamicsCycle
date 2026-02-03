import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .lv import drift_latent_lv

def neg_loglik_latent(params, X, Y, dt):
    a, b, c, d, logsigX, logsigY = params
    sigX = np.exp(logsigX)
    sigY = np.exp(logsigY)
    if sigX <= 0 or sigY <= 0:
        return np.inf

    ll = 0.0
    N = len(X) - 1
    for i in range(N):
        xi, yi = X[i], Y[i]
        xi1, yi1 = X[i + 1], Y[i + 1]

        muX, muY = drift_latent_lv(xi, yi, a, b, c, d)
        meanX = xi + muX * dt
        meanY = yi + muY * dt

        varX = (sigX * xi) ** 2 * dt
        varY = (sigY * yi) ** 2 * dt

        ll += -0.5 * np.log(2 * np.pi * varX) - 0.5 * ((xi1 - meanX) ** 2) / varX
        ll += -0.5 * np.log(2 * np.pi * varY) - 0.5 * ((yi1 - meanY) ** 2) / varY

    return -ll

def estimate_params_mle(X_obs, Y_obs, dt, init=None):
    if init is None:
        init = np.array([0.04, 0.07, 0.03, 0.05, np.log(0.02), np.log(0.02)], dtype=float)

    bnds = [
        (1e-6, 1.0),
        (1e-6, 1.0),
        (1e-6, 1.0),
        (1e-6, 1.0),
        (np.log(1e-6), np.log(1.0)),
        (np.log(1e-6), np.log(1.0)),
    ]

    res = minimize(
        neg_loglik_latent,
        init,
        args=(X_obs, Y_obs, dt),
        method="L-BFGS-B",
        bounds=bnds,
        options={"maxiter": 300},
    )

    a, b, c, d, logsigX, logsigY = res.x
    return np.array([a, b, c, d, np.exp(logsigX), np.exp(logsigY)]), res

def param_recovery_table(theta_true, theta_hat):
    names = ["a", "b", "c", "d", "sigma_X", "sigma_Y"]
    theta_true = np.array(theta_true, dtype=float)
    theta_hat = np.array(theta_hat, dtype=float)
    rel_err = np.abs(theta_hat - theta_true) / np.maximum(np.abs(theta_true), 1e-12)
    return pd.DataFrame({
        "Parameter": names,
        "True": theta_true,
        "Estimated": theta_hat,
        "AbsError": np.abs(theta_hat - theta_true),
        "RelError": rel_err
    })
