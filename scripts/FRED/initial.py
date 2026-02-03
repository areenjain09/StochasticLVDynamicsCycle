import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

UNRATE_CSV = "data/UNRATE.csv"
GDP_CSV    = "data/GDP.csv"
COE_CSV    = "data/COE.csv"

START_DATE = "1960-01-01"
END_DATE   = "2023-12-31"

DT = 1.0
SMOOTH_Q = 8
MAX_LAG = 24
SEED = 123
EPS = 1e-6

def load_fred_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if "DATE" in df.columns:
        date_col = "DATE"
    elif "observation_date" in df.columns:
        date_col = "observation_date"
    else:
        raise ValueError(f"{path}: can't find DATE/observation_date. Columns={df.columns.tolist()}")

    val_cols = [c for c in df.columns if c != date_col]
    if len(val_cols) != 1:
        raise ValueError(f"{path}: expected exactly 1 value column besides {date_col}, got {val_cols}")
    val_col = val_cols[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    s = pd.to_numeric(df[val_col].replace(".", np.nan), errors="coerce")
    s.name = val_col
    return s

def build_dataset():
    unrate_m = load_fred_csv(UNRATE_CSV).loc[START_DATE:END_DATE]
    gdp_q    = load_fred_csv(GDP_CSV).loc[START_DATE:END_DATE]
    coe_q    = load_fred_csv(COE_CSV).loc[START_DATE:END_DATE]

    E_q = (1.0 - unrate_m / 100.0).resample("QE").mean()
    W_q = (coe_q / gdp_q)

    E_qp = E_q.to_period("Q")
    W_qp = W_q.to_period("Q")

    df = pd.concat([E_qp, W_qp], axis=1).dropna()
    df.columns = ["E", "W"]
    df.index = df.index.to_timestamp(how="end")
    df["E"] = df["E"].clip(EPS, 1 - EPS)
    df["W"] = df["W"].clip(EPS, 1 - EPS)
    return df

def to_latent(E, W):
    X = E / (1.0 - E)
    Y = W / (1.0 - W)
    return X, Y

def to_bounded(X, Y):
    E = X / (1.0 + X)
    W = Y / (1.0 + Y)
    return E, W

def neg_loglike(theta, Xt, Yt):
    a, b, c, d, sx, sy = theta
    if min(theta) <= 0:
        return 1e18

    X0, X1 = Xt[:-1], Xt[1:]
    Y0, Y1 = Yt[:-1], Yt[1:]

    muX = a * X0 - b * X0 * Y0
    muY = -c * Y0 + d * X0 * Y0

    Xmean = X0 + muX * DT
    Ymean = Y0 + muY * DT

    Xvar = (sx * X0) ** 2 * DT
    Yvar = (sy * Y0) ** 2 * DT
    Xvar = np.maximum(Xvar, 1e-12)
    Yvar = np.maximum(Yvar, 1e-12)

    llX = -0.5 * (np.log(2 * np.pi * Xvar) + (X1 - Xmean) ** 2 / Xvar)
    llY = -0.5 * (np.log(2 * np.pi * Yvar) + (Y1 - Ymean) ** 2 / Yvar)
    return -(llX.sum() + llY.sum())

def estimate_params(df):
    Xobs, Yobs = to_latent(df["E"].values, df["W"].values)

    x0 = np.array([0.05, 0.08, 0.04, 0.06, 0.03, 0.03], dtype=float)
    bounds = [(1e-8, None)] * 6

    res = minimize(lambda th: neg_loglike(th, Xobs, Yobs),
                   x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 20000})
    if not res.success:
        raise RuntimeError("Estimation failed: " + str(res.message))

    a, b, c, d, sx, sy = res.x
    X_star = c / d
    Y_star = a / b
    E_star, W_star = to_bounded(np.array([X_star]), np.array([Y_star]))
    return a, b, c, d, sx, sy, float(E_star[0]), float(W_star[0])

def simulate_det(a, b, c, d, X0, Y0, T, dt=1.0):
    X = np.zeros(T); Y = np.zeros(T)
    X[0], Y[0] = X0, Y0
    for t in range(T - 1):
        muX = a * X[t] - b * X[t] * Y[t]
        muY = -c * Y[t] + d * X[t] * Y[t]
        X[t + 1] = max(X[t] + muX * dt, 1e-10)
        Y[t + 1] = max(Y[t] + muY * dt, 1e-10)
    return X, Y

def simulate_sde(a, b, c, d, sx, sy, X0, Y0, T, dt=1.0, seed=123):
    rng = np.random.default_rng(seed)
    X = np.zeros(T); Y = np.zeros(T)
    X[0], Y[0] = X0, Y0
    for t in range(T - 1):
        muX = a * X[t] - b * X[t] * Y[t]
        muY = -c * Y[t] + d * X[t] * Y[t]
        X[t + 1] = max(X[t] + muX * dt + sx * X[t] * np.sqrt(dt) * rng.normal(), 1e-10)
        Y[t + 1] = max(Y[t] + muY * dt + sy * Y[t] * np.sqrt(dt) * rng.normal(), 1e-10)
    return X, Y

def xcorr_leads(x, y, max_lag=24):
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    for L in lags:
        if L < 0:
            corrs.append(np.corrcoef(x[-L:], y[:L])[0, 1])
        elif L > 0:
            corrs.append(np.corrcoef(x[:-L], y[L:])[0, 1])
        else:
            corrs.append(np.corrcoef(x, y)[0, 1])
    corrs = np.array(corrs)
    best_lag = int(lags[np.nanargmax(corrs)])
    return lags, corrs, best_lag

def main():
    df = build_dataset()
    a, b, c, d, sx, sy, E_star, W_star = estimate_params(df)

    Xobs, Yobs = to_latent(df["E"].values, df["W"].values)
    X0, Y0 = Xobs[0], Yobs[0]
    Tn = len(df)

    X_det, Y_det = simulate_det(a, b, c, d, X0, Y0, Tn, dt=DT)
    E_det, W_det = to_bounded(X_det, Y_det)

    X_sim, Y_sim = simulate_sde(a, b, c, d, sx, sy, X0, Y0, Tn, dt=DT, seed=SEED)
    E_sim, W_sim = to_bounded(X_sim, Y_sim)
