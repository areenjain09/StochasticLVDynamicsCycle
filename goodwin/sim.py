import numpy as np
from scipy.integrate import solve_ivp
from .lv import drift_latent_lv, drift_bounded
from .transforms import bounded_from_latent

def simulate_bounded_ode(alpha, beta, gamma, delta, E0, W0, T=300.0, dt=0.01):
    N = int(T / dt)
    t = np.linspace(0.0, T, N + 1)
    E = np.zeros(N + 1)
    W = np.zeros(N + 1)
    E[0], W[0] = E0, W0

    for i in range(N):
        e, w = E[i], W[i]

        k1e, k1w = drift_bounded(e, w, alpha, beta, gamma, delta)
        k2e, k2w = drift_bounded(e + 0.5 * dt * k1e, w + 0.5 * dt * k1w, alpha, beta, gamma, delta)
        k3e, k3w = drift_bounded(e + 0.5 * dt * k2e, w + 0.5 * dt * k2w, alpha, beta, gamma, delta)
        k4e, k4w = drift_bounded(e + dt * k3e, w + dt * k3w, alpha, beta, gamma, delta)

        E[i + 1] = e + (dt / 6.0) * (k1e + 2 * k2e + 2 * k3e + k4e)
        W[i + 1] = w + (dt / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)

        E[i + 1] = np.clip(E[i + 1], 1e-10, 1.0 - 1e-10)
        W[i + 1] = np.clip(W[i + 1], 1e-10, 1.0 - 1e-10)

    return t, E, W

def simulate_deterministic_latent(theta_drift, X0=0.7, Y0=0.55, T=240.0, dt=0.02):
    a, b, c, d = theta_drift
    t_eval = np.arange(0.0, T + dt, dt)

    def rhs(t, z):
        X, Y = z
        dX, dY = drift_latent_lv(X, Y, a, b, c, d)
        return [dX, dY]

    sol = solve_ivp(rhs, (0.0, T), [X0, Y0], t_eval=t_eval, rtol=1e-9, atol=1e-12, method="RK45")
    t = sol.t
    X = np.maximum(sol.y[0], 1e-12)
    Y = np.maximum(sol.y[1], 1e-12)
    E = bounded_from_latent(X)
    W = bounded_from_latent(Y)
    return t, X, Y, E, W

def simulate_stochastic_latent(theta, X0=0.7, Y0=0.55, T=240.0, dt=0.02, seed=0):
    a, b, c, d, sigX, sigY = theta
    rng = np.random.default_rng(seed)

    N = int(T / dt)
    t = np.linspace(0.0, T, N + 1)
    X = np.zeros(N + 1)
    Y = np.zeros(N + 1)
    X[0], Y[0] = X0, Y0

    sqrt_dt = np.sqrt(dt)
    for i in range(N):
        x, y = X[i], Y[i]
        muX, muY = drift_latent_lv(x, y, a, b, c, d)
        x_next = x + muX * dt + sigX * x * sqrt_dt * rng.normal()
        y_next = y + muY * dt + sigY * y * sqrt_dt * rng.normal()
        X[i + 1] = max(x_next, 1e-12)
        Y[i + 1] = max(y_next, 1e-12)

    E = bounded_from_latent(X)
    W = bounded_from_latent(Y)
    return t, X, Y, E, W
