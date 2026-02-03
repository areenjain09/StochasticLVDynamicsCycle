import numpy as np
import matplotlib.pyplot as plt
from goodwin.sim import simulate_stochastic_latent
from goodwin.transforms import latent_from_bounded, bounded_from_latent

def H_latent(X, Y, a, b, c, d):
    X = np.maximum(X, 1e-12)
    Y = np.maximum(Y, 1e-12)
    return d * X - c * np.log(X) + b * Y - a * np.log(Y)

def H_bounded(E, W, a, b, c, d):
    X = latent_from_bounded(E)
    Y = latent_from_bounded(W)
    return H_latent(X, Y, a, b, c, d)

def main():
    theta_true = np.array([0.05, 0.08, 0.04, 0.06, 0.03, 0.03], dtype=float)
    a, b, c, d, sigX, sigY = theta_true

    X_star = c / d
    Y_star = a / b
    E_star = bounded_from_latent(np.array([X_star]))[0]
    W_star = bounded_from_latent(np.array([Y_star]))[0]

    X0, Y0 = 0.7, 0.55
    T, dt = 240.0, 0.02
    seed_obs = 12345

    t, X_obs, Y_obs, E_obs, W_obs = simulate_stochastic_latent(theta_true, X0=X0, Y0=Y0, T=T, dt=dt, seed=seed_obs)

    H_star = H_bounded(E_star, W_star, a, b, c, d)

    E_grid = np.linspace(0.05, 0.95, 320)
    W_grid = np.linspace(0.05, 0.95, 320)
    EE, WW = np.meshgrid(E_grid, W_grid)

    Z = H_bounded(EE, WW, a, b, c, d) - H_star
    lo, hi = np.quantile(Z.ravel(), [0.06, 0.94])
    Zc = np.clip(Z, lo, hi)

    plt.figure(figsize=(7.8, 6.2))
    cf = plt.contourf(EE, WW, Zc, levels=32, cmap="Purples", alpha=0.85)
    cbar = plt.colorbar(cf)
    cbar.set_label(r"$H(E,W)-H(E^*,W^*)$")

    plt.contour(EE, WW, Zc, levels=8, colors="black", linewidths=1.0, alpha=0.7)

    plt.plot(E_obs, W_obs, lw=1.2, label="Stochastic path")
    plt.scatter([E_star], [W_star], marker="x", s=140, color="red", label="Equilibrium")

    plt.xlim(0.05, 0.95)
    plt.ylim(0.05, 0.95)
    plt.xlabel(r"Employment rate $E$")
    plt.ylabel(r"Wage share $W$")
    plt.title("Stochastic Diffusion Across Invariant Contours")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("Fig_A5_stochastic_on_contours.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()


