import numpy as np
import matplotlib.pyplot as plt
from goodwin.sim import simulate_bounded_ode
from goodwin.lv import drift_bounded
from goodwin.transforms import E_from_X

def main():
    # params so equilibrium in latent (X*,Y*)=(1,1) => (E*,W*)=(0.5,0.5) if ratios match
    alpha, beta = 0.05, 0.05
    gamma, delta = 0.04, 0.04

    X_star, Y_star = gamma / delta, alpha / beta
    E_star, W_star = E_from_X(X_star), E_from_X(Y_star)

    T, dt = 400.0, 0.01
    values = np.linspace(0.3, 0.9, 5)
    colors = plt.cm.autumn_r(np.linspace(0.3, 1.0, len(values)))

    plt.figure(figsize=(8, 6))

    for v, col in zip(values, colors):
        E0 = np.clip(v * E_star, 1e-6, 1 - 1e-6)
        W0 = np.clip(v * W_star, 1e-6, 1 - 1e-6)
        t, E, W = simulate_bounded_ode(alpha, beta, gamma, delta, E0, W0, T=T, dt=dt)
        plt.plot(E, W, lw=3.5 * v, color=col, label=fr'$(E_0,W_0)=({E0:.2f},{W0:.2f})$')

    # Direction field
    x = np.linspace(0.02, 0.98, 22)
    y = np.linspace(0.02, 0.98, 22)
    EE, WW = np.meshgrid(x, y)
    dE, dW = drift_bounded(EE, WW, alpha, beta, gamma, delta)
    M = np.hypot(dE, dW)
    M[M == 0] = 1.0
    plt.quiver(EE, WW, dE / M, dW / M, M, pivot="mid", cmap="jet", alpha=0.9)

    plt.scatter([E_star], [W_star], marker="x", s=140, color="red", label="Equilibrium")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Employment rate $E$")
    plt.ylabel("Wage share $W$")
    plt.title("Bounded Goodwin/LV: Trajectories and Direction Field")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("Fig_phase_bounded_deterministic.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()


