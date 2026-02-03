# Stochastic Lotka-Volterra Dynamics in Macroeconomics: A Bounded Goodwin Wage-Employment Cycle

This repository implements a **bounded Goodwin / Lotka–Volterra macroeconomic model**
with both **deterministic** and **stochastic** dynamics, parameter estimation, and
diagnostic tools.

The model describes the joint evolution of:

- **Employment rate** \( E \in (0,1) \)
- **Wage share** \( W \in (0,1) \)

via a latent Lotka–Volterra system transformed to remain strictly within bounds.
The code supports simulation, inference, and empirical comparison.

---

## Model Overview

### Latent (unbounded) dynamics
\[
\begin{aligned}
dX &= (aX - bXY)\,dt + \sigma_X X\,dB_1 \\
dY &= (-cY + dXY)\,dt + \sigma_Y Y\,dB_2
\end{aligned}
\]

### Bounded observables
\[
E = \frac{X}{1+X}, \qquad W = \frac{Y}{1+Y}
\]

This guarantees:
- \(E,W \in (0,1)\)
- Interior equilibrium
- Well-defined stochastic dynamics

---

## Features

- Deterministic Goodwin/LV dynamics
- Stochastic Euler–Maruyama simulation
- Approximate maximum likelihood estimation (MLE)
- Cycle metrics (amplitude, period)
- Ensemble diagnostics and goodness-of-fit
- Phase plots, contour plots, and time-series figures
- Empirical application using FRED data

---

## Repository Structure


