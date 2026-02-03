import numpy as np
from .transforms import latent_from_bounded

def drift_latent_lv(X, Y, alpha, beta, gamma, delta):
    dX = alpha * X - beta * X * Y
    dY = -gamma * Y + delta * X * Y
    return dX, dY

def drift_bounded(E, W, alpha, beta, gamma, delta):
    # Map bounded -> latent
    X = latent_from_bounded(E)
    Y = latent_from_bounded(W)

    dX, dY = drift_latent_lv(X, Y, alpha, beta, gamma, delta)

  
    dE = (1.0 - E) ** 2 * dX
    dW = (1.0 - W) ** 2 * dY
    return dE, dW
