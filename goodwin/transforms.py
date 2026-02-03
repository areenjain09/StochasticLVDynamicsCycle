import numpy as np

def bounded_from_latent(X):
    X = np.maximum(X, 1e-12)
    return X / (1.0 + X)

def latent_from_bounded(E):
    E = np.clip(E, 1e-12, 1.0 - 1e-12)
    return E / (1.0 - E)

# Compatibility with your earlier naming
def X_from_E(E):
    return latent_from_bounded(E)

def E_from_X(X):
    return bounded_from_latent(X)
