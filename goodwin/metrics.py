import numpy as np
from scipy.signal import find_peaks

def cycle_metrics(t, E, W, burn_frac=0.2):
    n = len(t)
    burn = int(n * burn_frac)
    t2 = t[burn:]
    E2 = E[burn:]
    W2 = W[burn:]

    amp_E = float(E2.max() - E2.min())
    amp_W = float(W2.max() - W2.min())

    peaks, _ = find_peaks(E2, prominence=1e-4, distance=20)
    if len(peaks) >= 3:
        peak_times = t2[peaks]
        periods = np.diff(peak_times)
        period_mean = float(periods.mean())
        period_sd = float(periods.std(ddof=1)) if len(periods) > 1 else 0.0
    else:
        period_mean, period_sd = np.nan, np.nan

    return {
        "amp_E": amp_E,
        "amp_W": amp_W,
        "period_mean": period_mean,
        "period_sd": period_sd,
        "num_peaks": int(len(peaks)),
    }

def empirical_two_sided_pvalue(obs_value, sim_values):
    sim_values = np.asarray(sim_values)
    cdf = np.mean(sim_values <= obs_value)
    p_two = 2 * min(cdf, 1 - cdf)
    return float(p_two), float(cdf)
