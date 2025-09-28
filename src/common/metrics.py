import numpy as np
from scipy.stats import rankdata


# -------------------------
# Continuous Ranked Probability Score (CRPS)
# -------------------------
def crps_ensemble(forecasts: np.ndarray, observation: float) -> float:
    """
    Compute CRPS for an ensemble forecast.
    (Gneiting & Raftery, 2007)

    Args:
        forecasts: (N,) ensemble values
        observation: scalar truth

    Returns:
        float: CRPS value
    """
    forecasts = np.asarray(forecasts)
    N = len(forecasts)

    term1 = np.mean(np.abs(forecasts - observation))
    term2 = 0.5 * np.mean(np.abs(forecasts[:, None] - forecasts[None, :]))
    return term1 - term2


# -------------------------
# Energy Score (multivariate CRPS generalization)
# -------------------------
def energy_score(ensemble: np.ndarray, obs: np.ndarray, beta: float = 1.0) -> float:
    """
    Energy score for multivariate forecasts.

    Args:
        ensemble: (N, d) forecast ensemble
        obs: (d,) observation vector
        beta: power parameter (default 1.0 → energy score)

    Returns:
        float: energy score
    """
    ensemble = np.asarray(ensemble)
    obs = np.asarray(obs)

    N = ensemble.shape[0]

    term1 = np.mean(np.linalg.norm(ensemble - obs, axis=1) ** beta)
    term2 = 0.5 * np.mean(
        np.linalg.norm(ensemble[:, None, :] - ensemble[None, :, :], axis=2) ** beta
    )
    return term1 - term2


# -------------------------
# Wasserstein-1 Distance (Earth Mover)
# -------------------------
def wasserstein_1d(forecasts: np.ndarray, observation: float) -> float:
    """
    Compute 1D Wasserstein-1 (a.k.a. Earth Mover) distance between
    ensemble forecast distribution and observation.

    Args:
        forecasts: (N,) ensemble values
        observation: scalar truth

    Returns:
        float: W1 distance
    """
    forecasts = np.sort(np.asarray(forecasts))
    N = len(forecasts)
    obs_arr = np.repeat(observation, N)
    return np.mean(np.abs(forecasts - obs_arr))


# -------------------------
# PIT Histogram & Reliability Diagnostics
# -------------------------
def pit_values(ensemble: np.ndarray, observation: float) -> float:
    """
    Probability Integral Transform (PIT) value for one forecast–observation pair.
    Fraction of ensemble less than the observation.

    Args:
        ensemble: (N,) forecast ensemble
        observation: scalar truth

    Returns:
        float: PIT value in [0,1]
    """
    ensemble = np.asarray(ensemble)
    return np.mean(ensemble <= observation)


def pit_histogram(ensembles: np.ndarray, observations: np.ndarray, bins=10) -> np.ndarray:
    """
    Compute PIT histogram for many forecast–observation pairs.

    Args:
        ensembles: (M, N) forecasts, M cases × N ensemble members
        observations: (M,) truths
        bins: number of histogram bins

    Returns:
        hist: (bins,) PIT frequencies
        edges: bin edges
    """
    M, N = ensembles.shape
    pit_vals = [pit_values(ensembles[i], observations[i]) for i in range(M)]
    hist, edges = np.histogram(pit_vals, bins=bins, range=(0, 1), density=True)
    return hist, edges
