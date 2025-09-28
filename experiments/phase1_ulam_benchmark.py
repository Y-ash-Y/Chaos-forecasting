"""
phase1_ulam_benchmark.py

Ulam benchmarking script for Phase 1.

Usage (from repo root):
    python experiments/phase1_ulam_benchmark.py

What it does:
  1. Generate training trajectories (S short bursts) using mc_runner.run_ensemble (but shortened T)
  2. Build Ulam partition over 2D observables (theta1, theta2) (you can change dims)
  3. Estimate Ulam transition matrix P and initial histogram p0 via build_ulam_from_trajectories
  4. Generate an independent test ensemble (large) to serve as "truth" histograms at eval times
  5. For each eval horizon, compute:
        - empirical histogram from test ensemble
        - Ulam-predicted histogram by evolving p0 with P
        - scores: CRPS (marginal on theta1 via sampling from cell centers), W1 (1D on theta1), Energy Score (multivariate)
  6. Save results to experiments/results_ulam_phase1.npz

Notes:
  - This script assumes src is importable (run from repo root). If you get import errors,
    run: export PYTHONPATH=$(pwd)  (Linux) or set PYTHONPATH appropriately.
"""

import os
import time
import numpy as np

# Imports from your project modules
from src.chaotic.systems import DoublePendulum
from src.chaotic.mc_runner import run_ensemble
from src.chaotic.ulam_edmd import (
    build_ulam_partition,
    build_ulam_from_trajectories,
    predict_pdf_with_ulam,
    ulam_cell_centers,
    hist_from_states,
)
from src.common.metrics import crps_ensemble, wasserstein_1d, energy_score

# ----------------------------
# Config (tweak these)
# ----------------------------
# Training trajectories (for building Ulam)
S_train = 300           # number of short bursts used to estimate P
T_train = 40            # timesteps per burst (not seconds; we'll pick dt so T_train*dt is small)
dt_train = 0.01         # dt for training burst integration (keeps transitions small)
t_burst = (T_train - 1) * dt_train  # total burst duration in seconds

# Test ensemble (truth)
N_test = 500            # size of independent test ensemble (bigger = smoother empirical hist)
t_end_test = 8.0        # total test sim time in seconds
dt_test = 0.005

# Ulam partition (2D: theta1, theta2). You can increase bins for resolution.
# We set bounds reasonably large to capture most motion; adjust if your trajectories exceed them.
bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]
bins = [60, 60]         # 60x60 grid => 3600 cells (ok for moderate memory)

# Evaluation times (seconds)
eval_times = [1.0, 3.0, 5.0, 8.0]

# Output file
out_file = "experiments/results_ulam_phase1.npz"


# ----------------------------
# Helper: convert cell-histogram (p on Ulam grid) to sample forecasts for theta1
# ----------------------------
def sample_from_ulam_hist(p_hist, edges, n_samples=1000, rng_seed=0):
    """
    Draw samples in state-space from Ulam histogram p_hist (distribution over cells).
    We sample a cell proportional to p_hist, then sample uniformly within the cell.
    This heuristic gives a set of state samples for scoring 1D marginals (theta1).
    """
    rng = np.random.default_rng(rng_seed)
    M = p_hist.shape[0]
    # precompute cell centers and sizes
    centers = ulam_cell_centers(edges)  # shape (M, D)
    # sample cells
    if p_hist.sum() == 0:
        # edge case: empty histogram
        return np.zeros(n_samples)
    cells = rng.choice(np.arange(M), size=n_samples, p=p_hist)
    samples = centers[cells]  # pick center of chosen cell as representative (cheap)
    # For theta1 marginal, return first column
    return samples[:, 0]


# ----------------------------
# 1) Build training bursts (short trajectories) to estimate P
# ----------------------------
print("1) Building training bursts for Ulam estimation...")
system = DoublePendulum()
# We'll use mc_runner.run_ensemble but with short t_end such that each traj is a burst of T_train steps.
# run_ensemble uses parameters t_end (seconds) and dt; ensure T_train = t_end/dt + 1
res_train = run_ensemble(
    n_samples=S_train,
    t_end=t_burst,
    dt=dt_train,
    mean=[np.pi/2, 0.0, 0.1, 0.0],
    std=[0.005, 0.005, 0.005, 0.005],
)
# Build trajectories array: (S_train, T_train, D) where D = 2 (theta1, theta2) for Ulam
# Note: mc_runner returns full state (theta1, theta2...) in keys; we extract needed dims
times_train = res_train["times"]
thetas1_train = res_train["thetas1"]  # shape (S_train, T_train)
thetas2_train = res_train["thetas2"]
# stack into (S, T, D)
trajectories = np.stack([thetas1_train, thetas2_train], axis=2)  # shape (S, T, 2)

print(f"  Collected training bursts: S={S_train}, T={trajectories.shape[1]}, dt={dt_train}")

# ----------------------------
# 2) Build Ulam partition and estimate P
# ----------------------------
print("2) Building Ulam partition and estimating transition matrix P...")
edges = build_ulam_partition(bounds, bins)
P, p0 = build_ulam_from_trajectories(trajectories, edges)
M = P.shape[0]
print(f"  Ulam cells M={M}. Sum p0 = {p0.sum():.6f}")

# ----------------------------
# 3) Build independent test ensemble (truth histograms)
# ----------------------------
print("3) Building independent test ensemble (truth)...")
res_test = run_ensemble(
    n_samples=N_test,
    t_end=t_end_test,
    dt=dt_test,
    mean=[np.pi/2, 0.0, 0.1, 0.0],
    std=[0.01, 0.01, 0.01, 0.01],
)
times_test = res_test["times"]
thetas1_test = res_test["thetas1"]
thetas2_test = res_test["thetas2"]

# Precompute empirical histograms for each eval time (on Ulam grid)
eval_idx = [np.argmin(np.abs(times_test - t)) for t in eval_times]
empirical_histograms = []
for idx in eval_idx:
    states_at_t = np.stack([thetas1_test[:, idx], thetas2_test[:, idx]], axis=1)  # (N_test, 2)
    hist = hist_from_states(states_at_t, edges)  # (M,)
    empirical_histograms.append(hist)

print("  Built empirical histograms for eval times:", eval_times)

# ----------------------------
# 4) Predict with Ulam operator and score vs empirical histograms
# ----------------------------
print("4) Evolving p0 with Ulam operator and scoring against empirical histograms...")
results = {"eval_times": eval_times, "metrics": {}, "raw": {}}

# We'll iteratively evolve p_current forward and compare at eval steps.
p_current = p0.copy()
# Determine step size mismatch: operator step corresponds to dt_train; we must evolve the operator for appropriate number of small steps
# For each eval_time, compute number of operator steps k = round(eval_time / dt_train)
for t, target_hist in zip(eval_times, empirical_histograms):
    k_steps = int(np.round(t / dt_train))
    # Evolve p_current from time 0 to time t (compute P^k @ p0)
    p_pred = predict_pdf_with_ulam(P, p0, n_steps=k_steps)

    # Convert p_pred into forecast samples for marginals (theta1) for scoring
    samples_pred_theta1 = sample_from_ulam_hist(p_pred, edges, n_samples=2000, rng_seed=42)
    # For observation, sample from empirical histogram by sampling test ensemble states at the same time
    obs_states = np.stack([thetas1_test[:, np.argmin(np.abs(times_test - t))],
                           thetas2_test[:, np.argmin(np.abs(times_test - t))]], axis=1)
    # For CRPS we need a single observation. Use an ensemble of observations: score crps against each test member and average (proper Monte-Carlo approx).
    # Simpler: pick a random held-out obs (or loop). We'll compute CRPS averaged over a subset of test observations for stability.
    n_obs_for_crps = min(100, obs_states.shape[0])
    obs_indices = np.arange(n_obs_for_crps)
    crps_vals = []
    w1_vals = []
    energy_vals = []

    for oi in obs_indices:
        obs_theta1 = obs_states[oi, 0]
        crps_vals.append(crps_ensemble(samples_pred_theta1, obs_theta1))
        w1_vals.append(wasserstein_1d(samples_pred_theta1, obs_theta1))

    # Energy score: treat ensemble in full 2D. Build ensemble samples for 2D by sampling centers
    # Reuse same sampling but return both dims by sampling cell centers
    # sample_from_ulam_hist currently returns theta1 only; but we can create a quick 2D sampler here:
    centers = ulam_cell_centers(edges)  # (M,2)
    rng = np.random.default_rng(42)
    cells = rng.choice(np.arange(M), size=2000, p=p_pred)
    samples_pred_2d = centers[cells]  # (2000,2)
    # For obs ensemble, use a subset of test states as observations to compute energy_score (ensemble vs obs vector)
    obs_vec = obs_states[0]  # single obs vector, but energy_score expects ensemble vs obs vector shape
    es = energy_score(samples_pred_2d, obs_vec.reshape(-1))
    # store average metrics
    results["metrics"][t] = {
        "CRPS_mean": float(np.mean(crps_vals)),
        "CRPS_std": float(np.std(crps_vals)),
        "W1_mean": float(np.mean(w1_vals)),
        "W1_std": float(np.std(w1_vals)),
        "EnergyScore": float(es)
    }
    results["raw"][t] = {"p_pred": p_pred, "p_emp": target_hist}
    print(f"  t={t:.2f}s: k_steps={k_steps}, CRPS={results['metrics'][t]['CRPS_mean']:.4f}, W1={results['metrics'][t]['W1_mean']:.4f}, ES={es:.4f}")

# ----------------------------
# 5) Save outputs for plotting later
# ----------------------------
os.makedirs("experiments", exist_ok=True)
np.savez_compressed(out_file, **results)
print(f"Saved results to {out_file}. Done.")
