import numpy as np
from .systems import wrap_angle
from .integrators import simulate
from .systems import DoublePendulum

rng = np.random.default_rng(42)


def sample_initial_conditions(n, mean, std):
    """Sample n initial conditions from Gaussian priors."""
    mean = np.asarray(mean, dtype=float)
    std  = np.asarray(std, dtype=float)
    X0 = rng.normal(loc=mean, scale=std, size=(n, len(mean)))
    X0[:, 0] = wrap_angle(X0[:, 0])
    X0[:, 2] = wrap_angle(X0[:, 2])
    return X0


def unwrap_angles(theta_series):
    """Unwrap angle trajectory into continuous rotation count."""
    return np.unwrap(theta_series)


def detect_flip(times, theta_unwrapped, threshold=np.pi):
    """First time a flip occurs (|Δtheta| > threshold)."""
    ref = theta_unwrapped[0]
    delta = np.abs(theta_unwrapped - ref)
    idx = np.argmax(delta > threshold)
    if delta[idx] > threshold:
        return times[idx]
    return np.inf


def run_ensemble(system, n_samples=200, t_end=10.0, dt=0.005,
                 mean=[np.pi/2, 0.0, 0.1, 0.0],
                 std =[0.01,     0.01, 0.01, 0.01]):
    """
    Run ensemble simulations for a given system.

    Returns:
        dict with ensemble trajectories, flip times, energies
    """
    X0 = sample_initial_conditions(n_samples, mean, std)
    n_steps = int(np.ceil(t_end / dt)) + 1

    thetas1 = np.zeros((n_samples, n_steps))
    thetas2 = np.zeros((n_samples, n_steps))
    energies_all = np.zeros((n_samples, n_steps))
    flips_t1 = np.zeros(n_samples)
    flips_t2 = np.zeros(n_samples)

    for i in range(n_samples):
        times, traj, energies = simulate(system, X0[i], 0.0, t_end, dt)
        thetas1[i] = traj[:, 0]
        thetas2[i] = traj[:, 2]
        energies_all[i] = energies

        th1_unwrapped = unwrap_angles(thetas1[i])
        th2_unwrapped = unwrap_angles(thetas2[i])

        flips_t1[i] = detect_flip(times, th1_unwrapped)
        flips_t2[i] = detect_flip(times, th2_unwrapped)

    return {
        "times": times,
        "thetas1": thetas1,
        "thetas2": thetas2,
        "flips_t1": flips_t1,
        "flips_t2": flips_t2,
        "energies_all": energies_all
    }
