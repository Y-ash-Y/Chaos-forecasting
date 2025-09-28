import numpy as np
from .integrators import rk4_step
from .systems import wrap_angle

rng = np.random.default_rng(123)


def lyapunov_benettin(system, x0, dt=0.005, t_end=10.0, d0=1e-8, renorm_every=1):
    """
    Estimate the largest Lyapunov exponent (LLE) via Benettin's method.

    Args:
        system: system object with `.derivs` method
        x0: initial state vector (ndarray)
        dt: timestep
        t_end: total integration time
        d0: initial separation norm between trajectories
        renorm_every: steps after which we renormalize the perturbation

    Returns:
        times: array of times
        ftle: array of finite-time Lyapunov exponent estimates
        lle: final estimated largest Lyapunov exponent
    """
    # Create small random perturbation of norm d0
    delta = rng.normal(size=len(x0))
    delta = d0 * delta / np.linalg.norm(delta)

    x = x0.copy()
    y = x0 + delta

    n_steps = int(np.ceil(t_end / dt))
    times = np.arange(n_steps + 1) * dt
    ftle = np.zeros(n_steps + 1)

    sum_log = 0.0
    curr_norm = d0
    t = 0.0
    ftle[0] = 0.0

    for k in range(1, n_steps + 1):
        # advance both trajectories
        x = rk4_step(system.derivs, t, x, dt)
        y = rk4_step(system.derivs, t, y, dt)
        t += dt

        # compute difference
        diff = y - x
        # wrap angular components to stay consistent
        diff[0] = wrap_angle(diff[0])
        diff[2] = wrap_angle(diff[2])

        new_norm = np.linalg.norm(diff)

        if (k % renorm_every) == 0 and new_norm > 0:
            sum_log += np.log(new_norm / curr_norm)
            diff = (d0 / new_norm) * diff
            y = x + diff
            curr_norm = d0

        # finite-time LE up to time t
        ftle[k] = sum_log / t if t > 0 else 0.0

    lle = sum_log / t if t > 0 else np.nan
    return times, ftle, lle
