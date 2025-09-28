import numpy as np
from .systems import wrap_angle

def rk4_step(f, t: float, x: np.ndarray, dt: float) -> np.ndarray:
    """
    One step of RK4 integrator for x' = f(t,x).
    Wraps angular components to [-pi, pi].
    """
    k1 = f(t, x)
    k2 = f(t + 0.5*dt, x + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, x + 0.5*dt*k2)
    k4 = f(t + dt,     x + dt*k3)

    x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Wrap angle components (theta1, theta2)
    x_next[0] = wrap_angle(x_next[0])
    x_next[2] = wrap_angle(x_next[2])
    return x_next


def simulate(system, x0: np.ndarray, t0: float, t_end: float, dt: float):
    """
    Simulate a system forward using RK4.

    Args:
        system: object with `.derivs(t,x)` and `.total_energy(x)`
        x0: initial state vector
        t0: start time
        t_end: end time
        dt: timestep

    Returns:
        times: (N,) array
        traj: (N, len(x)) array of states
        energies: (N,) array of energies
    """
    n_steps = int(np.ceil((t_end - t0) / dt))
    times = t0 + np.arange(n_steps+1) * dt
    traj = np.zeros((n_steps+1, len(x0)))
    energies = np.zeros(n_steps+1)

    traj[0] = x0
    energies[0] = system.total_energy(x0)

    t = t0
    x = x0.copy()
    for i in range(1, n_steps+1):
        x = rk4_step(system.derivs, t, x, dt)
        t += dt
        traj[i] = x
        energies[i] = system.total_energy(x)

    return times, traj, energies
