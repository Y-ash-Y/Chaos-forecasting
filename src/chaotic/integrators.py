from scipy.integrate import solve_ivp
import numpy as np

def rk4(f, t0, y0, t1, dt):
    """Runge-Kutta 4th order integrator."""
    n_steps = int(np.ceil((t1 - t0) / dt))
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    
    t[0] = t0
    y[0] = y0
    
    for i in range(n_steps):
        t[i + 1] = t[i] + dt
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt / 2, y[i] + dt / 2 * k1)
        k3 = f(t[i] + dt / 2, y[i] + dt / 2 * k2)
        k4 = f(t[i] + dt, y[i] + dt * k3)
        y[i + 1] = y[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return t, y

def dopri5(f, t0, y0, t1, dt):
    """Dormand-Prince 5th order integrator."""
    sol = solve_ivp(f, [t0, t1], y0, method='DOP853', t_eval=np.arange(t0, t1, dt))
    return sol.t, sol.y.T

# Example usage:
# Define a simple harmonic oscillator
def harmonic_oscillator(t, y):
    return [y[1], -y[0]]

# Integrate using RK4
t_rk4, y_rk4 = rk4(harmonic_oscillator, 0, [1, 0], 10, 0.1)

# Integrate using DOPRI5
t_dopri5, y_dopri5 = dopri5(harmonic_oscillator, 0, [1, 0], 10, 0.1)