import numpy as np

def euler_maruyama(f, g, x0, t, dt, n):
    """
    Euler-Maruyama method for simulating stochastic differential equations (SDEs).

    Parameters:
    f : function
        Drift coefficient function.
    g : function
        Diffusion coefficient function.
    x0 : array_like
        Initial condition.
    t : array_like
        Time points at which to simulate.
    dt : float
        Time step size.
    n : int
        Number of paths to simulate.

    Returns:
    X : ndarray
        Simulated paths of the SDE.
    """
    m = len(t)
    d = len(x0)
    X = np.zeros((m, n, d))
    X[0] = x0

    for i in range(1, m):
        dW = np.random.normal(0, np.sqrt(dt), size=(n, d))
        X[i] = X[i-1] + f(X[i-1], t[i-1]) * dt + g(X[i-1], t[i-1]) * dW

    return X

def milstein(f, g, x0, t, dt, n):
    """
    Milstein method for simulating stochastic differential equations (SDEs).

    Parameters:
    f : function
        Drift coefficient function.
    g : function
        Diffusion coefficient function.
    x0 : array_like
        Initial condition.
    t : array_like
        Time points at which to simulate.
    dt : float
        Time step size.
    n : int
        Number of paths to simulate.

    Returns:
    X : ndarray
        Simulated paths of the SDE.
    """
    m = len(t)
    d = len(x0)
    X = np.zeros((m, n, d))
    X[0] = x0

    for i in range(1, m):
        dW = np.random.normal(0, np.sqrt(dt), size=(n, d))
        X[i] = X[i-1] + f(X[i-1], t[i-1]) * dt + g(X[i-1], t[i-1]) * dW + \
                 0.5 * g(X[i-1], t[i-1]) @ g(X[i-1], t[i-1]).T * (dW**2 - dt)

    return X