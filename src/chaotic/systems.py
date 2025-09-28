import numpy as np
# -------------------------------
# Double Pendulum System
# -------------------------------

def wrap_angle(theta: float) -> float:
    """Wrap angle into [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


class DoublePendulum:
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        self.m1, self.m2, self.l1, self.l2, self.g = m1, m2, l1, l2, g

    def derivs(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Equations of motion for the double pendulum.
        x = [theta1, omega1, theta2, omega2]
        """
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        th1, w1, th2, w2 = x
        delta = th2 - th1
        den1 = (2*m1 + m2 - m2 * np.cos(2*th1 - 2*th2))

        dw1 = (
            -g*(2*m1+m2)*np.sin(th1)
            - m2*g*np.sin(th1 - 2*th2)
            - 2*np.sin(delta)*m2*(w2**2*l2 + w1**2*l1*np.cos(delta))
        ) / (l1 * den1)

        dw2 = (
            2*np.sin(delta) * (
                w1**2*l1*(m1+m2)
                + g*(m1+m2)*np.sin(th1)
                + w2**2*l2*m2*np.cos(delta)
            )
        ) / (l2 * den1)

        return np.array([w1, dw1, w2, dw2], dtype=float)

    def total_energy(self, x: np.ndarray) -> float:
        """Compute total energy (K + U)."""
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        th1, w1, th2, w2 = x
        delta = th2 - th1

        T = 0.5*m1*(l1*w1)**2 \
            + 0.5*m2*((l1*w1)**2 + (l2*w2)**2 + 2*l1*l2*w1*w2*np.cos(delta))

        U = -(m1+m2)*g*l1*np.cos(th1) - m2*g*l2*np.cos(th2)
        return T + U
