from scipy.integrate import solve_ivp
import numpy as np

class DoublePendulum:
    def __init__(self, length1, length2, mass1, mass2, theta1, theta2, omega1, omega2):
        self.length1 = length1
        self.length2 = length2
        self.mass1 = mass1
        self.mass2 = mass2
        self.theta1 = theta1
        self.theta2 = theta2
        self.omega1 = omega1
        self.omega2 = omega2

    def equations(self, t, y):
        theta1, omega1, theta2, omega2 = y
        delta = theta2 - theta1

        denom1 = (self.mass1 + self.mass2) * self.length1 - self.mass2 * self.length1 * np.cos(delta) ** 2
        denom2 = (self.length2 / self.length1) * denom1

        dtheta1_dt = omega1
        domega1_dt = ((self.mass2 * self.length1 * omega1 ** 2 * np.sin(delta) * np.cos(delta) +
                       self.mass2 * 9.81 * np.sin(theta2) * np.cos(delta) +
                       self.mass2 * self.length2 * omega2 ** 2 * np.sin(delta) -
                       (self.mass1 + self.mass2) * 9.81 * np.sin(theta1)) / denom1)

        dtheta2_dt = omega2
        domega2_dt = ((-self.mass2 * self.length2 * omega2 ** 2 * np.sin(delta) * np.cos(delta) +
                       (self.mass1 + self.mass2) * 9.81 * np.sin(theta1) * np.cos(delta) -
                       (self.mass1 + self.mass2) * self.length1 * omega1 ** 2 * np.sin(delta) -
                       (self.mass1 + self.mass2) * 9.81 * np.sin(theta2)) / denom2)

        return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

    def simulate(self, t_span, t_eval):
        y0 = [self.theta1, self.omega1, self.theta2, self.omega2]
        sol = solve_ivp(self.equations, t_span, y0, t_eval=t_eval)
        return sol.t, sol.y

class DuffingOscillator:
    def __init__(self, mass, damping, stiffness, nonlinearity, initial_position, initial_velocity):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.nonlinearity = nonlinearity
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity

    def equations(self, t, y):
        position, velocity = y
        dposition_dt = velocity
        dvelocity_dt = (-self.damping * velocity - self.stiffness * position - self.nonlinearity * position ** 3) / self.mass
        return [dposition_dt, dvelocity_dt]

    def simulate(self, t_span, t_eval):
        y0 = [self.initial_position, self.initial_velocity]
        sol = solve_ivp(self.equations, t_span, y0, t_eval=t_eval)
        return sol.t, sol.y

class MagneticPendulum:
    def __init__(self, length, mass, angle, angular_velocity):
        self.length = length
        self.mass = mass
        self.angle = angle
        self.angular_velocity = angular_velocity

    def equations(self, t, y):
        angle, angular_velocity = y
        dangle_dt = angular_velocity
        dangular_velocity_dt = -9.81 / self.length * np.sin(angle)
        return [dangle_dt, dangular_velocity_dt]

    def simulate(self, t_span, t_eval):
        y0 = [self.angle, self.angular_velocity]
        sol = solve_ivp(self.equations, t_span, y0, t_eval=t_eval)
        return sol.t, sol.y