import unittest
from src.chaotic.systems import DoublePendulum, DuffingOscillator, MagneticPendulum
from src.chaotic.integrators import RK4, Dopri5

class TestChaoticSystems(unittest.TestCase):

    def setUp(self):
        self.double_pendulum = DoublePendulum()
        self.duffing_oscillator = DuffingOscillator()
        self.magnetic_pendulum = MagneticPendulum()

    def test_double_pendulum(self):
        initial_conditions = [1.0, 0.0, 0.0, 0.0]
        time_span = (0, 10)
        result = self.double_pendulum.integrate(initial_conditions, time_span, RK4)
        self.assertIsNotNone(result)

    def test_duffing_oscillator(self):
        initial_conditions = [0.5, 0.0]
        time_span = (0, 10)
        result = self.duffing_oscillator.integrate(initial_conditions, time_span, Dopri5)
        self.assertIsNotNone(result)

    def test_magnetic_pendulum(self):
        initial_conditions = [0.0, 0.0]
        time_span = (0, 10)
        result = self.magnetic_pendulum.integrate(initial_conditions, time_span, RK4)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()