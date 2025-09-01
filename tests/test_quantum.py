import unittest
from src.quantum.lindblad import Lindblad
from src.quantum.qtraj import QuantumTrajectory

class TestQuantumModule(unittest.TestCase):

    def setUp(self):
        self.lindblad_system = Lindblad(parameters={'gamma': 1.0})
        self.quantum_trajectory = QuantumTrajectory(initial_state=[1, 0])

    def test_lindblad_initialization(self):
        self.assertIsNotNone(self.lindblad_system)
        self.assertEqual(self.lindblad_system.parameters['gamma'], 1.0)

    def test_quantum_trajectory_initialization(self):
        self.assertIsNotNone(self.quantum_trajectory)
        self.assertEqual(self.quantum_trajectory.state, [1, 0])

    def test_lindblad_evolution(self):
        evolved_state = self.lindblad_system.evolve(time=1.0)
        self.assertIsNotNone(evolved_state)

    def test_quantum_trajectory_sampling(self):
        samples = self.quantum_trajectory.sample(num_samples=100)
        self.assertEqual(len(samples), 100)

if __name__ == '__main__':
    unittest.main()