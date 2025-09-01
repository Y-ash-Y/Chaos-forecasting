from typing import Callable, List
import numpy as np

class UlamEDMD:
    def __init__(self, dynamics: Callable, grid_points: int):
        self.dynamics = dynamics
        self.grid_points = grid_points
        self.grid = None
        self.operators = None

    def build_grid(self, state_space: np.ndarray):
        self.grid = np.linspace(state_space[0], state_space[1], self.grid_points)

    def compute_operators(self, time_steps: int):
        if self.grid is None:
            raise ValueError("Grid not built. Call build_grid() first.")
        
        self.operators = np.zeros((self.grid_points, self.grid_points))
        for i in range(self.grid_points):
            for j in range(self.grid_points):
                self.operators[i, j] = self._compute_operator(i, j, time_steps)

    def _compute_operator(self, i: int, j: int, time_steps: int) -> float:
        # Placeholder for operator computation logic
        return np.random.rand()  # Replace with actual computation

    def apply_operator(self, state: np.ndarray) -> np.ndarray:
        if self.operators is None:
            raise ValueError("Operators not computed. Call compute_operators() first.")
        
        return np.dot(self.operators, state)

    def run_dynamics(self, initial_state: np.ndarray, time_steps: int) -> List[np.ndarray]:
        states = [initial_state]
        for _ in range(time_steps):
            next_state = self.dynamics(states[-1])
            states.append(next_state)
        return states
