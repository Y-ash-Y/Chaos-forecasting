import numpy as np

class FokkerPlanckSolver:
    def __init__(self, grid_size, time_steps, dt, diffusion_coefficient):
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.dt = dt
        self.diffusion_coefficient = diffusion_coefficient
        self.grid = np.zeros(grid_size)
        self.initialize_distribution()

    def initialize_distribution(self):
        # Initialize a Gaussian distribution in the center of the grid
        center = self.grid_size // 2
        self.grid[center - 5:center + 5] = 1.0

    def step(self):
        new_grid = np.zeros(self.grid_size)
        for i in range(1, self.grid_size - 1):
            new_grid[i] = (self.grid[i] +
                           self.diffusion_coefficient * self.dt * 
                           (self.grid[i + 1] - 2 * self.grid[i] + self.grid[i - 1]))
        self.grid = new_grid

    def solve(self):
        for _ in range(self.time_steps):
            self.step()

    def get_distribution(self):
        return self.grid

# Example usage:
# solver = FokkerPlanckSolver(grid_size=100, time_steps=1000, dt=0.01, diffusion_coefficient=0.1)
# solver.solve()
# distribution = solver.get_distribution()