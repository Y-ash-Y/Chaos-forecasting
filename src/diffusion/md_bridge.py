# md_bridge.py

import numpy as np

class LennardJonesBath:
    def __init__(self, num_particles, temperature, cutoff):
        self.num_particles = num_particles
        self.temperature = temperature
        self.cutoff = cutoff
        self.positions = np.random.rand(num_particles, 3)  # Random initial positions
        self.velocities = np.random.normal(0, np.sqrt(temperature), (num_particles, 3))  # Maxwelian distribution

    def potential_energy(self):
        energy = 0.0
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                r = np.linalg.norm(self.positions[i] - self.positions[j])
                if r < self.cutoff:
                    energy += 4 * ((1/r)**12 - (1/r)**6)  # Lennard-Jones potential
        return energy

    def update_positions(self, dt):
        self.positions += self.velocities * dt

    def apply_boundary_conditions(self):
        # Simple periodic boundary conditions
        self.positions = np.mod(self.positions, 1.0)

    def run_simulation(self, num_steps, dt):
        for step in range(num_steps):
            self.update_positions(dt)
            self.apply_boundary_conditions()
            if step % 100 == 0:
                print(f"Step {step}: Potential Energy = {self.potential_energy()}")

# Example usage
if __name__ == "__main__":
    lj_bath = LennardJonesBath(num_particles=100, temperature=1.0, cutoff=2.5)
    lj_bath.run_simulation(num_steps=1000, dt=0.01)