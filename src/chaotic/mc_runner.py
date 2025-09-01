from itertools import product
import numpy as np

class MonteCarloRunner:
    def __init__(self, system, num_samples, variance_reduction=False):
        self.system = system
        self.num_samples = num_samples
        self.variance_reduction = variance_reduction
        self.samples = None

    def generate_samples(self):
        # Generate samples based on the chaotic system dynamics
        self.samples = [self.system.sample() for _ in range(self.num_samples)]

    def apply_variance_reduction(self):
        if not self.variance_reduction:
            return self.samples
        
        # Implement variance reduction technique (e.g., control variates)
        # Placeholder for actual variance reduction logic
        reduced_samples = self.samples  # Modify this line with actual logic
        return reduced_samples

    def run_simulation(self):
        self.generate_samples()
        reduced_samples = self.apply_variance_reduction()
        return reduced_samples

    def compute_statistics(self):
        if self.samples is None:
            raise ValueError("No samples generated. Run the simulation first.")
        
        mean = np.mean(self.samples, axis=0)
        variance = np.var(self.samples, axis=0)
        return mean, variance

# Example usage:
# if __name__ == "__main__":
#     from systems import SomeChaoticSystem
#     system = SomeChaoticSystem()
#     mc_runner = MonteCarloRunner(system, num_samples=1000, variance_reduction=True)
#     results = mc_runner.run_simulation()
#     mean, variance = mc_runner.compute_statistics()
#     print("Mean:", mean)
#     print("Variance:", variance)