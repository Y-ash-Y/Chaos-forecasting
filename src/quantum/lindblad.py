from scipy.linalg import expm
import numpy as np

class LindbladMasterEquation:
    def __init__(self, hamiltonian, jump_operators):
        self.hamiltonian = hamiltonian
        self.jump_operators = jump_operators

    def evolve(self, rho, dt):
        """
        Evolve the density matrix rho for a time step dt using the Lindblad master equation.
        """
        # Hamiltonian evolution
        H_term = -1j * (self.hamiltonian @ rho - rho @ self.hamiltonian)
        
        # Lindblad terms
        L_terms = sum(
            (jump @ rho @ jump.conj().T - 0.5 * (jump.conj().T @ jump @ rho + rho @ jump.conj().T @ jump))
            for jump in self.jump_operators)
        )
        
        # Total evolution
        total_evolution = H_term + L_terms
        rho_new = rho + total_evolution * dt
        
        # Normalize the density matrix
        return self.normalize_density_matrix(rho_new)

    def normalize_density_matrix(self, rho):
        """
        Normalize the density matrix to ensure it remains a valid quantum state.
        """
        trace = np.trace(rho)
        if trace > 0:
            return rho / trace
        else:
            raise ValueError("Density matrix has zero trace, cannot normalize.")

# Example usage
if __name__ == "__main__":
    # Define a simple Hamiltonian and jump operators for testing
    H = np.array([[1, 0], [0, -1]])  # Example Hamiltonian
    jump_ops = [np.array([[0, 1], [0, 0]])]  # Example jump operator

    # Initialize the Lindblad master equation
    lindblad = LindbladMasterEquation(H, jump_ops)

    # Initial density matrix (pure state)
    rho_initial = np.array([[1, 0], [0, 0]])

    # Evolve the state
    dt = 0.01
    rho_evolved = lindblad.evolve(rho_initial, dt)
    print("Evolved density matrix:\n", rho_evolved)