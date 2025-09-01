def compute_ftle(trajectory, time_step):
    """
    Compute the Finite-Time Lyapunov Exponent (FTLE) for a given trajectory.

    Parameters:
    trajectory (np.ndarray): The trajectory of the system as a 2D array (time x dimensions).
    time_step (float): The time step between points in the trajectory.

    Returns:
    np.ndarray: The FTLE values for the trajectory.
    """
    # Calculate the Jacobian matrix for the trajectory
    # (This is a placeholder for the actual Jacobian calculation)
    jacobian = np.gradient(trajectory, axis=0) / time_step

    # Compute the FTLE
    ftle = np.log(np.linalg.norm(jacobian, axis=1)) / time_step

    return ftle


def lyapunov_exponent(trajectory, time_step):
    """
    Calculate the Lyapunov exponent from a trajectory.

    Parameters:
    trajectory (np.ndarray): The trajectory of the system as a 2D array (time x dimensions).
    time_step (float): The time step between points in the trajectory.

    Returns:
    float: The Lyapunov exponent.
    """
    ftle_values = compute_ftle(trajectory, time_step)
    return np.mean(ftle_values)


def main():
    # Example usage of the functions
    import numpy as np

    # Generate a sample trajectory (placeholder)
    time = np.linspace(0, 10, 100)
    trajectory = np.sin(time)  # Replace with actual trajectory data

    time_step = time[1] - time[0]
    le = lyapunov_exponent(trajectory, time_step)
    print(f"Lyapunov Exponent: {le}")


if __name__ == "__main__":
    main()