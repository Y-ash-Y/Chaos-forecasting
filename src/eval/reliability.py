def calculate_reliability(predictions, observations):
    """
    Calculate the reliability of probabilistic forecasts.

    Parameters:
    predictions (array-like): Predicted probabilities.
    observations (array-like): Observed outcomes (0 or 1).

    Returns:
    reliability (float): The reliability score.
    """
    # Ensure predictions and observations are numpy arrays
    predictions = np.array(predictions)
    observations = np.array(observations)

    # Calculate reliability
    reliability = np.mean(predictions[observations == 1]) - np.mean(predictions[observations == 0])
    
    return reliability


def reliability_diagram(predictions, observations, n_bins=10):
    """
    Create a reliability diagram for visualizing the reliability of forecasts.

    Parameters:
    predictions (array-like): Predicted probabilities.
    observations (array-like): Observed outcomes (0 or 1).
    n_bins (int): Number of bins for the histogram.

    Returns:
    bin_centers (array): Centers of the bins.
    reliability (array): Reliability values for each bin.
    """
    # Ensure predictions and observations are numpy arrays
    predictions = np.array(predictions)
    observations = np.array(observations)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    reliability = np.zeros(n_bins)

    for i in range(n_bins):
        bin_mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if np.sum(bin_mask) > 0:
            reliability[i] = np.mean(observations[bin_mask])

    return bin_centers, reliability


def main():
    # Example usage
    predictions = [0.1, 0.4, 0.35, 0.8, 0.9]
    observations = [0, 0, 1, 1, 1]

    reliability_score = calculate_reliability(predictions, observations)
    print(f'Reliability Score: {reliability_score}')

    bin_centers, reliability_values = reliability_diagram(predictions, observations)
    print(f'Bin Centers: {bin_centers}')
    print(f'Reliability Values: {reliability_values}')


if __name__ == "__main__":
    main()