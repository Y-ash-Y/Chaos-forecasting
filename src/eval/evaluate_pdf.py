import os
import yaml
import numpy as np
from src.common.metrics import calculate_metrics  # Assuming this function exists
from src.common.utils import load_data  # Assuming this function exists

def evaluate_pdf(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    results = {}
    
    for system in config['systems']:
        results[system] = {}
        for method in config['methods']:
            data_path = os.path.join(config['data_dir'], system, f"{method}.npy")
            data = load_data(data_path)  # Load the data for the specific system and method
            
            metrics = calculate_metrics(data)  # Calculate metrics for the loaded data
            results[system][method] = metrics

    return results

if __name__ == "__main__":
    config_path = 'experiments/configs/example.yaml'  # Example path to the config file
    results = evaluate_pdf(config_path)
    print(results)  # Output the results for inspection