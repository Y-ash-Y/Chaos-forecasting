def sobol_sample(dim, n_samples):
    """Generate Sobol samples."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=dim, scramble=True)
    return sampler.random(n_samples)

def stratified_sample(data, n_samples):
    """Perform stratified sampling on the given data."""
    import numpy as np
    n_strata = len(data)
    samples = []
    for stratum in data:
        stratum_samples = np.random.choice(stratum, size=n_samples // n_strata, replace=False)
        samples.extend(stratum_samples)
    return samples

def set_seed(seed):
    """Set the random seed for reproducibility."""
    import numpy as np
    import random
    import torch

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)