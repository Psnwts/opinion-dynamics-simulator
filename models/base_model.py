import numpy as np
import random


def deffuant_simulation(pop_size: int , epsilon : float, iterations : int) -> np.ndarray:
    """
    Deffuant model: Basic opinion dynamics through pairwise interactions
    
    Individuals update opinions when difference is below epsilon threshold.
    Both individuals move to the average of their opinions.
    
    Args:
        pop_size: Number of individuals in population
        epsilon: Threshold for opinion difference (0 to 1)
        iterations: Number of simulation iterations
        
    Returns:
        history: Array of shape (iterations, pop_size) with opinion trajectories
    """
    
    # Initialize opinions uniformly in [0,1]
    opinions = np.random.rand(pop_size)

    # Store opinion history
    history = np.zeros((iterations, pop_size))

    # Simulation loop
    for t in range(iterations):
        history[t] = opinions.copy()  

        # Each iteration: pop_size // 2 pairwise interactions 
        for _ in range(pop_size//2):
            # Select two individuals from the population randomly
            i1, i2 = random.sample(range(pop_size), 2)
            
            # Check if opinions are close enough (distance < epsilon)
            if abs(opinions[i1] - opinions[i2]) < epsilon:
                opinions[i1] = opinions[i2] = (opinions[i1] + opinions[i2]) / 2
    return history