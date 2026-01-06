import numpy as np
import random
from typing import Tuple

class Individual:
    """
    Individual with opinion
    
    Attributes:
        opinion: Float in [0, 1] representing individual's opinion
    """
    def __init__(self):
        self.opinion = random.random()


def polarized_model(
    population_size: int,
    weight: float,
    center: float,
    num_iterations: int
) -> np.ndarray:
    """
    Polarized model: Extremists pull moderates toward poles
    
    Key mechanism:
    - Individuals only interact with others on the SAME side of center
    - The MORE EXTREME individual influences the less extreme one
    - Moderates are pulled toward extremes, not toward center
    
    This models radicalization and polarization dynamics.
    
    Args:
        population_size: Number of individuals
        weight: Influence strength of extreme individual (higher = stronger pull)
        center: Opinion center point (typically 0.5)
        num_iterations: Number of simulation iterations
        
    Returns:
        opinion_history: Array of shape (num_iterations, population_size)
    """
    # Create initial population
    population = [Individual() for _ in range(population_size)]
    
    # Store opinion history
    opinion_history = np.zeros((num_iterations, population_size))
    
    # Simulation loop
    for t in range(num_iterations):
        # Store current state
        opinion_history[t] = [x.opinion for x in population]
        
        # Each iteration: population_size // 2 pairwise interactions
        for _ in range(population_size // 2):
            # Select two individuals randomly
            i1, i2 = random.sample(range(population_size), 2)
            
            opinion1 = population[i1].opinion
            opinion2 = population[i2].opinion
            
            # Only interact if BOTH on same side of center (both left OR both right)
            same_side = (opinion1 < center and opinion2 < center) or \
                       (opinion1 > center and opinion2 > center)
            
            if same_side:
                # Calculate distance from center (extremity measure)
                dist1 = abs(opinion1 - center)
                dist2 = abs(opinion2 - center)
                
                # More extreme individual influences less extreme one
                if dist1 > dist2:
                    # Individual 1 is more extreme, pulls individual 2 away from center
                    population[i2].opinion = ((weight * opinion1) + opinion2) / (weight + 1)
                    
                elif dist2 > dist1:
                    # Individual 2 is more extreme, pulls individual 1 away from center
                    population[i1].opinion = ((weight * opinion2) + opinion1) / (weight + 1)
                
                # If dist1 == dist2, no update (both equally extreme)
    
    return opinion_history