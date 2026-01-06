import numpy as np
import random
from typing import Tuple

class Individual:
    """
    Individual with opinion and conviction level
    
    Attributes:
        opinion: Float in [0, 1] representing individual's opinion
        conviction: Float in [0, 1] representing strength of belief
    """
    def __init__(self):
        self.opinion = random.random()
        self.conviction = 0
        
    def init_conviction(self):
        """
        Initialize conviction based on opinion extremity
        
        Logic:
        - Moderate opinions (0.3-0.7): Low conviction (0-0.4)
        - Extreme opinions (>0.7 or <0.3): High conviction
        """
        if 0.3 <= self.opinion <= 0.7:
            # Moderate opinions have low, random conviction
            self.conviction = random.uniform(0, 0.4)
        else:
            # Extreme opinions have high conviction
            if self.opinion > 0.7:
                self.conviction = self.opinion
            else:  # self.opinion < 0.3
                self.conviction = 1 - self.opinion


def convinced_model(
    population_size: int,
    epsilon: float, 
    bornesup: float,  # param1 - conviction threshold
    distmax: float,   # param2 - distance threshold for conviction increase
    num_iterations: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deffuant model with conviction parameter
    
    Individuals with low conviction update opinions more easily.
    When opinions are very close, conviction increases slightly.
    
    Args:
        population_size: Number of individuals
        epsilon: Opinion distance threshold for interaction
        bornesup: Conviction upper bound (if conviction >= bornesup, opinion doesn't change)
        distmax: Distance threshold for conviction increase
        num_iterations: Number of simulation iterations
        
    Returns:
        opinion_history: Array of shape (num_iterations, population_size)
        conviction_history: Array of shape (num_iterations, population_size)
    """
    # Create initial population
    population = [Individual() for _ in range(population_size)]
    
    # Initialize convictions based on initial opinions
    for individual in population:
        individual.init_conviction()
    
    # Store opinion and conviction history
    opinion_history = np.zeros((num_iterations, population_size))
    conviction_history = np.zeros((num_iterations, population_size))
    
    # Store initial state
    opinion_history[0] = [x.opinion for x in population]
    conviction_history[0] = [x.conviction for x in population]
    
    # Simulation loop
    for t in range(num_iterations):
        # Store current state
        opinion_history[t] = [x.opinion for x in population]
        conviction_history[t] = [x.conviction for x in population]
        
        # Each iteration: population_size // 2 pairwise interactions
        for _ in range(population_size // 2):
            # Select two individuals randomly
            i1, i2 = random.sample(range(population_size), 2)
            
            opinion1 = population[i1].opinion
            opinion2 = population[i2].opinion
            conviction1 = population[i1].conviction
            conviction2 = population[i2].conviction
            
            # Check if opinions are close enough for interaction
            if abs(opinion1 - opinion2) < epsilon:
                
                # Individual 1 updates if conviction is below threshold
                if conviction1 < bornesup:
                    population[i1].opinion = (opinion1 + opinion2) / 2
                    
                    # Conviction increases if opinions are very close
                    if abs(opinion1 - opinion2) < distmax and population[i1].conviction + 0.1 < 1:
                        population[i1].conviction += 0.05
                
                # Individual 2 updates if conviction is below threshold
                if conviction2 < bornesup:
                    population[i2].opinion = (opinion1 + opinion2) / 2
                    
                    # Conviction increases if opinions are very close
                    if abs(opinion1 - opinion2) < distmax and population[i2].conviction + 0.1 < 1:
                        population[i2].conviction += 0.05
    
    return opinion_history, conviction_history

