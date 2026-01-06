import numpy as np
import random
from typing import List, Tuple, Optional

class Individual:
    """
    Individual that can be regular person or influencer
    
    Attributes:
        opinion: Float in [0, 1] representing opinion
        is_influencer: Boolean, True if this individual is an influencer
        weight: Influence strength (1.0 for regular, 2-8 for influencers)
        influenced_by: Tracks which influencer "claimed" this person (-1 if influencer, 
                       0 if unclaimed, >0 if claimed by specific influencer)
    """
    def __init__(self, is_influencer: bool = False):
        self.opinion = random.random()
        self.is_influencer = is_influencer
        
        # Influencers have higher weight (stronger influence)
        self.weight = 1.0 if not is_influencer else float(np.random.randint(2, 8))
        
        # Track which influencer has "claimed" this person
        # -1: This IS an influencer (can't be influenced)
        # 0: Not yet influenced by anyone
        # >0: Influenced by specific influencer (their weight)
        self.influenced_by = -1 if is_influencer else 0
        
    def interact(self, others: List['Individual'], allow_competition: bool = False):
        """
        Influencer interacts with group, pulling them toward influencer's opinion
        
        Args:
            others: List of individuals to influence
            allow_competition: If False, only influence unclaimed people
                             If True, influencer can always influence others
        """
        if not self.is_influencer:
            return  # Only influencers initiate interactions
        
        for other in others:
            # Skip if trying to influence another influencer
            if other.is_influencer:
                continue
            
            if allow_competition:
                # Version 1: Influencer can always influence anyone
                other.opinion = (self.weight * self.opinion + other.weight * other.opinion) / \
                               (self.weight + other.weight)
            else:
                # Version 2: "Claim" mechanism - first influencer to reach someone owns them
                
                # If person not yet claimed, claim them
                if other.influenced_by == 0:
                    other.influenced_by = self.weight
                
                # Only influence if this influencer "owns" them
                if other.influenced_by == self.weight:
                    other.opinion = (self.weight * self.opinion + other.weight * other.opinion) / \
                                   (self.weight + other.weight)


def influencer_model(
    population_size: int,
    num_influencers: int,
    num_iterations: int,
    influencer_opinions: Optional[List[float]] = None,
    influencer_weights: Optional[List[float]] = None,
    allow_competition: bool = False,
    group_size_range: Tuple[int, int] = (2, 5)
) -> Tuple[np.ndarray, List[Individual]]:
    """
    Influencer model: Strategic agents with high influence shape population opinions
    
    Mechanism:
    - Influencers have higher weight (2-8x) than regular people (1x)
    - Each iteration, influencers select random groups and interact
    - Opinions update via weighted averaging (stronger weight = more influence)
    - Optional "claiming" mechanism: first influencer to reach someone owns them
    
    This models: social media influencers, opinion leaders, marketing, propaganda
    
    Args:
        population_size: Total population size
        num_influencers: Number of influencers (typically 1-5)
        num_iterations: Simulation iterations
        influencer_opinions: Fixed opinions for influencers (optional)
                           If None, random opinions
        influencer_weights: Fixed weights for influencers (optional)
                          If None, random in [2, 8]
        allow_competition: If False, first influencer to reach someone "owns" them
                         If True, all influencers can always influence anyone
        group_size_range: (min, max) size of groups influencers interact with
        
    Returns:
        opinion_history: Array of shape (num_iterations, population_size)
        population: Final population (to inspect influencer states)
    """
    # Create population
    # First num_influencers are influencers, rest are regular people
    population = [Individual(is_influencer=(i < num_influencers)) 
                 for i in range(population_size)]
    
    # Set influencer opinions if specified
    if influencer_opinions is not None:
        for i, opinion in enumerate(influencer_opinions[:num_influencers]):
            population[i].opinion = opinion
    
    # Set influencer weights if specified
    if influencer_weights is not None:
        for i, weight in enumerate(influencer_weights[:num_influencers]):
            population[i].weight = weight
    
    # Store opinion history
    opinion_history = np.zeros((num_iterations, population_size))
    
    # Simulation loop
    for iteration in range(num_iterations):
        # Store current state
        for i, individual in enumerate(population):
            opinion_history[iteration, i] = individual.opinion
        
        # Each influencer acts
        for individual in population:
            if individual.is_influencer:
                # Influencer selects random group to interact with
                group_size = random.randint(*group_size_range)
                
                # Select group (excluding self)
                available = [p for p in population if p != individual]
                group = random.sample(available, min(group_size, len(available)))
                
                # Interact with group
                individual.interact(group, allow_competition=allow_competition)
    
    return opinion_history, population


def influencer_model_v1(
    population_size: int,
    num_influencers: int,
    influencer_opinion: float,
    num_iterations: int
) -> np.ndarray:
    """
    Simple influencer model (Version 1)
    
    Single influencer with fixed opinion pulls population toward their view.
    
    Args:
        population_size: Population size
        num_influencers: Number of influencers (typically 1)
        influencer_opinion: Fixed opinion for influencer
        num_iterations: Simulation iterations
        
    Returns:
        opinion_history: Array of shape (num_iterations, population_size)
    """
    history, _ = influencer_model(
        population_size=population_size,
        num_influencers=num_influencers,
        num_iterations=num_iterations,
        influencer_opinions=[influencer_opinion],
        allow_competition=True  # Version 1: no claiming
    )
    return history


def influencer_model_v2(
    population_size: int,
    num_influencers: int,
    influencer_opinions: List[float],
    influencer_weights: List[float],
    num_iterations: int
) -> Tuple[np.ndarray, List[Individual]]:
    """
    Competing influencers model (Version 2)
    
    Multiple influencers compete for followers. First influencer to reach 
    someone "claims" them and continues influencing only them.
    
    Models: Brand loyalty, first-mover advantage, competing ideologies
    
    Args:
        population_size: Population size
        num_influencers: Number of competing influencers
        influencer_opinions: Fixed opinion for each influencer
        influencer_weights: Influence strength for each influencer
        num_iterations: Simulation iterations
        
    Returns:
        opinion_history: Array of shape (num_iterations, population_size)
        population: Final population (to see who influenced whom)
    """
    return influencer_model(
        population_size=population_size,
        num_influencers=num_influencers,
        num_iterations=num_iterations,
        influencer_opinions=influencer_opinions,
        influencer_weights=influencer_weights,
        allow_competition=False  # Version 2: claiming mechanism
    )