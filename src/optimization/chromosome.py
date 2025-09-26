"""
Chromosome representation for hyperparameter optimization.
Handles encoding/decoding of both continuous and categorical parameters.
"""

import random
import copy
from typing import Dict, Any, List, Union, Tuple
import numpy as np


class HyperparameterSpace:
    """Defines the search space for hyperparameters."""
    
    def __init__(self):
        self.parameters = {}
    
    def add_continuous(self, name: str, min_val: float, max_val: float, log_scale: bool = False):
        """Add a continuous parameter to the search space."""
        self.parameters[name] = {
            'type': 'continuous',
            'min': min_val,
            'max': max_val,
            'log_scale': log_scale
        }
    
    def add_categorical(self, name: str, choices: List[Any]):
        """Add a categorical parameter to the search space."""
        self.parameters[name] = {
            'type': 'categorical',
            'choices': choices
        }
    
    def add_integer(self, name: str, min_val: int, max_val: int):
        """Add an integer parameter to the search space."""
        self.parameters[name] = {
            'type': 'integer',
            'min': min_val,
            'max': max_val
        }
    
    @classmethod
    def create_default_space(cls):
        """Create the default hyperparameter space for image inpainting model."""
        space = cls()
        
        # Learning rate (log scale)
        space.add_continuous('learning_rate', 1e-6, 1e-2, log_scale=True)
        
        # Batch size (categorical for memory constraints)
        space.add_categorical('batch_size', [1, 4, 8, 16, 32])
        
        # Image size (categorical)
        space.add_categorical('image_size', [256, 384, 512])
        
        # Loss weights (continuous)
        space.add_continuous('diffusion_weight', 0.1, 3.0)
        space.add_continuous('perceptual_weight', 0.0, 1.0)
        space.add_continuous('mask_weight', 1.0, 10.0)
        
        # Optimizer parameters
        space.add_continuous('weight_decay', 1e-6, 1e-1, log_scale=True)
        space.add_continuous('beta1', 0.8, 0.999)
        space.add_continuous('beta2', 0.9, 0.9999)
        
        # Training epochs (adaptive early stopping)
        space.add_integer('max_epochs', 5, 50)
        
        # Memory optimization settings
        space.add_categorical('gradient_checkpointing', [True, False])
        space.add_categorical('attention_slice_size', [1, 2, 4, 8])
        
        return space


class Chromosome:
    """Represents a solution (hyperparameter configuration) in the genetic algorithm."""
    
    def __init__(self, hyperparameter_space: HyperparameterSpace, genes: Dict[str, Any] = None):
        self.space = hyperparameter_space
        self.genes = genes if genes is not None else self._initialize_random()
        self.fitness_values = None  # Will store (speed_score, accuracy_score, stability_score)
        self.objectives = None      # Raw objective values
        self.rank = None           # NSGA-II rank
        self.crowding_distance = 0.0  # NSGA-II crowding distance
        self.dominated_count = 0   # Number of solutions that dominate this one
        self.dominating_set = []   # Solutions dominated by this one
    
    def _initialize_random(self) -> Dict[str, Any]:
        """Initialize chromosome with random values within the parameter space."""
        genes = {}
        
        for param_name, param_config in self.space.parameters.items():
            if param_config['type'] == 'continuous':
                if param_config.get('log_scale', False):
                    # Log scale sampling
                    log_min = np.log10(param_config['min'])
                    log_max = np.log10(param_config['max'])
                    log_val = random.uniform(log_min, log_max)
                    genes[param_name] = 10 ** log_val
                else:
                    genes[param_name] = random.uniform(param_config['min'], param_config['max'])
            
            elif param_config['type'] == 'categorical':
                genes[param_name] = random.choice(param_config['choices'])
            
            elif param_config['type'] == 'integer':
                genes[param_name] = random.randint(param_config['min'], param_config['max'])
        
        return genes
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Convert chromosome to configuration dictionary for training."""
        return copy.deepcopy(self.genes)
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Mutate the chromosome with given probability and strength."""
        for param_name, param_config in self.space.parameters.items():
            if random.random() < mutation_rate:
                if param_config['type'] == 'continuous':
                    current_val = self.genes[param_name]
                    
                    if param_config.get('log_scale', False):
                        # Log scale mutation
                        log_val = np.log10(current_val)
                        log_range = np.log10(param_config['max']) - np.log10(param_config['min'])
                        log_val += random.gauss(0, mutation_strength * log_range)
                        log_val = np.clip(log_val, np.log10(param_config['min']), np.log10(param_config['max']))
                        self.genes[param_name] = 10 ** log_val
                    else:
                        # Linear scale mutation
                        val_range = param_config['max'] - param_config['min']
                        current_val += random.gauss(0, mutation_strength * val_range)
                        self.genes[param_name] = np.clip(current_val, param_config['min'], param_config['max'])
                
                elif param_config['type'] == 'categorical':
                    self.genes[param_name] = random.choice(param_config['choices'])
                
                elif param_config['type'] == 'integer':
                    current_val = self.genes[param_name]
                    val_range = param_config['max'] - param_config['min']
                    delta = int(random.gauss(0, mutation_strength * val_range))
                    new_val = current_val + delta
                    self.genes[param_name] = np.clip(new_val, param_config['min'], param_config['max'])
    
    def crossover(self, other: 'Chromosome', crossover_rate: float = 0.7) -> Tuple['Chromosome', 'Chromosome']:
        """Perform crossover with another chromosome."""
        child1_genes = {}
        child2_genes = {}
        
        for param_name in self.genes.keys():
            if random.random() < crossover_rate:
                # Swap genes
                child1_genes[param_name] = other.genes[param_name]
                child2_genes[param_name] = self.genes[param_name]
            else:
                # Keep original genes
                child1_genes[param_name] = self.genes[param_name]
                child2_genes[param_name] = other.genes[param_name]
        
        child1 = Chromosome(self.space, child1_genes)
        child2 = Chromosome(self.space, child2_genes)
        
        return child1, child2
    
    def dominates(self, other: 'Chromosome') -> bool:
        """Check if this chromosome dominates another in multi-objective sense."""
        if self.fitness_values is None or other.fitness_values is None:
            return False
        
        # For minimization problems (we want to minimize negative values)
        # Higher fitness values are better, so we check if all our values are >= and at least one is >
        at_least_one_better = False
        for i in range(len(self.fitness_values)):
            if self.fitness_values[i] < other.fitness_values[i]:
                return False  # Other is better in at least one objective
            elif self.fitness_values[i] > other.fitness_values[i]:
                at_least_one_better = True
        
        return at_least_one_better
    
    def __str__(self) -> str:
        """String representation of the chromosome."""
        fitness_str = f"fitness={self.fitness_values}" if self.fitness_values else "fitness=None"
        rank_str = f"rank={self.rank}" if self.rank is not None else "rank=None"
        return f"Chromosome({fitness_str}, {rank_str}, genes={self.genes})"
    
    def __repr__(self) -> str:
        return self.__str__()