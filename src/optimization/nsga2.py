"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation for multi-objective optimization.
"""

import numpy as np
from typing import List, Set, Tuple
import logging


class NSGA2:
    """NSGA-II algorithm for multi-objective optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def non_dominated_sort(self, population: List) -> List[List[int]]:
        """
        Perform non-dominated sorting on the population.
        
        Args:
            population: List of chromosomes with fitness_values
            
        Returns:
            List of fronts, where each front is a list of chromosome indices
        """
        n = len(population)
        fronts = []
        
        # Initialize domination data for each chromosome
        for i, chromosome in enumerate(population):
            chromosome.dominated_count = 0
            chromosome.dominating_set = []
        
        # Calculate domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                if population[i].dominates(population[j]):
                    population[i].dominating_set.append(j)
                    population[j].dominated_count += 1
                elif population[j].dominates(population[i]):
                    population[j].dominating_set.append(i)
                    population[i].dominated_count += 1
        
        # Find first front (non-dominated solutions)
        first_front = []
        for i, chromosome in enumerate(population):
            if chromosome.dominated_count == 0:
                chromosome.rank = 1
                first_front.append(i)
        
        fronts.append(first_front)
        
        # Find subsequent fronts
        current_front = first_front
        front_rank = 1
        
        while current_front:
            next_front = []
            
            for i in current_front:
                for j in population[i].dominating_set:
                    population[j].dominated_count -= 1
                    if population[j].dominated_count == 0:
                        population[j].rank = front_rank + 1
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
                current_front = next_front
                front_rank += 1
            else:
                break
        
        return fronts
    
    def calculate_crowding_distance(self, population: List, front_indices: List[int]):
        """
        Calculate crowding distance for chromosomes in a front.
        
        Args:
            population: List of chromosomes
            front_indices: Indices of chromosomes in the current front
        """
        if len(front_indices) <= 2:
            # If front has 2 or fewer solutions, set maximum crowding distance
            for idx in front_indices:
                population[idx].crowding_distance = float('inf')
            return
        
        # Initialize crowding distances
        for idx in front_indices:
            population[idx].crowding_distance = 0.0
        
        # Check if all chromosomes in front have fitness values
        valid_fitness = [population[idx].fitness_values for idx in front_indices if population[idx].fitness_values is not None]
        if not valid_fitness:
            # If no valid fitness values, set all to zero
            return
        
        # Get number of objectives from first valid fitness
        num_objectives = len(valid_fitness[0])
        
        # Calculate crowding distance for each objective
        for obj_idx in range(num_objectives):
            # Filter front to include only chromosomes with valid fitness values
            valid_front_indices = [idx for idx in front_indices if population[idx].fitness_values is not None]
            
            if len(valid_front_indices) <= 2:
                for idx in valid_front_indices:
                    population[idx].crowding_distance = float('inf')
                continue
            
            # Sort front by current objective
            valid_front_indices.sort(key=lambda x: population[x].fitness_values[obj_idx])
            
            # Set boundary points to infinite distance
            population[valid_front_indices[0]].crowding_distance = float('inf')
            population[valid_front_indices[-1]].crowding_distance = float('inf')
            
            # Calculate objective range
            obj_values = [population[idx].fitness_values[obj_idx] for idx in valid_front_indices]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range == 0:
                continue  # All solutions have same objective value
            
            # Calculate crowding distance for intermediate points
            for i in range(1, len(valid_front_indices) - 1):
                if population[valid_front_indices[i]].crowding_distance != float('inf'):
                    distance = (population[valid_front_indices[i + 1]].fitness_values[obj_idx] - 
                              population[valid_front_indices[i - 1]].fitness_values[obj_idx]) / obj_range
                    population[valid_front_indices[i]].crowding_distance += distance
    
    def environmental_selection(self, population: List, population_size: int) -> List:
        """
        Select the best individuals for the next generation using NSGA-II selection.
        
        Args:
            population: Combined parent and offspring population
            population_size: Desired population size
            
        Returns:
            Selected population
        """
        if len(population) <= population_size:
            return population
        
        # Perform non-dominated sorting
        fronts = self.non_dominated_sort(population)
        
        # Select individuals front by front
        selected = []
        
        for front_indices in fronts:
            if len(selected) + len(front_indices) <= population_size:
                # Add entire front
                selected.extend([population[i] for i in front_indices])
            else:
                # Calculate crowding distance for the current front
                self.calculate_crowding_distance(population, front_indices)
                
                # Sort by crowding distance (descending)
                front_indices.sort(key=lambda x: population[x].crowding_distance, reverse=True)
                
                # Add individuals until population size is reached
                remaining_slots = population_size - len(selected)
                selected.extend([population[i] for i in front_indices[:remaining_slots]])
                break
        
        self.logger.info(f"Environmental selection: {len(population)} -> {len(selected)}")
        return selected
    
    def tournament_selection(self, population: List, tournament_size: int = 2):
        """
        Perform tournament selection based on NSGA-II criteria.
        
        Args:
            population: Population to select from
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected chromosome
        """
        import random
        
        # Randomly select tournament participants
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Find the best individual based on rank and crowding distance
        best = tournament[0]
        
        for individual in tournament[1:]:
            if self._is_better(individual, best):
                best = individual
        
        return best
    
    def _is_better(self, ind1, ind2) -> bool:
        """
        Check if individual 1 is better than individual 2 based on NSGA-II criteria.
        
        Args:
            ind1, ind2: Individuals to compare
            
        Returns:
            True if ind1 is better than ind2
        """
        # First criterion: rank (lower is better)
        if ind1.rank is not None and ind2.rank is not None:
            if ind1.rank < ind2.rank:
                return True
            elif ind1.rank > ind2.rank:
                return False
        
        # Second criterion: crowding distance (higher is better)
        if hasattr(ind1, 'crowding_distance') and hasattr(ind2, 'crowding_distance'):
            return ind1.crowding_distance > ind2.crowding_distance
        
        return False
    
    def get_pareto_front(self, population: List) -> List:
        """
        Get the Pareto front (first non-dominated front) from the population.
        
        Args:
            population: List of chromosomes
            
        Returns:
            List of chromosomes in the Pareto front
        """
        fronts = self.non_dominated_sort(population)
        
        if fronts:
            return [population[i] for i in fronts[0]]
        else:
            return []
    
    def calculate_hypervolume(self, pareto_front: List, reference_point: Tuple[float, ...]) -> float:
        """
        Calculate hypervolume indicator for the Pareto front.
        
        Args:
            pareto_front: List of chromosomes in Pareto front
            reference_point: Reference point for hypervolume calculation
            
        Returns:
            Hypervolume value
        """
        if not pareto_front:
            return 0.0
        
        # Simple 3D hypervolume calculation (for 3 objectives)
        # This is a simplified implementation - for production use, consider more sophisticated algorithms
        
        points = []
        for chromosome in pareto_front:
            if chromosome.fitness_values is not None:
                points.append(chromosome.fitness_values)
        
        if not points:
            return 0.0
        
        # Sort points by first objective
        points.sort(key=lambda x: x[0], reverse=True)
        
        hypervolume = 0.0
        
        for i, point in enumerate(points):
            # Calculate contribution of this point
            if all(point[j] >= reference_point[j] for j in range(len(point))):
                # Point dominates reference point
                volume = 1.0
                for j in range(len(point)):
                    if i == 0:
                        volume *= (point[j] - reference_point[j])
                    else:
                        volume *= max(0, point[j] - max(points[k][j] for k in range(i)))
                
                hypervolume += volume
        
        return hypervolume
    
    def get_diversity_metric(self, population: List) -> float:
        """
        Calculate diversity metric for the population.
        
        Args:
            population: List of chromosomes
            
        Returns:
            Diversity value (higher is better)
        """
        if len(population) < 2:
            return 0.0
        
        # Calculate average crowding distance
        crowding_distances = []
        for chromosome in population:
            if hasattr(chromosome, 'crowding_distance') and chromosome.crowding_distance != float('inf'):
                crowding_distances.append(chromosome.crowding_distance)
        
        if not crowding_distances:
            return 0.0
        
        return np.mean(crowding_distances)
    
    def print_population_stats(self, population: List, generation: int):
        """Print statistics about the current population."""
        fronts = self.non_dominated_sort(population)
        pareto_front = [population[i] for i in fronts[0]] if fronts else []
        
        self.logger.info(f"Generation {generation}:")
        self.logger.info(f"  Total population: {len(population)}")
        self.logger.info(f"  Number of fronts: {len(fronts)}")
        self.logger.info(f"  Pareto front size: {len(pareto_front)}")
        
        if pareto_front:
            # Calculate objective statistics for Pareto front
            objectives = np.array([chr.fitness_values for chr in pareto_front if chr.fitness_values is not None])
            if len(objectives) > 0:
                self.logger.info(f"  Pareto front objectives (mean ± std):")
                for i, obj_name in enumerate(['Speed', 'Accuracy', 'Stability']):
                    mean_val = np.mean(objectives[:, i])
                    std_val = np.std(objectives[:, i])
                    self.logger.info(f"    {obj_name}: {mean_val:.3f} ± {std_val:.3f}")