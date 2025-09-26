"""
Multi-Objective Genetic Algorithm (MOGA) implementation for hyperparameter optimization.
Uses NSGA-II for multi-objective optimization.
"""

import random
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np

from .chromosome import Chromosome, HyperparameterSpace
from .fitness import FitnessEvaluator, MultiObjectiveFitness
from .nsga2 import NSGA2


class MOGA:
    """Multi-Objective Genetic Algorithm for hyperparameter optimization."""
    
    def __init__(self,
                 hyperparameter_space: HyperparameterSpace,
                 fitness_evaluator: FitnessEvaluator,
                 population_size: int = 50,
                 num_generations: int = 20,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.1,
                 tournament_size: int = 2,
                 elite_size: int = 5,
                 results_dir: str = "outputs/moga_results",
                 random_seed: Optional[int] = None):
        """
        Initialize MOGA.
        
        Args:
            hyperparameter_space: Defines the search space
            fitness_evaluator: Evaluates fitness of configurations
            population_size: Size of the population
            num_generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            mutation_strength: Strength of mutations
            tournament_size: Size of tournament selection
            elite_size: Number of elite individuals to preserve
            results_dir: Directory to save results
            random_seed: Random seed for reproducibility
        """
        self.hyperparameter_space = hyperparameter_space
        self.fitness_evaluator = fitness_evaluator
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        
        # Setup results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize NSGA-II
        self.nsga2 = NSGA2()
        
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Evolution tracking
        self.population = []
        self.generation_stats = []
        self.best_solutions = []
        self.pareto_history = []
        
        self.logger.info(f"MOGA initialized with population_size={population_size}, "
                        f"num_generations={num_generations}")
    
    def initialize_population(self) -> List[Chromosome]:
        """Initialize the population with random chromosomes."""
        self.logger.info("Initializing population...")
        
        population = []
        for i in range(self.population_size):
            chromosome = Chromosome(self.hyperparameter_space)
            population.append(chromosome)
        
        self.logger.info(f"Population initialized with {len(population)} individuals")
        return population
    
    def evaluate_population(self, 
                          population: List[Chromosome],
                          train_loader,
                          val_loader,
                          model_factory_fn: Callable,
                          generation: int = 0) -> List[Chromosome]:
        """Evaluate fitness for all chromosomes in the population."""
        self.logger.info(f"Evaluating population of {len(population)} individuals...")
        
        evaluated_population = []
        
        for i, chromosome in enumerate(population):
            # Skip evaluation if already evaluated (for efficiency)
            if chromosome.fitness_values is not None:
                evaluated_population.append(chromosome)
                continue
            
            evaluation_id = f"gen{generation}_ind{i}"
            
            try:
                fitness = self.fitness_evaluator.evaluate_chromosome(
                    chromosome=chromosome,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model_factory_fn=model_factory_fn,
                    evaluation_id=evaluation_id
                )
                
                # Store fitness in chromosome
                chromosome.fitness_values = fitness.to_tuple()
                chromosome.objectives = fitness
                
                evaluated_population.append(chromosome)
                
                self.logger.info(f"Individual {i+1}/{len(population)} evaluated: "
                               f"Speed={fitness.speed_score:.3f}, "
                               f"Accuracy={fitness.accuracy_score:.3f}, "
                               f"Stability={fitness.stability_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate individual {i}: {e}")
                # Assign poor fitness to failed evaluations
                chromosome.fitness_values = (0.1, 0.1, 0.1)
                chromosome.objectives = MultiObjectiveFitness(0.1, 0.1, 0.1)
                evaluated_population.append(chromosome)
        
        return evaluated_population
    
    def selection(self, population: List[Chromosome]) -> List[Chromosome]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        
        for _ in range(self.population_size):
            parent = self.nsga2.tournament_selection(population, self.tournament_size)
            parents.append(parent)
        
        return parents
    
    def crossover_and_mutation(self, parents: List[Chromosome]) -> List[Chromosome]:
        """Create offspring through crossover and mutation."""
        offspring = []
        
        # Create pairs for crossover
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            if random.random() < self.crossover_rate:
                # Perform crossover
                child1, child2 = parent1.crossover(parent2, self.crossover_rate)
            else:
                # No crossover, create copies
                child1 = Chromosome(self.hyperparameter_space, parent1.genes.copy())
                child2 = Chromosome(self.hyperparameter_space, parent2.genes.copy())
            
            # Apply mutation
            child1.mutate(self.mutation_rate, self.mutation_strength)
            child2.mutate(self.mutation_rate, self.mutation_strength)
            
            offspring.extend([child1, child2])
        
        # Trim to population size
        return offspring[:self.population_size]
    
    def evolve(self,
               train_loader,
               val_loader, 
               model_factory_fn: Callable,
               checkpoint_interval: int = 5) -> Dict[str, Any]:
        """
        Run the complete evolutionary algorithm.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_factory_fn: Function to create model instances
            checkpoint_interval: Save checkpoints every N generations
            
        Returns:
            Dictionary with evolution results
        """
        self.logger.info(f"Starting evolution for {self.num_generations} generations...")
        start_time = time.time()
        
        # Initialize population
        self.population = self.initialize_population()
        
        # Evolution loop
        for generation in range(self.num_generations):
            generation_start = time.time()
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Generation {generation + 1}/{self.num_generations}")
            self.logger.info(f"{'='*60}")
            
            # Evaluate population
            self.population = self.evaluate_population(
                self.population, train_loader, val_loader, model_factory_fn, generation
            )
            
            # NSGA-II non-dominated sorting and crowding distance
            fronts = self.nsga2.non_dominated_sort(self.population)
            for front_indices in fronts:
                self.nsga2.calculate_crowding_distance(self.population, front_indices)
            
            # Track Pareto front
            pareto_front = self.nsga2.get_pareto_front(self.population)
            self.pareto_history.append([chr.genes.copy() for chr in pareto_front])
            
            # Calculate generation statistics
            gen_stats = self._calculate_generation_stats(self.population, generation)
            self.generation_stats.append(gen_stats)
            
            # Print statistics
            self.nsga2.print_population_stats(self.population, generation)
            
            # Create next generation (except for last generation)
            if generation < self.num_generations - 1:
                # Selection
                parents = self.selection(self.population)
                
                # Crossover and mutation
                offspring = self.crossover_and_mutation(parents)
                
                # Combine parents and offspring
                combined_population = self.population + offspring
                
                # Environmental selection (NSGA-II)
                self.population = self.nsga2.environmental_selection(
                    combined_population, self.population_size
                )
            
            generation_time = time.time() - generation_start
            self.logger.info(f"Generation {generation + 1} completed in {generation_time:.1f}s")
            
            # Save checkpoint
            if (generation + 1) % checkpoint_interval == 0:
                self._save_checkpoint(generation + 1)
        
        total_time = time.time() - start_time
        
        # Final results
        final_pareto_front = self.nsga2.get_pareto_front(self.population)
        results = self._compile_results(final_pareto_front, total_time)
        
        # Save final results
        self._save_final_results(results)
        
        self.logger.info(f"\nEvolution completed in {total_time:.1f}s")
        self.logger.info(f"Final Pareto front size: {len(final_pareto_front)}")
        
        return results
    
    def _calculate_generation_stats(self, population: List[Chromosome], generation: int) -> Dict[str, Any]:
        """Calculate statistics for the current generation."""
        fitness_values = [chr.fitness_values for chr in population if chr.fitness_values is not None]
        
        if not fitness_values:
            return {'generation': generation, 'population_size': len(population)}
        
        fitness_array = np.array(fitness_values)
        
        stats = {
            'generation': generation,
            'population_size': len(population),
            'speed_score': {
                'mean': float(np.mean(fitness_array[:, 0])),
                'std': float(np.std(fitness_array[:, 0])),
                'max': float(np.max(fitness_array[:, 0])),
                'min': float(np.min(fitness_array[:, 0]))
            },
            'accuracy_score': {
                'mean': float(np.mean(fitness_array[:, 1])),
                'std': float(np.std(fitness_array[:, 1])),
                'max': float(np.max(fitness_array[:, 1])),
                'min': float(np.min(fitness_array[:, 1]))
            },
            'stability_score': {
                'mean': float(np.mean(fitness_array[:, 2])),
                'std': float(np.std(fitness_array[:, 2])),
                'max': float(np.max(fitness_array[:, 2])),
                'min': float(np.min(fitness_array[:, 2]))
            }
        }
        
        return stats
    
    def _compile_results(self, pareto_front: List[Chromosome], total_time: float) -> Dict[str, Any]:
        """Compile final results from evolution."""
        results = {
            'evolution_summary': {
                'total_time': total_time,
                'num_generations': self.num_generations,
                'population_size': self.population_size,
                'final_population_size': len(self.population),
                'pareto_front_size': len(pareto_front)
            },
            'pareto_front': [],
            'generation_stats': self.generation_stats,
            'hyperparameter_space': {
                param: config for param, config in self.hyperparameter_space.parameters.items()
            },
            'algorithm_config': {
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'mutation_strength': self.mutation_strength,
                'tournament_size': self.tournament_size,
                'elite_size': self.elite_size
            }
        }
        
        # Add Pareto front solutions
        for i, chromosome in enumerate(pareto_front):
            solution = {
                'rank': i + 1,
                'hyperparameters': chromosome.genes,
                'fitness': {
                    'speed_score': chromosome.fitness_values[0],
                    'accuracy_score': chromosome.fitness_values[1], 
                    'stability_score': chromosome.fitness_values[2]
                }
            }
            
            if hasattr(chromosome, 'objectives') and chromosome.objectives:
                solution['detailed_metrics'] = chromosome.objectives.to_dict()
            
            results['pareto_front'].append(solution)
        
        # Add evaluation summary
        results['evaluation_summary'] = self.fitness_evaluator.get_evaluation_summary()
        
        return results
    
    def _save_checkpoint(self, generation: int):
        """Save checkpoint of current evolution state."""
        checkpoint = {
            'generation': generation,
            'population': [
                {
                    'genes': chr.genes,
                    'fitness_values': chr.fitness_values,
                    'rank': chr.rank,
                    'crowding_distance': getattr(chr, 'crowding_distance', 0.0)
                }
                for chr in self.population
            ],
            'generation_stats': self.generation_stats,
            'pareto_history': self.pareto_history
        }
        
        checkpoint_file = self.results_dir / f"checkpoint_gen_{generation}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final evolution results."""
        results_file = self.results_dir / "moga_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save Pareto front configurations separately for easy access
        pareto_configs = []
        for solution in results['pareto_front']:
            pareto_configs.append({
                'hyperparameters': solution['hyperparameters'],
                'fitness': solution['fitness']
            })
        
        pareto_file = self.results_dir / "pareto_front_configs.json"
        with open(pareto_file, 'w') as f:
            json.dump(pareto_configs, f, indent=2)
        
        self.logger.info(f"Results saved: {results_file}")
        self.logger.info(f"Pareto front configs saved: {pareto_file}")
    
    def load_checkpoint(self, checkpoint_file: str) -> int:
        """
        Load evolution state from checkpoint.
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Generation number to resume from
        """
        checkpoint_path = Path(checkpoint_file)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        # Restore population
        self.population = []
        for chr_data in checkpoint['population']:
            chromosome = Chromosome(self.hyperparameter_space, chr_data['genes'])
            chromosome.fitness_values = chr_data.get('fitness_values')
            chromosome.rank = chr_data.get('rank')
            chromosome.crowding_distance = chr_data.get('crowding_distance', 0.0)
            self.population.append(chromosome)
        
        # Restore stats
        self.generation_stats = checkpoint.get('generation_stats', [])
        self.pareto_history = checkpoint.get('pareto_history', [])
        
        generation = checkpoint['generation']
        self.logger.info(f"Checkpoint loaded from generation {generation}")
        
        return generation
    
    def get_best_configurations(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top-k best configurations from the current population.
        
        Args:
            top_k: Number of best configurations to return
            
        Returns:
            List of best configurations with their fitness values
        """
        if not self.population:
            return []
        
        # Get Pareto front
        pareto_front = self.nsga2.get_pareto_front(self.population)
        
        # Sort by a combined fitness score for ranking
        scored_solutions = []
        for chromosome in pareto_front:
            if chromosome.fitness_values:
                # Combined score (weighted sum)
                combined_score = (0.4 * chromosome.fitness_values[0] + 
                                0.4 * chromosome.fitness_values[1] + 
                                0.2 * chromosome.fitness_values[2])
                scored_solutions.append((combined_score, chromosome))
        
        # Sort by combined score (descending)
        scored_solutions.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k configurations
        best_configs = []
        for i, (score, chromosome) in enumerate(scored_solutions[:top_k]):
            config = {
                'rank': i + 1,
                'combined_score': score,
                'hyperparameters': chromosome.genes,
                'fitness': {
                    'speed_score': chromosome.fitness_values[0],
                    'accuracy_score': chromosome.fitness_values[1],
                    'stability_score': chromosome.fitness_values[2]
                }
            }
            
            if hasattr(chromosome, 'objectives') and chromosome.objectives:
                config['detailed_metrics'] = chromosome.objectives.to_dict()
            
            best_configs.append(config)
        
        return best_configs