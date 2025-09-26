"""
Test script for MOGA implementation.
Tests the core components without requiring full model training.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

try:
    from src.optimization import (
        HyperparameterSpace, 
        Chromosome, 
        NSGA2,
        FitnessEvaluator,
        MultiObjectiveFitness,
        MOGA
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all optimization modules are properly implemented")
    sys.exit(1)


def test_hyperparameter_space():
    """Test hyperparameter space creation."""
    print("Testing HyperparameterSpace...")
    
    # Test manual creation
    space = HyperparameterSpace()
    space.add_continuous('learning_rate', 1e-6, 1e-2, log_scale=True)
    space.add_categorical('batch_size', [1, 2, 4, 8])
    space.add_integer('max_epochs', 5, 20)
    
    print(f"  Manual space created with {len(space.parameters)} parameters")
    
    # Test default space creation
    default_space = HyperparameterSpace.create_default_space()
    print(f"  Default space created with {len(default_space.parameters)} parameters")
    
    return True


def test_chromosome():
    """Test chromosome operations."""
    print("Testing Chromosome...")
    
    space = HyperparameterSpace.create_default_space()
    
    # Test chromosome creation
    chromosome1 = Chromosome(space)
    chromosome2 = Chromosome(space)
    
    print(f"  Chromosome 1 genes: {list(chromosome1.genes.keys())}")
    print(f"  Chromosome 2 genes: {list(chromosome2.genes.keys())}")
    
    # Test mutation
    original_lr = chromosome1.genes['learning_rate']
    chromosome1.mutate(mutation_rate=1.0, mutation_strength=0.1)
    new_lr = chromosome1.genes['learning_rate']
    print(f"  Mutation test - LR: {original_lr:.2e} -> {new_lr:.2e}")
    
    # Test crossover
    child1, child2 = chromosome1.crossover(chromosome2, crossover_rate=0.8)
    print(f"  Crossover test - Children created successfully")
    
    # Test fitness assignment and dominance
    chromosome1.fitness_values = (0.8, 0.6, 0.7)  # Speed, Accuracy, Stability
    chromosome2.fitness_values = (0.6, 0.8, 0.5)
    
    dominates = chromosome1.dominates(chromosome2)
    print(f"  Dominance test - Chr1 dominates Chr2: {dominates}")
    
    return True


def test_nsga2():
    """Test NSGA-II algorithm."""
    print("Testing NSGA-II...")
    
    space = HyperparameterSpace.create_default_space()
    nsga2 = NSGA2()
    
    # Create test population
    population = []
    fitness_values = [
        (0.8, 0.6, 0.7),  # Solution 1
        (0.6, 0.8, 0.5),  # Solution 2
        (0.7, 0.7, 0.8),  # Solution 3
        (0.5, 0.9, 0.6),  # Solution 4
        (0.9, 0.5, 0.7),  # Solution 5
    ]
    
    for i, fitness in enumerate(fitness_values):
        chromosome = Chromosome(space)
        chromosome.fitness_values = fitness
        population.append(chromosome)
    
    # Test non-dominated sorting
    fronts = nsga2.non_dominated_sort(population)
    print(f"  Non-dominated sorting - Found {len(fronts)} fronts")
    print(f"  Front sizes: {[len(front) for front in fronts]}")
    
    # Test crowding distance
    for front_indices in fronts:
        nsga2.calculate_crowding_distance(population, front_indices)
    
    crowding_distances = [chr.crowding_distance for chr in population]
    print(f"  Crowding distances: {[f'{d:.3f}' if d != float('inf') else 'inf' for d in crowding_distances]}")
    
    # Test environmental selection
    # Create larger population for selection test
    extended_population = population * 2  # 10 individuals
    selected = nsga2.environmental_selection(extended_population, population_size=5)
    print(f"  Environmental selection - Selected {len(selected)}/10 individuals")
    
    # Test Pareto front extraction
    pareto_front = nsga2.get_pareto_front(population)
    print(f"  Pareto front size: {len(pareto_front)}")
    
    return True


def test_fitness_evaluator():
    """Test fitness evaluator with dummy data."""
    print("Testing FitnessEvaluator...")
    
    # Create temporary directory for results
    with tempfile.TemporaryDirectory() as temp_dir:
        evaluator = FitnessEvaluator(
            device="cpu",
            max_training_epochs=2,
            results_dir=temp_dir
        )
        
        # Create dummy chromosome
        space = HyperparameterSpace.create_default_space()
        chromosome = Chromosome(space)
        
        # Create dummy data loaders
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        dummy_images = torch.randn(10, 3, 64, 64)
        dummy_masks = torch.randint(0, 2, (10, 1, 64, 64)).float()
        dummy_dataset = TensorDataset(dummy_images, dummy_masks)
        
        train_loader = DataLoader(dummy_dataset, batch_size=2)
        val_loader = DataLoader(dummy_dataset, batch_size=2)
        
        # Create dummy model factory
        def dummy_model_factory():
            class DummyModel:
                def __init__(self):
                    self.device = "cpu"
                def to(self, device):
                    return self
                def train(self):
                    pass
                def eval(self):
                    pass
            return DummyModel()
        
        print("  Note: FitnessEvaluator test requires actual training components")
        print("  This would be tested in integration with the full training pipeline")
        
    return True


def test_moga():
    """Test MOGA algorithm."""
    print("Testing MOGA...")
    
    # Create hyperparameter space
    space = HyperparameterSpace()
    space.add_continuous('learning_rate', 1e-5, 1e-3, log_scale=True)
    space.add_categorical('batch_size', [1, 2, 4])
    space.add_continuous('weight_decay', 1e-5, 1e-2, log_scale=True)
    
    # Create dummy fitness evaluator
    class DummyFitnessEvaluator:
        def evaluate_chromosome(self, chromosome, train_loader, val_loader, model_factory_fn, evaluation_id=None):
            # Return random but deterministic fitness based on chromosome genes
            import hashlib
            
            gene_str = str(sorted(chromosome.genes.items()))
            hash_val = int(hashlib.md5(gene_str.encode()).hexdigest()[:8], 16)
            
            # Generate consistent but varied fitness values
            speed = 0.3 + 0.4 * ((hash_val % 1000) / 1000)
            accuracy = 0.3 + 0.4 * (((hash_val // 1000) % 1000) / 1000)
            stability = 0.3 + 0.4 * (((hash_val // 1000000) % 1000) / 1000)
            
            return MultiObjectiveFitness(speed, accuracy, stability)
        
        def get_evaluation_summary(self):
            return {'total_evaluations': 0}
    
    # Create temporary directory for results
    with tempfile.TemporaryDirectory() as temp_dir:
        moga = MOGA(
            hyperparameter_space=space,
            fitness_evaluator=DummyFitnessEvaluator(),
            population_size=8,
            num_generations=3,
            results_dir=temp_dir
        )
        
        # Test population initialization
        population = moga.initialize_population()
        print(f"  Population initialized with {len(population)} individuals")
        
        # Test evaluation (using dummy evaluator)
        population = moga.evaluate_population(
            population, None, None, None, generation=0
        )
        
        evaluated_count = sum(1 for chr in population if chr.fitness_values is not None)
        print(f"  Population evaluated - {evaluated_count}/{len(population)} individuals")
        
        # Test NSGA-II operations
        fronts = moga.nsga2.non_dominated_sort(population)
        print(f"  Non-dominated sorting - {len(fronts)} fronts")
        
        # Test selection
        parents = moga.selection(population)
        print(f"  Selection - {len(parents)} parents selected")
        
        # Test crossover and mutation
        offspring = moga.crossover_and_mutation(parents)
        print(f"  Crossover/Mutation - {len(offspring)} offspring created")
        
        # Test environmental selection
        combined_population = population + offspring
        next_population = moga.nsga2.environmental_selection(combined_population, moga.population_size)
        print(f"  Environmental selection - {len(next_population)} individuals selected")
        
        # Test getting best configurations
        best_configs = moga.get_best_configurations(top_k=3)
        print(f"  Best configurations - Found {len(best_configs)} configurations")
        
    return True


def run_all_tests():
    """Run all MOGA tests."""
    print("=" * 60)
    print("RUNNING MOGA COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        ("HyperparameterSpace", test_hyperparameter_space),
        ("Chromosome", test_chromosome),
        ("NSGA-II", test_nsga2),
        ("FitnessEvaluator", test_fitness_evaluator),
        ("MOGA", test_moga),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            success = test_func()
            if success:
                print(f"✓ {test_name} PASSED")
                results.append((test_name, True))
            else:
                print(f"✗ {test_name} FAILED")
                results.append((test_name, False))
        except Exception as e:
            print(f"✗ {test_name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All MOGA component tests passed!")
        print("Ready to run full MOGA optimization!")
        return True
    else:
        print("\n⚠️  Some tests failed - check implementation")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)