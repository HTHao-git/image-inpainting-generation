"""
Demo script for MOGA hyperparameter optimization.
Shows the MOGA system working with minimal configuration.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Add src to path
sys.path.append('src')

try:
    from src.optimization import MOGA, HyperparameterSpace, FitnessEvaluator, MultiObjectiveFitness
    from analysis.visualization import MOGAVisualizer
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class DemoFitnessEvaluator:
    """Demo fitness evaluator that simulates training without real models."""
    
    def __init__(self, results_dir="outputs/demo_evaluation"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_count = 0
        
    def evaluate_chromosome(self, chromosome, train_loader, val_loader, model_factory_fn, evaluation_id=None):
        """Simulate chromosome evaluation with realistic fitness relationships."""
        self.evaluation_count += 1
        
        # Extract hyperparameters
        genes = chromosome.genes
        
        # Simulate realistic fitness relationships
        # Speed is generally inverse to batch_size and image_size
        batch_size = genes.get('batch_size', 2)
        image_size = genes.get('image_size', 256)
        learning_rate = genes.get('learning_rate', 1e-4)
        
        # Speed score: smaller batch size and image size = faster training
        speed_base = 1.0 / (batch_size * (image_size / 256) ** 2)
        speed_score = min(0.95, max(0.1, speed_base + np.random.normal(0, 0.1)))
        
        # Accuracy score: moderate learning rate usually better
        lr_factor = max(0.1, 1.0 - abs(np.log10(learning_rate) + 4.5))  # Best around 1e-4.5
        accuracy_base = lr_factor * (batch_size / 8.0) ** 0.3  # Slight preference for larger batches
        accuracy_score = min(0.95, max(0.1, accuracy_base + np.random.normal(0, 0.1)))
        
        # Stability score: related to learning rate and other factors
        stability_base = 1.0 / (1.0 + 10 * abs(learning_rate - 1e-4))  # Stable around 1e-4
        stability_score = min(0.95, max(0.1, stability_base + np.random.normal(0, 0.05)))
        
        return MultiObjectiveFitness(
            speed_score=float(speed_score),
            accuracy_score=float(accuracy_score),
            stability_score=float(stability_score),
            training_time_per_epoch=10.0 / speed_score,  # Inverse relationship
            psnr=20.0 + 15.0 * accuracy_score,  # Realistic PSNR range
            ssim=0.5 + 0.4 * accuracy_score,    # Realistic SSIM range
            loss_variance=1.0 / stability_score  # Inverse relationship
        )
    
    def get_evaluation_summary(self):
        return {'total_evaluations': self.evaluation_count}


def run_moga_demo():
    """Run a complete MOGA demo."""
    print("🎯 MOGA Hyperparameter Optimization Demo")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a simplified hyperparameter space for demo
    print("Creating hyperparameter space...")
    space = HyperparameterSpace()
    space.add_continuous('learning_rate', 1e-5, 1e-3, log_scale=True)
    space.add_categorical('batch_size', [1, 2, 4, 8])
    space.add_categorical('image_size', [256, 384])
    space.add_continuous('diffusion_weight', 0.5, 2.0)
    space.add_continuous('weight_decay', 1e-5, 1e-2, log_scale=True)
    
    print(f"  ✓ Space created with {len(space.parameters)} parameters")
    
    # Create demo fitness evaluator
    print("Creating fitness evaluator...")
    fitness_evaluator = DemoFitnessEvaluator()
    print("  ✓ Demo fitness evaluator created")
    
    # Create MOGA instance with small parameters for demo
    print("Creating MOGA optimizer...")
    with tempfile.TemporaryDirectory() as temp_dir:
        moga = MOGA(
            hyperparameter_space=space,
            fitness_evaluator=fitness_evaluator,
            population_size=10,      # Small population for demo
            num_generations=5,       # Few generations for demo
            crossover_rate=0.8,
            mutation_rate=0.2,
            results_dir=temp_dir,
            random_seed=42
        )
        print(f"  ✓ MOGA created with population_size=10, num_generations=5")
        
        # Run evolution (simplified)
        print("\nRunning MOGA evolution...")
        print("-" * 40)
        
        # Initialize population
        population = moga.initialize_population()
        print(f"Generation 0: Initialized {len(population)} individuals")
        
        # Evolution loop
        for generation in range(5):
            print(f"\nGeneration {generation + 1}:")
            
            # Evaluate population
            for i, chromosome in enumerate(population):
                if chromosome.fitness_values is None:
                    fitness = fitness_evaluator.evaluate_chromosome(
                        chromosome, None, None, None, f"gen{generation}_ind{i}"
                    )
                    chromosome.fitness_values = fitness.to_tuple()
                    chromosome.objectives = fitness
            
            # NSGA-II operations
            fronts = moga.nsga2.non_dominated_sort(population)
            for front_indices in fronts:
                moga.nsga2.calculate_crowding_distance(population, front_indices)
            
            # Show progress
            pareto_front = moga.nsga2.get_pareto_front(population)
            print(f"  - Evaluated {len(population)} individuals")
            print(f"  - Found {len(fronts)} non-dominated fronts")
            print(f"  - Pareto front size: {len(pareto_front)}")
            
            # Show best individual in current generation
            if pareto_front:
                best = max(pareto_front, key=lambda x: sum(x.fitness_values))
                print(f"  - Best individual: Speed={best.fitness_values[0]:.3f}, "
                      f"Accuracy={best.fitness_values[1]:.3f}, Stability={best.fitness_values[2]:.3f}")
            
            # Create next generation (except for last generation)
            if generation < 4:
                parents = moga.selection(population)
                offspring = moga.crossover_and_mutation(parents)
                combined_population = population + offspring
                population = moga.nsga2.environmental_selection(combined_population, 10)
        
        # Final results
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETED!")
        print("=" * 60)
        
        final_pareto_front = moga.nsga2.get_pareto_front(population)
        print(f"Final Pareto front contains {len(final_pareto_front)} solutions")
        
        print("\nTop 3 Pareto-optimal solutions:")
        for i, solution in enumerate(final_pareto_front[:3]):
            print(f"\nSolution {i+1}:")
            print(f"  Fitness: Speed={solution.fitness_values[0]:.3f}, "
                  f"Accuracy={solution.fitness_values[1]:.3f}, Stability={solution.fitness_values[2]:.3f}")
            print("  Hyperparameters:")
            for param, value in solution.genes.items():
                if isinstance(value, float):
                    print(f"    {param}: {value:.2e}")
                else:
                    print(f"    {param}: {value}")
        
        # Calculate improvements over random baseline
        print("\n" + "-" * 40)
        print("IMPROVEMENT ANALYSIS")
        print("-" * 40)
        
        # Create random baseline for comparison
        baseline_fitness = []
        for _ in range(10):
            random_chromosome = moga.initialize_population()[0]
            fitness = fitness_evaluator.evaluate_chromosome(random_chromosome, None, None, None)
            baseline_fitness.append(fitness.to_tuple())
        
        baseline_avg = [sum(scores) / len(scores) for scores in zip(*baseline_fitness)]
        pareto_avg = [sum(scores) / len(scores) for scores in zip(*[s.fitness_values for s in final_pareto_front])]
        
        improvements = [(pareto_avg[i] - baseline_avg[i]) / baseline_avg[i] * 100 for i in range(3)]
        
        print(f"Speed improvement: {improvements[0]:+.1f}%")
        print(f"Accuracy improvement: {improvements[1]:+.1f}%") 
        print(f"Stability improvement: {improvements[2]:+.1f}%")
        
        print(f"\nTotal evaluations performed: {fitness_evaluator.evaluation_count}")
        
        # Save minimal results for visualization demo
        results = {
            'pareto_front': [
                {
                    'hyperparameters': sol.genes,
                    'fitness': {
                        'speed_score': sol.fitness_values[0],
                        'accuracy_score': sol.fitness_values[1],
                        'stability_score': sol.fitness_values[2]
                    }
                }
                for sol in final_pareto_front
            ],
            'evolution_summary': {
                'total_time': 60.0,  # Simulated
                'num_generations': 5,
                'population_size': 10,
                'pareto_front_size': len(final_pareto_front)
            }
        }
        
        # Save to actual results directory for visualization
        results_dir = Path("outputs/demo_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(results_dir / "moga_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDemo results saved to: {results_dir}")
        print("You can now use the visualization tools to analyze results!")
        
        return results


def demo_visualization():
    """Demo the visualization capabilities."""
    print("\n" + "🎨 VISUALIZATION DEMO")
    print("=" * 60)
    
    results_dir = "outputs/demo_results"
    if not Path(results_dir).exists() or not Path(results_dir + "/moga_results.json").exists():
        print("No demo results found. Run the MOGA demo first.")
        return
    
    try:
        # Create visualizer
        visualizer = MOGAVisualizer(results_dir)
        
        # Generate plots
        viz_dir = Path(results_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        print("Generating visualization plots...")
        
        # 2D Pareto front plots
        visualizer.plot_pareto_front_2d(save_path=viz_dir / "pareto_2d_speed_accuracy.png")
        print("  ✓ Speed vs Accuracy plot saved")
        
        visualizer.plot_pareto_front_2d(
            objectives=('speed_score', 'stability_score'),
            save_path=viz_dir / "pareto_2d_speed_stability.png"
        )
        print("  ✓ Speed vs Stability plot saved")
        
        # Hyperparameter distribution
        visualizer.plot_hyperparameter_distribution(save_path=viz_dir / "hyperparameter_dist.png")
        print("  ✓ Hyperparameter distribution plot saved")
        
        # Summary report
        report = visualizer.create_summary_report(save_path=viz_dir / "summary_report.txt")
        print("  ✓ Summary report saved")
        
        print(f"\nAll visualizations saved to: {viz_dir}")
        print("\nSample of summary report:")
        print("-" * 30)
        print(report[:500] + "..." if len(report) > 500 else report)
        
    except Exception as e:
        print(f"Visualization demo failed: {e}")


if __name__ == "__main__":
    import numpy as np
    
    # Run MOGA demo
    results = run_moga_demo()
    
    # Demo visualization
    demo_visualization()
    
    print("\n" + "🎉 MOGA DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("This demo showed:")
    print("✓ Hyperparameter space definition")
    print("✓ Multi-objective fitness evaluation")
    print("✓ NSGA-II genetic algorithm execution")
    print("✓ Pareto front discovery")
    print("✓ Results visualization")
    print("✓ Performance improvement analysis")
    print("\nThe full MOGA system is ready for real hyperparameter optimization!")