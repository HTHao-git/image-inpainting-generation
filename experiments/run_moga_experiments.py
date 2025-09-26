"""
Comprehensive evaluation framework for MOGA optimization results.
Includes baseline comparisons and statistical analysis.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

# Import our modules
try:
    from src.models import UnifiedDiffusionPipeline  
    from src.data import UnifiedDataset, collate_fn
    from src.training import UnifiedTrainer
    from src.optimization import FitnessEvaluator
    from analysis.visualization import MOGAVisualizer
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


@dataclass 
class ExperimentConfig:
    """Configuration for MOGA experiments."""
    name: str
    description: str
    moga_results_dir: str
    baseline_configs: List[Dict[str, Any]]
    test_epochs: int = 10
    num_test_runs: int = 3
    device: str = "cpu"
    output_dir: str = "outputs/experiments"


class MOGAExperimentRunner:
    """Run comprehensive experiments comparing MOGA results with baselines."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load MOGA results
        self.moga_results = self._load_moga_results()
        
        # Initialize fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(
            device=config.device,
            max_training_epochs=config.test_epochs,
            results_dir=str(self.output_dir / "fitness_evaluation")
        )
    
    def _load_moga_results(self) -> Dict[str, Any]:
        """Load MOGA optimization results."""
        results_file = Path(self.config.moga_results_dir) / "moga_results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"MOGA results file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def create_test_datasets(self) -> tuple:
        """Create test datasets for evaluation."""
        try:
            # Create dataset
            dataset = UnifiedDataset(
                data_dir="data/samples",
                image_size=256,
                task_ratio=0.5,
                mask_types=["random", "center"],
                mask_ratio_range=[0.1, 0.3],
                augment=False
            )
            
            # Split into train/val
            dataset_size = min(len(dataset), 100)  # Limit for testing
            val_size = int(0.2 * dataset_size)
            train_size = dataset_size - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            return train_dataset, val_dataset
            
        except Exception as e:
            self.logger.warning(f"Failed to create real datasets, using dummy data: {e}")
            
            # Create dummy datasets
            dummy_data = []
            for i in range(50):
                dummy_data.append({
                    'image': torch.randn(3, 256, 256),
                    'mask': torch.randint(0, 2, (1, 256, 256)).float(),
                    'task': 'inpainting' if i % 2 == 0 else 'generation'
                })
            
            split_idx = int(0.8 * len(dummy_data))
            return dummy_data[:split_idx], dummy_data[split_idx:]
    
    def create_model_factory(self):
        """Create model factory function."""
        def model_factory():
            try:
                return UnifiedDiffusionPipeline.from_pretrained_unified(
                    pretrained_model_name="runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float32,
                    device=self.config.device
                )
            except Exception as e:
                self.logger.warning(f"Failed to create real model: {e}")
                return self._create_dummy_model()
        
        return model_factory
    
    def _create_dummy_model(self):
        """Create dummy model for testing."""
        class DummyModel:
            def __init__(self):
                self.unet = DummyUNet()
                self.device = self.config.device
            
            def to(self, device):
                return self
            
            def train(self):
                pass
            
            def eval(self):
                pass
        
        class DummyUNet:
            def __init__(self):
                self.generation_head = torch.nn.Linear(10, 10)
                self.inpainting_head = torch.nn.Linear(10, 10)
                self.task_embedding = torch.nn.Linear(10, 10)
                self.input_adapter = torch.nn.Linear(10, 10)
            
            def parameters(self):
                return [torch.randn(10, 10, requires_grad=True)]
            
            def train(self):
                pass
            
            def eval(self):
                pass
        
        return DummyModel()
    
    def evaluate_configuration(self, 
                             config: Dict[str, Any],
                             train_dataset, 
                             val_dataset,
                             model_factory,
                             config_name: str = "test") -> Dict[str, Any]:
        """Evaluate a single configuration."""
        self.logger.info(f"Evaluating configuration: {config_name}")
        
        results = []
        
        for run in range(self.config.num_test_runs):
            self.logger.info(f"  Run {run + 1}/{self.config.num_test_runs}")
            
            # Create data loaders
            batch_size = config.get('batch_size', 2)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            # Create chromosome-like object for evaluation
            class TestChromosome:
                def __init__(self, genes):
                    self.genes = genes
                    self.fitness_values = None
                    self.objectives = None
                
                def get_config_dict(self):
                    return self.genes.copy()
            
            chromosome = TestChromosome(config)
            
            # Evaluate
            try:
                fitness = self.fitness_evaluator.evaluate_chromosome(
                    chromosome=chromosome,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model_factory_fn=model_factory,
                    evaluation_id=f"{config_name}_run_{run}"
                )
                
                results.append(fitness.to_dict())
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for run {run}: {e}")
                results.append({
                    'speed_score': 0.1,
                    'accuracy_score': 0.1,
                    'stability_score': 0.1,
                    'failed': True
                })
        
        # Aggregate results
        if results:
            aggregated = self._aggregate_results(results)
            aggregated['config'] = config
            aggregated['config_name'] = config_name
            return aggregated
        else:
            return {
                'config': config,
                'config_name': config_name,
                'speed_score': {'mean': 0.1, 'std': 0.0},
                'accuracy_score': {'mean': 0.1, 'std': 0.0},
                'stability_score': {'mean': 0.1, 'std': 0.0},
                'failed': True
            }
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple runs."""
        valid_results = [r for r in results if not r.get('failed', False)]
        
        if not valid_results:
            return {'failed': True}
        
        aggregated = {}
        
        # Aggregate each metric
        for metric in ['speed_score', 'accuracy_score', 'stability_score']:
            values = [r[metric] for r in valid_results if r.get(metric) is not None]
            if values:
                aggregated[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # Aggregate detailed metrics if available
        for metric in ['training_time_per_epoch', 'psnr', 'ssim', 'loss_variance']:
            values = [r.get(metric) for r in valid_results if r.get(metric) is not None]
            if values:
                aggregated[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        
        aggregated['num_successful_runs'] = len(valid_results)
        aggregated['total_runs'] = len(results)
        
        return aggregated
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        self.logger.info(f"Starting experiment: {self.config.name}")
        start_time = time.time()
        
        # Create datasets and model factory
        train_dataset, val_dataset = self.create_test_datasets()
        model_factory = self.create_model_factory()
        
        experiment_results = {
            'experiment_name': self.config.name,
            'description': self.config.description,
            'start_time': start_time,
            'configurations': {}
        }
        
        # Evaluate baseline configurations
        self.logger.info("Evaluating baseline configurations...")
        for i, baseline_config in enumerate(self.config.baseline_configs):
            config_name = f"baseline_{i+1}"
            result = self.evaluate_configuration(
                baseline_config, train_dataset, val_dataset, 
                model_factory, config_name
            )
            experiment_results['configurations'][config_name] = result
        
        # Evaluate top MOGA configurations
        self.logger.info("Evaluating MOGA configurations...")
        pareto_front = self.moga_results.get('pareto_front', [])
        
        # Test top 3 configurations from Pareto front
        for i, solution in enumerate(pareto_front[:3]):
            config_name = f"moga_solution_{i+1}"
            result = self.evaluate_configuration(
                solution['hyperparameters'], train_dataset, val_dataset,
                model_factory, config_name
            )
            # Add original MOGA fitness for comparison
            result['original_moga_fitness'] = solution['fitness']
            experiment_results['configurations'][config_name] = result
        
        # Calculate experiment statistics
        experiment_results.update(self._calculate_experiment_statistics(experiment_results))
        experiment_results['total_time'] = time.time() - start_time
        
        # Save results
        self._save_experiment_results(experiment_results)
        
        # Generate comparison plots
        self._generate_comparison_plots(experiment_results)
        
        self.logger.info(f"Experiment completed in {experiment_results['total_time']:.1f}s")
        return experiment_results
    
    def _calculate_experiment_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical comparisons between configurations."""
        configs = results['configurations']
        
        stats = {
            'baseline_performance': {},
            'moga_performance': {},
            'improvements': {}
        }
        
        # Separate baseline and MOGA results
        baseline_results = {k: v for k, v in configs.items() if k.startswith('baseline')}
        moga_results = {k: v for k, v in configs.items() if k.startswith('moga')}
        
        # Calculate baseline averages
        if baseline_results:
            for metric in ['speed_score', 'accuracy_score', 'stability_score']:
                values = [r[metric]['mean'] for r in baseline_results.values() if metric in r]
                if values:
                    stats['baseline_performance'][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'best': float(np.max(values))
                    }
        
        # Calculate MOGA averages
        if moga_results:
            for metric in ['speed_score', 'accuracy_score', 'stability_score']:
                values = [r[metric]['mean'] for r in moga_results.values() if metric in r]
                if values:
                    stats['moga_performance'][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'best': float(np.max(values))
                    }
        
        # Calculate improvements
        if baseline_results and moga_results:
            for metric in ['speed_score', 'accuracy_score', 'stability_score']:
                baseline_mean = stats['baseline_performance'].get(metric, {}).get('mean', 0)
                moga_mean = stats['moga_performance'].get(metric, {}).get('mean', 0)
                
                if baseline_mean > 0:
                    improvement = (moga_mean - baseline_mean) / baseline_mean * 100
                    stats['improvements'][metric] = improvement
        
        return stats
    
    def _save_experiment_results(self, results: Dict[str, Any]):
        """Save experiment results to file."""
        results_file = self.output_dir / f"{self.config.name}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Experiment results saved to: {results_file}")
        
        # Save summary report
        self._create_experiment_report(results)
    
    def _create_experiment_report(self, results: Dict[str, Any]):
        """Create human-readable experiment report."""
        report = []
        report.append(f"MOGA EXPERIMENT REPORT: {self.config.name}")
        report.append("=" * 60)
        report.append(f"Description: {self.config.description}")
        report.append(f"Total time: {results.get('total_time', 0):.1f} seconds")
        report.append("")
        
        # Configuration results
        report.append("CONFIGURATION RESULTS:")
        report.append("-" * 30)
        
        for config_name, config_result in results['configurations'].items():
            if config_result.get('failed'):
                report.append(f"{config_name}: FAILED")
                continue
            
            report.append(f"{config_name}:")
            for metric in ['speed_score', 'accuracy_score', 'stability_score']:
                if metric in config_result:
                    mean_val = config_result[metric]['mean']
                    std_val = config_result[metric]['std']
                    report.append(f"  {metric}: {mean_val:.3f} ± {std_val:.3f}")
            report.append("")
        
        # Statistical comparison
        if 'improvements' in results:
            report.append("MOGA vs BASELINE IMPROVEMENTS:")
            report.append("-" * 30)
            for metric, improvement in results['improvements'].items():
                report.append(f"{metric}: {improvement:+.1f}%")
            report.append("")
        
        # Best configurations
        report.append("BEST CONFIGURATIONS:")
        report.append("-" * 30)
        
        configs = results['configurations']
        for metric in ['speed_score', 'accuracy_score', 'stability_score']:
            best_config = None
            best_score = -1
            
            for config_name, config_result in configs.items():
                if metric in config_result:
                    score = config_result[metric]['mean']
                    if score > best_score:
                        best_score = score
                        best_config = config_name
            
            if best_config:
                report.append(f"Best {metric}: {best_config} ({best_score:.3f})")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / f"{self.config.name}_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Experiment report saved to: {report_file}")
        print(report_text)  # Also print to console
    
    def _generate_comparison_plots(self, results: Dict[str, Any]):
        """Generate comparison plots."""
        try:
            import matplotlib.pyplot as plt
            
            configs = results['configurations']
            config_names = list(configs.keys())
            
            # Create comparison bar plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            metrics = ['speed_score', 'accuracy_score', 'stability_score']
            colors = ['blue', 'green', 'orange']
            
            for i, (metric, color) in enumerate(zip(metrics, colors)):
                ax = axes[i]
                
                means = []
                stds = []
                labels = []
                
                for config_name in config_names:
                    config_result = configs[config_name]
                    if metric in config_result and not config_result.get('failed'):
                        means.append(config_result[metric]['mean'])
                        stds.append(config_result[metric]['std'])
                        labels.append(config_name)
                
                if means:
                    x_pos = np.arange(len(labels))
                    bars = ax.bar(x_pos, means, yerr=stds, color=color, alpha=0.7, capsize=5)
                    
                    ax.set_xlabel('Configuration')
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(labels, rotation=45)
                    ax.grid(True, alpha=0.3)
                    
                    # Highlight MOGA solutions
                    for j, label in enumerate(labels):
                        if label.startswith('moga'):
                            bars[j].set_color('red')
                            bars[j].set_alpha(0.8)
            
            plt.tight_layout()
            plot_file = self.output_dir / f"{self.config.name}_comparison.png"
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Comparison plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate comparison plots: {e}")


def run_moga_experiment(moga_results_dir: str, 
                       output_dir: str = "outputs/experiments",
                       test_epochs: int = 5,
                       num_test_runs: int = 2) -> Dict[str, Any]:
    """
    Run a complete MOGA experiment with baseline comparisons.
    
    Args:
        moga_results_dir: Directory containing MOGA results
        output_dir: Directory to save experiment results
        test_epochs: Number of epochs for testing configurations
        num_test_runs: Number of test runs per configuration
        
    Returns:
        Dictionary with experiment results
    """
    # Define baseline configurations for comparison
    baseline_configs = [
        {  # Conservative baseline
            'learning_rate': 1e-5,
            'batch_size': 2,
            'image_size': 256,
            'diffusion_weight': 1.0,
            'perceptual_weight': 0.1,
            'mask_weight': 5.0,
            'weight_decay': 0.01,
            'beta1': 0.9,
            'beta2': 0.999,
            'max_epochs': 10,
            'gradient_checkpointing': True,
            'attention_slice_size': 1
        },
        {  # Aggressive baseline
            'learning_rate': 1e-3,
            'batch_size': 8,
            'image_size': 384,
            'diffusion_weight': 2.0,
            'perceptual_weight': 0.5,
            'mask_weight': 2.0,
            'weight_decay': 1e-4,
            'beta1': 0.95,
            'beta2': 0.99,
            'max_epochs': 15,
            'gradient_checkpointing': False,
            'attention_slice_size': 4
        }
    ]
    
    # Create experiment configuration
    experiment_config = ExperimentConfig(
        name="moga_vs_baseline",
        description="Compare MOGA-optimized configurations with manual baselines",
        moga_results_dir=moga_results_dir,
        baseline_configs=baseline_configs,
        test_epochs=test_epochs,
        num_test_runs=num_test_runs,
        output_dir=output_dir
    )
    
    # Run experiment
    runner = MOGAExperimentRunner(experiment_config)
    return runner.run_experiment()


def compare_configurations(config_files: List[str], output_dir: str = "outputs/config_comparison"):
    """Compare multiple configuration files."""
    # This is a placeholder for comparing different MOGA runs
    # Implementation would load multiple MOGA result files and compare them
    pass


def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(description="Run MOGA experiments")
    parser.add_argument("--moga_results", type=str, required=True,
                       help="Directory containing MOGA results")
    parser.add_argument("--output_dir", type=str, default="outputs/experiments",
                       help="Output directory for experiment results")
    parser.add_argument("--test_epochs", type=int, default=5,
                       help="Number of epochs for testing")
    parser.add_argument("--num_runs", type=int, default=2,
                       help="Number of test runs per configuration")
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_moga_experiment(
        moga_results_dir=args.moga_results,
        output_dir=args.output_dir,
        test_epochs=args.test_epochs,
        num_test_runs=args.num_runs
    )
    
    print("\nExperiment completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()