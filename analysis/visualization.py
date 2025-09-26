"""
Visualization tools for MOGA optimization results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


class MOGAVisualizer:
    """Visualize MOGA optimization results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results = self._load_results()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _load_results(self) -> Dict[str, Any]:
        """Load MOGA results from file."""
        results_file = self.results_dir / "moga_results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def plot_pareto_front_2d(self, 
                           objectives: Tuple[str, str] = ('speed_score', 'accuracy_score'),
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot 2D Pareto front."""
        fig, ax = plt.subplots(figsize=figsize)
        
        pareto_front = self.results.get('pareto_front', [])
        if not pareto_front:
            ax.text(0.5, 0.5, 'No Pareto front data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Extract objective values
        obj1_values = [sol['fitness'][objectives[0]] for sol in pareto_front]
        obj2_values = [sol['fitness'][objectives[1]] for sol in pareto_front]
        
        # Plot Pareto front
        ax.scatter(obj1_values, obj2_values, c='red', s=100, alpha=0.7, 
                  label='Pareto Front', zorder=5)
        
        # Connect points to show front
        sorted_indices = np.argsort(obj1_values)
        sorted_obj1 = [obj1_values[i] for i in sorted_indices]
        sorted_obj2 = [obj2_values[i] for i in sorted_indices]
        ax.plot(sorted_obj1, sorted_obj2, 'r--', alpha=0.5, zorder=3)
        
        # Add point labels
        for i, (x, y) in enumerate(zip(obj1_values, obj2_values)):
            ax.annotate(f'P{i+1}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel(objectives[0].replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(objectives[1].replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Pareto Front: {objectives[0]} vs {objectives[1]}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pareto_front_3d(self, save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 9)) -> plt.Figure:
        """Plot 3D Pareto front with all three objectives."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        pareto_front = self.results.get('pareto_front', [])
        if not pareto_front:
            ax.text(0.5, 0.5, 0.5, 'No Pareto front data available')
            return fig
        
        # Extract objective values
        speed_values = [sol['fitness']['speed_score'] for sol in pareto_front]
        accuracy_values = [sol['fitness']['accuracy_score'] for sol in pareto_front]
        stability_values = [sol['fitness']['stability_score'] for sol in pareto_front]
        
        # Create color map based on combined fitness
        combined_scores = [s + a + st for s, a, st in zip(speed_values, accuracy_values, stability_values)]
        
        scatter = ax.scatter(speed_values, accuracy_values, stability_values,
                           c=combined_scores, cmap='viridis', s=100, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label('Combined Fitness Score', fontsize=10)
        
        # Add point labels
        for i, (x, y, z) in enumerate(zip(speed_values, accuracy_values, stability_values)):
            ax.text(x, y, z, f'P{i+1}', fontsize=8)
        
        ax.set_xlabel('Speed Score', fontsize=12)
        ax.set_ylabel('Accuracy Score', fontsize=12)
        ax.set_zlabel('Stability Score', fontsize=12)
        ax.set_title('3D Pareto Front: Speed vs Accuracy vs Stability', fontsize=14)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_evolution_history(self, save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot evolution history showing fitness improvement over generations."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        generation_stats = self.results.get('generation_stats', [])
        if not generation_stats:
            fig.suptitle('No generation statistics available')
            return fig
        
        generations = [stat['generation'] for stat in generation_stats]
        
        # Plot each objective
        objectives = ['speed_score', 'accuracy_score', 'stability_score']
        colors = ['blue', 'green', 'orange']
        
        for i, (obj, color) in enumerate(zip(objectives, colors)):
            ax = axes[i]
            
            means = [stat[obj]['mean'] for stat in generation_stats]
            stds = [stat[obj]['std'] for stat in generation_stats]
            maxs = [stat[obj]['max'] for stat in generation_stats]
            mins = [stat[obj]['min'] for stat in generation_stats]
            
            # Plot mean with error bars
            ax.errorbar(generations, means, yerr=stds, color=color, 
                       label='Mean ± Std', marker='o', markersize=4)
            
            # Plot max and min
            ax.plot(generations, maxs, color=color, linestyle='--', 
                   alpha=0.7, label='Max')
            ax.plot(generations, mins, color=color, linestyle=':', 
                   alpha=0.7, label='Min')
            
            ax.set_xlabel('Generation')
            ax.set_ylabel(obj.replace('_', ' ').title())
            ax.set_title(f'{obj.replace("_", " ").title()} Evolution')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot combined fitness evolution
        ax = axes[3]
        combined_means = []
        combined_stds = []
        
        for stat in generation_stats:
            # Weighted combination of objectives
            combined_mean = (0.4 * stat['speed_score']['mean'] + 
                           0.4 * stat['accuracy_score']['mean'] + 
                           0.2 * stat['stability_score']['mean'])
            combined_std = np.sqrt(0.4**2 * stat['speed_score']['std']**2 + 
                                 0.4**2 * stat['accuracy_score']['std']**2 + 
                                 0.2**2 * stat['stability_score']['std']**2)
            combined_means.append(combined_mean)
            combined_stds.append(combined_std)
        
        ax.errorbar(generations, combined_means, yerr=combined_stds, 
                   color='red', label='Combined Fitness', marker='o', markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Combined Fitness Score')
        ax.set_title('Combined Fitness Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_hyperparameter_distribution(self, save_path: Optional[str] = None,
                                       figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Plot distribution of hyperparameters in Pareto front."""
        pareto_front = self.results.get('pareto_front', [])
        if not pareto_front:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No Pareto front data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Extract hyperparameters
        hyperparams = [sol['hyperparameters'] for sol in pareto_front]
        param_names = list(hyperparams[0].keys())
        
        # Create subplots
        n_params = len(param_names)
        n_cols = 4
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, param_name in enumerate(param_names):
            ax = axes[i]
            values = [hp[param_name] for hp in hyperparams]
            
            if isinstance(values[0], (int, float)):
                # Numerical parameter - histogram
                ax.hist(values, bins=min(len(set(values)), 10), alpha=0.7, edgecolor='black')
                ax.set_xlabel(param_name.replace('_', ' ').title())
                ax.set_ylabel('Count')
            else:
                # Categorical parameter - bar plot
                unique_values, counts = np.unique(values, return_counts=True)
                ax.bar(range(len(unique_values)), counts, alpha=0.7)
                ax.set_xticks(range(len(unique_values)))
                ax.set_xticklabels(unique_values, rotation=45)
                ax.set_ylabel('Count')
            
            ax.set_title(param_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_hyperparameter_correlation(self, save_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """Plot correlation between hyperparameters and fitness scores."""
        pareto_front = self.results.get('pareto_front', [])
        if not pareto_front:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No Pareto front data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create DataFrame
        data = []
        for sol in pareto_front:
            row = sol['hyperparameters'].copy()
            row.update(sol['fitness'])
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Select only numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlation_df = df[numerical_cols]
        
        # Calculate correlation matrix
        corr_matrix = correlation_df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Hyperparameter-Fitness Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_report(self, save_path: Optional[str] = None) -> str:
        """Create a text summary report of the MOGA results."""
        report = []
        report.append("MOGA OPTIMIZATION SUMMARY REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Evolution summary
        evolution_summary = self.results.get('evolution_summary', {})
        report.append("EVOLUTION SUMMARY:")
        report.append(f"  Total time: {evolution_summary.get('total_time', 0):.1f} seconds")
        report.append(f"  Generations: {evolution_summary.get('num_generations', 0)}")
        report.append(f"  Population size: {evolution_summary.get('population_size', 0)}")
        report.append(f"  Final Pareto front size: {evolution_summary.get('pareto_front_size', 0)}")
        report.append("")
        
        # Pareto front analysis
        pareto_front = self.results.get('pareto_front', [])
        if pareto_front:
            report.append("PARETO FRONT ANALYSIS:")
            
            # Calculate statistics
            speed_scores = [sol['fitness']['speed_score'] for sol in pareto_front]
            accuracy_scores = [sol['fitness']['accuracy_score'] for sol in pareto_front]
            stability_scores = [sol['fitness']['stability_score'] for sol in pareto_front]
            
            report.append(f"  Speed Score - Mean: {np.mean(speed_scores):.3f}, "
                         f"Std: {np.std(speed_scores):.3f}, Range: [{np.min(speed_scores):.3f}, {np.max(speed_scores):.3f}]")
            report.append(f"  Accuracy Score - Mean: {np.mean(accuracy_scores):.3f}, "
                         f"Std: {np.std(accuracy_scores):.3f}, Range: [{np.min(accuracy_scores):.3f}, {np.max(accuracy_scores):.3f}]")
            report.append(f"  Stability Score - Mean: {np.mean(stability_scores):.3f}, "
                         f"Std: {np.std(stability_scores):.3f}, Range: [{np.min(stability_scores):.3f}, {np.max(stability_scores):.3f}]")
            report.append("")
            
            # Top solutions
            report.append("TOP 3 SOLUTIONS:")
            for i, sol in enumerate(pareto_front[:3]):
                report.append(f"  Solution {i+1}:")
                report.append(f"    Speed: {sol['fitness']['speed_score']:.3f}, "
                             f"Accuracy: {sol['fitness']['accuracy_score']:.3f}, "
                             f"Stability: {sol['fitness']['stability_score']:.3f}")
                report.append(f"    Key hyperparameters:")
                for key, value in sol['hyperparameters'].items():
                    if key in ['learning_rate', 'batch_size', 'image_size']:
                        report.append(f"      {key}: {value}")
                report.append("")
        
        # Evaluation summary
        eval_summary = self.results.get('evaluation_summary', {})
        if eval_summary:
            report.append("EVALUATION SUMMARY:")
            report.append(f"  Total evaluations: {eval_summary.get('total_evaluations', 0)}")
            report.append(f"  Average speed score: {eval_summary.get('avg_speed_score', 0):.3f}")
            report.append(f"  Average accuracy score: {eval_summary.get('avg_accuracy_score', 0):.3f}")
            report.append(f"  Average stability score: {eval_summary.get('avg_stability_score', 0):.3f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def generate_all_plots(self, output_dir: Optional[str] = None):
        """Generate all visualization plots and save to directory."""
        if output_dir is None:
            output_dir = self.results_dir / "visualizations"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating visualizations in {output_path}...")
        
        # Generate all plots
        self.plot_pareto_front_2d(save_path=output_path / "pareto_front_2d_speed_accuracy.png")
        self.plot_pareto_front_2d(objectives=('speed_score', 'stability_score'),
                                save_path=output_path / "pareto_front_2d_speed_stability.png")
        self.plot_pareto_front_2d(objectives=('accuracy_score', 'stability_score'),
                                save_path=output_path / "pareto_front_2d_accuracy_stability.png")
        self.plot_pareto_front_3d(save_path=output_path / "pareto_front_3d.png")
        self.plot_evolution_history(save_path=output_path / "evolution_history.png")
        self.plot_hyperparameter_distribution(save_path=output_path / "hyperparameter_distribution.png")
        self.plot_hyperparameter_correlation(save_path=output_path / "hyperparameter_correlation.png")
        
        # Generate summary report
        self.create_summary_report(save_path=output_path / "summary_report.txt")
        
        print(f"All visualizations saved to {output_path}")


def plot_pareto_front(results_file: str, save_path: Optional[str] = None) -> plt.Figure:
    """Convenience function to plot Pareto front from results file."""
    visualizer = MOGAVisualizer(Path(results_file).parent)
    return visualizer.plot_pareto_front_3d(save_path=save_path)


def plot_evolution_history(results_file: str, save_path: Optional[str] = None) -> plt.Figure:
    """Convenience function to plot evolution history from results file."""
    visualizer = MOGAVisualizer(Path(results_file).parent)
    return visualizer.plot_evolution_history(save_path=save_path)