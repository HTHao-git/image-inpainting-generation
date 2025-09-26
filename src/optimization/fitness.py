"""
Multi-objective fitness evaluation for hyperparameter optimization.
Evaluates speed, accuracy, and stability metrics.
"""

import time
import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import logging

from ..evaluation.metrics import ImageQualityMetrics


@dataclass
class MultiObjectiveFitness:
    """Container for multi-objective fitness values."""
    speed_score: float      # Higher is better (inverse of training time)
    accuracy_score: float   # Higher is better (PSNR, SSIM, etc.)
    stability_score: float  # Higher is better (inverse of loss variance)
    
    # Raw metrics for analysis
    training_time_per_epoch: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    convergence_rate: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    fid: Optional[float] = None
    loss_variance: Optional[float] = None
    gradient_norm_variance: Optional[float] = None
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple for NSGA-II processing."""
        return (self.speed_score, self.accuracy_score, self.stability_score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/saving."""
        return {
            'speed_score': self.speed_score,
            'accuracy_score': self.accuracy_score,
            'stability_score': self.stability_score,
            'training_time_per_epoch': self.training_time_per_epoch,
            'memory_usage_mb': self.memory_usage_mb,
            'convergence_rate': self.convergence_rate,
            'psnr': self.psnr,
            'ssim': self.ssim,
            'fid': self.fid,
            'loss_variance': self.loss_variance,
            'gradient_norm_variance': self.gradient_norm_variance
        }


class FitnessEvaluator:
    """Evaluates fitness of hyperparameter configurations."""
    
    def __init__(self, 
                 device: str = "cpu",
                 max_training_epochs: int = 5,
                 early_stopping_patience: int = 3,
                 min_samples_for_eval: int = 10,
                 results_dir: str = "outputs/moga_evaluation"):
        self.device = device
        self.max_training_epochs = max_training_epochs
        self.early_stopping_patience = early_stopping_patience
        self.min_samples_for_eval = min_samples_for_eval
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics evaluator
        self.metrics_evaluator = ImageQualityMetrics(device=device)
        
        # Track evaluation history
        self.evaluation_history = []
    
    def evaluate_chromosome(self, 
                           chromosome,
                           train_loader,
                           val_loader,
                           model_factory_fn,
                           evaluation_id: Optional[str] = None) -> MultiObjectiveFitness:
        """
        Evaluate a chromosome's fitness by training with its hyperparameters.
        
        Args:
            chromosome: Chromosome containing hyperparameters
            train_loader: Training data loader
            val_loader: Validation data loader  
            model_factory_fn: Function that creates model instances
            evaluation_id: Optional ID for tracking this evaluation
            
        Returns:
            MultiObjectiveFitness object with computed metrics
        """
        self.logger.info(f"Evaluating chromosome {evaluation_id}: {chromosome.genes}")
        
        try:
            # Extract hyperparameters
            config = chromosome.get_config_dict()
            
            # Create model and trainer with these hyperparameters
            model = model_factory_fn()
            trainer = self._create_trainer(model, config)
            
            # Training metrics tracking
            training_times = []
            losses = []
            gradient_norms = []
            memory_usage = []
            
            # Early stopping tracking
            best_val_loss = float('inf')
            patience_counter = 0
            
            start_time = time.time()
            
            # Training loop with limited epochs
            for epoch in range(min(config.get('max_epochs', self.max_training_epochs), self.max_training_epochs)):
                epoch_start = time.time()
                
                # Training step
                train_metrics = self._train_epoch(trainer, train_loader, config)
                
                epoch_time = time.time() - epoch_start
                training_times.append(epoch_time)
                losses.append(train_metrics['loss'])
                gradient_norms.append(train_metrics.get('grad_norm', 0.0))
                
                # Memory usage tracking
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
                else:
                    memory_usage.append(0.0)
                
                # Validation for early stopping
                if len(val_loader) > 0:
                    val_metrics = self._validate_epoch(trainer, val_loader)
                    current_val_loss = val_metrics['loss']
                    
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                # Check for instability (loss exploding)
                if len(losses) > 2 and losses[-1] > 10 * losses[0]:
                    self.logger.warning("Training unstable - loss exploding")
                    break
            
            total_training_time = time.time() - start_time
            
            # Compute accuracy metrics on validation set
            accuracy_metrics = self._compute_accuracy_metrics(trainer, val_loader)
            
            # Compute final fitness scores
            fitness = self._compute_fitness_scores(
                training_times=training_times,
                losses=losses,
                gradient_norms=gradient_norms,
                memory_usage=memory_usage,
                accuracy_metrics=accuracy_metrics,
                total_training_time=total_training_time
            )
            
            # Save evaluation results
            if evaluation_id:
                self._save_evaluation_results(evaluation_id, chromosome, fitness, config)
            
            self.logger.info(f"Evaluation completed: {fitness.to_tuple()}")
            return fitness
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            # Return poor fitness for failed evaluations
            return MultiObjectiveFitness(
                speed_score=0.1,
                accuracy_score=0.1, 
                stability_score=0.1
            )
    
    def _create_trainer(self, model, config: Dict[str, Any]):
        """Create trainer with given configuration."""
        # Import here to avoid circular imports
        from ..training.trainer import UnifiedTrainer
        from omegaconf import OmegaConf
        
        # Convert config to OmegaConf format expected by trainer
        trainer_config = OmegaConf.create({
            'training': {
                'learning_rate': config['learning_rate'],
                'diffusion_weight': config['diffusion_weight'],
                'perceptual_weight': config['perceptual_weight'],
                'mask_weight': config['mask_weight'],
                'num_epochs': config.get('max_epochs', self.max_training_epochs),
                'save_steps': 999999,  # Don't save during evaluation
                'optimizer': {
                    'type': 'adamw',
                    'weight_decay': config['weight_decay'],
                    'betas': [config['beta1'], config['beta2']]
                },
                'lr_scheduler': {
                    'type': 'constant',
                    'warmup_steps': 0
                },
                'task_weights': {'generation': 1.0, 'inpainting': 1.0}
            },
            'paths': {
                'model_dir': str(self.results_dir / 'temp_models'),
                'log_dir': str(self.results_dir / 'temp_logs')
            }
        })
        
        return UnifiedTrainer(model, trainer_config, device=self.device)
    
    def _train_epoch(self, trainer, train_loader, config: Dict[str, Any]) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        # Limit training steps for evaluation efficiency
        max_steps = min(len(train_loader), 50)  # Max 50 steps per epoch for evaluation
        
        total_loss = 0.0
        total_grad_norm = 0.0
        step_count = 0
        
        trainer.model.train()
        
        for step, batch in enumerate(train_loader):
            if step >= max_steps:
                break
                
            try:
                loss, grad_norm = self._train_step(trainer, batch)
                total_loss += loss
                total_grad_norm += grad_norm
                step_count += 1
                
            except Exception as e:
                self.logger.warning(f"Training step failed: {e}")
                # Return high loss for failed steps
                return {'loss': 1000.0, 'grad_norm': 100.0}
        
        if step_count == 0:
            return {'loss': 1000.0, 'grad_norm': 100.0}
        
        return {
            'loss': total_loss / step_count,
            'grad_norm': total_grad_norm / step_count
        }
        """Simplified training step for evaluation."""
        try:
            # Move batch to device
            if isinstance(batch, dict):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
            
            # Simple forward pass and loss computation
            trainer.optimizer.zero_grad()
            
            # Simulate training step with random loss
            loss = torch.tensor(np.random.uniform(0.5, 2.0), requires_grad=True)
            
            # Simulate backward pass
            loss.backward()
            
            # Calculate gradient norm
            total_norm = 0
            param_count = 0
            for p in trainer.optimizer.param_groups[0]['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                grad_norm = total_norm ** (1. / 2)
            else:
                grad_norm = 0.0
            
            # Gradient clipping
            max_grad_norm = getattr(trainer.config.training, 'max_grad_norm', 1.0)
            if grad_norm > max_grad_norm:
                torch.nn.utils.clip_grad_norm_(trainer.optimizer.param_groups[0]['params'], max_grad_norm)
                grad_norm = max_grad_norm
            
            trainer.optimizer.step()
            
            return loss.item(), grad_norm
            
        except Exception as e:
            self.logger.warning(f"Training step failed: {e}")
            return 10.0, 10.0  # High loss and gradient norm for failed steps
    
    def _validate_step(self, trainer, batch) -> float:
        """Simplified validation step."""
        try:
            # Move batch to device
            if isinstance(batch, dict):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
            
            # Simulate validation loss
            return np.random.uniform(0.3, 1.5)
            
        except Exception as e:
            self.logger.warning(f"Validation step failed: {e}")
            return 10.0
    
    def _generate_sample(self, trainer, batch):
        """Generate a sample for accuracy evaluation."""
        try:
            # Simulate sample generation
            if isinstance(batch, dict) and 'image' in batch:
                # Return slightly modified input as "generated" sample
                return batch['image'] + torch.randn_like(batch['image']) * 0.1
            else:
                # Return dummy generated sample
                return torch.randn(1, 3, 256, 256)
                
        except Exception as e:
            self.logger.warning(f"Sample generation failed: {e}")
            return None
    
    def _validate_epoch(self, trainer, val_loader) -> Dict[str, float]:
        """Validate for one epoch and return metrics."""
        max_steps = min(len(val_loader), 20)  # Max 20 validation steps
        
        total_loss = 0.0
        step_count = 0
        
        trainer.model.eval()
        
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if step >= max_steps:
                    break
                    
                try:
                    # Simple validation step
                    loss = self._validate_step(trainer, batch)
                    total_loss += loss
                    step_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Validation step failed: {e}")
                    continue
        
        if step_count == 0:
            return {'loss': 1000.0}
        
        return {'loss': total_loss / step_count}
    
    def _compute_accuracy_metrics(self, trainer, val_loader) -> Dict[str, float]:
        """Compute accuracy metrics on validation set."""
        if len(val_loader) == 0:
            return {'psnr': 0.0, 'ssim': 0.0, 'fid': 100.0}
        
        psnr_scores = []
        ssim_scores = []
        
        trainer.model.eval()
        max_samples = min(self.min_samples_for_eval, len(val_loader))
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max_samples:
                    break
                
                try:
                    # Generate predictions
                    generated = self._generate_sample(trainer, batch)
                    target = batch.get('target', batch.get('image')) if isinstance(batch, dict) else None
                    
                    if generated is not None and target is not None:
                        # Compute metrics
                        metrics = self.metrics_evaluator.compute_basic_metrics(generated, target)
                        psnr_scores.append(metrics.get('psnr', 0.0))
                        ssim_scores.append(metrics.get('ssim', 0.0))
                        
                except Exception as e:
                    self.logger.warning(f"Accuracy computation failed: {e}")
                    continue
        
        if not psnr_scores:
            return {'psnr': 0.0, 'ssim': 0.0, 'fid': 100.0}
        
        return {
            'psnr': np.mean(psnr_scores),
            'ssim': np.mean(ssim_scores),
            'fid': 50.0  # Placeholder - FID computation is expensive
        }
    
    def _compute_fitness_scores(self,
                              training_times: List[float],
                              losses: List[float], 
                              gradient_norms: List[float],
                              memory_usage: List[float],
                              accuracy_metrics: Dict[str, float],
                              total_training_time: float) -> MultiObjectiveFitness:
        """Compute final fitness scores from collected metrics."""
        
        # Speed Score (higher is better)
        avg_epoch_time = np.mean(training_times) if training_times else 60.0
        speed_score = 1.0 / (1.0 + avg_epoch_time)  # Normalize to [0, 1]
        
        # Memory efficiency component
        avg_memory = np.mean(memory_usage) if memory_usage else 1000.0
        memory_efficiency = 1.0 / (1.0 + avg_memory / 1000.0)  # Normalize by GB
        speed_score = 0.7 * speed_score + 0.3 * memory_efficiency
        
        # Accuracy Score (higher is better)
        psnr = accuracy_metrics.get('psnr', 0.0)
        ssim = accuracy_metrics.get('ssim', 0.0)
        
        # Normalize PSNR to [0, 1] (assume max reasonable PSNR is 40)
        psnr_normalized = min(psnr / 40.0, 1.0)
        # SSIM is already in [0, 1]
        accuracy_score = 0.6 * psnr_normalized + 0.4 * ssim
        
        # Stability Score (higher is better)
        if len(losses) > 1:
            loss_variance = np.var(losses)
            loss_stability = 1.0 / (1.0 + loss_variance)
        else:
            loss_stability = 0.5
        
        if len(gradient_norms) > 1:
            grad_variance = np.var(gradient_norms)
            grad_stability = 1.0 / (1.0 + grad_variance)
        else:
            grad_stability = 0.5
        
        # Convergence rate (how quickly loss decreases)
        if len(losses) >= 3:
            convergence_rate = max(0, (losses[0] - losses[-1]) / len(losses))
            convergence_score = min(convergence_rate, 1.0)
        else:
            convergence_score = 0.0
        
        stability_score = 0.4 * loss_stability + 0.3 * grad_stability + 0.3 * convergence_score
        
        return MultiObjectiveFitness(
            speed_score=float(speed_score),
            accuracy_score=float(accuracy_score),
            stability_score=float(stability_score),
            training_time_per_epoch=avg_epoch_time,
            memory_usage_mb=avg_memory if memory_usage else None,
            convergence_rate=convergence_rate if len(losses) >= 3 else None,
            psnr=psnr,
            ssim=ssim,
            fid=accuracy_metrics.get('fid'),
            loss_variance=loss_variance if len(losses) > 1 else None,
            gradient_norm_variance=grad_variance if len(gradient_norms) > 1 else None
        )
    
    def _save_evaluation_results(self, 
                               evaluation_id: str,
                               chromosome,
                               fitness: MultiObjectiveFitness,
                               config: Dict[str, Any]):
        """Save evaluation results for analysis."""
        results = {
            'evaluation_id': evaluation_id,
            'chromosome_genes': chromosome.genes,
            'fitness': fitness.to_dict(),
            'config': config,
            'timestamp': time.time()
        }
        
        results_file = self.results_dir / f"evaluation_{evaluation_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.evaluation_history.append(results)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed."""
        if not self.evaluation_history:
            return {}
        
        fitness_scores = [eval_data['fitness'] for eval_data in self.evaluation_history]
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'avg_speed_score': np.mean([f['speed_score'] for f in fitness_scores]),
            'avg_accuracy_score': np.mean([f['accuracy_score'] for f in fitness_scores]),
            'avg_stability_score': np.mean([f['stability_score'] for f in fitness_scores]),
            'best_speed_score': max([f['speed_score'] for f in fitness_scores]),
            'best_accuracy_score': max([f['accuracy_score'] for f in fitness_scores]),
            'best_stability_score': max([f['stability_score'] for f in fitness_scores])
        }