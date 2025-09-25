"""
Loss functions for unified training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
import numpy as np


class DiffusionLoss(nn.Module):
    """
    Standard diffusion loss (MSE between predicted and actual noise)
    """
    
    def __init__(self, prediction_type: str = "epsilon"):
        super().__init__()
        self.prediction_type = prediction_type
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute diffusion loss
        
        Args:
            noise_pred: Predicted noise from model
            noise_target: Ground truth noise
        """
        return self.mse_loss(noise_pred, noise_target)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG features
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        # For CPU training, we'll use a simplified version
        # In practice, you'd use a pretrained VGG or similar
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Simplified perceptual loss (just MSE for now)
        # In a full implementation, you'd extract VGG features
        return self.mse_loss(pred, target)


class MaskLoss(nn.Module):
    """
    Mask-aware loss for inpainting tasks
    """
    
    def __init__(self, mask_weight: float = 10.0):
        super().__init__()
        self.mask_weight = mask_weight
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mask-weighted loss
        
        Args:
            pred: Predicted values
            target: Ground truth values  
            mask: Binary mask (1 = masked region, 0 = unmasked)
        """
        # Basic MSE loss
        mse = self.mse_loss(pred, target)
        
        # Weight masked regions more heavily
        mask_expanded = mask.expand_as(mse)
        weighted_loss = mse * (1 + self.mask_weight * mask_expanded)
        
        return weighted_loss.mean()


class UnifiedLoss(nn.Module):
    """
    Unified loss function that handles both generation and inpainting
    """
    
    def __init__(
        self,
        diffusion_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        mask_weight: float = 5.0,
        task_weights: Optional[Dict[str, float]] = None,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.diffusion_weight = diffusion_weight
        self.perceptual_weight = perceptual_weight
        self.mask_weight = mask_weight
        
        # Default task weights
        self.task_weights = task_weights or {
            'generation': 1.0,
            'inpainting': 1.0
        }
        
        # Loss components
        self.diffusion_loss = DiffusionLoss()
        self.perceptual_loss = PerceptualLoss(device)
        self.mask_loss = MaskLoss(mask_weight)
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_types: list,
        masks: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute unified loss
        
        Args:
            predictions: Dict containing model predictions
            targets: Dict containing ground truth targets
            task_types: List of task types for each sample in batch
            masks: Mask tensors for inpainting tasks
        """
        
        total_loss = 0.0
        loss_dict = {}
        
        # Main diffusion loss
        diffusion_loss = self.diffusion_loss(
            predictions['noise_pred'],
            targets['noise']
        )
        total_loss += self.diffusion_weight * diffusion_loss
        loss_dict['diffusion_loss'] = diffusion_loss
        
        # Task-specific losses
        generation_loss = 0.0
        inpainting_loss = 0.0
        gen_count = 0
        inp_count = 0
        
        batch_size = len(task_types)
        
        for i in range(batch_size):
            task_type = task_types[i]
            
            if task_type == 'generation':
                # Standard generation loss (already included in diffusion loss)
                gen_count += 1
                
            elif task_type == 'inpainting':
                # Additional mask-aware loss for inpainting
                if masks is not None:
                    mask_i = masks[i:i+1]  # Single sample mask
                    pred_i = predictions['noise_pred'][i:i+1]
                    target_i = targets['noise'][i:i+1]
                    
                    mask_loss_val = self.mask_loss(pred_i, target_i, mask_i)
                    inpainting_loss += mask_loss_val
                    inp_count += 1
        
        # Average task-specific losses
        if gen_count > 0:
            generation_loss = generation_loss / gen_count if generation_loss > 0 else diffusion_loss
            total_loss += self.task_weights['generation'] * generation_loss
            loss_dict['generation_loss'] = generation_loss
        
        if inp_count > 0:
            inpainting_loss = inpainting_loss / inp_count
            total_loss += self.task_weights['inpainting'] * inpainting_loss
            loss_dict['inpainting_loss'] = inpainting_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


class LossLogger:
    """Helper class to track and log losses"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = {}
        self.counts = {}
    
    def update(self, loss_dict: Dict[str, torch.Tensor]):
        """Update running averages"""
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if key not in self.losses:
                self.losses[key] = 0.0
                self.counts[key] = 0
            
            self.losses[key] += value
            self.counts[key] += 1
    
    def get_averages(self) -> Dict[str, float]:
        """Get average losses"""
        averages = {}
        for key in self.losses:
            if self.counts[key] > 0:
                averages[key] = self.losses[key] / self.counts[key]
            else:
                averages[key] = 0.0
        return averages
    
    def log_summary(self, epoch: int, step: int, prefix: str = ""):
        """Print loss summary"""
        averages = self.get_averages()
        
        log_str = f"{prefix}Epoch {epoch}, Step {step}: "
        for key, value in averages.items():
            log_str += f"{key}: {value:.6f}, "
        
        print(log_str.rstrip(", "))
        return averages