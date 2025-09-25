"""
Learning rate schedulers
"""

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LinearLR, 
    SequentialLR,
    ConstantLR
)
from typing import Dict, Any


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 100,
    warmup_steps: int = 1000,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        num_epochs: Total number of training epochs
        warmup_steps: Number of warmup steps
    """
    
    if scheduler_type == "cosine":
        if warmup_steps > 0:
            # Warmup + Cosine schedule
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=warmup_steps
            )
            
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_steps,
                eta_min=optimizer.param_groups[0]['lr'] * 0.01
            )
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=optimizer.param_groups[0]['lr'] * 0.01
            )
    
    elif scheduler_type == "linear":
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_epochs
        )
    
    elif scheduler_type == "constant":
        scheduler = ConstantLR(optimizer, factor=1.0)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def get_optimizer(
    model_parameters,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get optimizer
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay
    """
    
    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer