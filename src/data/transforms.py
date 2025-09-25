"""
Image transforms and augmentations
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(image_size: int = 512, mode: str = "train"):
    """
    Get image transforms for training or validation
    
    Args:
        image_size: Target image size
        mode: 'train' or 'val'
    """
    
    if mode == "train":
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
    
    return transform


def get_mask_transforms(image_size: int = 512):
    """Get transforms specifically for masks"""
    
    transform = A.Compose([
        A.Resize(image_size, image_size),
        # Don't normalize masks, just convert to tensor
        ToTensorV2(),
    ])
    
    return transform


class ToLatent:
    """Convert images to latent space using VAE encoder"""
    
    def __init__(self, vae, device="cpu"):
        self.vae = vae
        self.device = device
        self.vae.eval()
    
    def __call__(self, image_tensor):
        """
        Convert image tensor to latent representation
        
        Args:
            image_tensor: Tensor of shape (C, H, W) with values in [-1, 1]
        """
        with torch.no_grad():
            # Add batch dimension
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Encode to latent
            latent = self.vae.encode(image_tensor).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
            
            # Remove batch dimension if it was added
            if latent.shape[0] == 1:
                latent = latent.squeeze(0)
            
            return latent