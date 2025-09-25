"""
Dataset classes for unified training
"""

import os
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np

from .transforms import get_transforms, get_mask_transforms
from .utils import create_mask, load_image_from_path, generate_caption


class GenerationDataset(Dataset):
    """Dataset for text-to-image generation"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        mode: str = "train"
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.mode = mode
        
        self.transforms = get_transforms(image_size, mode)
        
        # Load image paths
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            import glob
            self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        
        print(f"Found {len(self.image_paths)} images for generation dataset")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        image = load_image_from_path(image_path)
        
        # Apply transforms
        transformed = self.transforms(image=np.array(image))
        image_tensor = transformed['image']
        
        # Generate caption (in practice, load from annotations)
        caption = generate_caption(image_path)
        
        return {
            'image': image_tensor,
            'prompt': caption,
            'task_type': 'generation',
            'image_path': image_path
        }


class InpaintingDataset(Dataset):
    """Dataset for image inpainting"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        mode: str = "train",
        mask_types: List[str] = ["random", "center", "edges", "irregular"],
        mask_ratio_range: Tuple[float, float] = (0.1, 0.4)
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.mode = mode
        self.mask_types = mask_types
        self.mask_ratio_range = mask_ratio_range
        
        self.image_transforms = get_transforms(image_size, mode)
        self.mask_transforms = get_mask_transforms(image_size)
        
        # Load image paths
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            import glob
            self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        
        print(f"Found {len(self.image_paths)} images for inpainting dataset")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        image = load_image_from_path(image_path)
        
        # Create mask
        mask_type = random.choice(self.mask_types)
        mask_ratio = random.uniform(*self.mask_ratio_range)
        mask = create_mask(
            image_size=(self.image_size, self.image_size),
            mask_type=mask_type,
            mask_ratio=mask_ratio
        )
        
        # Apply transforms
        image_transformed = self.image_transforms(image=np.array(image))
        image_tensor = image_transformed['image']
        
        mask_transformed = self.mask_transforms(image=np.array(mask))
        mask_tensor = mask_transformed['image']
        
        # Ensure mask is single channel and float32
        if mask_tensor.shape[0] == 3:
            mask_tensor = mask_tensor[0:1]  # Take first channel only
        
        # Convert mask to float32 and normalize to [0, 1]
        mask_tensor = mask_tensor.float() / 255.0 if mask_tensor.max() > 1.0 else mask_tensor.float()
        
        # Create masked image
        masked_image = image_tensor * (1 - mask_tensor)
        
        # Generate inpainting prompt
        caption = generate_caption(image_path)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'masked_image': masked_image,
            'prompt': caption,
            'task_type': 'inpainting',
            'image_path': image_path
        }


class UnifiedDataset(Dataset):
    """
    Unified dataset that combines generation and inpainting tasks
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        mode: str = "train",
        task_ratio: float = 0.5,  # Ratio of generation vs inpainting
        **kwargs
    ):
        self.task_ratio = task_ratio
        
        # Create individual datasets
        self.generation_dataset = GenerationDataset(data_dir, image_size, mode)
        self.inpainting_dataset = InpaintingDataset(data_dir, image_size, mode, **kwargs)
        
        # Calculate total length
        gen_len = len(self.generation_dataset)
        inp_len = len(self.inpainting_dataset)
        
        # Balance datasets based on task_ratio
        self.gen_samples = int(task_ratio * max(gen_len, inp_len) * 2)
        self.inp_samples = int((1 - task_ratio) * max(gen_len, inp_len) * 2)
        
        self.total_length = self.gen_samples + self.inp_samples
        
        print(f"Unified dataset: {self.gen_samples} generation + {self.inp_samples} inpainting = {self.total_length} total")
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        if idx < self.gen_samples:
            # Generation task
            gen_idx = idx % len(self.generation_dataset)
            return self.generation_dataset[gen_idx]
        else:
            # Inpainting task
            inp_idx = (idx - self.gen_samples) % len(self.inpainting_dataset)
            return self.inpainting_dataset[inp_idx]