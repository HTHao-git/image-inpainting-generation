"""
Data utilities and helper functions
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import random
from typing import List, Dict, Any, Optional, Tuple


def create_mask(
    image_size: Tuple[int, int] = (512, 512),
    mask_type: str = "random",
    mask_ratio: float = 0.2,
    **kwargs
) -> Image.Image:
    """
    Create various types of masks for inpainting
    
    Args:
        image_size: (height, width) of the mask
        mask_type: Type of mask ('random', 'center', 'edges', 'irregular')
        mask_ratio: Approximate ratio of masked area
    """
    
    height, width = image_size
    mask = Image.new('L', (width, height), 0)  # Black background
    draw = ImageDraw.Draw(mask)
    
    if mask_type == "center":
        # Center square mask
        mask_size = int(min(height, width) * np.sqrt(mask_ratio))
        x1 = (width - mask_size) // 2
        y1 = (height - mask_size) // 2
        x2 = x1 + mask_size
        y2 = y1 + mask_size
        draw.rectangle([x1, y1, x2, y2], fill=255)
        
    elif mask_type == "random":
        # Random rectangular masks
        num_masks = random.randint(1, 5)
        total_area = height * width * mask_ratio
        
        for _ in range(num_masks):
            mask_w = random.randint(20, width // 3)
            mask_h = random.randint(20, height // 3)
            
            # Limit total area
            if mask_w * mask_h > total_area / num_masks * 2:
                scale = np.sqrt((total_area / num_masks * 2) / (mask_w * mask_h))
                mask_w = int(mask_w * scale)
                mask_h = int(mask_h * scale)
            
            x1 = random.randint(0, max(1, width - mask_w))
            y1 = random.randint(0, max(1, height - mask_h))
            x2 = x1 + mask_w
            y2 = y1 + mask_h
            
            draw.rectangle([x1, y1, x2, y2], fill=255)
    
    elif mask_type == "edges":
        # Edge masks (top, bottom, left, right)
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        thickness = int(min(height, width) * mask_ratio)
        
        if edge == 'top':
            draw.rectangle([0, 0, width, thickness], fill=255)
        elif edge == 'bottom':
            draw.rectangle([0, height-thickness, width, height], fill=255)
        elif edge == 'left':
            draw.rectangle([0, 0, thickness, height], fill=255)
        elif edge == 'right':
            draw.rectangle([width-thickness, 0, width, height], fill=255)
    
    elif mask_type == "irregular":
        # Irregular brush-like masks
        num_strokes = random.randint(5, 15)
        
        for _ in range(num_strokes):
            # Random starting point
            x, y = random.randint(0, width), random.randint(0, height)
            
            # Random stroke parameters
            length = random.randint(20, min(width, height) // 4)
            thickness = random.randint(5, 20)
            
            # Draw stroke with random direction
            angle = random.uniform(0, 2 * np.pi)
            end_x = x + int(length * np.cos(angle))
            end_y = y + int(length * np.sin(angle))
            
            # Ensure end point is within bounds
            end_x = max(0, min(width, end_x))
            end_y = max(0, min(height, end_y))
            
            # Draw line with thickness
            for i in range(-thickness//2, thickness//2 + 1):
                for j in range(-thickness//2, thickness//2 + 1):
                    if 0 <= x+i < width and 0 <= y+j < height:
                        if 0 <= end_x+i < width and 0 <= end_y+j < height:
                            try:
                                draw.line([x+i, y+j, end_x+i, end_y+j], fill=255, width=1)
                            except:
                                pass
    
    return mask


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for the unified dataset
    """
    
    # Separate items by task type
    generation_items = [item for item in batch if item['task_type'] == 'generation']
    inpainting_items = [item for item in batch if item['task_type'] == 'inpainting']
    
    collated = {
        'images': [],
        'prompts': [],
        'task_types': [],
        'masks': [],
        'masked_images': []
    }
    
    # Process generation items
    for item in generation_items:
        collated['images'].append(item['image'])
        collated['prompts'].append(item['prompt'])
        collated['task_types'].append(item['task_type'])
        collated['masks'].append(None)
        collated['masked_images'].append(None)
    
    # Process inpainting items
    for item in inpainting_items:
        collated['images'].append(item['image'])
        collated['prompts'].append(item['prompt'])
        collated['task_types'].append(item['task_type'])
        collated['masks'].append(item['mask'])
        collated['masked_images'].append(item['masked_image'])
    
    # Stack tensors
    collated['images'] = torch.stack(collated['images'])
    
    # Handle masks and masked images (some may be None for generation tasks)
    masks = []
    masked_images = []
    
    for mask, masked_img in zip(collated['masks'], collated['masked_images']):
        if mask is not None:
            masks.append(mask)
            masked_images.append(masked_img)
        else:
            # Create dummy tensors for generation tasks
            dummy_mask = torch.zeros(1, collated['images'].shape[-2], collated['images'].shape[-1])
            dummy_masked = torch.zeros_like(collated['images'][0])
            masks.append(dummy_mask)
            masked_images.append(dummy_masked)
    
    collated['masks'] = torch.stack(masks)
    collated['masked_images'] = torch.stack(masked_images)
    
    return collated


def load_image_from_path(image_path: str) -> Image.Image:
    """Load and convert image to RGB"""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a default image
        return Image.new('RGB', (512, 512), color=(128, 128, 128))


def generate_caption(image_path: str = None) -> str:
    """
    Generate simple captions for images
    For now, returns random captions. In practice, you'd use a captioning model
    """
    
    captions = [
        "a beautiful landscape with mountains and trees",
        "a person walking in a park",
        "a cat sitting on a windowsill", 
        "a colorful flower garden",
        "a modern building in the city",
        "a sunset over the ocean",
        "a forest with tall trees",
        "a lake surrounded by nature",
        "a street with old architecture",
        "a field of green grass",
        "a painting of abstract shapes",
        "a room with modern furniture",
        "a bridge over a river",
        "a market with fresh fruits",
        "a library with many books"
    ]
    
    return random.choice(captions)