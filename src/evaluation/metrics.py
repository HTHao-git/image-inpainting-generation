"""
Comprehensive evaluation metrics for unified diffusion model - FIXED IMPORTS
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from typing import List, Dict, Tuple, Optional
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import metric libraries with graceful fallbacks
try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Note: pytorch-fid not available. Install with: pip install pytorch-fid")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Note: lpips not available. Install with: pip install lpips")


class ImageQualityMetrics:
    """Compute image quality metrics"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Initialize LPIPS if available
        if HAS_LPIPS:
            try:
                self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            except:
                self.lpips_fn = None
                print("Warning: LPIPS initialization failed")
        else:
            self.lpips_fn = None
        
        # Initialize FID components if available
        if HAS_FID:
            try:
                self.inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
                self.inception_model.eval()
            except:
                self.inception_model = None
                print("Warning: Inception model initialization failed")
        else:
            self.inception_model = None
        
        # Basic metrics always available
        self.mse_fn = nn.MSELoss()
        self.mae_fn = nn.L1Loss()
        
        print(f"Metrics initialized:")
        print(f"  LPIPS: {'✓' if self.lpips_fn is not None else '✗'}")
        print(f"  FID: {'✓' if self.inception_model is not None else '✗'}")
        print(f"  Basic: ✓ (MSE, MAE, PSNR, SSIM)")
    
    def compute_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0  # Assuming normalized images
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()
    
    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute Structural Similarity Index"""
        # Simplified SSIM implementation
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)
        
        sigma1_sq = torch.var(img1)
        sigma2_sq = torch.var(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim.item()
    
    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute LPIPS (perceptual distance)"""
        if self.lpips_fn is None:
            return -1.0
        
        try:
            with torch.no_grad():
                # Ensure correct format for LPIPS (requires [-1, 1] range)
                if img1.max() <= 1.0:
                    img1 = img1 * 2.0 - 1.0
                    img2 = img2 * 2.0 - 1.0
                
                lpips_dist = self.lpips_fn(img1, img2)
                return lpips_dist.item()
        except:
            return -1.0
    
    def compute_basic_metrics(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute basic reconstruction metrics"""
        
        metrics = {}
        
        # MSE and MAE
        metrics['mse'] = self.mse_fn(generated, target).item()
        metrics['mae'] = self.mae_fn(generated, target).item()
        
        # PSNR
        metrics['psnr'] = self.compute_psnr(generated, target)
        
        # SSIM
        metrics['ssim'] = self.compute_ssim(generated, target)
        
        # LPIPS (if available)
        if self.lpips_fn is not None:
            metrics['lpips'] = self.compute_lpips(generated, target)
        
        return metrics


class GenerationMetrics:
    """Metrics specific to image generation quality"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.quality_metrics = ImageQualityMetrics(device)
    
    def compute_inception_score(self, images: List[torch.Tensor]) -> Tuple[float, float]:
        """Compute Inception Score (IS)"""
        try:
            from torchvision.models import inception_v3
            inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
            inception_model.eval()
            
            pred_scores = []
            with torch.no_grad():
                for img in images:
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    
                    # Resize to inception input size
                    img_resized = torch.nn.functional.interpolate(
                        img, size=(299, 299), mode='bilinear', align_corners=False
                    )
                    
                    pred = inception_model(img_resized)
                    pred = torch.nn.functional.softmax(pred, dim=1)
                    pred_scores.append(pred.cpu().numpy())
            
            pred_scores = np.concatenate(pred_scores, axis=0)
            
            # Compute IS
            marginal = np.mean(pred_scores, axis=0)
            kl_divs = []
            for i in range(pred_scores.shape[0]):
                kl_div = np.sum(pred_scores[i] * np.log(pred_scores[i] / marginal + 1e-10))
                kl_divs.append(kl_div)
            
            is_mean = np.exp(np.mean(kl_divs))
            is_std = np.std(kl_divs)
            
            return is_mean, is_std
            
        except Exception as e:
            print(f"IS computation failed: {e}")
            return -1.0, -1.0
    
    def compute_diversity_metrics(self, images: List[torch.Tensor]) -> Dict[str, float]:
        """Compute diversity metrics for generated images"""
        
        if len(images) < 2:
            return {'diversity_lpips': 0.0, 'diversity_mse': 0.0}
        
        lpips_distances = []
        mse_distances = []
        
        # Compare all pairs
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                img1, img2 = images[i], images[j]
                
                # Ensure same shape
                if img1.shape != img2.shape:
                    continue
                
                # MSE distance
                mse_dist = torch.mean((img1 - img2) ** 2).item()
                mse_distances.append(mse_dist)
                
                # LPIPS distance (if available)
                if self.quality_metrics.lpips_fn is not None:
                    lpips_dist = self.quality_metrics.compute_lpips(
                        img1.unsqueeze(0), img2.unsqueeze(0)
                    )
                    if lpips_dist > 0:  # Valid LPIPS score
                        lpips_distances.append(lpips_dist)
        
        metrics = {
            'diversity_mse': np.mean(mse_distances) if mse_distances else 0.0,
        }
        
        if lpips_distances:
            metrics['diversity_lpips'] = np.mean(lpips_distances)
        
        return metrics


class InpaintingMetrics:
    """Metrics specific to inpainting quality"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.quality_metrics = ImageQualityMetrics(device)
    
    def compute_inpainting_metrics(
        self, 
        inpainted: torch.Tensor, 
        original: torch.Tensor, 
        mask: torch.Tensor
    ) -> Dict[str, float]:
        """Compute inpainting-specific metrics"""
        
        metrics = {}
        
        # Metrics for inpainted region only
        inpainted_region = inpainted * mask
        original_region = original * mask
        
        if torch.sum(mask) > 0:  # Ensure mask has content
            # Basic metrics on inpainted region
            region_metrics = self.quality_metrics.compute_basic_metrics(
                inpainted_region, original_region
            )
            
            for key, value in region_metrics.items():
                metrics[f'inpaint_{key}'] = value
        
        # Metrics for preserved region (should be unchanged)
        preserved_mask = 1.0 - mask
        inpainted_preserved = inpainted * preserved_mask
        original_preserved = original * preserved_mask
        
        if torch.sum(preserved_mask) > 0:
            preserved_mse = torch.mean((inpainted_preserved - original_preserved) ** 2).item()
            metrics['preservation_mse'] = preserved_mse
        
        # Overall image metrics
        overall_metrics = self.quality_metrics.compute_basic_metrics(inpainted, original)
        for key, value in overall_metrics.items():
            metrics[f'overall_{key}'] = value
        
        return metrics


class TrainingMetrics:
    """Track training progress and model performance"""
    
    def __init__(self, save_dir: str = "outputs/metrics"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics_history = {
            'epoch': [],
            'step': [],
            'timestamp': [],
            'train_loss': [],
            'val_loss': [],
            'generation_metrics': [],
            'inpainting_metrics': []
        }
    
    def log_training_step(
        self, 
        epoch: int, 
        step: int, 
        train_loss: float, 
        val_loss: Optional[float] = None
    ):
        """Log basic training metrics"""
        
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['step'].append(step)
        self.metrics_history['timestamp'].append(datetime.now().isoformat())
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
    
    def log_evaluation_metrics(
        self, 
        generation_metrics: Dict[str, float],
        inpainting_metrics: Dict[str, float]
    ):
        """Log evaluation metrics"""
        
        self.metrics_history['generation_metrics'].append(generation_metrics)
        self.metrics_history['inpainting_metrics'].append(inpainting_metrics)
    
    def save_metrics(self):
        """Save metrics to file"""
        
        import json
        
        # Save as JSON
        metrics_file = os.path.join(self.save_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"Metrics saved to {metrics_file}")
    
    def plot_training_curves(self):
        """Plot training progress"""
        
        if not self.metrics_history['epoch']:
            print("No metrics to plot")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Training loss
            axes[0, 0].plot(self.metrics_history['epoch'], self.metrics_history['train_loss'], 'b-', label='Train Loss')
            if any(v is not None for v in self.metrics_history['val_loss']):
                val_losses = [v for v in self.metrics_history['val_loss'] if v is not None]
                val_epochs = [e for e, v in zip(self.metrics_history['epoch'], self.metrics_history['val_loss']) if v is not None]
                axes[0, 0].plot(val_epochs, val_losses, 'r-', label='Val Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Other plots with basic data
            axes[0, 1].text(0.5, 0.5, 'Generation\nMetrics', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Generation Metrics')
            
            axes[1, 0].text(0.5, 0.5, 'Inpainting\nMetrics', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Inpainting Metrics')
            
            axes[1, 1].text(0.5, 0.5, 'Additional\nMetrics', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Additional Metrics')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.save_dir, "training_curves.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {plot_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"Plotting failed: {e}")