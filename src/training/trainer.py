"""
Unified trainer for generation and inpainting - FIXED IMPORTS
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from typing import Dict, Any, Optional
from tqdm import tqdm
import wandb
from omegaconf import DictConfig

# Use absolute imports instead of relative imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.losses import UnifiedLoss, LossLogger
from training.scheduler import get_optimizer, get_scheduler


class UnifiedTrainer:
    """
    Trainer for unified diffusion model
    """
    
    def __init__(
        self,
        model,  # UnifiedDiffusionPipeline - avoid import here
        config: DictConfig,
        device: str = "cpu"
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup loss function
        self.loss_fn = UnifiedLoss(
            diffusion_weight=config.training.get('diffusion_weight', 1.0),
            perceptual_weight=config.training.get('perceptual_weight', 0.1),
            mask_weight=config.training.get('mask_weight', 5.0),
            task_weights=config.training.get('task_weights', {'generation': 1.0, 'inpainting': 1.0}),
            device=device
        )
        
        # Setup optimizer - Only train our custom components
        trainable_params = []
        trainable_params.extend(self.model.unet.generation_head.parameters())
        trainable_params.extend(self.model.unet.inpainting_head.parameters())
        trainable_params.extend(self.model.unet.task_embedding.parameters())
        trainable_params.extend(self.model.unet.input_adapter.parameters())
        
        self.optimizer = get_optimizer(
            trainable_params,  # Only our custom components
            optimizer_type=config.training.optimizer.type,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.optimizer.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=config.training.lr_scheduler.type,
            num_epochs=config.training.num_epochs,
            warmup_steps=config.training.lr_scheduler.warmup_steps
        )
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        
        # Logging
        self.loss_logger = LossLogger()
        
        # Create output directories
        os.makedirs(config.paths.model_dir, exist_ok=True)
        os.makedirs(config.paths.log_dir, exist_ok=True)
        
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"Total model parameters: {sum(p.numel() for p in self.model.unet.parameters()):,}")
    
    def get_trainable_state_dict(self):
        """Get only the trainable parts of the model"""
        trainable_state = {}
        
        # Save our custom components
        trainable_state['generation_head'] = self.model.unet.generation_head.state_dict()
        trainable_state['inpainting_head'] = self.model.unet.inpainting_head.state_dict()
        trainable_state['task_embedding'] = self.model.unet.task_embedding.state_dict()
        trainable_state['input_adapter'] = self.model.unet.input_adapter.state_dict()
        
        return trainable_state
    
    def load_trainable_state_dict(self, trainable_state):
        """Load only the trainable parts of the model"""
        
        # Load our custom components
        if 'generation_head' in trainable_state:
            self.model.unet.generation_head.load_state_dict(trainable_state['generation_head'])
        
        if 'inpainting_head' in trainable_state:
            self.model.unet.inpainting_head.load_state_dict(trainable_state['inpainting_head'])
        
        if 'task_embedding' in trainable_state:
            self.model.unet.task_embedding.load_state_dict(trainable_state['task_embedding'])
        
        if 'input_adapter' in trainable_state:
            self.model.unet.input_adapter.load_state_dict(trainable_state['input_adapter'])
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save lightweight checkpoint with only trainable components"""
        
        # Lightweight checkpoint
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'trainable_state_dict': self.get_trainable_state_dict(),  # Only our parts!
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'pretrained_model_name': self.config.model.pretrained_model_name  # To reload backbone
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
        
        # Check file size
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Checkpoint saved: {path} ({file_size_mb:.1f} MB)")
    
    def load_checkpoint(self, path: str):
        """Load lightweight checkpoint"""
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load our trainable components
            if 'trainable_state_dict' in checkpoint:
                self.load_trainable_state_dict(checkpoint['trainable_state_dict'])
                print("Loaded trainable model components")
            elif 'model_state_dict' in checkpoint:
                # Fallback for old checkpoints
                print("Warning: Loading old-format checkpoint...")
                missing_keys, unexpected_keys = self.model.unet.load_state_dict(
                    checkpoint['model_state_dict'], strict=False
                )
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
            
            # Load optimizer and scheduler
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
            
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")
            
            # Load training state
            self.current_epoch = checkpoint.get('epoch', 0)
            self.current_step = checkpoint.get('step', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"Checkpoint loaded: {path} ({file_size_mb:.1f} MB)")
            print(f"Resuming from epoch {self.current_epoch}, step {self.current_step}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Single training step"""
        
        # Move batch to device and ensure proper data types
        images = batch['images'].to(self.device, dtype=torch.float32)
        prompts = batch['prompts']
        task_types = batch['task_types']
        
        # Ensure masks and masked_images are float32
        masks = None
        masked_images = None
        
        if batch['masks'][0] is not None:
            masks = batch['masks'].to(self.device, dtype=torch.float32)
        
        if batch['masked_images'][0] is not None:
            masked_images = batch['masked_images'].to(self.device, dtype=torch.float32)
        
        batch_size = images.shape[0]
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.model.vae.encode(images).latent_dist.sample()
            latents = latents * self.model.vae.config.scaling_factor
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.model.scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        )
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.model.scheduler.add_noise(latents, noise, timesteps)
        
        # Encode prompts
        with torch.no_grad():
            prompt_embeds = []
            for prompt in prompts:
                prompt_embed = self.model.encode_prompt(
                    [prompt], 
                    device=self.device,
                    do_classifier_free_guidance=False
                )
                prompt_embeds.append(prompt_embed)
            prompt_embeds = torch.cat(prompt_embeds, dim=0)
        
        # Prepare model inputs based on task types
        predictions = []
        latent_masks = []  # Store properly resized masks for loss computation
        
        for i in range(batch_size):
            task_type = task_types[i]
            
            # Get single sample inputs
            sample = noisy_latents[i:i+1]
            timestep = timesteps[i:i+1]
            encoder_hidden_states = prompt_embeds[i:i+1]
            
            # Task-specific inputs
            if task_type == 'inpainting' and masks is not None and masked_images is not None:
                # Encode masked image to latents
                with torch.no_grad():
                    masked_latents = self.model.vae.encode(masked_images[i:i+1]).latent_dist.sample()
                    masked_latents = masked_latents * self.model.vae.config.scaling_factor
                
                # Resize mask to latent dimensions and ensure float32
                mask_latent = torch.nn.functional.interpolate(
                    masks[i:i+1], size=sample.shape[-2:], mode="nearest"
                ).float()
                latent_masks.append(mask_latent)
                
                pred = self.model.unet(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    task_type=task_type,
                    mask=mask_latent,
                    masked_image=masked_latents
                )
            else:
                # For generation tasks, create dummy mask in latent space
                dummy_mask = torch.zeros(1, 1, sample.shape[-2], sample.shape[-1], device=self.device, dtype=torch.float32)
                latent_masks.append(dummy_mask)
                
                pred = self.model.unet(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    task_type=task_type
                )
            
            predictions.append(pred)
        
        # Concatenate predictions and masks
        noise_pred = torch.cat(predictions, dim=0)
        latent_masks = torch.cat(latent_masks, dim=0)
        
        # Compute loss with properly sized masks
        loss_dict = self.loss_fn(
            predictions={'noise_pred': noise_pred},
            targets={'noise': noise},
            task_types=task_types,
            masks=latent_masks  # Pass latent-space masks
        )
        
        return loss_dict
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        # Freeze backbone, only train our components
        self.model.unet.backbone.eval()
        for param in self.model.unet.backbone.parameters():
            param.requires_grad = False
        
        # Train our components
        self.model.unet.generation_head.train()
        self.model.unet.inpainting_head.train()
        self.model.unet.task_embedding.requires_grad_(True)
        self.model.unet.input_adapter.train()
        
        self.loss_logger.reset()
        
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(epoch_pbar):
            # Forward pass
            loss_dict = self.train_step(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            if hasattr(self.config.training, 'max_grad_norm'):
                # Only clip gradients of trainable parameters
                trainable_params = []
                trainable_params.extend(self.model.unet.generation_head.parameters())
                trainable_params.extend(self.model.unet.inpainting_head.parameters())
                trainable_params.extend(self.model.unet.task_embedding.parameters())
                trainable_params.extend(self.model.unet.input_adapter.parameters())
                
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, 
                    self.config.training.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Update learning rate
            if hasattr(self.scheduler, 'step') and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            # Log losses
            self.loss_logger.update(loss_dict)
            self.current_step += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Log to wandb if available
            if hasattr(self, 'use_wandb') and self.use_wandb:
                wandb.log({
                    'train/loss': loss_dict['total_loss'].item(),
                    'train/lr': current_lr,
                    'step': self.current_step
                })
            
            # Save checkpoint periodically
            if self.current_step % self.config.training.save_steps == 0:
                checkpoint_path = os.path.join(
                    self.config.paths.checkpoint_dir,
                    f"checkpoint_step_{self.current_step}.pth"
                )
                self.save_checkpoint(checkpoint_path)
        
        # Get epoch averages
        epoch_averages = self.loss_logger.get_averages()
        
        return epoch_averages
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        
        self.model.unet.eval()
        val_logger = LossLogger()
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                loss_dict = self.train_step(batch)
                val_logger.update(loss_dict)
        
        return val_logger.get_averages()