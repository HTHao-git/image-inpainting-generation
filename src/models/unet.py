"""
Unified U-Net with specialized heads for generation and inpainting
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor


class TaskSpecificHead(nn.Module):
    """Task-specific head for generation or inpainting"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task_type: str,
        additional_layers: int = 2
    ):
        super().__init__()
        self.task_type = task_type
        
        layers = []
        current_channels = in_channels
        
        for i in range(additional_layers):
            # Calculate appropriate number of groups for GroupNorm
            # num_groups must divide num_channels evenly
            num_groups = min(8, current_channels)
            while current_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            
            layers.extend([
                nn.Conv2d(current_channels, current_channels, 3, padding=1),
                nn.GroupNorm(num_groups, current_channels),
                nn.SiLU(),
            ])
            
        # Final output layer
        layers.append(nn.Conv2d(current_channels, out_channels, 3, padding=1))
        
        self.layers = nn.Sequential(*layers)
        
        # Task-specific conditioning
        if task_type == "inpainting":
            # Additional mask conditioning
            self.mask_conv = nn.Conv2d(1, current_channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.task_type == "inpainting" and mask is not None:
            # Incorporate mask information
            mask_features = self.mask_conv(mask)
            x = x + mask_features
            
        return self.layers(x)


class UnifiedUNet(nn.Module):
    """
    Unified U-Net for both image generation and inpainting
    Built on top of HuggingFace's UNet2DConditionModel
    """
    
    def __init__(
        self,
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
        in_channels: int = 4,
        out_channels: int = 4,
        cross_attention_dim: int = 768,
        **kwargs
    ):
        super().__init__()
        
        # Load pretrained U-Net backbone
        self.backbone = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            in_channels=in_channels,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True
        )
        
        # Get backbone output channels
        backbone_out_channels = self.backbone.config.out_channels
        
        # Task-specific heads
        self.generation_head = TaskSpecificHead(
            in_channels=backbone_out_channels,
            out_channels=out_channels,
            task_type="generation",
            additional_layers=2
        )
        
        self.inpainting_head = TaskSpecificHead(
            in_channels=backbone_out_channels,
            out_channels=out_channels,
            task_type="inpainting", 
            additional_layers=2
        )
        
        # Task conditioning
        self.task_embedding = nn.Embedding(2, cross_attention_dim)  # 0: generation, 1: inpainting
        
        # Pre-create input adapter for inpainting (instead of creating on-the-fly)
        # This handles the case where input channels don't match backbone expectations
        inpainting_input_channels = in_channels + in_channels + 1  # sample + masked_image + mask
        if inpainting_input_channels != self.backbone.config.in_channels:
            self.input_adapter = nn.Conv2d(
                inpainting_input_channels, 
                self.backbone.config.in_channels, 
                1, 
                bias=False
            )
        else:
            self.input_adapter = nn.Identity()
        
        # Store config for compatibility
        self.config = self.backbone.config
    
    @property
    def dtype(self):
        """Return the dtype of the model parameters"""
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        """Return the device of the model parameters"""
        return next(self.parameters()).device
    
    def to(self, *args, **kwargs):
        """Override to method to ensure proper device/dtype handling"""
        super().to(*args, **kwargs)
        # Also move backbone explicitly
        self.backbone = self.backbone.to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True):
        """Set training mode"""
        super().train(mode)
        self.backbone.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        super().eval()
        self.backbone.eval()
        return self
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        task_type: str = "generation",
        mask: Optional[torch.Tensor] = None,
        masked_image: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through unified model
        
        Args:
            sample: Noisy latent input
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings
            task_type: "generation" or "inpainting"
            mask: Inpainting mask (for inpainting task)
            masked_image: Masked image latents (for inpainting task)
        """
        
        # Prepare input based on task
        if task_type == "inpainting" and masked_image is not None and mask is not None:
            # Concatenate masked image and mask to input
            # Resize mask to match latent dimensions
            if mask.shape[-2:] != sample.shape[-2:]:
                mask = torch.nn.functional.interpolate(
                    mask, size=sample.shape[-2:], mode='nearest'
                )
            
            # Concatenate along channel dimension
            model_input = torch.cat([sample, masked_image, mask], dim=1)
            
            # Use the pre-created input adapter
            model_input = self.input_adapter(model_input)
        else:
            model_input = sample
        
        # Add task conditioning to text embeddings
        task_id = 0 if task_type == "generation" else 1
        task_emb = self.task_embedding(torch.tensor([task_id], device=sample.device))
        task_emb = task_emb.unsqueeze(0).expand(encoder_hidden_states.shape[0], -1, -1)
        
        # Concatenate task embedding with text embeddings
        conditioned_embeddings = torch.cat([encoder_hidden_states, task_emb], dim=1)
        
        # Forward through backbone
        backbone_output = self.backbone(
            sample=model_input,
            timestep=timestep,
            encoder_hidden_states=conditioned_embeddings,
            return_dict=False
        )[0]
        
        # Route through appropriate head
        if task_type == "generation":
            output = self.generation_head(backbone_output)
        elif task_type == "inpainting":
            output = self.inpainting_head(backbone_output, mask=mask)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        return output
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.backbone.enable_gradient_checkpointing()
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        if hasattr(self.backbone, 'disable_gradient_checkpointing'):
            self.backbone.disable_gradient_checkpointing()
    
    def set_attention_slice(self, slice_size):
        """Set attention slicing for memory efficiency"""
        if hasattr(self.backbone, 'set_attention_slice'):
            self.backbone.set_attention_slice(slice_size)
    
    def set_use_memory_efficient_attention_xformers(self, valid: bool, attention_op=None):
        """Enable xformers memory efficient attention"""
        if hasattr(self.backbone, 'set_use_memory_efficient_attention_xformers'):
            self.backbone.set_use_memory_efficient_attention_xformers(valid, attention_op)