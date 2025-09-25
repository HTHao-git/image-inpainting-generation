"""
Diffusion Pipeline for Unified Generation and Inpainting
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Tuple
from diffusers import (
    DDPMScheduler, 
    AutoencoderKL, 
    DiffusionPipeline
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer
)
from diffusers.utils import logging
import numpy as np
from PIL import Image

from .unet import UnifiedUNet

logger = logging.get_logger(__name__)


class UnifiedDiffusionPipeline(DiffusionPipeline):
    """
    Unified pipeline for both text-to-image generation and inpainting
    """
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UnifiedUNet,
        scheduler: DDPMScheduler,
        safety_checker = None,
        feature_extractor = None,
        requires_safety_checker: bool = False,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.requires_safety_checker = requires_safety_checker
    
    @property
    def _execution_device(self):
        """Get the execution device from the pipeline components"""
        # Try to get device from registered modules
        if hasattr(self.unet, 'device'):
            return self.unet.device
        elif hasattr(self.vae, 'device'):
            return self.vae.device
        elif hasattr(self.text_encoder, 'device'):
            return self.text_encoder.device
        else:
            # Fallback to CPU
            return torch.device('cpu')
    
    @classmethod
    def from_pretrained_unified(
        cls,
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
        torch_dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        """Create pipeline with our unified UNet"""
        
        # Convert device to torch.device if string
        if isinstance(device, str):
            device = torch.device(device)
        
        # Load standard components
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name, 
            subfolder="vae",
            torch_dtype=torch_dtype
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name, 
            subfolder="text_encoder",
            torch_dtype=torch_dtype
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name, 
            subfolder="tokenizer"
        )
        scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name, 
            subfolder="scheduler"
        )
        
        # Create our unified UNet
        unet = UnifiedUNet(pretrained_model_name=pretrained_model_name)
        
        # Create pipeline
        pipeline = cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            **kwargs
        )
        
        # Move to device - let diffusers handle this
        try:
            pipeline = pipeline.to(device)
        except Exception as e:
            print(f"Warning: Could not move pipeline to device {device}: {e}")
            print("Continuing with default device placement...")
        
        return pipeline
    
    def enable_model_cpu_offload(self, gpu_id: int = 0):
        """Enable CPU offloading for memory efficiency"""
        # For CPU-only training, this is a no-op
        pass
    
    def enable_attention_slicing(self, slice_size: Union[str, int] = "auto"):
        """Enable attention slicing for memory efficiency"""
        if hasattr(self.unet, 'set_attention_slice'):
            self.unet.set_attention_slice(slice_size)
    
    def disable_attention_slicing(self):
        """Disable attention slicing"""
        if hasattr(self.unet, 'set_attention_slice'):
            self.unet.set_attention_slice(None)
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> torch.Tensor:
        """Encode text prompt to embeddings"""
        
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        
        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        
        # Get text embeddings
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]
        
        # Duplicate for multiple images per prompt
        if num_images_per_prompt > 1:
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        # Handle classifier-free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size if negative_prompt is None else negative_prompt
            
            if isinstance(uncond_tokens, str):
                uncond_tokens = [uncond_tokens]
            
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
            
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            
            if num_images_per_prompt > 1:
                seq_len = negative_prompt_embeds.shape[1]
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            
            # Concatenate for classifier-free guidance
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        return prompt_embeds
    
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prepare initial latents for generation"""
        
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Length of generator list {len(generator)} != batch_size {batch_size}")
        
        if latents is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        
        # Scale initial noise by scheduler's standard deviation
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_mask_and_masked_image(
        self,
        image: Union[torch.Tensor, Image.Image],
        mask: Union[torch.Tensor, Image.Image],
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare mask and masked image for inpainting"""
        
        # Convert PIL to tensor if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1).unsqueeze(0)
        
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
            mask = torch.from_numpy(mask).float() / 255.0
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(0)
        
        # Resize to target dimensions
        if image.shape[-2:] != (height, width):
            image = torch.nn.functional.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)
        
        if mask.shape[-2:] != (height, width):
            mask = torch.nn.functional.interpolate(mask, size=(height, width), mode="nearest")
        
        # Ensure mask is binary
        mask = (mask > 0.5).float()
        
        # Create masked image
        masked_image = image * (1 - mask)
        
        # Move to device and convert dtype
        mask = mask.to(device=device, dtype=dtype)
        masked_image = masked_image.to(device=device, dtype=dtype)
        
        return mask, masked_image
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[Union[torch.Tensor, Image.Image]] = None,
        mask_image: Optional[Union[torch.Tensor, Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[callable] = None,
        callback_steps: int = 1,
        **kwargs,
    ):
        """
        Main inference call - handles both generation and inpainting
        """
        
        # Determine task type
        task_type = "inpainting" if (image is not None and mask_image is not None) else "generation"
        
        # Set default dimensions
        height = height or self.unet.backbone.config.sample_size * self.vae_scale_factor
        width = width or self.unet.backbone.config.sample_size * self.vae_scale_factor
        
        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1
        
        device = self._execution_device
        
        # Classifier-free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Encode prompt
        if prompt is not None:
            prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )
        else:
            # Use empty prompt for unconditional generation
            prompt_embeds = self.encode_prompt(
                [""],
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )
        
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Prepare latents
        num_channels_latents = self.unet.backbone.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # Prepare inpainting inputs if needed
        mask = None
        masked_image_latents = None
        
        if task_type == "inpainting":
            mask, masked_image = self.prepare_mask_and_masked_image(
                image, mask_image, height, width, device, prompt_embeds.dtype
            )
            
            # Encode masked image to latents
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator)
            masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
            
            # Resize mask to latent dimensions
            mask = torch.nn.functional.interpolate(
                mask, size=latents.shape[-2:], mode="nearest"
            )
        
        # Extra step kwargs for scheduler
        extra_step_kwargs = {}
        if "eta" in self.scheduler.step.__code__.co_varnames:
            extra_step_kwargs["eta"] = eta
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Expand conditioning for classifier-free guidance
            if task_type == "inpainting" and masked_image_latents is not None:
                masked_image_input = torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
                mask_input = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
            else:
                masked_image_input = None
                mask_input = None
            
            # Predict noise
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                task_type=task_type,
                mask=mask_input,
                masked_image=masked_image_input,
            )
            
            # Classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            
            # Call callback
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        # Decode latents to images
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            
            if output_type == "pil":
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = (image * 255).round().astype("uint8")
                image = [Image.fromarray(img) for img in image]
        
        if not return_dict:
            return (image,)
        
        # Return format compatible with diffusers
        from diffusers.utils import BaseOutput
        
        class PipelineOutput(BaseOutput):
            images: Union[List[Image.Image], torch.Tensor]
        
        return PipelineOutput(images=image)