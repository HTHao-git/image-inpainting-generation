"""
Main training script for Unified Diffusion Model
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf
import wandb
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.models import UnifiedDiffusionPipeline
from src.data import UnifiedDataset, collate_fn
from src.training import UnifiedTrainer


def setup_logging(config):
    """Setup logging configuration"""
    
    log_level = getattr(logging, config.logging.log_level.upper())
    
    # Create logs directory
    os.makedirs(config.paths.log_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.paths.log_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def setup_wandb(config):
    """Setup Weights & Biases logging"""
    
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            config=OmegaConf.to_container(config, resolve=True),
            name=f"unified-diffusion-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        return True
    return False


def create_datasets(config, logger):
    """Create training and validation datasets"""
    
    logger.info(f"Creating datasets from {config.data.data_dir}")
    
    # Create full dataset
    full_dataset = UnifiedDataset(
        data_dir=config.data.data_dir,
        image_size=config.data.image_size,
        mode="train",
        task_ratio=config.data.task_ratio,
        mask_types=config.data.mask_types,
        mask_ratio_range=config.data.mask_ratio_range
    )
    
    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(total_size * config.data.val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    logger.info(f"Dataset split: {train_size} train, {val_size} validation")
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config):
    """Create data loaders"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=False  # Set to True if using GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    return train_loader, val_loader


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth') and 'checkpoint_epoch_' in file:
            try:
                epoch = int(file.split('checkpoint_epoch_')[1].split('.pth')[0])
                checkpoints.append((epoch, os.path.join(checkpoint_dir, file)))
            except:
                continue
    
    if checkpoints:
        # Return the checkpoint with the highest epoch number
        latest = max(checkpoints, key=lambda x: x[0])
        return latest[1]
    
    return None


def generate_samples(pipeline, config, epoch, device, logger):
    """Generate sample images during training"""
    
    logger.info("Generating sample images...")
    
    sample_dir = Path(config.paths.sample_dir) / f"epoch_{epoch}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline.unet.eval()
    
    with torch.no_grad():
        for i, prompt in enumerate(config.validation.sample_prompts[:config.validation.num_sample_images]):
            try:
                # Generate image
                result = pipeline(
                    prompt=prompt,
                    height=512,
                    width=512,
                    num_inference_steps=20,  # More steps for better quality
                    guidance_scale=7.5,
                    num_images_per_prompt=1
                )
                
                # Save image
                image = result.images[0]
                image_path = sample_dir / f"sample_{i}_{prompt[:30].replace(' ', '_')}.png"
                image.save(image_path)
                
                logger.info(f"Saved sample: {image_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate sample for prompt '{prompt}': {e}")
    
    pipeline.unet.train()


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Train Unified Diffusion Model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override device if specified
    if args.device:
        config.device = args.device
    
    # Override resume checkpoint if specified
    if args.resume:
        config.resume.checkpoint_path = args.resume
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting Unified Diffusion Model Training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Device: {config.device}")
    
    # Setup wandb
    use_wandb = setup_wandb(config)
    
    # Create output directories
    for path_key in ['model_dir', 'log_dir', 'checkpoint_dir', 'sample_dir']:
        os.makedirs(config.paths[path_key], exist_ok=True)
    
    # Set device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config, logger)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Loading pretrained model...")
    pipeline = UnifiedDiffusionPipeline.from_pretrained_unified(
        pretrained_model_name=config.model.pretrained_model_name,
        torch_dtype=torch.float32,
        device=device
    )
    logger.info("Model loaded successfully")
    
    # Create trainer
    trainer = UnifiedTrainer(pipeline, config, device=str(device))
    
    # Setup wandb watching
    if use_wandb:
        wandb.watch(pipeline.unet, log="all", log_freq=config.training.log_steps)
        trainer.use_wandb = True
    
    # Handle resume
    if config.resume.auto_resume:
        latest_checkpoint = find_latest_checkpoint(config.paths.checkpoint_dir)
        if latest_checkpoint:
            config.resume.checkpoint_path = latest_checkpoint
    
    if config.resume.checkpoint_path and os.path.exists(config.resume.checkpoint_path):
        logger.info(f"Resuming from checkpoint: {config.resume.checkpoint_path}")
        trainer.load_checkpoint(config.resume.checkpoint_path)
    
    # Training loop with sample generation
    logger.info("Starting training...")
    
    for epoch in range(trainer.current_epoch, config.training.num_epochs):
        trainer.current_epoch = epoch
        
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader)
        
        # Log epoch summary
        logger.info(f"Epoch {epoch} Summary:")
        for key, value in train_metrics.items():
            logger.info(f"  {key}: {value:.6f}")
        
        # Validation
        if config.validation.run_validation and epoch % config.validation.val_steps == 0:
            val_metrics = trainer.validate(val_loader)
            logger.info(f"Validation Summary:")
            for key, value in val_metrics.items():
                logger.info(f"  val_{key}: {value:.6f}")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()}
                })
        else:
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **{f'train/{k}': v for k, v in train_metrics.items()}
                })
        
        # Generate samples
        if epoch % config.validation.sample_every_n_epochs == 0:
            generate_samples(pipeline, config, epoch, device, logger)
        
        # Save checkpoint
        is_best = train_metrics.get('total_loss', float('inf')) < trainer.best_loss
        if is_best:
            trainer.best_loss = train_metrics['total_loss']
        
        checkpoint_path = os.path.join(
            config.paths.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pth"
        )
        trainer.save_checkpoint(checkpoint_path, is_best=is_best)
        
        logger.info(f"Epoch {epoch} completed\n" + "="*50)
    
    logger.info("Training completed!")
    
    # Final sample generation
    generate_samples(pipeline, config, "final", device, logger)
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()