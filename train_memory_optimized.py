"""
Memory-optimized training script for systems with limited RAM
"""

import os
import sys
import gc
import torch
from omegaconf import OmegaConf

# Memory optimization settings
torch.set_num_threads(2)  # Limit CPU threads
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# Add src to path
sys.path.append('src')

from src.models import UnifiedDiffusionPipeline
from src.data import UnifiedDataset, collate_fn
from src.training import UnifiedTrainer
from torch.utils.data import DataLoader, random_split


def create_minimal_config():
    """Create a memory-optimized configuration"""
    return OmegaConf.create({
        'model': {
            'pretrained_model_name': "runwayml/stable-diffusion-v1-5",
            'in_channels': 4,
            'out_channels': 4,
            'cross_attention_dim': 768
        },
        'data': {
            'data_dir': "data/samples",
            'image_size': 256,  # Smaller images
            'batch_size': 1,    # Smallest batch
            'num_workers': 0,
            'task_ratio': 0.5,
            'mask_types': ["random", "center"],
            'mask_ratio_range': [0.1, 0.3],
            'val_split': 0.1,
            'shuffle': True
        },
        'training': {
            'num_epochs': 3,
            'learning_rate': 1e-5,
            'save_steps': 20,
            'log_steps': 5,
            'max_grad_norm': 1.0,
            'diffusion_weight': 1.0,
            'perceptual_weight': 0.0,  # Disable perceptual loss to save memory
            'mask_weight': 2.0,
            'task_weights': {'generation': 1.0, 'inpainting': 1.0},
            'optimizer': {'type': 'adamw', 'weight_decay': 0.01},
            'lr_scheduler': {'type': 'constant', 'warmup_steps': 10}
        },
        'paths': {
            'model_dir': 'outputs/models',
            'log_dir': 'outputs/logs',
            'checkpoint_dir': 'outputs/checkpoints',
            'sample_dir': 'outputs/samples'
        },
        'validation': {
            'run_validation': False,  # Disable validation to save memory
            'sample_every_n_epochs': 3
        }
    })


def setup_memory_optimization():
    """Setup various memory optimizations"""
    
    # Clear any existing cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Garbage collection
    gc.collect()
    
    # Set memory allocation strategy
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


def create_memory_efficient_pipeline():
    """Create pipeline with memory optimizations"""
    
    print("Creating memory-efficient pipeline...")
    
    # Load with explicit memory settings
    pipeline = UnifiedDiffusionPipeline.from_pretrained_unified(
        pretrained_model_name="runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        device="cpu"
    )
    
    # Enable memory efficient attention if available
    try:
        pipeline.unet.enable_gradient_checkpointing()
        print("✓ Gradient checkpointing enabled")
    except:
        print("⚠ Gradient checkpointing not available")
    
    try:
        pipeline.unet.set_attention_slice("auto")
        print("✓ Attention slicing enabled")
    except:
        print("⚠ Attention slicing not available")
    
    return pipeline


def train_with_memory_monitoring():
    """Train with memory monitoring and cleanup"""
    
    # Setup
    setup_memory_optimization()
    config = create_minimal_config()
    
    # Create directories
    for path_key in ['model_dir', 'log_dir', 'checkpoint_dir', 'sample_dir']:
        os.makedirs(config.paths[path_key], exist_ok=True)
    
    print("Memory-Optimized Training Configuration:")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Image size: {config.data.image_size}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Validation: {config.validation.run_validation}")
    
    try:
        # Create dataset
        print("\nCreating dataset...")
        full_dataset = UnifiedDataset(
            data_dir=config.data.data_dir,
            image_size=config.data.image_size,
            mode="train",
            task_ratio=config.data.task_ratio
        )
        
        # Use smaller subset for testing
        subset_size = min(10, len(full_dataset))
        indices = list(range(subset_size))
        
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, indices[:8])
        val_dataset = Subset(full_dataset, indices[8:]) if len(indices) > 8 else None
        
        print(f"Using subset: {len(train_dataset)} training samples")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,  # Don't shuffle to reduce memory
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=False
        )
        
        print(f"Data loader created: {len(train_loader)} batches")
        
        # Create pipeline
        print("\nLoading model...")
        pipeline = create_memory_efficient_pipeline()
        
        # Create trainer
        print("Creating trainer...")
        trainer = UnifiedTrainer(pipeline, config, device="cpu")
        
        # Monitor memory before training
        def print_memory_usage():
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage: {memory_mb:.1f} MB")
        
        print_memory_usage()
        
        # Training loop with memory management
        print("\nStarting memory-optimized training...")
        
        for epoch in range(config.training.num_epochs):
            print(f"\n--- Epoch {epoch} ---")
            trainer.current_epoch = epoch
            
            # Train with frequent cleanup
            trainer.model.unet.train()
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                print(f"  Batch {batch_idx+1}/{len(train_loader)}")
                
                try:
                    # Forward pass
                    loss_dict = trainer.train_step(batch)
                    
                    # Backward pass
                    trainer.optimizer.zero_grad()
                    loss_dict['total_loss'].backward()
                    trainer.optimizer.step()
                    
                    epoch_losses.append(loss_dict['total_loss'].item())
                    
                    # Memory cleanup
                    del loss_dict
                    gc.collect()
                    
                    if batch_idx % 2 == 0:  # Print every 2 batches
                        print(f"    Loss: {epoch_losses[-1]:.6f}")
                        print_memory_usage()
                
                except RuntimeError as e:
                    if "out of memory" in str(e) or "not enough memory" in str(e):
                        print(f"    Memory error in batch {batch_idx}: {e}")
                        # Try to recover
                        gc.collect()
                        continue
                    else:
                        raise e
            
            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
            print(f"  Epoch {epoch} avg loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if epoch % 2 == 0:  # Save every 2 epochs
                checkpoint_path = os.path.join(
                    config.paths.checkpoint_dir,
                    f"checkpoint_epoch_{epoch}_mem_opt.pth"
                )
                trainer.save_checkpoint(checkpoint_path)
        
        print("\nMemory-optimized training completed!")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train_with_memory_monitoring()