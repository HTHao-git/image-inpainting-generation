"""
Quick test script to verify training works end-to-end
"""

import os
import sys
import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append('src')

from src.models import UnifiedDiffusionPipeline
from src.data import UnifiedDataset, collate_fn
from src.training import UnifiedTrainer
from torch.utils.data import DataLoader


def quick_training_test():
    """Run a quick training test with minimal configuration"""
    
    print("="*60)
    print("QUICK TRAINING TEST")
    print("="*60)
    
    # Minimal config
    config = OmegaConf.create({
        'training': {
            'diffusion_weight': 1.0,
            'perceptual_weight': 0.1,
            'mask_weight': 5.0,
            'task_weights': {'generation': 1.0, 'inpainting': 1.0},
            'learning_rate': 1e-5,
            'num_epochs': 2,  # Just 2 epochs
            'save_steps': 10,
            'log_steps': 5,
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'optimizer': {'type': 'adamw', 'weight_decay': 0.01},
            'lr_scheduler': {'type': 'cosine', 'warmup_steps': 5}
        },
        'paths': {
            'model_dir': 'outputs/quick_test/models',
            'log_dir': 'outputs/quick_test/logs',
            'checkpoint_dir': 'outputs/quick_test/checkpoints',
            'sample_dir': 'outputs/quick_test/samples'
        },
        'validation': {
            'run_validation': True,
            'val_steps': 1,
            'sample_every_n_epochs': 1,
            'num_sample_images': 2,
            'sample_prompts': [
                "a beautiful sunset",
                "a cat on a table"
            ]
        }
    })
    
    # Create directories
    for path_key in ['model_dir', 'log_dir', 'checkpoint_dir', 'sample_dir']:
        os.makedirs(config.paths[path_key], exist_ok=True)
    
    device = "cpu"
    
    try:
        # Create dataset (should use existing sample data)
        print("Creating dataset...")
        dataset = UnifiedDataset("data/samples", image_size=256, mode="train")
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0
        )
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Create model
        print("Loading model...")
        pipeline = UnifiedDiffusionPipeline.from_pretrained_unified()
        print("✓ Model loaded")
        
        # Create trainer
        print("Creating trainer...")
        trainer = UnifiedTrainer(pipeline, config, device=device)
        print("✓ Trainer created")
        
        # Run a few training steps
        print("Running training steps...")
        
        # Test 3 batches only
        test_batches = []
        for i, batch in enumerate(dataloader):
            test_batches.append(batch)
            if i >= 2:  # Just 3 batches
                break
        
        for i, batch in enumerate(test_batches):
            print(f"  Training step {i+1}/3...")
            
            # Forward pass
            loss_dict = trainer.train_step(batch)
            
            # Backward pass
            trainer.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            trainer.optimizer.step()
            
            print(f"    Loss: {loss_dict['total_loss'].item():.6f}")
        
        print("✓ Training steps completed")
        
        # Test checkpoint save/load
        print("Testing checkpoint...")
        checkpoint_path = os.path.join(config.paths.checkpoint_dir, "test_checkpoint.pth")
        trainer.save_checkpoint(checkpoint_path)
        
        # Modify something to test loading
        original_step = trainer.current_step
        trainer.current_step = 999
        
        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)
        
        if trainer.current_step == original_step:
            print("✓ Checkpoint save/load working")
        else:
            print("✗ Checkpoint issue")
            return False
        
        # Test sample generation
        print("Testing sample generation...")
        try:
            with torch.no_grad():
                result = pipeline(
                    prompt="a simple test image",
                    height=256,
                    width=256,
                    num_inference_steps=2,  # Very quick
                    guidance_scale=7.5,
                    num_images_per_prompt=1
                )
                
                # Save sample
                sample_path = os.path.join(config.paths.sample_dir, "test_sample.png")
                result.images[0].save(sample_path)
                print(f"✓ Sample saved to {sample_path}")
        
        except Exception as e:
            print(f"✗ Sample generation failed: {e}")
            return False
        
        print("\n" + "="*60)
        print("QUICK TRAINING TEST PASSED!")
        print("Ready for full training!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nQUICK TRAINING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_training_test()
    
    if success:
        print("\nYou can now run full training with:")
        print("   python train.py")
        print("\nTo customize training:")
        print("   python train.py --config configs/train_config.yaml")
        print("\nTo resume training:")
        print("   python train.py --resume outputs/checkpoints/checkpoint_epoch_X.pth")
    else:
        print("\nPlease fix the issues before running full training")