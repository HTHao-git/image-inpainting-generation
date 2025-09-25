"""
Test training components
"""

import torch
from torch.utils.data import DataLoader
import os
from omegaconf import DictConfig, OmegaConf

def test_loss_functions():
    """Test loss functions"""
    print("Testing loss functions...")
    
    from src.training.losses import UnifiedLoss, DiffusionLoss
    
    try:
        # Test basic diffusion loss
        diffusion_loss = DiffusionLoss()
        
        noise_pred = torch.randn(2, 4, 32, 32)
        noise_target = torch.randn(2, 4, 32, 32)
        
        loss = diffusion_loss(noise_pred, noise_target)
        print(f"✓ Diffusion loss: {loss.item():.6f}")
        
        # Test unified loss
        unified_loss = UnifiedLoss()
        
        predictions = {'noise_pred': noise_pred}
        targets = {'noise': noise_target}
        task_types = ['generation', 'inpainting']
        masks = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        loss_dict = unified_loss(predictions, targets, task_types, masks)
        
        print("✓ Unified loss components:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizers_schedulers():
    """Test optimizers and schedulers"""
    print("\nTesting optimizers and schedulers...")
    
    from src.training.scheduler import get_optimizer, get_scheduler
    
    try:
        # Create dummy model parameters
        dummy_params = [torch.randn(10, requires_grad=True)]
        
        # Test optimizer
        optimizer = get_optimizer(
            dummy_params,
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=0.01
        )
        print(f"✓ Optimizer created: {type(optimizer).__name__}")
        
        # Test scheduler
        scheduler = get_scheduler(
            optimizer,
            scheduler_type="cosine",
            num_epochs=10,
            warmup_steps=5
        )
        print(f"✓ Scheduler created: {type(scheduler).__name__}")
        
        # Test a few steps
        for step in range(3):
            lr = optimizer.param_groups[0]['lr']
            print(f"  Step {step}: LR = {lr:.2e}")
            scheduler.step()
        
        return True
        
    except Exception as e:
        print(f"✗ Optimizer/scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_setup():
    """Test trainer initialization"""
    print("\nTesting trainer setup...")
    
    from src.models import UnifiedDiffusionPipeline
    from src.training.trainer import UnifiedTrainer
    
    try:
        # Create minimal config
        config = OmegaConf.create({
            'training': {
                'diffusion_weight': 1.0,
                'perceptual_weight': 0.1,
                'mask_weight': 5.0,
                'task_weights': {'generation': 1.0, 'inpainting': 1.0},
                'learning_rate': 1e-4,
                'num_epochs': 2,
                'save_steps': 100,
                'optimizer': {'type': 'adamw', 'weight_decay': 0.01},
                'lr_scheduler': {'type': 'cosine', 'warmup_steps': 10}
            },
            'paths': {
                'model_dir': 'outputs/models',
                'log_dir': 'outputs/logs'
            }
        })
        
        # Create pipeline (this will take a moment)
        print("  Creating pipeline...")
        pipeline = UnifiedDiffusionPipeline.from_pretrained_unified()
        print("  ✓ Pipeline created")
        
        # Create trainer
        print("  Creating trainer...")
        trainer = UnifiedTrainer(pipeline, config, device="cpu")
        print("  ✓ Trainer created")
        
        # Test checkpoint save/load
        checkpoint_path = "outputs/test_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)
        print("  ✓ Checkpoint saved")
        
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            print("  ✓ Checkpoint loaded")
            os.remove(checkpoint_path)  # Cleanup
        
        return True
        
    except Exception as e:
        print(f"✗ Trainer setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    from src.models import UnifiedDiffusionPipeline
    from src.training.trainer import UnifiedTrainer
    from src.data import UnifiedDataset, collate_fn
    
    try:
        # Create minimal config
        config = OmegaConf.create({
            'training': {
                'diffusion_weight': 1.0,
                'perceptual_weight': 0.1,
                'mask_weight': 5.0,
                'task_weights': {'generation': 1.0, 'inpainting': 1.0},
                'learning_rate': 1e-4,
                'num_epochs': 1,
                'save_steps': 100,
                'batch_size': 2,
                'optimizer': {'type': 'adamw', 'weight_decay': 0.01},
                'lr_scheduler': {'type': 'cosine', 'warmup_steps': 10}
            },
            'paths': {
                'model_dir': 'outputs/models',
                'log_dir': 'outputs/logs'
            }
        })
        
        # Create small dataset
        dataset = UnifiedDataset("data/samples", image_size=256, mode="train")
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Create pipeline and trainer
        pipeline = UnifiedDiffusionPipeline.from_pretrained_unified()
        trainer = UnifiedTrainer(pipeline, config, device="cpu")
        
        # Test single training step
        print("  Running training step...")
        batch = next(iter(dataloader))
        
        loss_dict = trainer.train_step(batch)
        
        print("  ✓ Training step completed")
        print("  Loss components:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_training_tests():
    """Run all training tests"""
    print("="*60)
    print("TESTING TRAINING COMPONENTS")
    print("="*60)
    
    # Ensure output directories exist
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    
    tests = [
        ("Loss Functions", test_loss_functions),
        ("Optimizers & Schedulers", test_optimizers_schedulers),
        ("Trainer Setup", test_trainer_setup),
        ("Training Step", test_training_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        
        result = test_func()
        results.append((test_name, result))
        
        if result:
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n{'='*60}")
    print("TRAINING TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        icon = "✓" if passed else "✗"
        print(f"{icon} {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = run_training_tests()
    
    if success:
        print("\n🎉 Training tests passed!")
        print("Ready to run full training!")
    else:
        print("\n⚠️  Training tests failed - need to debug")