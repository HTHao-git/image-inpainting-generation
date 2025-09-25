"""
Simple test to verify all imports work
"""

import os
import sys

# Add src to path
sys.path.insert(0, 'src')

def test_individual_imports():
    """Test each import individually"""
    
    print("TESTING INDIVIDUAL IMPORTS")
    print("="*40)
    
    # Test 1: Basic modules
    try:
        from models import UnifiedDiffusionPipeline
        print("✓ models.UnifiedDiffusionPipeline")
    except Exception as e:
        print(f"❌ models: {e}")
    
    try:
        from data import UnifiedDataset, collate_fn
        print("✓ data.UnifiedDataset, collate_fn")
    except Exception as e:
        print(f"❌ data: {e}")
    
    try:
        from training import UnifiedTrainer
        print("✓ training.UnifiedTrainer")
    except Exception as e:
        print(f"❌ training: {e}")
    
    # Test 2: Evaluation module components
    try:
        from evaluation.metrics import ImageQualityMetrics
        print("✓ evaluation.metrics.ImageQualityMetrics")
    except Exception as e:
        print(f"❌ evaluation.metrics: {e}")
    
    try:
        from evaluation.evaluator import ModelEvaluator
        print("✓ evaluation.evaluator.ModelEvaluator")
    except Exception as e:
        print(f"❌ evaluation.evaluator: {e}")
    
    # Test 3: Full evaluation import
    try:
        from evaluation import ModelEvaluator
        print("✓ evaluation.ModelEvaluator (full import)")
    except Exception as e:
        print(f"❌ evaluation full import: {e}")


def create_minimal_test():
    """Create a minimal working test"""
    
    print("\nCREATING MINIMAL TEST")
    print("="*40)
    
    try:
        # Just test the basic functionality without complex imports
        from models import UnifiedDiffusionPipeline
        
        print("Loading basic model...")
        pipeline = UnifiedDiffusionPipeline.from_pretrained_unified()
        
        print("Testing basic generation...")
        result = pipeline(
            prompt="a simple test", 
            height=256, 
            width=256, 
            num_inference_steps=2
        )
        
        print("✓ Basic model functionality works")
        print("✓ Ready to add evaluation system")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


if __name__ == "__main__":
    test_individual_imports()
    
    if create_minimal_test():
        print("\n🎉 Basic functionality confirmed!")
        print("📝 Next: Fix evaluation imports and rerun")
    else:
        print("\n⚠️  Basic functionality has issues")
        print("📝 Need to fix basic imports first")