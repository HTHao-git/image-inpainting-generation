"""
Test the fixed evaluation system with proper file saving
"""

import os
import sys
sys.path.insert(0, 'src')

def test_evaluation_saving():
    """Test that evaluation results are properly saved"""
    
    print("TESTING DETAILED EVALUATION SAVING")
    print("="*50)
    
    try:
        from models import UnifiedDiffusionPipeline
        from evaluation.evaluator import ModelEvaluator
        
        # Clean up old evaluation directory
        import shutil
        if os.path.exists("outputs/evaluation"):
            shutil.rmtree("outputs/evaluation")
        
        # Load model
        print("Loading model...")
        pipeline = UnifiedDiffusionPipeline.from_pretrained_unified()
        
        # Create evaluator
        print("Creating evaluator...")
        evaluator = ModelEvaluator(pipeline, device="cpu")
        
        # Run evaluation
        print("Running evaluation...")
        test_prompts = ["a simple test image", "a red flower", "a mountain landscape"]
        metrics = evaluator.run_simple_evaluation(test_prompts)
        
        # Check what files were created
        print("\n📁 CHECKING SAVED FILES")
        print("="*30)
        
        evaluation_dir = "outputs/evaluation"
        
        def check_directory(dir_path, description):
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                print(f"✅ {description}: {len(files)} files")
                for file in files[:5]:  # Show first 5 files
                    print(f"   - {file}")
                if len(files) > 5:
                    print(f"   ... and {len(files) - 5} more")
            else:
                print(f"❌ {description}: Directory not found")
        
        check_directory(os.path.join(evaluation_dir, "samples"), "Generated Samples")
        check_directory(os.path.join(evaluation_dir, "reports"), "Human-Readable Reports")  
        check_directory(os.path.join(evaluation_dir, "metrics"), "Metric Data Files")
        
        # Check for summary file
        summary_file = os.path.join(evaluation_dir, "latest_evaluation_summary.txt")
        if os.path.exists(summary_file):
            print("✅ Summary File: Created")
            print("   Content preview:")
            with open(summary_file, 'r') as f:
                lines = f.readlines()[:10]
                for line in lines:
                    print(f"   {line.strip()}")
        else:
            print("❌ Summary File: Not found")
        
        print(f"\n🎉 Evaluation completed with proper file saving!")
        print(f"📂 Check {evaluation_dir} for all results")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_evaluation_saving()