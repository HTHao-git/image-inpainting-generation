"""
Fixed ModelEvaluator with proper JSON serialization
"""

import torch
import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy/torch types to JSON-serializable Python types"""
    
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else float(obj.item())
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

class ModelEvaluator:
    # ... (keep your existing __init__ and other methods)
    
    def save_detailed_results(self, metrics: Dict[str, float], evaluation_type: str = "general"):
        """Save detailed evaluation results with proper JSON serialization"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert all metrics to JSON-serializable format
        serializable_metrics = convert_to_serializable(metrics)
        
        # Save metrics as JSON
        metrics_file = os.path.join(self.save_dir, "metrics", f"{evaluation_type}_metrics_{timestamp}.json")
        
        detailed_results = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": evaluation_type,
            "device": self.device,
            "metrics": serializable_metrics,
            "interpretation": self.interpret_metrics(serializable_metrics)
        }
        
        # Ensure everything is serializable
        detailed_results = convert_to_serializable(detailed_results)
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            print(f"📊 Detailed metrics saved to: {metrics_file}")
        except Exception as e:
            print(f"❌ Failed to save JSON metrics: {e}")
            # Fallback: save simplified version
            simplified_results = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {k: float(v) if isinstance(v, (int, float, np.float32, np.float64)) else str(v) 
                          for k, v in metrics.items()}
            }
            
            with open(metrics_file.replace('.json', '_simplified.json'), 'w') as f:
                json.dump(simplified_results, f, indent=2)
        
        return detailed_results