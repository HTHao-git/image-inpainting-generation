# Create an updated test_setup.py
import sys
import torch
import os
from pathlib import Path

def load_env_file():
    """Load .env file manually since python-dotenv isn't in requirements"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ .env file loaded")
    else:
        print("⚠️  .env file not found")

def test_setup():
    print("🔍 Testing Image Inpainting Generation Setup")
    print("="*50)
    
    # Load environment variables
    load_env_file()
    
    # Python version
    print(f"✅ Python version: {sys.version}")
    
    # PyTorch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # CUDA
    cuda_available = torch.cuda.is_available()
    print(f"{'✅' if cuda_available else '⚠️ '} CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("   💡 Running on CPU (will be slower but functional)")
    
    # Test imports
    try:
        import diffusers
        print(f"✅ Diffusers version: {diffusers.__version__}")
    except ImportError:
        print("❌ Diffusers not installed - run: pip install diffusers==0.11.1")
    
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
    
    # Environment variables
    print("\n🔧 Environment Variables:")
    env_vars = {
        'WANDB_API_KEY': 'Weights & Biases API key',
        'HF_TOKEN': 'Hugging Face token', 
        'DATA_ROOT': 'Data directory path',
        'OUTPUT_ROOT': 'Output directory path'
    }
    
    for var, description in env_vars.items():
        value = os.getenv(var, '')
        if value:
            # Hide sensitive tokens partially
            if 'TOKEN' in var or 'KEY' in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "Set"
            else:
                display_value = value
            print(f"   ✅ {var}: {display_value}")
        else:
            print(f"   ⚠️{var}: Not set ({description})")
    
    # Directory structure
    print("\n📁 Directory Structure:")
    required_dirs = ['src', 'configs', 'data', 'outputs', 'scripts', 'notebooks', 'tests']
    for dir_name in required_dirs:
        exists = Path(dir_name).exists()
        print(f"   {'✅' if exists else '❌'} {dir_name}: {'Exists' if exists else 'Missing'}")
    
    # Test basic functionality
    print("\n🧪 Basic Functionality Test:")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1, 3, 64, 64).to(device)
        print(f"Created tensor on {device} with shape {x.shape}")
    except Exception as e:
        print(f"❌ Tensor creation failed: {e}")
    
    print(f"\nSetup Status:")
    if not Path('.env').exists():
        print(".env file missing")
    elif cuda_available:
        print("Ready for GPU training!")
    else:
        print("⚠️CPU-only setup (functional but slower)")
    
    print(f"\n💡 Next Steps:")
    if not Path('.env').exists():
        print("   1. Create .env file with your API keys")
    if 'diffusers' not in sys.modules:
        print("   2. Install diffusers: pip install diffusers==0.11.1")
    if not cuda_available:
        print("   3. For GPU training, install CUDA-enabled PyTorch later")
    print("   4. Test model loading and basic training loop")

if __name__ == "__main__":
    test_setup()