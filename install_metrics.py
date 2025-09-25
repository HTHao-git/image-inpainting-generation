"""
Install additional packages for advanced metrics
"""

def install_metric_packages():
    """Install packages for comprehensive metrics"""
    
    packages = [
        "pytorch-fid",      # For FID score
        "lpips",           # For perceptual distance
        "scikit-image",    # For additional image metrics
        "matplotlib",      # For plotting
        "seaborn",         # For better plots
        "psutil",          # For memory monitoring
    ]
    
    install_commands = []
    for package in packages:
        install_commands.append(f"pip install {package}")
    
    print("METRIC PACKAGES INSTALLATION")
    print("="*50)
    print("Run these commands to install advanced metrics:")
    print()
    for cmd in install_commands:
        print(cmd)
    
    print("\nOptional (for even more metrics):")
    print("pip install torchmetrics")  # Additional ML metrics
    print("pip install wandb")         # For experiment tracking

if __name__ == "__main__":
    install_metric_packages()