# Image Inpainting and Generation Model

A PyTorch implementation of an image inpainting and generation model based on diffusion models (DDPM) with controllable synthesis capabilities.

## Features

- **Unified Model**: Single model for both image generation and inpainting
- **Controllable Synthesis**: Text-guided and spatially-controlled image editing
- **DDPM Architecture**: Based on Denoising Diffusion Probabilistic Models
- **HuggingFace Integration**: Leverages pre-trained models and pipelines

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HTHao-git/image-inpainting-generation.git
cd image-inpainting-generation
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from src.inference.pipeline import InpaintingPipeline

# Initialize pipeline
pipeline = InpaintingPipeline()

# Generate image
image = pipeline.generate("A beautiful landscape", height=512, width=512)

# Inpaint image
inpainted = pipeline.inpaint(image, mask, "A serene lake")
```

## Project Structure

- `src/models/`: Core model implementations
- `src/data/`: Data loading and preprocessing
- `src/training/`: Training scripts and utilities
- `src/inference/`: Inference pipelines
- `configs/`: Configuration files
- `scripts/`: Training and inference scripts
- `notebooks/`: Jupyter notebooks for exploration

## Training

```bash
python scripts/train.py --config configs/training_config.yaml
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License