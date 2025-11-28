# Project Structure Documentation

This document describes the modular architecture of the image denoising methods repository.

## Directory Structure

```
image-denoising-methods/
├── denoisers/                  # Denoising method implementations
│   ├── __init__.py            # Package initialization
│   ├── ideal_denoiser.py      # Ideal denoiser (Equation 57 from EDM paper)
│   └── edm_denoiser.py        # EDM neural network-based denoiser
│
├── utils/                      # Utility modules
│   ├── __init__.py            # Package initialization
│   ├── noise_utils.py         # Noise generation utilities
│   ├── image_utils.py         # Image loading and processing
│   └── visualization.py       # Visualization and plotting tools
│
├── draft_codes/                # Draft implementations and experiments
│   ├── edm_denoiser_gradient.py
│   └── *.ipynb                # Jupyter notebooks
│
├── data/                       # Dataset storage (auto-downloaded)
├── results/                    # Output directory for generated images
├── __pycache__/               # Python bytecode cache
│
├── generate_edm_figure1.py    # Main script for reproducing EDM Figure 1
├── example_usage.py           # Usage examples for all modules
│
├── requirements.txt           # Python dependencies
├── README.md                  # Main documentation
├── README_FIGURE1.md          # Figure 1 specific documentation
├── MATHEMATICAL_BACKGROUND.md # Mathematical background
├── QUICKSTART.md              # Quick start guide
└── PROJECT_STRUCTURE.md       # This file
```

## Module Descriptions

### 1. Denoisers Package (`denoisers/`)

This package contains different denoising method implementations that can be easily extended.

#### `ideal_denoiser.py`

Implements the theoretical ideal denoiser from Equation 57 of the EDM paper.

**Key Function:**
- `ideal_denoiser(x_noisy, sigma, x_all)` - Computes the closed-form ideal denoiser

**Features:**
- Exact implementation of Equation 57
- Numerically stable using log-sum-exp trick
- Works with batch inputs
- Fully documented with examples

**Usage:**
```python
from denoisers.ideal_denoiser import ideal_denoiser
denoised = ideal_denoiser(noisy_images, sigma=2.0, train_images)
```

#### `edm_denoiser.py`

Provides functionality for using pretrained EDM neural network denoisers.

**Key Functions:**
- `load_edm_model(model_path, url, device)` - Load pretrained EDM models
- `edm_denoise(model, noisy_images, sigma)` - Denoise using EDM model
- `compute_score_gradient(model, x, sigma)` - Compute score gradients
- `gradient_ascent_denoise(model, x_init, sigma, ...)` - Iterative denoising
- `load_pretrained_edm(model_name, device)` - Convenient loading of standard models

**Features:**
- Support for pretrained EDM models from NVlabs
- Automatic model downloading and caching
- Score gradient computation
- Gradient ascent optimization
- Predefined model configurations

**Usage:**
```python
from denoisers.edm_denoiser import load_pretrained_edm, edm_denoise

# Load pretrained model
model, config = load_pretrained_edm('cifar10-uncond')

# Denoise
denoised = edm_denoise(model, noisy_images, sigma=3.0)
```

### 2. Utils Package (`utils/`)

Common utilities used across different denoising methods.

#### `noise_utils.py`

Functions for adding noise to images.

**Key Function:**
- `add_gaussian_noise(images, sigma)` - Add Gaussian noise to images

**Features:**
- Vectorized implementation
- Handles zero noise case
- Works with any image dimension

**Usage:**
```python
from utils.noise_utils import add_gaussian_noise
noisy_images = add_gaussian_noise(clean_images, sigma=2.0)
```

#### `image_utils.py`

Image loading, processing, and normalization utilities.

**Key Functions:**
- `load_cifar10_dataset(root, normalize)` - Load CIFAR-10 dataset
- `normalize_for_display(images)` - Normalize images to [0,1] for visualization

**Features:**
- Automatic dataset downloading
- Configurable normalization
- Progress bars for loading
- Min-max normalization for display

**Usage:**
```python
from utils.image_utils import load_cifar10_dataset, normalize_for_display

train_imgs, test_imgs = load_cifar10_dataset(root="./data")
display_imgs = normalize_for_display(images)
```

#### `visualization.py`

Visualization and plotting utilities.

**Key Function:**
- `create_labeled_figure(noisy_grid, denoised_grid, sigma_values, save_dir)` - Create publication-quality figures

**Features:**
- Automatic grid layout
- Sigma value labels
- High-quality output
- Customizable styling

**Usage:**
```python
from utils.visualization import create_labeled_figure
create_labeled_figure(noisy_grid, denoised_grid, sigmas, "./results")
```

## Main Scripts

### `generate_edm_figure1.py`

Main script for reproducing Figure 1 from the EDM paper.

**Purpose:** Generate a comprehensive visualization of the ideal denoiser's performance across multiple noise levels.

**Key Features:**
- Uses modular components from `denoisers/` and `utils/`
- Processes multiple test images and sigma values
- Generates three output images:
  - `figure1_noisy.png` - Grid of noisy images
  - `figure1_denoised.png` - Grid of denoised images
  - `figure1_combined.png` - Labeled combined visualization

**Usage:**
```bash
python generate_edm_figure1.py
```

### `example_usage.py`

Comprehensive examples demonstrating all module functionality.

**Includes:**
1. Example 1: Basic ideal denoiser usage
2. Example 2: EDM pretrained model usage
3. Example 3: Gradient ascent denoising
4. Example 4: Batch processing
5. Example 5: Custom noise utilities

**Usage:**
```bash
python example_usage.py
```

Or import specific examples:
```python
from example_usage import example_1_ideal_denoiser
example_1_ideal_denoiser()
```

## Design Principles

### 1. Modularity

Each denoising method is self-contained in its own module, making it easy to:
- Add new methods
- Test methods independently
- Reuse code across different experiments

### 2. Separation of Concerns

- **Denoisers:** Core denoising algorithms
- **Utils:** Shared utilities (noise, data loading, visualization)
- **Scripts:** High-level workflows using the modules

### 3. Extensibility

To add a new denoising method:

1. Create a new file in `denoisers/` (e.g., `denoisers/my_denoiser.py`)
2. Implement your denoising function(s)
3. Add to `denoisers/__init__.py`:
```python
from .my_denoiser import my_denoise_function
__all__.append('my_denoise_function')
```
4. Use shared utilities from `utils/` for common tasks

### 4. Clean Code Practices

- Comprehensive docstrings with examples
- Type hints where beneficial
- Consistent naming conventions
- Modular functions with single responsibilities
- Extensive inline documentation

## Adding New Denoising Methods

### Template for a New Denoiser

```python
# denoisers/my_new_denoiser.py
"""
My New Denoiser Implementation

Description of the method and references.
"""

import torch

def my_denoise(noisy_images, sigma, **kwargs):
    """
    Denoise images using my new method.
    
    Parameters:
    -----------
    noisy_images : torch.Tensor
        Noisy input images of shape (batch_size, C, H, W)
    sigma : float
        Noise level
    **kwargs : additional arguments
        
    Returns:
    --------
    denoised : torch.Tensor
        Denoised images of shape (batch_size, C, H, W)
        
    Examples:
    ---------
    >>> from denoisers.my_new_denoiser import my_denoise
    >>> denoised = my_denoise(noisy_images, sigma=2.0)
    """
    # Implementation here
    denoised = your_algorithm(noisy_images, sigma, **kwargs)
    return denoised
```

### Integration Steps

1. Add the new module to `denoisers/`
2. Update `denoisers/__init__.py`
3. Create examples in `example_usage.py`
4. Add tests if applicable
5. Update documentation

## Common Workflows

### Workflow 1: Compare Multiple Denoisers

```python
from denoisers.ideal_denoiser import ideal_denoiser
from denoisers.edm_denoiser import load_pretrained_edm, edm_denoise
from utils.noise_utils import add_gaussian_noise
from utils.image_utils import load_cifar10_dataset

# Load data
train_imgs, test_imgs = load_cifar10_dataset()
test_img = test_imgs[0:1]

# Add noise
sigma = 2.0
noisy = add_gaussian_noise(test_img, sigma)

# Method 1: Ideal denoiser
denoised_ideal = ideal_denoiser(noisy, sigma, train_imgs)

# Method 2: EDM denoiser
model, _ = load_pretrained_edm('cifar10-uncond')
denoised_edm = edm_denoise(model, noisy, sigma)

# Compare results
```

### Workflow 2: Experiment with Different Noise Levels

```python
from denoisers.ideal_denoiser import ideal_denoiser
from utils.noise_utils import add_gaussian_noise
from utils.image_utils import load_cifar10_dataset, normalize_for_display
from torchvision.utils import save_image

# Load data
train_imgs, test_imgs = load_cifar10_dataset()

# Test multiple sigma values
sigmas = [0.5, 1.0, 2.0, 5.0, 10.0]

for sigma in sigmas:
    noisy = add_gaussian_noise(test_imgs[0:1], sigma)
    denoised = ideal_denoiser(noisy, sigma, train_imgs)
    save_image(normalize_for_display(denoised), f"result_sigma{sigma}.png")
```

## Dependencies

See `requirements.txt` for the full list. Key dependencies:

- `torch` - PyTorch for tensor operations
- `torchvision` - CIFAR-10 dataset and image utilities
- `matplotlib` - Visualization
- `numpy` - Numerical operations
- `tqdm` - Progress bars
- `dnnlib` (optional) - For EDM model downloads

## Testing

To verify the installation and setup:

```bash
# Test ideal denoiser
python -c "from denoisers.ideal_denoiser import ideal_denoiser; print('✓ Ideal denoiser OK')"

# Test utilities
python -c "from utils.noise_utils import add_gaussian_noise; print('✓ Utils OK')"

# Run full example
python example_usage.py
```

## Future Extensions

Possible additions to the framework:

1. **New Denoisers:**
   - BM3D
   - Non-local means
   - Wavelet-based methods
   - Other diffusion-based methods

2. **New Utilities:**
   - Different noise types (Poisson, salt-and-pepper)
   - Image quality metrics (PSNR, SSIM)
   - More datasets (ImageNet, custom datasets)

3. **Advanced Features:**
   - Training scripts for custom denoisers
   - Hyperparameter tuning utilities
   - Benchmarking framework
   - Web interface for visualization

## References

- **EDM Paper:** Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
- **EDM Repository:** https://github.com/NVlabs/edm
- **Paper Link:** https://arxiv.org/abs/2206.00364

## Contributing

When adding new features:

1. Follow the existing code style
2. Add comprehensive docstrings
3. Include usage examples
4. Update this documentation
5. Test thoroughly

## License

This project follows the same license as the EDM paper (CC BY-NC-SA 4.0).

