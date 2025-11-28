# Image Denoising Methods - Modular Framework

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2206.00364)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains a **modular framework** for implementing and comparing various image denoising methods, with a focus on methods from the paper:

> **Elucidating the Design Space of Diffusion-Based Generative Models**  
> Tero Karras, Miika Aittala, Timo Aila, Samuli Laine  
> NeurIPS 2022

## 📖 Overview

This framework provides:

1. **Ideal Denoiser**: Theoretical optimal denoiser using closed-form solution (Equation 57)
2. **EDM Denoiser**: Pretrained neural network-based denoiser from the EDM paper
3. **Modular Architecture**: Easy to extend with new denoising methods
4. **Common Utilities**: Shared tools for noise generation, data loading, and visualization

### Ideal Denoiser Formula

```
D(x; σ) = Σᵢ [xᵢ · exp(-||x - xᵢ||² / (2σ²))] / Σᵢ [exp(-||x - xᵢ||² / (2σ²))]
```

## 🎯 Features

- ✅ **Modular Design**: Easy to add new denoising methods
- ✅ **Clean Code**: Well-documented with comprehensive examples
- ✅ **Multiple Methods**: Ideal denoiser + EDM neural network denoiser
- ✅ **Reproduces EDM Figure 1**: Generate paper figures with one command
- ✅ **Efficient**: Optimized PyTorch implementation
- ✅ **Flexible**: CPU and GPU support, batch processing
- ✅ **Extensible**: Template for adding new methods

## 📁 Project Structure

```
image-denoising-methods/
├── denoisers/                   # Denoising method implementations
│   ├── __init__.py
│   ├── ideal_denoiser.py       # Ideal denoiser (Equation 57)
│   └── edm_denoiser.py         # EDM neural network denoiser
│
├── utils/                       # Common utilities
│   ├── __init__.py
│   ├── noise_utils.py          # Noise generation
│   ├── image_utils.py          # Data loading and processing
│   └── visualization.py        # Plotting and visualization
│
├── draft_codes/                 # Experimental code
├── data/                        # Dataset storage (auto-downloaded)
├── results/                     # Output directory
│
├── generate_edm_figure1.py     # Reproduce EDM Figure 1
├── example_usage.py            # Comprehensive usage examples
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── PROJECT_STRUCTURE.md        # Detailed architecture documentation
├── README_FIGURE1.md           # Figure 1 documentation
├── MATHEMATICAL_BACKGROUND.md  # Mathematical details
└── QUICKSTART.md               # Quick start guide
```

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for detailed architecture documentation.

## 🚀 Quick Start

### Installation

```bash
# Navigate to the repository
cd image-denoising-methods

# Install dependencies
pip install -r requirements.txt

# Optional: For EDM pretrained models
pip install git+https://github.com/NVlabs/edm.git
```

### Generate Figure 1 from EDM Paper

```bash
# Generate Figure 1 with default settings
python generate_edm_figure1.py
```

This will:
1. Download CIFAR-10 dataset (if needed)
2. Generate noisy images with σ = [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]
3. Denoise using the ideal denoiser
4. Save results to `./results/` directory

**Expected runtime:** ~5-10 minutes on CPU, ~2-3 minutes on GPU

### Run Usage Examples

```bash
# Run all examples demonstrating different features
python example_usage.py
```

This demonstrates:
- Ideal denoiser usage
- EDM pretrained model usage
- Gradient ascent denoising
- Batch processing
- Custom noise utilities

### Use as a Library

```python
# Import denoising methods
from denoisers.ideal_denoiser import ideal_denoiser
from denoisers.edm_denoiser import load_pretrained_edm, edm_denoise

# Import utilities
from utils.noise_utils import add_gaussian_noise
from utils.image_utils import load_cifar10_dataset

# Load data
train_imgs, test_imgs = load_cifar10_dataset(root="./data")

# Add noise
noisy = add_gaussian_noise(test_imgs[0:1], sigma=2.0)

# Denoise with ideal denoiser
denoised_ideal = ideal_denoiser(noisy, sigma=2.0, x_all=train_imgs)

# Denoise with EDM model (requires dnnlib)
model, _ = load_pretrained_edm('cifar10-uncond')
denoised_edm = edm_denoise(model, noisy, sigma=2.0)
```

## 📊 Results

The script generates three output images:

| File | Description |
|------|-------------|
| `figure1_noisy.png` | Grid of noisy images (2 rows × 11 columns) |
| `figure1_denoised.png` | Grid of denoised images (2 rows × 11 columns) |
| `figure1_combined.png` | Combined visualization with sigma labels |

### Example Output

The generated images show:
- **Row 1-2:** Two different CIFAR-10 test images
- **Columns:** Different noise levels (σ = 0 to 50)

Each image demonstrates how the ideal denoiser performs across varying noise levels.

## 💡 Usage Examples

### Example 1: Basic Ideal Denoiser

```python
from denoisers.ideal_denoiser import ideal_denoiser
from utils.noise_utils import add_gaussian_noise
from utils.image_utils import load_cifar10_dataset

# Load data
train_images, test_images = load_cifar10_dataset(root="./data")

# Add noise and denoise
sigma = 2.0
noisy_img = add_gaussian_noise(test_images[0:1], sigma)
denoised_img = ideal_denoiser(noisy_img, sigma, train_images)
```

### Example 2: EDM Pretrained Model

```python
from denoisers.edm_denoiser import load_pretrained_edm, edm_denoise
from utils.noise_utils import add_gaussian_noise

# Load pretrained model
model, config = load_pretrained_edm('cifar10-uncond')

# Denoise
noisy_img = add_gaussian_noise(test_images[0:1], sigma=3.0)
denoised_img = edm_denoise(model, noisy_img, sigma=3.0)
```

### Example 3: Gradient Ascent Denoising

```python
from denoisers.edm_denoiser import load_pretrained_edm, gradient_ascent_denoise

model, _ = load_pretrained_edm('cifar10-uncond')
denoised, trajectory = gradient_ascent_denoise(
    model, noisy_img, sigma=3.0, num_steps=10, lr=1.0
)
```

See [`example_usage.py`](example_usage.py) for more comprehensive examples.

## 🧪 Testing

Verify your installation:

```bash
# Test imports
python -c "from denoisers.ideal_denoiser import ideal_denoiser; print('✓ OK')"
python -c "from denoisers.edm_denoiser import load_edm_model; print('✓ OK')"
python -c "from utils.noise_utils import add_gaussian_noise; print('✓ OK')"

# Run comprehensive examples
python example_usage.py

# Generate Figure 1 (full test)
python generate_edm_figure1.py
```

## 🔧 Adding New Denoising Methods

The modular architecture makes it easy to add new methods:

### Step 1: Create a new denoiser module

```python
# denoisers/my_denoiser.py
"""
My Custom Denoiser Implementation
"""

import torch

def my_denoise(noisy_images, sigma, **kwargs):
    """
    Denoise images using my custom method.
    
    Parameters:
    -----------
    noisy_images : torch.Tensor
        Noisy images of shape (batch_size, C, H, W)
    sigma : float
        Noise level
        
    Returns:
    --------
    denoised : torch.Tensor
        Denoised images
    """
    # Your implementation here
    denoised = your_algorithm(noisy_images, sigma)
    return denoised
```

### Step 2: Register in `denoisers/__init__.py`

```python
from .my_denoiser import my_denoise
__all__.append('my_denoise')
```

### Step 3: Use it!

```python
from denoisers.my_denoiser import my_denoise
from utils.noise_utils import add_gaussian_noise

noisy = add_gaussian_noise(images, sigma=2.0)
denoised = my_denoise(noisy, sigma=2.0)
```

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for more details on extending the framework.

## 📝 Implementation Details

### Mathematical Formula

The ideal denoiser implements Equation 57 from the paper:

```
D(x; σ) = E[x' | x] where x = x' + n, n ~ N(0, σ²I)
```

### Numerical Stability

The implementation uses the **log-sum-exp trick** to prevent numerical overflow:

```python
# Compute log probabilities
log_probs = -||x - xᵢ||² / (2σ²)

# Subtract max for stability
delta = max(log_probs)
weights = exp(log_probs - delta)

# Weighted average
D(x; σ) = Σᵢ [weights_i · xᵢ] / Σᵢ [weights_i]
```

### Computational Complexity

- **Time complexity:** O(N × B × C × H × W)
  - N: number of training images (50,000 for CIFAR-10)
  - B: batch size
  - C × H × W: image dimensions (3 × 32 × 32 for CIFAR-10)

- **Memory complexity:** O(N × C × H × W)
  - Stores entire training set in memory

**Note:** This approach is only feasible for small datasets like CIFAR-10. For larger datasets (e.g., ImageNet), the ideal denoiser becomes computationally intractable.

## 🔧 Configuration

You can customize the generation by modifying `generate_edm_figure1.py`:

```python
# In main() function:
sigma_values = [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]  # Noise levels
test_indices = [20, 21]                                    # Test image indices
save_dir = "./results"                                     # Output directory
```

## 🎓 Background

### What is the Ideal Denoiser?

The ideal denoiser is a **theoretical upper bound** on denoising performance. It assumes:
1. Access to the entire training distribution
2. Perfect knowledge of the noise level σ
3. Ability to compute exact expectations

In practice, neural networks trained as denoisers approximate this ideal denoiser.

### Why CIFAR-10?

The paper states:
> "Note that Eq. 57 is feasible to compute in practice for small datasets—we show the results for CIFAR-10 in Figure 1b."

CIFAR-10 is small enough (50,000 training images of 32×32 pixels) to compute the ideal denoiser for all test images and all sigma values in reasonable time.

## 📚 Related Files

- **`README_FIGURE1.md`**: Detailed documentation for Figure 1 generation
- **`edm_denoiser_gradient.py`**: Related work on EDM denoiser gradients
- **Jupyter notebooks**: Interactive exploration of the concepts

## 🤝 Acknowledgments

This implementation is based on:
- Original EDM paper by Karras et al. (NeurIPS 2022)
- Equation 57 from Appendix B.3 of the paper
- Discussions in [EDM GitHub Issue #26](https://github.com/NVlabs/edm/issues/26)

## 📖 Citation

If you use this code, please cite the original EDM paper:

```bibtex
@inproceedings{Karras2022edm,
  author    = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
  title     = {Elucidating the Design Space of Diffusion-Based Generative Models},
  booktitle = {Proc. NeurIPS},
  year      = {2022}
}
```

## 📄 License

This project is provided for research and educational purposes. The original EDM paper and code are licensed under CC BY-NC-SA 4.0.

## 🐛 Issues & Contributions

If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## 📞 Contact

For questions about this implementation, please refer to:
- Original EDM repository: https://github.com/NVlabs/edm
- Paper: https://arxiv.org/abs/2206.00364

---

**Note:** This is an independent implementation based on the mathematical formulas in the paper. The original EDM repository does not include code for generating Figure 1.

