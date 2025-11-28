# EDM Ideal Denoiser Implementation

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2206.00364)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains a clean implementation of the **ideal denoiser** from the paper:

> **Elucidating the Design Space of Diffusion-Based Generative Models**  
> Tero Karras, Miika Aittala, Timo Aila, Samuli Laine  
> NeurIPS 2022

## 📖 Overview

The ideal denoiser is a theoretical denoiser that uses the entire training dataset to denoise images. It computes the exact expected value of clean images given noisy observations, using a closed-form solution (Equation 57 from the paper):

```
D(x; σ) = Σᵢ [xᵢ · exp(-||x - xᵢ||² / (2σ²))] / Σᵢ [exp(-||x - xᵢ||² / (2σ²))]
```

This implementation reproduces **Figure 1** from the EDM paper, demonstrating the ideal denoiser's performance on CIFAR-10.

## 🎯 Features

- ✅ Clean, well-documented implementation of Equation 57
- ✅ Reproduces Figure 1 from the paper
- ✅ Comprehensive unit tests
- ✅ Multiple usage examples
- ✅ Efficient computation with PyTorch
- ✅ Support for both CPU and GPU
- ✅ Batch processing support

## 📁 Project Structure

```
ideal-and-edm-denoiser/
├── generate_edm_figure1.py      # Main script to generate Figure 1
├── test_ideal_denoiser.py       # Unit tests for ideal_denoiser function
├── test_figure1_quick.py        # Quick test with reduced dataset
├── example_usage.py             # Various usage examples
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── README_FIGURE1.md            # Detailed documentation for Figure 1
```

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the repository
cd ideal-and-edm-denoiser

# Install dependencies
pip install -r requirements.txt
```

### Generate Figure 1

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

### Quick Test

For a faster test with reduced dataset:

```bash
# Quick test with only 1000 training images
python test_figure1_quick.py
```

**Expected runtime:** ~1-2 minutes

### Unit Tests

Verify the implementation:

```bash
# Run unit tests
python test_ideal_denoiser.py
```

All tests should pass ✓

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

See `example_usage.py` for various usage patterns:

```python
from generate_edm_figure1 import ideal_denoiser, add_gaussian_noise

# Load data
train_images, test_images = load_cifar10_dataset(root="./data")

# Select a test image
test_img = test_images[0:1]  # Shape: (1, 3, 32, 32)

# Add noise
sigma = 2.0
noisy_img = add_gaussian_noise(test_img, sigma)

# Denoise
denoised_img = ideal_denoiser(noisy_img, sigma, train_images)
```

Run examples:

```bash
python example_usage.py
```

## 🧪 Testing

The repository includes comprehensive tests:

### 1. Unit Tests
```bash
python test_ideal_denoiser.py
```

Tests include:
- ✅ Basic functionality
- ✅ Zero noise case
- ✅ High noise case
- ✅ Batch processing

### 2. Quick Integration Test
```bash
python test_figure1_quick.py
```

### 3. Full Figure Generation
```bash
python generate_edm_figure1.py
```

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

