# Quick Start Guide

This guide will get you up and running with the EDM Ideal Denoiser in 5 minutes.

## ⚡ 30-Second Quickstart

```bash
# 1. Install dependencies
pip install torch torchvision matplotlib numpy tqdm

# 2. Run unit tests (verify implementation)
python test_ideal_denoiser.py

# 3. Generate Figure 1 (full version - takes ~5 min)
python generate_edm_figure1.py
```

## 📋 Step-by-Step Guide

### Step 1: Setup Environment

```bash
# Navigate to project directory
cd ideal-and-edm-denoiser

# Install dependencies
pip install -r requirements.txt

# Or with conda:
conda install pytorch torchvision matplotlib numpy tqdm -c pytorch
```

### Step 2: Verify Installation

Run the unit tests to ensure everything is working:

```bash
python test_ideal_denoiser.py
```

**Expected output:**
```
======================================================================
Testing ideal_denoiser Implementation
======================================================================

Test 1: Basic functionality
--------------------------------------------------
✓ Shape test passed
✓ Denoising reduces MSE

Test 2: Zero noise case
--------------------------------------------------
✓ Zero noise test passed

Test 3: High noise case
--------------------------------------------------
✓ High noise test passed

Test 4: Batch processing
--------------------------------------------------
✓ Batch processing test passed

======================================================================
All tests completed!
======================================================================
```

### Step 3: Quick Test (Optional)

For a faster test with reduced dataset:

```bash
python test_figure1_quick.py
```

This uses only 1,000 training images and takes ~1-2 minutes.

### Step 4: Generate Figure 1

Run the main script to generate Figure 1 from the EDM paper:

```bash
python generate_edm_figure1.py
```

**What happens:**
1. Downloads CIFAR-10 dataset (~170 MB) if not already present
2. Loads 50,000 training images
3. Selects 2 test images
4. Adds noise with 11 different sigma values
5. Denoises using the ideal denoiser
6. Saves results to `./results/`

**Time:** ~5-10 minutes on CPU, ~2-3 minutes on GPU

**Output files:**
- `results/figure1_noisy.png` - Grid of noisy images
- `results/figure1_denoised.png` - Grid of denoised images
- `results/figure1_combined.png` - Combined visualization with labels

### Step 5: Explore Examples

Try different usage patterns:

```bash
python example_usage.py
```

This runs various examples showing how to use the ideal denoiser programmatically.

## 🎯 What You'll Get

### Figure 1: Noisy Images
```
σ=0    σ=0.2  σ=0.5  σ=1    σ=2    σ=3    σ=5    σ=7    σ=10   σ=20   σ=50
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 🚗│ 🚗│ 🚗│ 🚗│ 🚗│ 🚗│ 🚗│ 📰│ 📰│ 📰│ 📰│  Image 1
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 🐸│ 🐸│ 🐸│ 🐸│ 🐸│ 🐸│ 🐸│ 📰│ 📰│ 📰│ 📰│  Image 2
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

### Figure 1: Denoised Images
```
σ=0    σ=0.2  σ=0.5  σ=1    σ=2    σ=3    σ=5    σ=7    σ=10   σ=20   σ=50
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 🚗│ 🚗│ 🚗│ 🚗│ 🚗│ 🚗│ 🚗│ 🚗│ 🚗│ 🚗│ 🖼️│  Image 1 (denoised)
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 🐸│ 🐸│ 🐸│ 🐸│ 🐸│ 🐸│ 🐸│ 🐸│ 🐸│ 🐸│ 🖼️│  Image 2 (denoised)
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

*(Emojis are for illustration - actual output shows CIFAR-10 images)*

## 🔧 Troubleshooting

### Issue: Out of Memory

**Problem:** GPU runs out of memory during processing.

**Solution:** Use CPU instead:
```python
# In generate_edm_figure1.py, line ~272:
device = 'cpu'  # Force CPU usage
```

Or reduce batch size by processing one image at a time.

### Issue: Slow Execution

**Problem:** Script takes too long to run.

**Solutions:**

1. **Use GPU:** If you have a GPU, the script will automatically use it.

2. **Reduce training set size:**
```python
# In generate_edm_figure1.py, modify generate_figure1():
train_images = train_images[:10000]  # Use only 10,000 images
```

3. **Reduce sigma values:**
```python
# In main():
sigma_values = [0, 1, 3, 10]  # Use only 4 sigma values
```

### Issue: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:** Install dependencies:
```bash
pip install torch torchvision matplotlib numpy tqdm
```

### Issue: CIFAR-10 Download Fails

**Problem:** Network issues downloading CIFAR-10.

**Solution:** Download manually from https://www.cs.toronto.edu/~kriz/cifar.html and place in `./data/cifar-10-batches-py/`

## 📚 Next Steps

After the quick start, explore:

1. **Read the README:** `README.md` for full documentation
2. **Mathematical background:** `MATHEMATICAL_BACKGROUND.md` for theory
3. **Figure 1 details:** `README_FIGURE1.md` for detailed explanation
4. **Usage examples:** `example_usage.py` for various use cases

## 💡 Usage Examples

### Example 1: Denoise a Single Image

```python
from generate_edm_figure1 import (
    load_cifar10_dataset, 
    ideal_denoiser, 
    add_gaussian_noise
)

# Load data
train_images, test_images = load_cifar10_dataset()

# Select test image
test_img = test_images[0:1]  # Shape: (1, 3, 32, 32)

# Add noise
sigma = 2.0
noisy_img = add_gaussian_noise(test_img, sigma)

# Denoise
denoised_img = ideal_denoiser(noisy_img, sigma, train_images)
```

### Example 2: Batch Processing

```python
# Denoise multiple images at once
test_imgs = test_images[0:10]  # 10 images
sigma = 3.0

noisy_imgs = add_gaussian_noise(test_imgs, sigma)
denoised_imgs = ideal_denoiser(noisy_imgs, sigma, train_images)
```

### Example 3: Custom Sigma Values

```python
from generate_edm_figure1 import generate_figure1

# Generate figure with custom sigma values
generate_figure1(
    train_images=train_images,
    test_images=test_images,
    sigma_values=[0.1, 0.5, 1.0, 5.0, 20.0],  # Custom sigmas
    test_indices=[0, 10],
    save_dir="./my_results",
    device='cuda'
)
```

## 🎓 Understanding the Output

### Noisy Images

Shows how Gaussian noise affects images at different noise levels (sigma):
- **σ = 0:** No noise (original image)
- **σ = 1-3:** Mild noise (image still recognizable)
- **σ = 5-10:** Moderate noise (image degraded)
- **σ = 20-50:** Severe noise (image mostly destroyed)

### Denoised Images

Shows how the ideal denoiser recovers images:
- **Low sigma:** Nearly perfect recovery
- **Medium sigma:** Good recovery
- **High sigma:** Partial recovery (converges to dataset mean)

## 📊 Performance Expectations

| Dataset Size | Sigma Values | Time (CPU) | Time (GPU) |
|-------------|-------------|-----------|-----------|
| 1,000 images | 4 sigmas | ~30 sec | ~10 sec |
| 10,000 images | 4 sigmas | ~3 min | ~1 min |
| 50,000 images | 4 sigmas | ~15 min | ~5 min |
| 50,000 images | 11 sigmas | ~40 min | ~12 min |

*Timings are approximate and depend on hardware.*

## ✅ Checklist

Before running:
- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] At least 2 GB free disk space (for CIFAR-10 dataset)
- [ ] At least 4 GB free RAM (8 GB recommended)

## 🚀 Advanced Usage

For advanced users, see:
- `example_usage.py` - Various usage patterns
- `MATHEMATICAL_BACKGROUND.md` - Mathematical theory
- `README.md` - Complete documentation

## ❓ FAQ

**Q: Can I use this with other datasets?**  
A: Yes! Modify `load_cifar10_dataset()` to load your dataset. The ideal denoiser works with any image dataset.

**Q: How do I change the test images?**  
A: Modify `test_indices` in `main()` function of `generate_edm_figure1.py`.

**Q: Can I run this without a GPU?**  
A: Yes! The script works on CPU, just slower.

**Q: Why does it take so long?**  
A: The ideal denoiser computes distances to all training images for each noisy image. This is computationally expensive but necessary for the exact solution.

---

**Ready to start?** Run `python test_ideal_denoiser.py` to begin!

