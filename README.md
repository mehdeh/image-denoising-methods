# Image Denoising Methods Comparison

A framework for comparing different image denoising methods on CIFAR-10 dataset.

## 📖 Overview

This repository provides a comparison framework for evaluating multiple image denoising methods:

1. **Ideal Denoiser** - Theoretical optimal denoiser using closed-form solution (Equation 57 from EDM paper)
2. **EDM Denoiser** - Pretrained neural network-based denoiser from the EDM paper (one-step)
3. **Gradient Ascent Denoiser** - Iterative denoising using score function optimization

The goal is to visually compare the performance of different denoising approaches across various noise levels.

## 📁 Project Structure

```
image-denoising-methods/
├── ideal_denoiser/              # Ideal denoiser module
│   └── ideal_denoiser.py       # Implementation from ideal-denoiser repository
│
├── edm_denoiser/                # EDM neural network denoiser package
│   ├── __init__.py             # Package exports
│   ├── core.py                 # Main denoising function
│   ├── model_loader.py         # Model loading and downloading
│   ├── score.py                # Score function and gradient ascent
│   ├── utils.py                # Helper utilities
│   └── edm/                    # EDM dependencies (CC BY-NC-SA 4.0)
│       ├── dnnlib/             # Deep learning utilities from NVlabs/edm
│       ├── torch_utils/        # PyTorch utilities from NVlabs/edm
│       ├── LICENSE.txt         # EDM license information
│       └── NOTICE.txt          # Attribution and citation
│
├── utils/                       # Common utilities
│   ├── __init__.py
│   ├── noise_utils.py          # Noise generation
│   ├── image_utils.py          # Data loading and processing
│   ├── model_utils.py          # Model download utilities
│   └── visualization.py        # Plotting and visualization
│
├── data/                        # Dataset storage (auto-downloaded)
├── pretrain_models/             # Pretrained EDM models (auto-downloaded)
├── results/                     # Output directory for comparison results
│
├── compare_denoisers.py        # Main script to compare all denoising methods
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Navigate to the repository
cd image-denoising-methods

# Install dependencies
pip install -r requirements.txt
```

### Run Comparison

```bash
# Compare all three denoising methods side-by-side
python compare_denoisers.py
```

This will:
1. Download CIFAR-10 dataset (if needed)
2. Load pretrained EDM model (downloads ~226MB on first run)
3. Generate noisy images at multiple noise levels
4. Denoise using all three methods
5. Create comparison visualizations
6. Save results to `./results/` directory

**Expected runtime:** ~30-40 minutes on CPU, ~8-10 minutes on GPU

**Output files:**
- `comparison_train.png` - Comparison for training images
- `comparison_test.png` - Comparison for test images

Each figure shows:
- **Row 1:** Noisy images at different noise levels
- **Row 2:** Ideal denoiser results (closed-form solution)
- **Row 3:** EDM denoiser results (one-step neural network)
- **Row 4:** Gradient ascent denoiser (iterative optimization, 10 steps)

## 📊 Results

### Training Set Comparison

<img src="results/comparison_train.png" width="40%">

### Test Set Comparison

<img src="results/comparison_test.png" width="40%">

The comparison visualizations show how each denoising method performs across different noise levels (σ = 0, 0.2, 0.5, 1, 2, 3, 5). The ideal denoiser represents the theoretical upper bound, while EDM and gradient ascent denoisers show practical learned approaches.

## 🔧 Configuration

You can customize the comparison by modifying the configuration in `compare_denoisers.py`:

```python
config = {
    'data_root': "./data",
    'save_dir': "./results",
    'sigma_values': [0, 0.2, 0.5, 1, 2, 3, 5],
    'max_samples_for_selection': 10,
    'train_selection_indices': [2, 3, 4],
    'test_selection_indices': [2, 3, 4],
    'ideal_denoiser_subset_size': 1000,
    'grad_ascent_steps': 10,
    'grad_ascent_lr': 1.0
}
```

## 📚 References

### Ideal Denoiser
- **Source:** [ideal-denoiser](https://github.com/mehdeh/ideal-denoiser) repository
- **Based on:** Equation 57 from EDM paper
- **License:** Free to use without restrictions

### EDM (Elucidating the Design Space of Diffusion-Based Generative Models)
- **Paper:** [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- **Authors:** Tero Karras, Miika Aittala, Timo Aila, Samuli Laine
- **Conference:** NeurIPS 2022
- **Official Repository:** [NVlabs/edm](https://github.com/NVlabs/edm)

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

### Main Project Code
The comparison framework and utilities are provided freely without restrictions.

### EDM Components (`edm_denoiser/edm/`)
The EDM components in `edm_denoiser/edm/` directory are from [NVlabs/edm](https://github.com/NVlabs/edm):

- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Copyright:** © 2022, NVIDIA CORPORATION & AFFILIATES
- **Usage:** Free for non-commercial research and educational purposes

For full license details, see `edm_denoiser/edm/LICENSE.txt` or visit http://creativecommons.org/licenses/by-nc-sa/4.0/

### Ideal Denoiser
The ideal denoiser module (`ideal_denoiser/ideal_denoiser.py`) is from the [ideal-denoiser](https://github.com/mehdeh/ideal-denoiser) repository and is provided freely without license restrictions.

## 🐛 Issues & Contributions

If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

---

**Note:** This framework focuses on visual comparison of different denoising methods. For detailed implementation of the ideal denoiser, see the [ideal-denoiser](https://github.com/mehdeh/ideal-denoiser) repository. For the full EDM implementation, see the [official EDM repository](https://github.com/NVlabs/edm).
