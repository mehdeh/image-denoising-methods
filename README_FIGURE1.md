# EDM Figure 1 Reproduction

This code reproduces **Figure 1** from the paper:

> **Elucidating the Design Space of Diffusion-Based Generative Models**  
> Tero Karras, Miika Aittala, Timo Aila, Samuli Laine  
> NeurIPS 2022  
> Paper: https://arxiv.org/abs/2206.00364  
> GitHub: https://github.com/NVlabs/edm

## What is Figure 1?

Figure 1 in the EDM paper demonstrates the **ideal denoiser** performance on CIFAR-10 dataset. The ideal denoiser uses a closed-form solution (Equation 57 in Appendix B.3) to denoise images corrupted with different levels of Gaussian noise.

The ideal denoiser formula is:

```
D(x; σ) = Σᵢ [xᵢ · exp(-||x - xᵢ||² / (2σ²))] / Σᵢ [exp(-||x - xᵢ||² / (2σ²))]
```

where:
- `x` is the noisy input
- `σ` is the noise level
- `xᵢ` are samples from the training set (CIFAR-10 training data)

## Installation

### Requirements

```bash
pip install torch torchvision matplotlib numpy tqdm
```

Or using conda:

```bash
conda install pytorch torchvision matplotlib numpy tqdm -c pytorch
```

## Usage

### Basic Usage

Simply run the script:

```bash
python generate_edm_figure1.py
```

This will:
1. Download CIFAR-10 dataset (if not already downloaded)
2. Load training and test sets
3. Select 2 test images (indices 20, 21)
4. Add Gaussian noise with σ = [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]
5. Denoise using the ideal denoiser
6. Save results to `./results/` directory

### Output Files

The script generates three output images:

1. **`figure1_noisy.png`**: Grid of noisy images (2 rows × 11 columns)
2. **`figure1_denoised.png`**: Grid of denoised images (2 rows × 11 columns)
3. **`figure1_combined.png`**: Combined visualization with sigma labels

### Customization

You can modify the configuration in the `main()` function:

```python
# Configuration
data_root = "./data"              # Where CIFAR-10 data is stored
save_dir = "./results"            # Where to save output images
sigma_values = [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]  # Noise levels
test_indices = [20, 21]           # Which test images to use
```

## Code Structure

### Main Functions

- **`load_cifar10_dataset()`**: Loads CIFAR-10 training and test sets
- **`ideal_denoiser()`**: Implements Equation 57 from the paper
- **`add_gaussian_noise()`**: Adds Gaussian noise to images
- **`normalize_for_display()`**: Normalizes images to [0, 1] for visualization
- **`generate_figure1()`**: Main function that generates the figure
- **`create_labeled_figure()`**: Creates a combined visualization with labels

### Implementation Details

The ideal denoiser implementation follows these steps:

1. **Compute distances**: Calculate L2 distance between noisy input and all training images
2. **Compute weights**: Use Gaussian kernel `exp(-||x - xᵢ||² / (2σ²))`
3. **Numerical stability**: Apply log-sum-exp trick to avoid overflow
4. **Weighted average**: Compute weighted average of training images

## Notes

- The computation is feasible for CIFAR-10 because it's a small dataset (50,000 training images)
- For larger datasets, the ideal denoiser becomes computationally intractable
- The script uses CPU by default; it will automatically use GPU if available
- Processing all sigma values takes approximately 5-10 minutes on CPU

## Comparison with Original Paper

The original EDM repository (https://github.com/NVlabs/edm) does **not** include code for generating Figure 1. This implementation is based on:

1. The mathematical formula in Appendix B.3 (Equation 57)
2. The description in the paper: *"Note that Eq. 57 is feasible to compute in practice for small datasets—we show the results for CIFAR-10 in Figure 1b."*
3. Related discussions in [GitHub Issue #26](https://github.com/NVlabs/edm/issues/26)

## Citation

If you use this code, please cite the original EDM paper:

```bibtex
@inproceedings{Karras2022edm,
  author    = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
  title     = {Elucidating the Design Space of Diffusion-Based Generative Models},
  booktitle = {Proc. NeurIPS},
  year      = {2022}
}
```

## License

This implementation is provided for research and educational purposes. The original EDM paper and code are licensed under CC BY-NC-SA 4.0.

