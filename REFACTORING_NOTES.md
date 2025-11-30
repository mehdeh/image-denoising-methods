# Refactoring Notes - Modular Code Improvements

## Overview

The codebase has been refactored to improve modularity and code organization by moving shared functionality to the `utils/` folder.

## Changes Made

### 1. New Module: `utils/model_utils.py`

**Purpose:** Handle model downloading and management

**Functions:**
- `download_file(url, destination)` - Download files with progress bar using urllib
- `ensure_model_downloaded(model_path, url)` - Check if model exists, download if needed

**Benefits:**
- No longer requires `dnnlib` for model downloads
- Uses standard Python libraries (`urllib`) for better compatibility
- Provides progress bar for large downloads
- Cleaner error handling

### 2. Enhanced: `utils/visualization.py`

**Added:**
- `create_comparison_figure()` - Create 3-row comparison visualizations
  - Row 1: Noisy images
  - Row 2: Ideal denoiser results
  - Row 3: EDM denoiser results

**Benefits:**
- Reusable visualization function
- Consistent styling across scripts
- Reduced code duplication

### 3. Updated: `utils/__init__.py`

**Exported functions:**
```python
from .noise_utils import add_gaussian_noise
from .image_utils import load_cifar10_dataset, load_cifar10_subset, normalize_for_display
from .visualization import create_labeled_figure, create_comparison_figure
from .model_utils import download_file, ensure_model_downloaded
```

### 4. Refactored: `compare_denoisers.py`

**Improvements:**
- Modular function structure
- Clear separation of concerns:
  - `process_images_at_sigma()` - Process single noise level
  - `generate_denoiser_comparison()` - Main comparison logic
  - `load_edm_model()` - Model loading with error handling
  - `setup_data_subsets()` - Data loading logic
  - `main()` - Orchestration only
- Configuration dictionary for easy parameter adjustment
- Better error messages and user guidance
- Uses utility functions from `utils/` folder

### 5. Updated: `denoisers/edm_denoiser.py`

**Changes:**
- Now uses `utils.model_utils` for downloading models
- Falls back to `dnnlib` if `model_utils` is not available
- Better error handling and user messages

## Code Organization

```
utils/
├── __init__.py              # Exports all utility functions
├── noise_utils.py           # Noise generation
├── image_utils.py           # Image loading and processing
├── visualization.py         # Visualization functions (2-row and 3-row)
└── model_utils.py           # Model downloading (NEW)
```

## Clean Code Practices Applied

✅ **Single Responsibility Principle**
- Each function has one clear purpose
- Small, focused functions

✅ **DRY (Don't Repeat Yourself)**
- Shared visualization code moved to utilities
- Reusable model download function

✅ **Descriptive Naming**
- Clear function and variable names
- Type hints for better readability

✅ **Documentation**
- Comprehensive docstrings in English
- Usage examples in docstrings
- Inline comments where needed

✅ **Error Handling**
- Graceful error handling with helpful messages
- Clear installation instructions

✅ **Modularity**
- Easy to test individual components
- Easy to extend with new functionality
- Loose coupling between modules

## Usage Example

### Before (in main script):
```python
# All visualization code embedded in main script
fig, axes = plt.subplots(3, 1, figsize=(20, 12))
# ... 50+ lines of plotting code ...
```

### After (using utilities):
```python
# Clean and simple
from utils.visualization import create_comparison_figure

create_comparison_figure(
    noisy_grid, ideal_grid, edm_grid,
    sigma_values, save_path, num_sigmas
)
```

## Installation and Running

### Install EDM Dependencies (Required)

```bash
pip install git+https://github.com/NVlabs/edm.git
```

### Run the Comparison Script

```bash
cd /home/ubuntu/repos/image-denoising-methods
conda activate myenv
python compare_denoisers.py
```

**What happens:**
1. Model file is automatically downloaded (~226MB) if not present
2. CIFAR-10 dataset is downloaded if needed
3. Comparison visualizations are generated
4. Results saved to `./results/denoiser_comparison/`

## Benefits of This Refactoring

1. **Maintainability**: Easier to update and fix bugs
2. **Testability**: Individual functions can be tested in isolation
3. **Reusability**: Utility functions can be used in other scripts
4. **Readability**: Main script is cleaner and easier to understand
5. **Extensibility**: Easy to add new denoisers or visualization types
6. **Reliability**: Better error handling and user feedback

## Migration Guide

If you have existing scripts using the old structure:

### Old import:
```python
# Old: Everything in main script
def create_comparison_figure(...):
    # 50+ lines of code
```

### New import:
```python
# New: Import from utilities
from utils.visualization import create_comparison_figure
from utils.model_utils import ensure_model_downloaded

# Use with one line
create_comparison_figure(noisy, ideal, edm, sigmas, path, n)
```

## Notes

- The EDM model pickle file requires `torch_utils` from the EDM repository
- First run will download ~226MB model file
- Model download uses standard `urllib` (no external dependencies)
- All utility functions have comprehensive docstrings and examples

