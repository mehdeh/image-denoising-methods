# Migration Guide: Modular Structure

This document explains the changes made to modularize the codebase and how to update existing code.

## What Changed?

The repository has been reorganized into a modular structure to:
1. Make it easier to add new denoising methods
2. Separate concerns (denoisers vs utilities)
3. Improve code reusability
4. Follow clean code principles

## New Structure

### Before (Old Structure)
```
image-denoising-methods/
├── generate_edm_figure1.py    # All code in one file
└── draft_codes/
    └── edm_denoiser_gradient.py
```

### After (New Modular Structure)
```
image-denoising-methods/
├── denoisers/                  # NEW: Denoising methods
│   ├── ideal_denoiser.py
│   └── edm_denoiser.py
├── utils/                      # NEW: Common utilities
│   ├── noise_utils.py
│   ├── image_utils.py
│   └── visualization.py
├── generate_edm_figure1.py     # UPDATED: Now uses modules
├── example_usage.py            # NEW: Comprehensive examples
└── test_modular_structure.py   # NEW: Tests
```

## Migration Examples

### Old Code (Before Modularization)

```python
# Old way - importing from generate_edm_figure1.py
from generate_edm_figure1 import (
    ideal_denoiser,
    add_gaussian_noise,
    load_cifar10_dataset,
    normalize_for_display
)

# Use functions
train, test = load_cifar10_dataset("./data")
noisy = add_gaussian_noise(test[0:1], sigma=2.0)
denoised = ideal_denoiser(noisy, 2.0, train)
```

### New Code (After Modularization)

```python
# New way - importing from organized modules
from denoisers.ideal_denoiser import ideal_denoiser
from utils.noise_utils import add_gaussian_noise
from utils.image_utils import load_cifar10_dataset, normalize_for_display

# Use functions (same as before)
train, test = load_cifar10_dataset("./data")
noisy = add_gaussian_noise(test[0:1], sigma=2.0)
denoised = ideal_denoiser(noisy, 2.0, train)
```

## Function Mapping

Here's where each function moved to:

| Old Location | New Location | Function |
|-------------|-------------|----------|
| `generate_edm_figure1.py` | `denoisers/ideal_denoiser.py` | `ideal_denoiser()` |
| `generate_edm_figure1.py` | `utils/noise_utils.py` | `add_gaussian_noise()` |
| `generate_edm_figure1.py` | `utils/image_utils.py` | `load_cifar10_dataset()` |
| `generate_edm_figure1.py` | `utils/image_utils.py` | `normalize_for_display()` |
| `generate_edm_figure1.py` | `utils/visualization.py` | `create_labeled_figure()` |
| `draft_codes/edm_denoiser_gradient.py` | `denoisers/edm_denoiser.py` | EDM functions |

## Key Changes to generate_edm_figure1.py

The main script `generate_edm_figure1.py` still works exactly the same way:

```bash
python generate_edm_figure1.py
```

**What changed internally:**
- Now imports from modular packages instead of defining functions inline
- Same functionality, cleaner code
- Easier to maintain and extend

## New Features

### 1. EDM Denoiser Module

A complete module for working with EDM pretrained models:

```python
from denoisers.edm_denoiser import load_pretrained_edm, edm_denoise

# Load pretrained model
model, config = load_pretrained_edm('cifar10-uncond')

# Denoise
denoised = edm_denoise(model, noisy_images, sigma=3.0)
```

### 2. Gradient Ascent Denoising

New functionality for iterative denoising:

```python
from denoisers.edm_denoiser import gradient_ascent_denoise

denoised, trajectory = gradient_ascent_denoise(
    model, noisy_img, sigma=3.0, num_steps=10, lr=1.0
)
```

### 3. Package-level Imports

You can now import directly from packages:

```python
# Instead of:
from denoisers.ideal_denoiser import ideal_denoiser
from utils.noise_utils import add_gaussian_noise

# You can do:
from denoisers import ideal_denoiser
from utils import add_gaussian_noise
```

## Backward Compatibility

**Important:** If you have existing code that imports from `generate_edm_figure1.py`, you need to update your imports.

### Quick Fix Script

If you have many files to update, here's a pattern:

```bash
# Find all files that import from generate_edm_figure1
grep -r "from generate_edm_figure1 import" .

# Update them to use new imports
# (manual update recommended for accuracy)
```

## Testing Your Migration

1. **Test imports:**
```bash
python test_modular_structure.py
```

2. **Test existing functionality:**
```bash
python generate_edm_figure1.py
```

3. **Try new examples:**
```bash
python example_usage.py
```

## Adding New Denoising Methods

The new structure makes it easy to add new methods:

1. Create `denoisers/my_method.py`
2. Implement your denoising function
3. Add to `denoisers/__init__.py`
4. Use common utilities from `utils/`

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for detailed instructions.

## Benefits of the New Structure

1. **Modularity**: Each component has a single responsibility
2. **Reusability**: Utilities are shared across methods
3. **Extensibility**: Easy to add new denoisers
4. **Testability**: Each module can be tested independently
5. **Documentation**: Better organized and discoverable
6. **Clean Code**: Follows software engineering best practices

## Need Help?

- See [`example_usage.py`](example_usage.py) for comprehensive examples
- Check [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for architecture details
- Run [`test_modular_structure.py`](test_modular_structure.py) to verify setup

## Summary

The modularization **does not break** the main functionality:
- ✅ `generate_edm_figure1.py` still works the same
- ✅ All functions are still available (just in new locations)
- ✅ New features added (EDM denoiser, gradient ascent, etc.)
- ✅ Better organization for future development

The main change is **where you import from**, not **what you import**.

