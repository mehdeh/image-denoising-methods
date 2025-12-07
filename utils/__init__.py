"""
Utility modules for image processing and visualization.

This package contains common utilities used across different denoising methods:
- noise_utils: Gaussian noise generation for images
- image_utils: CIFAR-10 data loading, normalization, and processing
- visualization: Publication-quality figure generation for denoising comparisons
- model_utils: Pretrained model downloading and management utilities (EDM models)

The structure is modular to facilitate testing and reuse across different denoising methods.
"""

# Core utilities (matching ideal-denoiser structure)
from .noise_utils import add_gaussian_noise
from .image_utils import (
    load_cifar10_dataset,
    load_cifar10_subset,
    normalize_for_display
)
from .visualization import (
    create_labeled_figure,
    create_comparison_figure
)

# Additional utilities specific to multi-denoiser comparison
from .model_utils import (
    download_file,
    ensure_model_downloaded
)

__all__ = [
    # Noise utilities
    'add_gaussian_noise',
    
    # Image utilities
    'load_cifar10_dataset',
    'load_cifar10_subset',
    'normalize_for_display',
    
    # Visualization utilities
    'create_labeled_figure',
    'create_comparison_figure',
    
    # Model utilities
    'download_file',
    'ensure_model_downloaded'
]

