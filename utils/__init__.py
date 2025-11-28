"""
Utility modules for image processing and visualization.

This package contains common utilities used across different denoising methods:
- noise_utils: Functions for adding noise to images
- image_utils: Image loading, normalization, and processing
- visualization: Visualization and plotting utilities
"""

from .noise_utils import add_gaussian_noise
from .image_utils import load_cifar10_dataset, normalize_for_display
from .visualization import create_labeled_figure

__all__ = [
    'add_gaussian_noise',
    'load_cifar10_dataset',
    'normalize_for_display',
    'create_labeled_figure'
]

