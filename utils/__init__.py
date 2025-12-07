"""
Utility modules for image processing and visualization.

This package contains common utilities used across different denoising methods:
- core: Core utilities (noise generation, image loading, normalization)
- visualization: Visualization and plotting utilities
- processing: Image processing pipelines for denoising experiments
"""

from .core import (
    add_gaussian_noise,
    load_cifar10_dataset,
    load_cifar10_subset,
    normalize_for_display
)
from .visualization import create_labeled_figure, create_comparison_figure
from .processing import process_images_at_sigma, generate_denoiser_comparison

__all__ = [
    # Core utilities
    'add_gaussian_noise',
    'load_cifar10_dataset',
    'load_cifar10_subset',
    'normalize_for_display',
    
    # Visualization utilities
    'create_labeled_figure',
    'create_comparison_figure',
    
    # Processing utilities
    'process_images_at_sigma',
    'generate_denoiser_comparison'
]
