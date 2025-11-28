"""
Visualization utilities for denoising experiments.

This module provides functions for creating visualizations, plots, and grids
of images for comparing different denoising methods and noise levels.
"""

import torch
import matplotlib.pyplot as plt
import os


def create_labeled_figure(noisy_grid, denoised_grid, sigma_values, save_dir):
    """
    Create a combined figure with labels for sigma values.
    
    This function creates a publication-quality figure showing both noisy and
    denoised images in a grid format with labeled sigma values.
    
    Parameters:
    -----------
    noisy_grid : torch.Tensor
        Grid of noisy images (C, H, W) after make_grid
    denoised_grid : torch.Tensor
        Grid of denoised images (C, H, W) after make_grid
    sigma_values : list
        List of sigma values used for noise levels
    save_dir : str
        Directory to save the figure
        
    Examples:
    ---------
    >>> import torch
    >>> from torchvision.utils import make_grid
    >>> from utils.visualization import create_labeled_figure
    >>> 
    >>> # Create sample grids
    >>> images = torch.randn(22, 3, 32, 32)  # 2 images × 11 sigma values
    >>> noisy_grid = make_grid(images, nrow=11, padding=2, pad_value=1.0)
    >>> denoised_grid = make_grid(images, nrow=11, padding=2, pad_value=1.0)
    >>> 
    >>> # Create labeled figure
    >>> sigma_values = [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]
    >>> create_labeled_figure(noisy_grid, denoised_grid, sigma_values, "./results")
    """
    fig, axes = plt.subplots(2, 1, figsize=(20, 8))
    
    # Convert grids to numpy
    noisy_np = noisy_grid.permute(1, 2, 0).cpu().numpy()
    denoised_np = denoised_grid.permute(1, 2, 0).cpu().numpy()
    
    # Plot noisy images
    axes[0].imshow(noisy_np)
    axes[0].set_title("Noisy Images (x + σ·ε, where ε ~ N(0, I))", fontsize=14, pad=10)
    axes[0].axis('off')
    
    # Plot denoised images
    axes[1].imshow(denoised_np)
    axes[1].set_title("Ideal Denoiser Output D(x; σ) - Eq. 57", fontsize=14, pad=10)
    axes[1].axis('off')
    
    # Add sigma labels at the top
    num_sigmas = len(sigma_values)
    for idx, sigma in enumerate(sigma_values):
        x_pos = (idx + 0.5) / num_sigmas
        fig.text(x_pos, 0.98, f'σ={sigma}', ha='center', va='top', fontsize=10, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    combined_path = os.path.join(save_dir, "figure1_combined.png")
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined figure to: {combined_path}")

