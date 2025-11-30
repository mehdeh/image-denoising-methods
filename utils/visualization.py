"""
Visualization utilities for denoising experiments.

This module provides functions for creating visualizations, plots, and grids
of images for comparing different denoising methods and noise levels.
"""

import torch
import matplotlib.pyplot as plt
import os


def create_labeled_figure(noisy_grid, denoised_grid, sigma_values, save_path, num_sigmas):
    """
    Create a combined figure with labels for sigma values.
    
    This function creates a publication-quality figure showing both noisy and
    denoised images in a grid format with labeled sigma values. The sigma labels
    are aligned with the actual image columns.
    
    Parameters:
    -----------
    noisy_grid : torch.Tensor
        Grid of noisy images (C, H, W) after make_grid
    denoised_grid : torch.Tensor
        Grid of denoised images (C, H, W) after make_grid
    sigma_values : list
        List of sigma values used for noise levels
    save_path : str
        Full path to save the figure
    num_sigmas : int
        Number of sigma values (columns in the grid)
    """
    fig, axes = plt.subplots(2, 1, figsize=(20, 8))
    
    # Convert grids to numpy
    noisy_np = noisy_grid.permute(1, 2, 0).cpu().numpy()
    denoised_np = denoised_grid.permute(1, 2, 0).cpu().numpy()
    
    # Plot noisy images
    axes[0].imshow(noisy_np, aspect='auto')
    axes[0].set_title("Noisy Images (x + σ·ε, where ε ~ N(0, I))", fontsize=14, pad=10)
    axes[0].axis('off')
    
    # Plot denoised images
    axes[1].imshow(denoised_np, aspect='auto')
    axes[1].set_title("Ideal Denoiser Output D(x; σ) - Eq. 57", fontsize=14, pad=10)
    axes[1].axis('off')
    
    # Apply tight_layout first to get final axes positions
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Add sigma labels at the top, aligned with actual image columns
    # Get the position of the axes in figure coordinates after tight_layout
    bbox = axes[0].get_position()
    for idx, sigma in enumerate(sigma_values):
        # Calculate position of each column center in figure coordinates
        # Each column takes up (1/num_sigmas) of the axes width
        x_pos_fig = bbox.x0 + (idx + 0.5) / num_sigmas * bbox.width
        fig.text(x_pos_fig, 0.98, f'σ={sigma}', ha='center', va='top', fontsize=10, weight='bold')
    
    # Save figure (bbox_inches='tight' may change layout, so we use it carefully)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved combined figure to: {save_path}")


def create_comparison_figure(
    noisy_grid: torch.Tensor,
    ideal_grid: torch.Tensor,
    edm_grid: torch.Tensor,
    sigma_values: list,
    save_path: str,
    num_sigmas: int
) -> None:
    """
    Create a 3-row comparison figure with labels for sigma values.
    
    This function creates a publication-quality figure showing noisy images,
    ideal denoiser results, and EDM denoiser results in a grid format with
    labeled sigma values aligned with image columns.
    
    Parameters:
    -----------
    noisy_grid : torch.Tensor
        Grid of noisy images (C, H, W) after make_grid
    ideal_grid : torch.Tensor
        Grid of ideal denoiser results (C, H, W) after make_grid
    edm_grid : torch.Tensor
        Grid of EDM denoiser results (C, H, W) after make_grid
    sigma_values : list
        List of sigma values used for noise levels
    save_path : str
        Full path to save the figure
    num_sigmas : int
        Number of sigma values (columns in the grid)
        
    Examples:
    ---------
    >>> import torch
    >>> from torchvision.utils import make_grid
    >>> from utils.visualization import create_comparison_figure
    >>> 
    >>> # Create dummy grids
    >>> noisy = make_grid(torch.randn(9, 3, 32, 32), nrow=3)
    >>> ideal = make_grid(torch.randn(9, 3, 32, 32), nrow=3)
    >>> edm = make_grid(torch.randn(9, 3, 32, 32), nrow=3)
    >>> 
    >>> create_comparison_figure(noisy, ideal, edm, [0, 1, 2], "comparison.png", 3)
    """
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    
    # Convert grids to numpy
    noisy_np = noisy_grid.permute(1, 2, 0).cpu().numpy()
    ideal_np = ideal_grid.permute(1, 2, 0).cpu().numpy()
    edm_np = edm_grid.permute(1, 2, 0).cpu().numpy()
    
    # Plot noisy images
    axes[0].imshow(noisy_np, aspect='auto')
    axes[0].set_title(
        "Noisy Images (x + σ·ε, where ε ~ N(0, I))",
        fontsize=14,
        pad=10
    )
    axes[0].axis('off')
    
    # Plot ideal denoiser results
    axes[1].imshow(ideal_np, aspect='auto')
    axes[1].set_title(
        "Ideal Denoiser Output D(x; σ) - Eq. 57 (Closed-form)",
        fontsize=14,
        pad=10
    )
    axes[1].axis('off')
    
    # Plot EDM denoiser results
    axes[2].imshow(edm_np, aspect='auto')
    axes[2].set_title(
        "EDM Denoiser Output D(x; σ) - Pretrained Neural Network",
        fontsize=14,
        pad=10
    )
    axes[2].axis('off')
    
    # Apply tight_layout first to get final axes positions
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Add sigma labels at the top, aligned with actual image columns
    bbox = axes[0].get_position()
    for idx, sigma in enumerate(sigma_values):
        # Calculate position of each column center in figure coordinates
        x_pos_fig = bbox.x0 + (idx + 0.5) / num_sigmas * bbox.width
        fig.text(
            x_pos_fig,
            0.98,
            f'σ={sigma}',
            ha='center',
            va='top',
            fontsize=10,
            weight='bold'
        )
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved comparison figure to: {save_path}")

