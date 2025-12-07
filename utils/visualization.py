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
    *grids: torch.Tensor,
    sigma_values: list,
    save_path: str,
    num_sigmas: int,
    denoiser_names: list = None
) -> None:
    """
    Create a multi-row comparison figure with labels for sigma values.
    
    This function creates a publication-quality figure showing noisy images
    and results from multiple denoisers in a grid format with labeled sigma
    values aligned with image columns.
    
    Parameters:
    -----------
    *grids : torch.Tensor
        Variable number of image grids (C, H, W) after make_grid.
        First grid should be noisy images, followed by denoiser results.
    sigma_values : list
        List of sigma values used for noise levels
    save_path : str
        Full path to save the figure
    num_sigmas : int
        Number of sigma values (columns in the grid)
    denoiser_names : list, optional
        List of denoiser names corresponding to grids (excluding noisy).
        If None, uses default names ['ideal', 'edm', 'grad-ascent']
        
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
    >>> create_comparison_figure(noisy, ideal, edm, 
    ...     sigma_values=[0, 1, 2], 
    ...     save_path="comparison.png", 
    ...     num_sigmas=3,
    ...     denoiser_names=['ideal', 'edm'])
    """
    # Denoiser title mappings
    denoiser_title_map = {
        'ideal': "Ideal Denoiser Output D(x; σ) - Eq. 57 (Closed-form)",
        'edm': "EDM Denoiser Output D(x; σ) - Pretrained Neural Network (One-step)",
        'grad-ascent': "Gradient Ascent Denoiser - Iterative Optimization (x ← x + lr·∇log p(x; σ))"
    }
    
    num_rows = len(grids)
    fig, axes = plt.subplots(num_rows, 1, figsize=(20, 4 * num_rows))
    
    # Handle single row case (axes is not an array)
    if num_rows == 1:
        axes = [axes]
    
    # If no denoiser names provided, use defaults
    if denoiser_names is None:
        denoiser_names = ['ideal', 'edm', 'grad-ascent'][:num_rows - 1]
    
    # Plot noisy images (first grid)
    noisy_np = grids[0].permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(noisy_np, aspect='auto')
    axes[0].set_title(
        "Noisy Images (x + σ·ε, where ε ~ N(0, I))",
        fontsize=14,
        pad=10
    )
    axes[0].axis('off')
    
    # Plot denoiser results (remaining grids)
    for i, (grid, denoiser_name) in enumerate(zip(grids[1:], denoiser_names), start=1):
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(grid_np, aspect='auto')
        
        # Get title from mapping or use default
        title = denoiser_title_map.get(
            denoiser_name,
            f"{denoiser_name.capitalize()} Denoiser Output"
        )
        axes[i].set_title(title, fontsize=14, pad=10)
        axes[i].axis('off')
    
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

