"""
Image processing utilities for denoising experiments.

This module provides common functions for processing images with noise and denoising,
including batch processing at various noise levels and grid generation.
"""

import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import os

from ideal_denoiser.ideal_denoiser import ideal_denoiser
from edm_denoiser import edm_denoise, gradient_ascent_denoise
from .core import add_gaussian_noise, normalize_for_display
from .visualization import create_comparison_figure


def process_images_at_sigma(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    edm_model: torch.nn.Module,
    sigma: float,
    device: str,
    enabled_denoisers: list,
    grad_ascent_steps: int = 10,
    grad_ascent_lr: float = 1.0
) -> dict:
    """
    Process images at a specific noise level with all enabled denoisers.
    
    Parameters:
    -----------
    selected_images : torch.Tensor
        Clean images to process
    train_images : torch.Tensor
        Training images for ideal denoiser reference
    edm_model : torch.nn.Module
        Pretrained EDM model
    sigma : float
        Noise level
    device : str
        Device to run computations on
    enabled_denoisers : list
        List of enabled denoisers ('ideal', 'edm', 'grad-ascent')
    grad_ascent_steps : int
        Number of gradient ascent iterations (default: 10)
    grad_ascent_lr : float
        Learning rate for gradient ascent (default: 1.0)
        
    Returns:
    --------
    dict : Dictionary containing results from each enabled denoiser
    
    Examples:
    ---------
    >>> import torch
    >>> from utils.processing import process_images_at_sigma
    >>> 
    >>> clean = torch.randn(3, 3, 32, 32)
    >>> train = torch.randn(1000, 3, 32, 32)
    >>> model = load_edm_model('cpu')
    >>> results = process_images_at_sigma(clean, train, model, 2.0, 'cpu', ['ideal', 'edm'])
    """
    results = {}
    
    # Handle sigma = 0 case
    if sigma == 0:
        results['noisy'] = selected_images.clone()
        if 'ideal' in enabled_denoisers:
            results['ideal'] = selected_images.clone()
        if 'edm' in enabled_denoisers:
            results['edm'] = selected_images.clone()
        if 'grad-ascent' in enabled_denoisers:
            results['grad-ascent'] = selected_images.clone()
        return results
    
    # Add noise (in float32, matching how data is loaded)
    noisy_batch = add_gaussian_noise(selected_images, sigma)
    results['noisy'] = noisy_batch
    
    # Denoise with ideal denoiser
    if 'ideal' in enabled_denoisers:
        with torch.no_grad():
            ideal_denoised_batch = ideal_denoiser(
                noisy_batch,
                sigma,
                train_images
            )
        results['ideal'] = ideal_denoised_batch
    
    # Denoise with EDM denoiser (one-step)
    if 'edm' in enabled_denoisers:
        with torch.no_grad():
            edm_denoised_batch = edm_denoise(
                edm_model,
                noisy_batch,
                sigma
            )
        results['edm'] = edm_denoised_batch
    
    # Denoise with gradient ascent (EDM score-based)
    if 'grad-ascent' in enabled_denoisers:
        with torch.no_grad():
            grad_ascent_denoised_batch = gradient_ascent_denoise(
                edm_model,
                noisy_batch,
                sigma,
                num_steps=grad_ascent_steps,
                lr=grad_ascent_lr,
                return_trajectory=False,
                use_float64=True
            )
        results['grad-ascent'] = grad_ascent_denoised_batch
    
    return results


def generate_denoiser_comparison(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    edm_model: torch.nn.Module,
    sigma_values: list,
    dataset_name: str,
    save_path: str,
    device: str = 'cpu',
    enabled_denoisers: list = ['ideal', 'edm', 'grad-ascent'],
    grad_ascent_steps: int = 10,
    grad_ascent_lr: float = 1.0
) -> dict:
    """
    Generate comparison of enabled denoising methods.
    
    This function processes selected images by:
    1. Adding Gaussian noise at various sigma levels
    2. Denoising with enabled methods
    3. Creating comparative visualizations
    
    Parameters:
    -----------
    selected_images : torch.Tensor
        Selected images to process (from train or test set)
        Shape: (num_images, C, H, W)
    train_images : torch.Tensor
        CIFAR-10 training images (used for ideal denoiser)
        Shape: (num_train, C, H, W)
    edm_model : torch.nn.Module
        Pretrained EDM denoising model
    sigma_values : list
        List of noise levels to test
    dataset_name : str
        Name of the dataset ('train' or 'test') for naming output file
    save_path : str
        Full path to save output image
    device : str
        Device to run computations on ('cpu' or 'cuda')
    enabled_denoisers : list
        List of enabled denoisers (default: ['ideal', 'edm', 'grad-ascent'])
    grad_ascent_steps : int
        Number of gradient ascent iterations (default: 10)
    grad_ascent_lr : float
        Learning rate for gradient ascent (default: 1.0)
        
    Returns:
    --------
    dict : Dictionary containing grids for each denoiser
    
    Examples:
    ---------
    >>> import torch
    >>> from utils.processing import generate_denoiser_comparison
    >>> 
    >>> selected = torch.randn(3, 3, 32, 32)
    >>> train = torch.randn(1000, 3, 32, 32)
    >>> model = load_edm_model('cpu')
    >>> grids = generate_denoiser_comparison(
    ...     selected, train, model, [0, 1, 2], "test", "output.png", "cpu"
    ... )
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Move to device
    train_images = train_images.to(device)
    selected_images = selected_images.to(device)
    
    num_images = len(selected_images)
    num_sigmas = len(sigma_values)
    
    print(f"\nGenerating comparison with {num_images} images and {num_sigmas} sigma values...")
    print(f"Sigma values: {sigma_values}")
    print(f"Enabled denoisers: {enabled_denoisers}")
    
    # Storage for results
    all_results = {denoiser: [] for denoiser in ['noisy'] + enabled_denoisers}
    
    # Process each sigma value with batch of all images
    for sigma in tqdm(sigma_values, desc="Processing sigma values"):
        sigma_results = process_images_at_sigma(
            selected_images,
            train_images,
            edm_model,
            sigma,
            device,
            enabled_denoisers,
            grad_ascent_steps,
            grad_ascent_lr
        )
        
        for key, value in sigma_results.items():
            all_results[key].append(value)
    
    # Stack and organize images: transpose from (num_sigmas, num_images, C, H, W)
    # to (num_images, num_sigmas, C, H, W) then flatten to grid format
    grids = {}
    for key, results_list in all_results.items():
        stacked = torch.stack(results_list, dim=0).transpose(0, 1)
        grid = stacked.reshape(-1, *stacked.shape[2:])
        display = normalize_for_display(grid)
        grid_img = make_grid(display, nrow=num_sigmas, padding=2, pad_value=1.0)
        grids[key] = grid_img
    
    # Create combined visualization with labels
    print("\nCreating comparison figure...")
    
    # Build list of grids in order: noisy, then enabled denoisers
    grid_list = [grids['noisy']]
    for denoiser in enabled_denoisers:
        grid_list.append(grids[denoiser])
    
    create_comparison_figure(
        *grid_list,
        sigma_values=sigma_values,
        save_path=save_path,
        num_sigmas=num_sigmas,
        denoiser_names=enabled_denoisers
    )
    
    print(f"✓ Saved comparison to: {save_path}")
    
    return grids


__all__ = ['process_images_at_sigma', 'generate_denoiser_comparison']
