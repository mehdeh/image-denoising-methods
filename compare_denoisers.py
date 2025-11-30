"""
Compare Ideal Denoiser and EDM Denoiser Side-by-Side

This script compares two denoising methods on CIFAR-10 images:
1. Ideal denoiser (closed-form solution from EDM paper Eq. 57)
2. EDM pretrained denoiser (learned neural network)

For 3 train and 3 test images with various noise levels, it generates visualizations
showing:
- Row 1: Noisy images at different sigma levels
- Row 2: Results from ideal denoiser
- Row 3: Results from EDM denoiser

This allows direct visual comparison of theoretical vs. learned denoising performance.

Reference:
    Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
    Paper: https://arxiv.org/abs/2206.00364
"""

import torch
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Import from modular structure
from denoisers.ideal_denoiser import ideal_denoiser
from denoisers.edm_denoiser import load_pretrained_edm, edm_denoise
from utils.noise_utils import add_gaussian_noise
from utils.image_utils import load_cifar10_subset, normalize_for_display


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


def generate_denoiser_comparison(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    edm_model: torch.nn.Module,
    sigma_values: list,
    dataset_name: str,
    save_dir: str,
    device: str = 'cpu'
) -> tuple:
    """
    Generate comparison of ideal and EDM denoisers.
    
    This function processes selected images by:
    1. Adding Gaussian noise at various sigma levels
    2. Denoising with ideal denoiser (closed-form solution)
    3. Denoising with EDM pretrained model
    4. Creating comparative visualizations
    
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
    save_dir : str
        Directory to save output images
    device : str
        Device to run computations on ('cpu' or 'cuda')
        
    Returns:
    --------
    tuple : (noisy_grid, ideal_grid, edm_grid)
        Three grids containing noisy, ideal denoised, and EDM denoised images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Move to device
    train_images = train_images.to(device)
    selected_images = selected_images.to(device)
    
    num_images = len(selected_images)
    num_sigmas = len(sigma_values)
    
    print(f"\nGenerating comparison with {num_images} images and {num_sigmas} sigma values...")
    print(f"Sigma values: {sigma_values}")
    
    # Storage for results
    noisy_images_all = []
    ideal_denoised_all = []
    edm_denoised_all = []
    
    # Process each sigma value with batch of all images
    for sigma in tqdm(sigma_values, desc="Processing sigma values"):
        # Add noise to all images at once
        noisy_batch = add_gaussian_noise(selected_images, sigma)
        
        # Denoise using ideal denoiser
        if sigma == 0:
            ideal_denoised_batch = selected_images.clone()
            edm_denoised_batch = selected_images.clone()
        else:
            # Ideal denoiser
            with torch.no_grad():
                ideal_denoised_batch = ideal_denoiser(
                    noisy_batch,
                    sigma,
                    train_images
                )
            
            # EDM denoiser
            with torch.no_grad():
                edm_denoised_batch = edm_denoise(
                    edm_model,
                    noisy_batch,
                    sigma
                )
        
        noisy_images_all.append(noisy_batch)
        ideal_denoised_all.append(ideal_denoised_batch)
        edm_denoised_all.append(edm_denoised_batch)
    
    # Stack all images and transpose to organize by image rows, sigma columns
    noisy_stack = torch.stack(noisy_images_all, dim=0).transpose(0, 1)
    ideal_stack = torch.stack(ideal_denoised_all, dim=0).transpose(0, 1)
    edm_stack = torch.stack(edm_denoised_all, dim=0).transpose(0, 1)
    
    # Flatten to grid format
    noisy_grid = noisy_stack.reshape(-1, *noisy_stack.shape[2:])
    ideal_grid = ideal_stack.reshape(-1, *ideal_stack.shape[2:])
    edm_grid = edm_stack.reshape(-1, *edm_stack.shape[2:])
    
    # Normalize for display
    noisy_display = normalize_for_display(noisy_grid)
    ideal_display = normalize_for_display(ideal_grid)
    edm_display = normalize_for_display(edm_grid)
    
    # Create grids
    print("\nCreating image grids...")
    noisy_grid_img = make_grid(noisy_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    ideal_grid_img = make_grid(ideal_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    edm_grid_img = make_grid(edm_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    
    # Create combined visualization with labels
    combined_path = os.path.join(save_dir, f"comparison_{dataset_name}.png")
    create_comparison_figure(
        noisy_grid_img,
        ideal_grid_img,
        edm_grid_img,
        sigma_values,
        combined_path,
        num_sigmas
    )
    
    return noisy_grid_img, ideal_grid_img, edm_grid_img


def main():
    """
    Main function to compare ideal and EDM denoisers.
    
    Generates comparison figures for both training and test datasets,
    showing noisy images, ideal denoiser results, and EDM denoiser results
    side-by-side for visual quality assessment.
    """
    # Configuration
    data_root = "./data"
    save_dir = "./results/denoiser_comparison"
    sigma_values = [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]
    
    # Image selection parameters
    max_samples_for_selection = 10
    train_selection_indices = [2, 3, 4]
    test_selection_indices = [2, 3, 4]
    
    # Ideal denoiser parameters
    ideal_denoiser_subset_size = 1000
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load EDM model
    print("\n" + "="*80)
    print("Loading pretrained EDM model...")
    print("="*80)
    
    try:
        edm_model, edm_config = load_pretrained_edm('cifar10-uncond', device=device)
        print(f"✓ EDM model loaded successfully")
        print(f"  Architecture: {edm_config['architecture']}")
        print(f"  Resolution: {edm_config['resolution']}x{edm_config['resolution']}")
        print(f"  Conditional: {edm_config['conditional']}")
    except Exception as e:
        print(f"✗ Failed to load EDM model: {e}")
        print("\nPlease ensure you have the required dependencies:")
        print("  pip install git+https://github.com/NVlabs/edm.git")
        return
    
    # Load image subsets
    print("\n" + "="*80)
    print("Loading image subsets...")
    print("="*80)
    
    train_subset = load_cifar10_subset(
        root=data_root,
        normalize=True,
        train=True,
        max_samples=max_samples_for_selection
    )
    test_subset = load_cifar10_subset(
        root=data_root,
        normalize=True,
        train=False,
        max_samples=max_samples_for_selection
    )
    
    # Select specific images
    train_selected = train_subset[train_selection_indices]
    test_selected = test_subset[test_selection_indices]
    
    # Load training images for ideal denoiser reference
    print(f"\nLoading {ideal_denoiser_subset_size} training images for ideal denoiser...")
    train_images_for_denoiser = load_cifar10_subset(
        root=data_root,
        normalize=True,
        train=True,
        max_samples=ideal_denoiser_subset_size
    )
    
    # Generate comparison for training set
    print("\n" + "="*80)
    print("Generating Denoiser Comparison: Training Set")
    print("="*80)
    
    generate_denoiser_comparison(
        selected_images=train_selected,
        train_images=train_images_for_denoiser,
        edm_model=edm_model,
        sigma_values=sigma_values,
        dataset_name="train",
        save_dir=save_dir,
        device=device
    )
    
    # Generate comparison for test set
    print("\n" + "="*80)
    print("Generating Denoiser Comparison: Test Set")
    print("="*80)
    
    generate_denoiser_comparison(
        selected_images=test_selected,
        train_images=train_images_for_denoiser,
        edm_model=edm_model,
        sigma_values=sigma_values,
        dataset_name="test",
        save_dir=save_dir,
        device=device
    )
    
    print("\n" + "="*80)
    print("Comparison generation completed successfully!")
    print("="*80)
    print(f"\nOutput files saved in: {save_dir}/")
    print("- comparison_train.png: Comparison for training set")
    print("- comparison_test.png: Comparison for test set")
    print("\nEach figure shows:")
    print("  Row 1: Noisy images at different noise levels")
    print("  Row 2: Ideal denoiser results (closed-form solution)")
    print("  Row 3: EDM denoiser results (pretrained neural network)")


if __name__ == "__main__":
    main()

