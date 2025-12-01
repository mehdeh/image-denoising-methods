"""
Compare Multiple Denoising Methods Side-by-Side

This script compares three denoising methods on CIFAR-10 images:
1. Ideal denoiser (closed-form solution from EDM paper Eq. 57)
2. EDM pretrained denoiser (learned neural network - one-step)
3. Gradient ascent denoiser (iterative optimization using score function)

For 3 train and 3 test images with various noise levels, it generates visualizations
showing:
- Row 1: Noisy images at different sigma levels
- Row 2: Results from ideal denoiser
- Row 3: Results from EDM denoiser (one-step)
- Row 4: Results from gradient ascent denoiser (iterative)

This allows direct visual comparison of theoretical vs. learned vs. iterative denoising.

Reference:
    Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
    Paper: https://arxiv.org/abs/2206.00364
"""

import torch
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import os

# Import from modular structure
from denoisers.ideal_denoiser import ideal_denoiser
from denoisers.edm_denoiser import load_pretrained_edm, edm_denoise, gradient_ascent_denoise
from utils.noise_utils import add_gaussian_noise
from utils.image_utils import load_cifar10_subset, normalize_for_display
from utils.visualization import create_comparison_figure


def process_images_at_sigma(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    edm_model: torch.nn.Module,
    sigma: float,
    device: str,
    grad_ascent_steps: int = 10,
    grad_ascent_lr: float = 0.1
) -> tuple:
    """
    Process images at a specific noise level with all denoisers.
    
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
    grad_ascent_steps : int
        Number of gradient ascent iterations (default: 10)
    grad_ascent_lr : float
        Learning rate for gradient ascent (default: 1.0)
        
    Returns:
    --------
    tuple : (noisy, ideal_denoised, edm_denoised, grad_ascent_denoised)
        Four tensors containing the noisy images and all denoised versions
    """
    # Handle sigma = 0 case
    if sigma == 0:
        return (
            selected_images.clone(),
            selected_images.clone(),
            selected_images.clone(),
            selected_images.clone(),
            selected_images.clone()
        )
    
    # Add noise (in float32, matching how data is loaded)
    noisy_batch = add_gaussian_noise(selected_images, sigma)
    
    # Denoise with ideal denoiser
    with torch.no_grad():
        ideal_denoised_batch = ideal_denoiser(
            noisy_batch,
            sigma,
            train_images
        )
    
    # Denoise with EDM denoiser (one-step)
    with torch.no_grad():
        edm_denoised_batch = edm_denoise(
            edm_model,
            noisy_batch,
            sigma
        )
    
    # Denoise with gradient ascent (EDM score-based, new implementation)
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
    
    return noisy_batch, ideal_denoised_batch, edm_denoised_batch, grad_ascent_denoised_batch


def generate_denoiser_comparison(
    selected_images: torch.Tensor,
    train_images: torch.Tensor,
    edm_model: torch.nn.Module,
    sigma_values: list,
    dataset_name: str,
    save_dir: str,
    device: str = 'cpu',
    grad_ascent_steps: int = 10,
    grad_ascent_lr: float = 1.0
) -> tuple:
    """
    Generate comparison of ideal, EDM, and gradient ascent denoisers.
    
    This function processes selected images by:
    1. Adding Gaussian noise at various sigma levels
    2. Denoising with ideal denoiser (closed-form solution)
    3. Denoising with EDM pretrained model (one-step)
    4. Denoising with gradient ascent (iterative)
    5. Creating comparative visualizations
    
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
    grad_ascent_steps : int
        Number of gradient ascent iterations (default: 10)
    grad_ascent_lr : float
        Learning rate for gradient ascent (default: 1.0)
        
    Returns:
    --------
    tuple : (noisy_grid, ideal_grid, edm_grid, grad_ascent_grid)
        Four grids containing noisy, ideal, EDM, and gradient ascent denoised images
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
    grad_ascent_denoised_all = []
    
    # Process each sigma value with batch of all images
    for sigma in tqdm(sigma_values, desc="Processing sigma values"):
        noisy, ideal_denoised, edm_denoised, grad_ascent_denoised = process_images_at_sigma(
            selected_images,
            train_images,
            edm_model,
            sigma,
            device,
            grad_ascent_steps,
            grad_ascent_lr
        )
        
        noisy_images_all.append(noisy)
        ideal_denoised_all.append(ideal_denoised)
        edm_denoised_all.append(edm_denoised)
        grad_ascent_denoised_all.append(grad_ascent_denoised)
    
    # Stack and organize images: transpose from (num_sigmas, num_images, C, H, W)
    # to (num_images, num_sigmas, C, H, W) then flatten to grid format
    noisy_stack = torch.stack(noisy_images_all, dim=0).transpose(0, 1)
    ideal_stack = torch.stack(ideal_denoised_all, dim=0).transpose(0, 1)
    edm_stack = torch.stack(edm_denoised_all, dim=0).transpose(0, 1)
    grad_ascent_stack = torch.stack(grad_ascent_denoised_all, dim=0).transpose(0, 1)
    
    # Flatten to grid format
    noisy_grid = noisy_stack.reshape(-1, *noisy_stack.shape[2:])
    ideal_grid = ideal_stack.reshape(-1, *ideal_stack.shape[2:])
    edm_grid = edm_stack.reshape(-1, *edm_stack.shape[2:])
    grad_ascent_grid = grad_ascent_stack.reshape(-1, *grad_ascent_stack.shape[2:])
    
    # Normalize for display
    noisy_display = normalize_for_display(noisy_grid)
    ideal_display = normalize_for_display(ideal_grid)
    edm_display = normalize_for_display(edm_grid)
    grad_ascent_display = normalize_for_display(grad_ascent_grid)
    
    # Create grids
    print("\nCreating image grids...")
    noisy_grid_img = make_grid(noisy_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    ideal_grid_img = make_grid(ideal_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    edm_grid_img = make_grid(edm_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    grad_ascent_grid_img = make_grid(grad_ascent_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    
    # Create combined visualization with labels
    combined_path = os.path.join(save_dir, f"comparison_{dataset_name}.png")
    create_comparison_figure(
        noisy_grid_img,
        ideal_grid_img,
        edm_grid_img,
        grad_ascent_grid_img,
        sigma_values,
        combined_path,
        num_sigmas
    )
    
    return noisy_grid_img, ideal_grid_img, edm_grid_img, grad_ascent_grid_img


def load_edm_model(device: str):
    """
    Load pretrained EDM model with error handling.
    
    Parameters:
    -----------
    device : str
        Device to load model on
        
    Returns:
    --------
    tuple : (model, config) or (None, None) if loading fails
    """
    print("\n" + "="*80)
    print("Loading pretrained EDM model...")
    print("="*80)
    
    try:
        model, config = load_pretrained_edm('cifar10-uncond', device=device)
        print(f"✓ EDM model loaded successfully")
        print(f"  Architecture: {config['architecture']}")
        print(f"  Resolution: {config['resolution']}x{config['resolution']}")
        print(f"  Conditional: {config['conditional']}")
        return model, config
    except ModuleNotFoundError as e:
        print(f"✗ Failed to load EDM model: {e}")
        print("\nThe EDM pretrained models require the EDM codebase to be installed.")
        print("\nPlease install EDM dependencies:")
        print("  pip install git+https://github.com/NVlabs/edm.git")
        print("\nNote: The model file will be downloaded automatically (~226MB)")
        print("      if not already present in ./pretrain_models/")
        return None, None
    except Exception as e:
        print(f"✗ Failed to load EDM model: {e}")
        print("\nPlease ensure you have the required dependencies.")
        return None, None


def setup_data_subsets(data_root: str, config: dict) -> tuple:
    """
    Load data subsets for comparison.
    
    Parameters:
    -----------
    data_root : str
        Root directory for CIFAR-10 data
    config : dict
        Configuration dictionary with selection parameters
        
    Returns:
    --------
    tuple : (train_selected, test_selected, train_images_for_denoiser)
    """
    print("\n" + "="*80)
    print("Loading image subsets...")
    print("="*80)
    
    # Load small subsets for selection
    train_subset = load_cifar10_subset(
        root=data_root,
        normalize=True,
        train=True,
        max_samples=config['max_samples_for_selection']
    )
    test_subset = load_cifar10_subset(
        root=data_root,
        normalize=True,
        train=False,
        max_samples=config['max_samples_for_selection']
    )
    
    # Select specific images
    train_selected = train_subset[config['train_selection_indices']]
    test_selected = test_subset[config['test_selection_indices']]
    
    # Load training images for ideal denoiser reference
    print(f"\nLoading {config['ideal_denoiser_subset_size']} training images for ideal denoiser...")
    train_images_for_denoiser = load_cifar10_subset(
        root=data_root,
        normalize=True,
        train=True,
        max_samples=config['ideal_denoiser_subset_size']
    )
    
    return train_selected, test_selected, train_images_for_denoiser


def main():
    """
    Main function to compare ideal, EDM, and gradient ascent denoisers.
    
    Generates comparison figures for both training and test datasets,
    showing noisy images, ideal denoiser results, EDM denoiser results,
    and gradient ascent denoiser results side-by-side for visual quality assessment.
    """
    # Configuration
    config = {
        'data_root': "./data",
        'save_dir': "./results/denoiser_comparison",
        'sigma_values': [0, 0.2, 0.5, 1, 2, 3, 5],#7, 10, 20, 50],
        'max_samples_for_selection': 10,
        'train_selection_indices': [2, 3, 4],
        'test_selection_indices': [2, 3, 4],
        'ideal_denoiser_subset_size': 1000,
        'grad_ascent_steps': 10,
        'grad_ascent_lr': 1.0
    }
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load EDM model
    edm_model, edm_config = load_edm_model(device)
    if edm_model is None:
        return
    
    # Load data subsets
    train_selected, test_selected, train_images_for_denoiser = setup_data_subsets(
        config['data_root'],
        config
    )
    
    # Generate comparison for training set
    print("\n" + "="*80)
    print("Generating Denoiser Comparison: Training Set")
    print("="*80)
    
    generate_denoiser_comparison(
        selected_images=train_selected,
        train_images=train_images_for_denoiser,
        edm_model=edm_model,
        sigma_values=config['sigma_values'],
        dataset_name="train",
        save_dir=config['save_dir'],
        device=device,
        grad_ascent_steps=config['grad_ascent_steps'],
        grad_ascent_lr=config['grad_ascent_lr']
    )
    
    # Generate comparison for test set
    print("\n" + "="*80)
    print("Generating Denoiser Comparison: Test Set")
    print("="*80)
    
    generate_denoiser_comparison(
        selected_images=test_selected,
        train_images=train_images_for_denoiser,
        edm_model=edm_model,
        sigma_values=config['sigma_values'],
        dataset_name="test",
        save_dir=config['save_dir'],
        device=device,
        grad_ascent_steps=config['grad_ascent_steps'],
        grad_ascent_lr=config['grad_ascent_lr']
    )
    
    print("\n" + "="*80)
    print("Comparison generation completed successfully!")
    print("="*80)
    print(f"\nOutput files saved in: {config['save_dir']}/")
    print("- comparison_train.png: Comparison for training set")
    print("- comparison_test.png: Comparison for test set")
    print("\nEach figure shows:")
    print("  Row 1: Noisy images at different noise levels")
    print("  Row 2: Ideal denoiser results (closed-form solution)")
    print("  Row 3: EDM denoiser results (one-step neural network)")
    print(f"  Row 4: Gradient ascent denoiser ({config['grad_ascent_steps']} steps, lr={config['grad_ascent_lr']})")


if __name__ == "__main__":
    main()
