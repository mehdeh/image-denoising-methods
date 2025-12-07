"""
Compare Multiple Denoising Methods Side-by-Side

This script compares three denoising methods on CIFAR-10 images:
1. Ideal denoiser (closed-form solution from EDM paper Eq. 57)
2. EDM pretrained denoiser (learned neural network - one-step)
3. Gradient ascent denoiser (iterative optimization using score function)

For selected train and test images with various noise levels, it generates visualizations
showing:
- Row 1: Noisy images at different sigma levels
- Row 2: Results from ideal denoiser
- Row 3: Results from EDM denoiser (one-step)
- Row 4: Results from gradient ascent denoiser (iterative)

This allows direct visual comparison of theoretical vs. learned vs. iterative denoising.

Reference:
    Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
    Paper: https://arxiv.org/abs/2206.00364

Usage:
    python compare_denoisers.py --num-images 3 --train-size 1000
    python compare_denoisers.py --sigma-list 0 0.5 1 2 5 10 --device cuda
    python compare_denoisers.py --grad-ascent-steps 20 --grad-ascent-lr 0.5
"""

import torch
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import os
import argparse
from datetime import datetime

# Import from modular structure
from ideal_denoiser.ideal_denoiser import ideal_denoiser
from edm_denoiser import load_pretrained_edm, edm_denoise, gradient_ascent_denoise
from utils.noise_utils import add_gaussian_noise
from utils.image_utils import load_cifar10_subset, normalize_for_display
from utils.visualization import create_comparison_figure


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare multiple denoising methods on CIFAR-10 images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data',
        help='Root directory for CIFAR-10 data'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results',
        help='Directory to save output images'
    )
    
    # Image selection parameters
    parser.add_argument(
        '--num-images',
        type=int,
        default=3,
        help='Number of images to denoise from each dataset (train/test)'
    )
    
    # Denoiser parameters
    parser.add_argument(
        '--train-size',
        type=int,
        default=1000,
        help='Number of training images to use for ideal denoiser reference'
    )
    parser.add_argument(
        '--sigma-list',
        type=float,
        nargs='+',
        default=[0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 15],
        help='List of sigma (noise level) values to test'
    )
    
    # Gradient ascent parameters
    parser.add_argument(
        '--grad-ascent-steps',
        type=int,
        default=10,
        help='Number of gradient ascent iterations'
    )
    parser.add_argument(
        '--grad-ascent-lr',
        type=float,
        default=1.0,
        help='Learning rate for gradient ascent'
    )
    
    # Denoiser selection (all enabled by default)
    parser.add_argument(
        '--denoisers',
        type=str,
        nargs='+',
        default=['ideal', 'edm', 'grad-ascent'],
        choices=['ideal', 'edm', 'grad-ascent'],
        help='Denoisers to use in comparison (by default all are enabled)'
    )
    
    # Device parameters
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu or cuda). If not specified, auto-detect.'
    )
    
    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (if not set, default is 42)'
    )
    
    return parser.parse_args()


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


def main():
    """
    Main function to compare enabled denoising methods with CLI arguments.
    
    Generates comparison figures for both training and test datasets,
    showing noisy images and results from enabled denoisers side-by-side
    for visual quality assessment.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Device selection
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*80)
    print("Denoiser Comparison Framework")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Data root: {args.data_root}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Number of images: {args.num_images}")
    print(f"  Training images for ideal denoiser: {args.train_size}")
    print(f"  Sigma values: {args.sigma_list}")
    print(f"  Gradient ascent steps: {args.grad_ascent_steps}")
    print(f"  Gradient ascent learning rate: {args.grad_ascent_lr}")
    print(f"  Enabled denoisers: {args.denoisers}")
    print(f"  Random seed: {args.seed}")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load EDM model (required for EDM and gradient ascent denoisers)
    if 'edm' in args.denoisers or 'grad-ascent' in args.denoisers:
        edm_model, edm_config = load_edm_model(device)
        if edm_model is None:
            print("\n✗ Cannot proceed without EDM model for EDM/gradient ascent denoisers")
            return
    else:
        edm_model = None
    
    # Load data subsets
    print("\n" + "="*80)
    print("Loading Data Subsets")
    print("="*80)
    
    print("\nLoading CIFAR-10 training subset for random selection and ideal denoiser...")
    train_subset = load_cifar10_subset(
        root=args.data_root,
        normalize=True,
        train=True,
        max_samples=args.train_size
    )
    print("\nLoading full CIFAR-10 test set for random selection...")
    test_subset = load_cifar10_subset(
        root=args.data_root,
        normalize=True,
        train=False,
        max_samples=None
    )
    
    # Generate separate random indices for train and test sets
    num_train_available = len(train_subset)
    num_test_available = len(test_subset)
    
    if args.num_images > num_train_available:
        raise ValueError(
            f"Requested num-images={args.num_images} but only {num_train_available} training samples are available."
        )
    if args.num_images > num_test_available:
        raise ValueError(
            f"Requested num-images={args.num_images} but only {num_test_available} test samples are available."
        )
    
    train_indices = np.random.choice(num_train_available, size=args.num_images, replace=False)
    test_indices = np.random.choice(num_test_available, size=args.num_images, replace=False)
    
    print(f"\n  Randomly selected train indices: {train_indices.tolist()}")
    print(f"  Randomly selected test indices: {test_indices.tolist()}")
    
    train_selected = train_subset[train_indices]
    test_selected = test_subset[test_indices]
    
    print(f"Selected {len(train_selected)} training images")
    print(f"Selected {len(test_selected)} test images")
    
    # Training images for ideal denoiser reference: use the same train subset
    train_images_for_denoiser = train_subset
    
    # Generate timestamp for output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive filename with key config parameters
    sigma_min = min(args.sigma_list)
    sigma_max = max(args.sigma_list)
    denoisers_str = '-'.join(args.denoisers)
    filename_base = f"{timestamp}_n{args.num_images}_s{sigma_min}-{sigma_max}_train{args.train_size}_{denoisers_str}"
    
    # Generate output for training set
    print("\n" + "="*80)
    print("Processing Training Set")
    print("="*80)
    
    train_output_path = os.path.join(args.save_dir, f"{filename_base}_train.png")
    generate_denoiser_comparison(
        selected_images=train_selected,
        train_images=train_images_for_denoiser,
        edm_model=edm_model,
        sigma_values=args.sigma_list,
        dataset_name="train",
        save_path=train_output_path,
        device=device,
        enabled_denoisers=args.denoisers,
        grad_ascent_steps=args.grad_ascent_steps,
        grad_ascent_lr=args.grad_ascent_lr
    )
    
    # Generate output for test set
    print("\n" + "="*80)
    print("Processing Test Set")
    print("="*80)
    
    test_output_path = os.path.join(args.save_dir, f"{filename_base}_test.png")
    generate_denoiser_comparison(
        selected_images=test_selected,
        train_images=train_images_for_denoiser,
        edm_model=edm_model,
        sigma_values=args.sigma_list,
        dataset_name="test",
        save_path=test_output_path,
        device=device,
        enabled_denoisers=args.denoisers,
        grad_ascent_steps=args.grad_ascent_steps,
        grad_ascent_lr=args.grad_ascent_lr
    )
    
    print("\n" + "="*80)
    print("Comparison Completed Successfully!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  Training set: {train_output_path}")
    print(f"  Test set: {test_output_path}")
    print(f"\nEach figure shows:")
    print(f"  Row 1: Noisy images at different noise levels")
    enabled_row = 2
    for denoiser in args.denoisers:
        denoiser_name = {
            'ideal': 'Ideal denoiser (closed-form solution)',
            'edm': 'EDM denoiser (one-step neural network)',
            'grad-ascent': f'Gradient ascent denoiser ({args.grad_ascent_steps} steps, lr={args.grad_ascent_lr})'
        }[denoiser]
        print(f"  Row {enabled_row}: {denoiser_name}")
        enabled_row += 1
    print()


if __name__ == "__main__":
    main()
