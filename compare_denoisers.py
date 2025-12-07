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
import numpy as np
import os
import argparse
from datetime import datetime

# Import from modular structure
from utils.core import load_cifar10_subset
from utils.model_utils import load_edm_model
from utils.processing import generate_denoiser_comparison


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
