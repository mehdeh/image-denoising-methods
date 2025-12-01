"""
Test script to verify the gradient_ascent_denoise function works correctly.

This script tests the fixed implementation by:
1. Loading a pretrained EDM model
2. Loading CIFAR-10 test images
3. Adding Gaussian noise
4. Denoising using gradient ascent
5. Displaying/saving results
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from local denoisers module
from denoisers.edm_denoiser import load_edm_model, gradient_ascent_denoise


def normalize_batch_01(x: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalization for a batch of images so that each image
    is scaled between 0 and 1 independently.
    """
    batch_size = x.shape[0]
    x_reshaped = x.view(batch_size, -1)
    min_vals = x_reshaped.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    max_vals = x_reshaped.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    return (x - min_vals) / (max_vals - min_vals + 1e-8)


def load_cifar10_subset(root="./data", train=False, selected_indices=[20, 21, 22, 23]):
    """Load a subset of CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transform
    )
    
    inputs = torch.stack([dataset[i][0] for i in selected_indices])
    print(f"Loaded {len(selected_indices)} CIFAR-10 test images: {inputs.shape}")
    return inputs


def add_gaussian_noise(images: torch.Tensor, sigma: float, device: torch.device):
    """Add Gaussian noise to images."""
    noise = torch.randn_like(images, device=device)
    noisy_images = images + sigma * noise
    return noisy_images


def save_image_grid(images: torch.Tensor, save_path: str, nrow: int = 4):
    """Save a grid of images."""
    from torchvision.utils import make_grid
    
    # Normalize for visualization
    images_norm = normalize_batch_01(images.cpu())
    
    grid = make_grid(images_norm, nrow=nrow, padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved image to: {save_path}")


def test_gradient_ascent_denoising():
    """Main test function."""
    print("=" * 70)
    print("Testing Gradient Ascent Denoising")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Configuration
    model_path = "./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl"
    model_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl"
    sigma_value = 3.0
    num_steps = 10
    lr = 1.0
    
    # Create results directory
    results_dir = Path("./results/gradient_ascent_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n{'='*70}")
    print("Loading EDM model...")
    print(f"{'='*70}")
    model = load_edm_model(model_path, model_url, device)
    print("Model loaded successfully!")
    
    # Load test images
    print(f"\n{'='*70}")
    print("Loading CIFAR-10 test images...")
    print(f"{'='*70}")
    clean_images = load_cifar10_subset()
    clean_images = clean_images.to(device)
    
    # Add noise
    print(f"\n{'='*70}")
    print(f"Adding Gaussian noise (σ = {sigma_value})...")
    print(f"{'='*70}")
    noisy_images = add_gaussian_noise(clean_images, sigma_value, device)
    
    # Compute noise level (PSNR-like metric)
    mse = torch.mean((noisy_images - clean_images) ** 2).item()
    print(f"Noise MSE: {mse:.6f}")
    
    # Save clean and noisy images
    save_image_grid(clean_images, results_dir / "01_clean_images.png")
    save_image_grid(noisy_images, results_dir / "02_noisy_images.png")
    
    # Denoise using gradient ascent
    print(f"\n{'='*70}")
    print(f"Running gradient ascent denoising...")
    print(f"  Number of steps: {num_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Using float64: True")
    print(f"{'='*70}")
    
    denoised_images, trajectory = gradient_ascent_denoise(
        model=model,
        x_init=noisy_images,
        sigma=sigma_value,
        num_steps=num_steps,
        lr=lr,
        class_labels=None,
        return_trajectory=True,
        use_float64=True
    )
    
    print(f"\nGradient ascent completed!")
    print(f"Trajectory length: {len(trajectory)} (including initial state)")
    
    # Save denoised images
    save_image_grid(denoised_images, results_dir / "03_denoised_images.png")
    
    # Compute denoising quality metrics
    print(f"\n{'='*70}")
    print("Denoising Quality Metrics")
    print(f"{'='*70}")
    
    # MSE between noisy and clean
    mse_noisy = torch.mean((noisy_images - clean_images) ** 2).item()
    # MSE between denoised and clean
    mse_denoised = torch.mean((denoised_images - clean_images) ** 2).item()
    
    print(f"MSE (noisy vs clean):    {mse_noisy:.6f}")
    print(f"MSE (denoised vs clean): {mse_denoised:.6f}")
    print(f"MSE reduction:           {(mse_noisy - mse_denoised):.6f} ({(1 - mse_denoised/mse_noisy)*100:.2f}%)")
    
    # Save intermediate trajectory images (every 2 steps)
    print(f"\n{'='*70}")
    print("Saving trajectory images...")
    print(f"{'='*70}")
    for i, img in enumerate(trajectory[::2]):  # Every 2 steps
        save_image_grid(img, results_dir / f"trajectory_step_{i*2:02d}.png")
    
    # Create comparison figure
    print(f"\n{'='*70}")
    print("Creating comparison figure...")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle('Gradient Ascent Denoising Results', fontsize=16)
    
    for i in range(4):
        # Clean image
        img_clean = normalize_batch_01(clean_images[i:i+1]).cpu().squeeze().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img_clean)
        axes[0, i].set_title(f'Clean {i+1}')
        axes[0, i].axis('off')
        
        # Noisy image
        img_noisy = normalize_batch_01(noisy_images[i:i+1]).cpu().squeeze().permute(1, 2, 0).numpy()
        axes[1, i].imshow(img_noisy)
        axes[1, i].set_title(f'Noisy (σ={sigma_value})')
        axes[1, i].axis('off')
        
        # Denoised image
        img_denoised = normalize_batch_01(denoised_images[i:i+1]).cpu().squeeze().permute(1, 2, 0).numpy()
        axes[2, i].imshow(img_denoised)
        axes[2, i].set_title(f'Denoised ({num_steps} steps)')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    comparison_path = results_dir / "comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure to: {comparison_path}")
    
    print(f"\n{'='*70}")
    print("Test completed successfully!")
    print(f"All results saved to: {results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_gradient_ascent_denoising()

