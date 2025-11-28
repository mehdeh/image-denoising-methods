"""
Generate Figure 1 from EDM Paper (Elucidating the Design Space of Diffusion-Based Generative Models)

This script reproduces the ideal denoiser visualization from the paper by:
1. Loading two sample images from CIFAR-10 test set
2. Adding Gaussian noise with various sigma values
3. Denoising using the ideal denoiser (closed-form solution from Eq. 57)
4. Visualizing both noisy and denoised images in grid format

The ideal denoiser is computed using the entire CIFAR-10 training set as the reference distribution.

Reference:
    Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
    Paper: https://arxiv.org/abs/2206.00364
    Ideal Denoiser Formula: Appendix B.3, Equation 57
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os


def load_cifar10_dataset(root="./data", normalize=True):
    """
    Load CIFAR-10 training and test datasets.
    
    Parameters:
    -----------
    root : str
        Root directory where CIFAR-10 data will be downloaded/stored
    normalize : bool
        Whether to apply normalization to [-1, 1] range
        
    Returns:
    --------
    train_images : torch.Tensor
        Training images of shape (50000, 3, 32, 32)
    test_images : torch.Tensor
        Test images of shape (10000, 3, 32, 32)
    """
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.ToTensor()
    
    # Load training set
    print("Loading CIFAR-10 training set...")
    trainset = torchvision.datasets.CIFAR10(
        root=root, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Load test set
    print("Loading CIFAR-10 test set...")
    testset = torchvision.datasets.CIFAR10(
        root=root, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Convert to tensors
    print("Converting datasets to tensors...")
    train_images = torch.stack([trainset[i][0] for i in tqdm(range(len(trainset)), desc="Train")])
    test_images = torch.stack([testset[i][0] for i in tqdm(range(len(testset)), desc="Test")])
    
    print(f"Training images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")
    
    return train_images, test_images


def ideal_denoiser(x_noisy, sigma, x_all):
    """
    Ideal denoiser using closed-form solution from EDM paper (Eq. 57).
    
    This computes D(x; sigma) = E[x' | x], where x' ~ p_data and x = x' + n with n ~ N(0, sigma^2 I).
    
    The formula computes:
    D(x; sigma) = sum_i [x_i * exp(-||x - x_i||^2 / (2*sigma^2))] / sum_i [exp(-||x - x_i||^2 / (2*sigma^2))]
    
    Parameters:
    -----------
    x_noisy : torch.Tensor
        Noisy input images of shape (batch_size, C, H, W)
    sigma : float or torch.Tensor
        Noise level (standard deviation)
    x_all : torch.Tensor
        All training images used as reference distribution of shape (num_samples, C, H, W)
        
    Returns:
    --------
    denoised : torch.Tensor
        Denoised images of shape (batch_size, C, H, W)
    """
    # Compute squared L2 distance between noisy images and all training images
    # x_all: (N, C, H, W), x_noisy: (B, C, H, W)
    # Result: (N, B)
    norm2 = ((x_all[:, None, :, :, :] - x_noisy[None, :, :, :, :]) ** 2).sum(dim=(2, 3, 4))
    
    # Compute log probabilities: log p(x | x_i) = -||x - x_i||^2 / (2*sigma^2)
    sigma_norm2 = -norm2 / (2 * sigma ** 2)
    
    # Numerical stability: subtract max value before exp (log-sum-exp trick)
    delta = torch.max(sigma_norm2, dim=0, keepdim=True)[0]
    
    # Compute exp of log probabilities
    exp_norm2 = (sigma_norm2 - delta).exp()
    
    # Compute weighted sum: numerator and denominator
    # exp_norm2: (N, B) -> (N, B, 1, 1, 1)
    # x_all: (N, C, H, W) -> (N, 1, C, H, W)
    numerator = exp_norm2[:, :, None, None, None] * x_all[:, None, :, :, :]  # (N, B, C, H, W)
    denominator = exp_norm2.sum(dim=0)  # (B,)
    
    # Compute denoised images
    denoised = numerator.sum(dim=0) / denominator[:, None, None, None]  # (B, C, H, W)
    
    return denoised


def add_gaussian_noise(images, sigma):
    """
    Add Gaussian noise to images.
    
    Parameters:
    -----------
    images : torch.Tensor
        Clean images of shape (batch_size, C, H, W)
    sigma : float
        Standard deviation of Gaussian noise
        
    Returns:
    --------
    noisy_images : torch.Tensor
        Noisy images of shape (batch_size, C, H, W)
    """
    if sigma == 0:
        return images.clone()
    
    noise = torch.randn_like(images) * sigma
    return images + noise


def normalize_for_display(images):
    """
    Min-max normalize images to [0, 1] range for display.
    Each image in the batch is normalized independently.
    
    Parameters:
    -----------
    images : torch.Tensor
        Images of shape (batch_size, C, H, W)
        
    Returns:
    --------
    normalized : torch.Tensor
        Normalized images in [0, 1] range
    """
    batch_size = images.shape[0]
    images_flat = images.view(batch_size, -1)
    
    min_vals = images_flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    max_vals = images_flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    
    normalized = (images - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized


def generate_figure1(train_images, test_images, 
                     sigma_values=[0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50],
                     test_indices=[20, 21],
                     save_dir="./results",
                     device='cpu'):
    """
    Generate Figure 1 from EDM paper showing ideal denoiser performance.
    
    Parameters:
    -----------
    train_images : torch.Tensor
        CIFAR-10 training images (used for ideal denoiser)
    test_images : torch.Tensor
        CIFAR-10 test images (source of test samples)
    sigma_values : list
        List of noise levels to test
    test_indices : list
        Indices of test images to use
    save_dir : str
        Directory to save output images
    device : str
        Device to run computations on ('cpu' or 'cuda')
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Select test images
    selected_images = test_images[test_indices].to(device)
    train_images = train_images.to(device)
    
    num_images = len(test_indices)
    num_sigmas = len(sigma_values)
    
    print(f"\nGenerating Figure 1 with {num_images} images and {num_sigmas} sigma values...")
    print(f"Sigma values: {sigma_values}")
    
    # Storage for results
    noisy_images_all = []
    denoised_images_all = []
    
    # Process each image
    for img_idx, img in enumerate(selected_images):
        print(f"\nProcessing image {img_idx + 1}/{num_images}...")
        
        noisy_row = []
        denoised_row = []
        
        # Process each sigma value
        for sigma in tqdm(sigma_values, desc=f"Image {img_idx + 1}"):
            # Add noise
            img_batch = img.unsqueeze(0)  # (1, C, H, W)
            noisy_img = add_gaussian_noise(img_batch, sigma)
            
            # Denoise using ideal denoiser
            if sigma == 0:
                denoised_img = img_batch.clone()
            else:
                with torch.no_grad():
                    denoised_img = ideal_denoiser(noisy_img, sigma, train_images)
            
            noisy_row.append(noisy_img.squeeze(0))
            denoised_row.append(denoised_img.squeeze(0))
        
        noisy_images_all.append(torch.stack(noisy_row))
        denoised_images_all.append(torch.stack(denoised_row))
    
    # Stack all images
    noisy_images_grid = torch.cat(noisy_images_all, dim=0)  # (num_images * num_sigmas, C, H, W)
    denoised_images_grid = torch.cat(denoised_images_all, dim=0)  # (num_images * num_sigmas, C, H, W)
    
    # Normalize for display
    noisy_images_display = normalize_for_display(noisy_images_grid)
    denoised_images_display = normalize_for_display(denoised_images_grid)
    
    # Create grids
    print("\nCreating image grids...")
    noisy_grid = make_grid(noisy_images_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    denoised_grid = make_grid(denoised_images_display, nrow=num_sigmas, padding=2, pad_value=1.0)
    
    # Save grids
    noisy_path = os.path.join(save_dir, "figure1_noisy.png")
    denoised_path = os.path.join(save_dir, "figure1_denoised.png")
    
    save_image(noisy_grid, noisy_path)
    save_image(denoised_grid, denoised_path)
    
    print(f"\nSaved noisy images grid to: {noisy_path}")
    print(f"Saved denoised images grid to: {denoised_path}")
    
    # Create combined visualization with labels
    create_labeled_figure(noisy_grid, denoised_grid, sigma_values, save_dir)
    
    return noisy_grid, denoised_grid


def create_labeled_figure(noisy_grid, denoised_grid, sigma_values, save_dir):
    """
    Create a combined figure with labels for sigma values.
    
    Parameters:
    -----------
    noisy_grid : torch.Tensor
        Grid of noisy images
    denoised_grid : torch.Tensor
        Grid of denoised images
    sigma_values : list
        List of sigma values used
    save_dir : str
        Directory to save the figure
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


def main():
    """
    Main function to generate Figure 1 from EDM paper.
    """
    # Configuration
    data_root = "./data"
    save_dir = "./results"
    sigma_values = [0, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50]
    test_indices = [20, 21]  # Indices of test images to visualize
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load CIFAR-10 dataset
    train_images, test_images = load_cifar10_dataset(root=data_root, normalize=True)
    
    # Generate Figure 1
    print("\n" + "="*80)
    print("Generating EDM Figure 1: Ideal Denoiser Visualization")
    print("="*80)
    
    noisy_grid, denoised_grid = generate_figure1(
        train_images=train_images,
        test_images=test_images,
        sigma_values=sigma_values,
        test_indices=test_indices,
        save_dir=save_dir,
        device=device
    )
    
    print("\n" + "="*80)
    print("Figure generation completed successfully!")
    print("="*80)
    print(f"\nOutput files saved in: {save_dir}/")
    print("- figure1_noisy.png: Grid of noisy images")
    print("- figure1_denoised.png: Grid of denoised images") 
    print("- figure1_combined.png: Combined visualization with labels")


if __name__ == "__main__":
    main()

