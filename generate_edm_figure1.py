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
from torchvision.utils import make_grid, save_image
import numpy as np
from tqdm import tqdm
import os

# Import from modular structure
from denoisers.ideal_denoiser import ideal_denoiser
from utils.noise_utils import add_gaussian_noise
from utils.image_utils import load_cifar10_dataset, normalize_for_display
from utils.visualization import create_labeled_figure


# Note: Functions now imported from modular structure
# - ideal_denoiser from denoisers.ideal_denoiser
# - add_gaussian_noise from utils.noise_utils
# - normalize_for_display from utils.image_utils
# - load_cifar10_dataset from utils.image_utils


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

