"""
Example Usage of Modular Denoising Methods

This script demonstrates how to use the modular denoising framework
for various denoising tasks.
"""

import torch
import numpy as np

# Import denoiser modules
from denoisers.ideal_denoiser import ideal_denoiser
from denoisers.edm_denoiser import (
    load_pretrained_edm,
    edm_denoise,
    gradient_ascent_denoise
)

# Import utility modules
from utils.noise_utils import add_gaussian_noise
from utils.image_utils import load_cifar10_dataset, normalize_for_display
from utils.visualization import create_labeled_figure

from torchvision.utils import save_image
import os


def example_1_ideal_denoiser():
    """
    Example 1: Using the ideal denoiser with CIFAR-10
    """
    print("="*80)
    print("Example 1: Ideal Denoiser")
    print("="*80)
    
    # Load a subset of CIFAR-10 for faster testing
    print("\nLoading CIFAR-10 dataset...")
    train_images, test_images = load_cifar10_dataset(root="./data", normalize=True)
    
    # Use a smaller subset for faster computation
    train_subset = train_images[:1000]  # Use 1000 training images
    test_image = test_images[0:1]  # Use 1 test image
    
    # Add noise
    sigma = 2.0
    print(f"\nAdding Gaussian noise with σ={sigma}")
    noisy_image = add_gaussian_noise(test_image, sigma)
    
    # Denoise using ideal denoiser
    print("Denoising with ideal denoiser...")
    with torch.no_grad():
        denoised_image = ideal_denoiser(noisy_image, sigma, train_subset)
    
    # Save results
    os.makedirs("./results/examples", exist_ok=True)
    save_image(normalize_for_display(test_image), "./results/examples/example1_clean.png")
    save_image(normalize_for_display(noisy_image), "./results/examples/example1_noisy.png")
    save_image(normalize_for_display(denoised_image), "./results/examples/example1_denoised.png")
    
    print("✓ Results saved to ./results/examples/")
    print()


def example_2_edm_denoiser():
    """
    Example 2: Using pretrained EDM denoiser
    """
    print("="*80)
    print("Example 2: EDM Denoiser (Pretrained)")
    print("="*80)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load pretrained EDM model
    print("\nLoading pretrained EDM model...")
    print("Note: This will download the model if not already cached (~300MB)")
    
    try:
        model, config = load_pretrained_edm('cifar10-uncond', device=device)
        print(f"✓ Model loaded: {config}")
        
        # Load test image
        _, test_images = load_cifar10_dataset(root="./data", normalize=True)
        test_image = test_images[0:1].to(device)
        
        # Add noise
        sigma = 3.0
        print(f"\nAdding Gaussian noise with σ={sigma}")
        noisy_image = add_gaussian_noise(test_image, sigma)
        
        # Denoise using EDM
        print("Denoising with EDM model...")
        with torch.no_grad():
            denoised_image = edm_denoise(model, noisy_image, sigma)
        
        # Save results
        os.makedirs("./results/examples", exist_ok=True)
        save_image(normalize_for_display(test_image.cpu()), "./results/examples/example2_clean.png")
        save_image(normalize_for_display(noisy_image.cpu()), "./results/examples/example2_noisy.png")
        save_image(normalize_for_display(denoised_image.cpu()), "./results/examples/example2_denoised.png")
        
        print("✓ Results saved to ./results/examples/")
        
    except Exception as e:
        print(f"⚠ Could not run EDM example: {e}")
        print("This is normal if you don't have dnnlib installed.")
        print("Install with: pip install git+https://github.com/NVlabs/edm.git")
    
    print()


def example_3_gradient_ascent():
    """
    Example 3: Gradient ascent denoising with EDM
    """
    print("="*80)
    print("Example 3: EDM Gradient Ascent Denoising")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    try:
        # Load pretrained EDM model
        print("\nLoading pretrained EDM model...")
        model, config = load_pretrained_edm('cifar10-uncond', device=device)
        
        # Load test image
        _, test_images = load_cifar10_dataset(root="./data", normalize=True)
        test_image = test_images[0:1].to(device)
        
        # Add noise
        sigma = 3.0
        print(f"\nAdding Gaussian noise with σ={sigma}")
        noisy_image = add_gaussian_noise(test_image, sigma)
        
        # Denoise using gradient ascent
        print("Denoising with gradient ascent (10 steps)...")
        denoised_image, trajectory = gradient_ascent_denoise(
            model, noisy_image, sigma, num_steps=10, lr=1.0
        )
        
        # Save trajectory
        os.makedirs("./results/examples", exist_ok=True)
        for i, img in enumerate(trajectory):
            if i % 2 == 0:  # Save every other step
                save_image(
                    normalize_for_display(img.cpu()), 
                    f"./results/examples/example3_step{i:02d}.png"
                )
        
        print(f"✓ Saved {len(trajectory)} trajectory images")
        
    except Exception as e:
        print(f"⚠ Could not run gradient ascent example: {e}")
    
    print()


def example_4_batch_processing():
    """
    Example 4: Batch processing multiple images
    """
    print("="*80)
    print("Example 4: Batch Processing")
    print("="*80)
    
    # Load dataset
    train_images, test_images = load_cifar10_dataset(root="./data", normalize=True)
    
    # Use a subset
    train_subset = train_images[:1000]
    test_batch = test_images[0:4]  # 4 test images
    
    # Add noise with different sigma values
    sigma_values = [1.0, 2.0, 3.0, 5.0]
    
    print(f"\nProcessing {len(test_batch)} images with σ={sigma_values}")
    
    results = []
    for sigma in sigma_values:
        noisy_batch = add_gaussian_noise(test_batch, sigma)
        
        with torch.no_grad():
            denoised_batch = ideal_denoiser(noisy_batch, sigma, train_subset)
        
        results.append(denoised_batch)
    
    # Save results
    os.makedirs("./results/examples", exist_ok=True)
    for i, sigma in enumerate(sigma_values):
        save_image(
            normalize_for_display(results[i]), 
            f"./results/examples/example4_sigma{sigma}.png",
            nrow=2
        )
    
    print(f"✓ Processed {len(test_batch)} images with {len(sigma_values)} noise levels")
    print()


def example_5_custom_noise():
    """
    Example 5: Using custom noise utilities
    """
    print("="*80)
    print("Example 5: Custom Noise Utilities")
    print("="*80)
    
    # Create a synthetic test image
    test_image = torch.randn(1, 3, 32, 32)
    
    # Test different noise levels
    sigma_values = [0, 0.5, 1.0, 2.0, 5.0]
    
    print(f"\nTesting {len(sigma_values)} noise levels: {sigma_values}")
    
    noisy_images = []
    for sigma in sigma_values:
        noisy = add_gaussian_noise(test_image, sigma)
        noisy_images.append(noisy)
    
    # Concatenate and save
    all_noisy = torch.cat(noisy_images, dim=0)
    os.makedirs("./results/examples", exist_ok=True)
    save_image(
        normalize_for_display(all_noisy),
        "./results/examples/example5_noise_levels.png",
        nrow=len(sigma_values)
    )
    
    print(f"✓ Generated images with {len(sigma_values)} noise levels")
    print()


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("MODULAR DENOISING FRAMEWORK - USAGE EXAMPLES")
    print("="*80 + "\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    example_1_ideal_denoiser()
    example_2_edm_denoiser()
    example_3_gradient_ascent()
    example_4_batch_processing()
    example_5_custom_noise()
    
    print("="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\nResults saved in: ./results/examples/")
    print("\nTo use individual examples, you can import and call them:")
    print("  from example_usage import example_1_ideal_denoiser")
    print("  example_1_ideal_denoiser()")


if __name__ == "__main__":
    main()

