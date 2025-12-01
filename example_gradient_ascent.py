"""
Simple example demonstrating the gradient ascent denoising function.

This is a minimal example to quickly test the gradient_ascent_denoise() function.
For comprehensive testing, see test_gradient_ascent_fix.py
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path

from denoisers.edm_denoiser import load_edm_model, gradient_ascent_denoise


def main():
    """Run a simple gradient ascent denoising example."""
    
    print("\n" + "=" * 70)
    print("Gradient Ascent Denoising - Simple Example")
    print("=" * 70 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\n1. Loading EDM model...")
    model_path = "./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl"
    model_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl"
    model = load_edm_model(model_path, model_url, device)
    print("   ✓ Model loaded")
    
    # Load a test image from CIFAR-10
    print("\n2. Loading CIFAR-10 test image...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    clean_img = dataset[0][0].unsqueeze(0).to(device)  # First test image
    print(f"   ✓ Loaded image with shape: {clean_img.shape}")
    
    # Add Gaussian noise
    print("\n3. Adding Gaussian noise...")
    sigma = 3.0
    noise = torch.randn_like(clean_img)
    noisy_img = clean_img + sigma * noise
    
    # Calculate noise level
    mse_noise = torch.mean((noisy_img - clean_img) ** 2).item()
    print(f"   σ = {sigma}")
    print(f"   Noise MSE = {mse_noise:.4f}")
    
    # Denoise using gradient ascent
    print("\n4. Denoising with gradient ascent...")
    print(f"   Steps: 10")
    print(f"   Learning rate: 1.0")
    print(f"   Precision: float64")
    
    denoised_img = gradient_ascent_denoise(
        model=model,
        x_init=noisy_img,
        sigma=sigma,
        num_steps=10,
        lr=1.0,
        use_float64=True
    )
    
    # Calculate denoising quality
    mse_denoised = torch.mean((denoised_img - clean_img) ** 2).item()
    reduction = (mse_noise - mse_denoised) / mse_noise * 100
    
    print(f"\n5. Results:")
    print(f"   MSE (noisy):    {mse_noise:.4f}")
    print(f"   MSE (denoised): {mse_denoised:.4f}")
    print(f"   Reduction:      {reduction:.2f}%")
    
    # Save visualization
    print("\n6. Saving visualization...")
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    # Normalize for display
    def normalize(x):
        x = x.squeeze().cpu()
        x = (x - x.min()) / (x.max() - x.min())
        return x.permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(normalize(clean_img))
    axes[0].set_title('Clean Image')
    axes[0].axis('off')
    
    axes[1].imshow(normalize(noisy_img))
    axes[1].set_title(f'Noisy (σ={sigma})')
    axes[1].axis('off')
    
    axes[2].imshow(normalize(denoised_img))
    axes[2].set_title(f'Denoised ({reduction:.1f}% reduction)')
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = results_dir / "gradient_ascent_example.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved to: {save_path}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

