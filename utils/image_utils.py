"""
Image utilities for loading, processing, and normalizing images.

This module provides common image processing functions used across different
denoising methods.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


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
        
    Examples:
    ---------
    >>> from utils.image_utils import load_cifar10_dataset
    >>> 
    >>> # Load normalized dataset
    >>> train_imgs, test_imgs = load_cifar10_dataset(root="./data", normalize=True)
    >>> print(f"Train: {train_imgs.shape}, Test: {test_imgs.shape}")
    >>> 
    >>> # Load unnormalized dataset
    >>> train_raw, test_raw = load_cifar10_dataset(root="./data", normalize=False)
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
        
    Examples:
    ---------
    >>> import torch
    >>> from utils.image_utils import normalize_for_display
    >>> 
    >>> # Normalize batch of images for visualization
    >>> images = torch.randn(10, 3, 32, 32)  # Random images
    >>> normalized = normalize_for_display(images)
    >>> assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    """
    batch_size = images.shape[0]
    images_flat = images.view(batch_size, -1)
    
    min_vals = images_flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    max_vals = images_flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
    
    normalized = (images - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized

