"""
EDM (Elucidating the Design Space of Diffusion Models) Denoiser Implementation.

This module provides functionality to use pretrained EDM models for denoising.
Based on the official implementation from NVlabs/edm repository.

Reference:
    Paper: https://arxiv.org/abs/2206.00364
    GitHub: https://github.com/NVlabs/edm
"""

import torch
import pickle
import os
import sys


def load_edm_model(model_path, url=None, device=None):
    """
    Load a pretrained EDM denoising model from a local path or URL.
    
    If the model file does not exist locally and a URL is provided,
    it will attempt to download the model from the URL and save it locally.
    
    Parameters:
    -----------
    model_path : str
        Local path where the model file should be stored and loaded from
    url : str, optional
        URL to download the model file if it does not already exist locally
    device : torch.device or str, optional
        Device to which the model will be moved (default: 'cuda' if available, else 'cpu')
        
    Returns:
    --------
    net : torch.nn.Module
        Loaded EDM denoising model in eval mode
        
    Examples:
    ---------
    >>> import torch
    >>> from denoisers.edm_denoiser import load_edm_model
    >>> 
    >>> # Load pretrained CIFAR-10 model
    >>> model_path = "./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl"
    >>> model_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl"
    >>> 
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> model = load_edm_model(model_path, model_url, device)
    >>> 
    >>> # Use the model for denoising
    >>> noisy_img = torch.randn(1, 3, 32, 32).to(device)
    >>> sigma = torch.tensor([2.0]).to(device)
    >>> denoised = model(noisy_img, sigma, class_labels=None)
    
    Notes:
    ------
    Requires dnnlib from the EDM repository for URL downloads.
    Install via: pip install git+https://github.com/NVlabs/edm.git
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Check if the model file already exists
    if os.path.exists(model_path):
        print(f"Loading EDM model from local path: {model_path}")
        with open(model_path, "rb") as f:
            net = pickle.load(f)['ema'].to(device)
    elif url is not None:
        print(f"Model not found at {model_path}. Downloading from {url}...")
        try:
            import dnnlib
            with dnnlib.util.open_url(url) as f:
                net = pickle.load(f)['ema'].to(device)
            
            # Save the downloaded model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump({'ema': net}, f)
            print(f"Model downloaded and saved at {model_path}")
        except ImportError:
            raise ImportError(
                "dnnlib is required to download models from URLs. "
                "Install it via: pip install git+https://github.com/NVlabs/edm.git"
            )
    else:
        raise FileNotFoundError(
            f"Model not found at {model_path} and no URL provided for download."
        )
    
    net.eval()
    return net


def edm_denoise(model, noisy_images, sigma, class_labels=None):
    """
    Denoise images using a pretrained EDM model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Pretrained EDM denoising model
    noisy_images : torch.Tensor
        Noisy input images of shape (batch_size, C, H, W)
    sigma : float or torch.Tensor
        Noise level (standard deviation). Can be a scalar or tensor of shape (batch_size,)
    class_labels : torch.Tensor, optional
        Class labels for conditional models (shape: batch_size,)
        For unconditional models, set to None
        
    Returns:
    --------
    denoised : torch.Tensor
        Denoised images of shape (batch_size, C, H, W)
        
    Examples:
    ---------
    >>> import torch
    >>> from denoisers.edm_denoiser import load_edm_model, edm_denoise
    >>> from utils.noise_utils import add_gaussian_noise
    >>> 
    >>> # Load model
    >>> model_path = "./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl"
    >>> model = load_edm_model(model_path)
    >>> 
    >>> # Create noisy image
    >>> clean_img = torch.randn(1, 3, 32, 32).cuda()
    >>> sigma = 2.0
    >>> noisy_img = add_gaussian_noise(clean_img, sigma)
    >>> 
    >>> # Denoise
    >>> with torch.no_grad():
    >>>     denoised = edm_denoise(model, noisy_img, sigma)
    """
    # Convert sigma to tensor if needed
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor([sigma], dtype=torch.float32, device=noisy_images.device)
    
    # Ensure sigma has the right shape
    if sigma.dim() == 0:
        sigma = sigma.unsqueeze(0)
    
    # Replicate sigma for batch if needed
    if sigma.shape[0] == 1 and noisy_images.shape[0] > 1:
        sigma = sigma.repeat(noisy_images.shape[0])
    
    # Denoise using the model
    with torch.no_grad():
        denoised = model(noisy_images, sigma, class_labels=class_labels)
    
    return denoised


def compute_score_gradient(model, x, sigma, class_labels=None):
    """
    Compute the score gradient (gradient of log probability density) at x.
    
    The score is defined as: ∇_x log p(x; σ) = -(x - D(x; σ)) / σ²
    where D(x; σ) is the denoiser output.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Pretrained EDM denoising model
    x : torch.Tensor
        Input images of shape (batch_size, C, H, W), requires_grad=True
    sigma : float or torch.Tensor
        Noise level (standard deviation)
    class_labels : torch.Tensor, optional
        Class labels for conditional models
        
    Returns:
    --------
    score : torch.Tensor
        Score gradient of shape (batch_size, C, H, W)
        
    Examples:
    ---------
    >>> import torch
    >>> from denoisers.edm_denoiser import load_edm_model, compute_score_gradient
    >>> 
    >>> # Load model
    >>> model = load_edm_model("./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl")
    >>> 
    >>> # Compute score at a point
    >>> x = torch.randn(1, 3, 32, 32, requires_grad=True).cuda()
    >>> sigma = 2.0
    >>> score = compute_score_gradient(model, x, sigma)
    >>> 
    >>> # Use for gradient ascent
    >>> learning_rate = 0.1
    >>> x_updated = x + learning_rate * score
    """
    # Convert sigma to tensor if needed
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor([sigma], dtype=torch.float32, device=x.device)
    
    # Get denoised output
    denoised = model(x, sigma, class_labels=class_labels)
    
    # Compute score: ∇_x log p(x; σ) = -(x - D(x; σ)) / σ²
    score = -(x - denoised) / (sigma ** 2)
    
    return score


def gradient_ascent_denoise(model, x_init, sigma, num_steps=10, lr=1.0, class_labels=None):
    """
    Perform gradient ascent on the log probability to denoise images.
    
    This iteratively moves x towards higher probability regions by following
    the score gradient: x_{t+1} = x_t + lr * ∇_x log p(x_t; σ)
    
    Parameters:
    -----------
    model : torch.nn.Module
        Pretrained EDM denoising model
    x_init : torch.Tensor
        Initial noisy images of shape (batch_size, C, H, W)
    sigma : float or torch.Tensor
        Noise level (standard deviation)
    num_steps : int
        Number of gradient ascent iterations
    lr : float
        Learning rate for gradient updates
    class_labels : torch.Tensor, optional
        Class labels for conditional models
        
    Returns:
    --------
    x_final : torch.Tensor
        Denoised images after gradient ascent
    trajectory : list of torch.Tensor
        List of intermediate images at each step
        
    Examples:
    ---------
    >>> import torch
    >>> from denoisers.edm_denoiser import load_edm_model, gradient_ascent_denoise
    >>> from utils.noise_utils import add_gaussian_noise
    >>> 
    >>> # Setup
    >>> model = load_edm_model("./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl")
    >>> clean_img = torch.randn(1, 3, 32, 32).cuda()
    >>> noisy_img = add_gaussian_noise(clean_img, sigma=3.0)
    >>> 
    >>> # Gradient ascent denoising
    >>> denoised, trajectory = gradient_ascent_denoise(
    >>>     model, noisy_img, sigma=3.0, num_steps=10, lr=1.0
    >>> )
    """
    x_current = x_init.clone().detach()
    trajectory = [x_current.clone()]
    
    # Convert sigma to tensor
    if not isinstance(sigma, torch.Tensor):
        sigma_tensor = torch.tensor([sigma], dtype=torch.float32, device=x_init.device)
    else:
        sigma_tensor = sigma
    
    for step in range(num_steps):
        # Get denoised output
        with torch.no_grad():
            denoised = model(x_current, sigma_tensor, class_labels=class_labels)
        
        # Compute score gradient
        score = -(x_current - denoised) / (sigma_tensor ** 2)
        
        # Gradient ascent step
        x_current = x_current + lr * score
        trajectory.append(x_current.clone())
    
    return x_current, trajectory


# Predefined model configurations
EDM_PRETRAINED_MODELS = {
    'cifar10-uncond': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-uncond-ve.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': False
    },
    'cifar10-cond': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-cond-ve.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': True
    }
}


def load_pretrained_edm(model_name='cifar10-uncond', device=None):
    """
    Load a pretrained EDM model by name.
    
    Parameters:
    -----------
    model_name : str
        Name of the pretrained model. Options: 'cifar10-uncond', 'cifar10-cond'
    device : torch.device or str, optional
        Device to load the model on
        
    Returns:
    --------
    model : torch.nn.Module
        Loaded pretrained model
    config : dict
        Model configuration
        
    Examples:
    ---------
    >>> from denoisers.edm_denoiser import load_pretrained_edm
    >>> 
    >>> model, config = load_pretrained_edm('cifar10-uncond')
    >>> print(f"Resolution: {config['resolution']}")
    >>> print(f"Conditional: {config['conditional']}")
    """
    if model_name not in EDM_PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available models: {list(EDM_PRETRAINED_MODELS.keys())}"
        )
    
    config = EDM_PRETRAINED_MODELS[model_name]
    model = load_edm_model(config['path'], config['url'], device)
    
    return model, config

