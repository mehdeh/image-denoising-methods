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
from typing import Optional, Tuple, List, Union

# Import model download utility
try:
    from utils.model_utils import ensure_model_downloaded
    HAS_MODEL_UTILS = True
except ImportError:
    HAS_MODEL_UTILS = False


def _convert_to_tensor(
    sigma: Union[float, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert sigma to a tensor with proper device and dtype.
    
    Parameters:
    -----------
    sigma : float or torch.Tensor
        Noise level value
    device : torch.device
        Target device
    dtype : torch.dtype
        Target dtype (default: float32)
        
    Returns:
    --------
    sigma_tensor : torch.Tensor
        Converted sigma tensor of shape (1,)
    """
    if not isinstance(sigma, torch.Tensor):
        sigma_tensor = torch.tensor([sigma], dtype=dtype, device=device)
    else:
        sigma_tensor = sigma.to(device=device, dtype=dtype)
    
    # Ensure it's at least 1D
    if sigma_tensor.dim() == 0:
        sigma_tensor = sigma_tensor.unsqueeze(0)
    
    return sigma_tensor


def load_edm_model(
    model_path: str,
    url: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None
) -> torch.nn.Module:
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
    # Set default device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Check if the model file already exists, download if needed
    if not os.path.exists(model_path) and url is not None:
        # Try using the model_utils download first
        if HAS_MODEL_UTILS:
            print(f"Model not found at {model_path}. Downloading...")
            if not ensure_model_downloaded(model_path, url):
                raise RuntimeError(f"Failed to download model from {url}")
        else:
            # Fallback to dnnlib if available
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
                net.eval()
                return net
            except ImportError:
                raise ImportError(
                    "Neither model_utils nor dnnlib is available. "
                    "Please ensure utils.model_utils is accessible or "
                    "install dnnlib via: pip install git+https://github.com/NVlabs/edm.git"
                )
    
    # Load the model from disk
    if os.path.exists(model_path):
        print(f"Loading EDM model from: {model_path}")
        with open(model_path, "rb") as f:
            net = pickle.load(f)['ema'].to(device)
    else:
        raise FileNotFoundError(
            f"Model not found at {model_path} and no URL provided for download."
        )
    
    net.eval()
    return net


def edm_denoise(
    model: torch.nn.Module,
    noisy_images: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    class_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Denoise images using a pretrained EDM model.
    
    This function applies the EDM denoiser D(x; σ) to noisy images.
    The caller is responsible for wrapping this in torch.no_grad() if needed.
    
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
        For unconditional models, set to None (default)
        
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
    >>> # Denoise (wrap in no_grad for inference)
    >>> with torch.no_grad():
    >>>     denoised = edm_denoise(model, noisy_img, sigma)
    
    Notes:
    ------
    - The model should already be in eval mode (handled by load_edm_model)
    - For inference, wrap the call in torch.no_grad() context
    - Sigma is automatically broadcast to match batch size if needed
    """
    # Convert sigma to tensor with proper device and dtype
    sigma_tensor = _convert_to_tensor(sigma, noisy_images.device, noisy_images.dtype)
    
    # Replicate sigma for batch if needed
    batch_size = noisy_images.shape[0]
    if sigma_tensor.shape[0] == 1 and batch_size > 1:
        sigma_tensor = sigma_tensor.repeat(batch_size)
    
    # Validate shapes
    if sigma_tensor.shape[0] != batch_size:
        raise ValueError(
            f"Sigma batch size ({sigma_tensor.shape[0]}) must match "
            f"noisy_images batch size ({batch_size}) or be 1"
        )
    
    # Denoise using the model
    denoised = model(noisy_images, sigma_tensor, class_labels=class_labels)
    
    return denoised


def compute_score_gradient(
    model: torch.nn.Module,
    x: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    class_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the score function (gradient of log probability density) at x.
    
    The score is defined as: ∇_x log p(x; σ) = -(x - D(x; σ)) / σ²
    where D(x; σ) is the denoiser output. This is derived from Tweedie's formula.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Pretrained EDM denoising model
    x : torch.Tensor
        Input images of shape (batch_size, C, H, W)
        Note: Does NOT need requires_grad=True (computed analytically)
    sigma : float or torch.Tensor
        Noise level (standard deviation)
    class_labels : torch.Tensor, optional
        Class labels for conditional models (default: None)
        
    Returns:
    --------
    score : torch.Tensor
        Score function (gradient direction) of shape (batch_size, C, H, W)
        
    Examples:
    ---------
    >>> import torch
    >>> from denoisers.edm_denoiser import load_edm_model, compute_score_gradient
    >>> 
    >>> # Load model
    >>> model = load_edm_model("./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl")
    >>> 
    >>> # Compute score at a point
    >>> x = torch.randn(1, 3, 32, 32).cuda()
    >>> sigma = 2.0
    >>> score = compute_score_gradient(model, x, sigma)
    >>> 
    >>> # Use for gradient ascent
    >>> learning_rate = 1.0
    >>> x_updated = x + learning_rate * score
    
    Notes:
    ------
    - The score is computed analytically using the denoiser output
    - No backpropagation is needed through the denoiser
    - For gradient ascent, use this in a loop to iteratively move x
    - Reference: EDM paper (Karras et al., 2022), Equation 4
    """
    # Convert sigma to tensor with proper device and dtype
    sigma_tensor = _convert_to_tensor(sigma, x.device, x.dtype)
    
    # Replicate sigma for batch if needed
    batch_size = x.shape[0]
    if sigma_tensor.shape[0] == 1 and batch_size > 1:
        sigma_tensor = sigma_tensor.repeat(batch_size)
    
    # Get denoised output (no gradient needed)
    with torch.no_grad():
        denoised = model(x, sigma_tensor, class_labels=class_labels)
    
    # Compute score: ∇_x log p(x; σ) = (D(x; σ) - x) / σ²
    # Note: Reshape sigma for broadcasting
    sigma_sq = (sigma_tensor ** 2).view(-1, 1, 1, 1)
    score = (denoised - x) / sigma_sq
    
    return score


def gradient_ascent_denoise(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    num_steps: int = 10,
    lr: float = 1.0,
    class_labels: Optional[torch.Tensor] = None,
    return_trajectory: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Perform gradient ascent on the log probability to denoise images.
    
    This iteratively moves x towards higher probability regions by following
    the score function: x_{t+1} = x_t + lr * ∇_x log p(x_t; σ)
    
    This implements Algorithm 1 from the EDM paper for fixed sigma denoising.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Pretrained EDM denoising model
    x_init : torch.Tensor
        Initial noisy images of shape (batch_size, C, H, W)
    sigma : float or torch.Tensor
        Noise level (standard deviation)
    num_steps : int
        Number of gradient ascent iterations (default: 10)
    lr : float
        Learning rate for gradient updates (default: 1.0)
    class_labels : torch.Tensor, optional
        Class labels for conditional models (default: None)
    return_trajectory : bool
        If True, return the full trajectory of intermediate images (default: False)
        
    Returns:
    --------
    x_final : torch.Tensor
        Denoised images after gradient ascent
    trajectory : list of torch.Tensor (optional)
        List of intermediate images at each step (only if return_trajectory=True)
        
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
    >>>     model, noisy_img, sigma=3.0, num_steps=10, lr=1.0, return_trajectory=True
    >>> )
    >>> 
    >>> # Or without trajectory
    >>> denoised = gradient_ascent_denoise(model, noisy_img, sigma=3.0, num_steps=10)
    
    Notes:
    ------
    - This is different from the full EDM sampler which uses a noise schedule
    - Here we denoise at a fixed noise level using gradient ascent
    - Useful for visualization and understanding the denoising process
    """
    x_current = x_init.clone().detach()
    trajectory = [x_current.clone()] if return_trajectory else []
    
    # Convert sigma to tensor with proper device and dtype
    sigma_tensor = _convert_to_tensor(sigma, x_init.device, x_init.dtype)
    
    # Replicate sigma for batch if needed
    batch_size = x_init.shape[0]
    if sigma_tensor.shape[0] == 1 and batch_size > 1:
        sigma_tensor = sigma_tensor.repeat(batch_size)
    
    # Precompute for efficiency
    sigma_sq = (sigma_tensor ** 2).view(-1, 1, 1, 1)
    
    for step in range(num_steps):
        # Compute score gradient using the denoiser
        with torch.no_grad():
            denoised = model(x_current, sigma_tensor, class_labels=class_labels)
            
            # Score: ∇_x log p(x; σ) = (D(x; σ) - x) / σ²
            score = (denoised - x_current) / sigma_sq
            
            # Gradient ascent step
            x_current = x_current + lr * score
        
        if return_trajectory:
            trajectory.append(x_current.clone())
    
    if return_trajectory:
        return x_current, trajectory
    else:
        return x_current


# Predefined model configurations
EDM_PRETRAINED_MODELS = {
    'cifar10-uncond-vp': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-uncond-vp.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': False,
        'architecture': 'ddpmpp'
    },
    'cifar10-uncond-ve': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-uncond-ve.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': False,
        'architecture': 'ncsnpp'
    },
    'cifar10-cond-vp': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-cond-vp.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': True,
        'architecture': 'ddpmpp'
    },
    'cifar10-cond-ve': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-cond-ve.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': True,
        'architecture': 'ncsnpp'
    },
    # Legacy aliases for backward compatibility
    'cifar10-uncond': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-uncond-ve.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': False,
        'architecture': 'ncsnpp'
    },
    'cifar10-cond': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-cond-ve.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': True,
        'architecture': 'ncsnpp'
    }
}


def load_pretrained_edm(
    model_name: str = 'cifar10-uncond',
    device: Optional[Union[torch.device, str]] = None
) -> Tuple[torch.nn.Module, dict]:
    """
    Load a pretrained EDM model by name.
    
    This is a convenience function that loads commonly used pretrained models
    from the official EDM repository.
    
    Parameters:
    -----------
    model_name : str
        Name of the pretrained model (default: 'cifar10-uncond')
        Available models:
            - 'cifar10-uncond-vp': Unconditional CIFAR-10, VP parameterization
            - 'cifar10-uncond-ve': Unconditional CIFAR-10, VE parameterization
            - 'cifar10-cond-vp': Conditional CIFAR-10, VP parameterization
            - 'cifar10-cond-ve': Conditional CIFAR-10, VE parameterization
            - 'cifar10-uncond': Alias for 'cifar10-uncond-ve'
            - 'cifar10-cond': Alias for 'cifar10-cond-ve'
    device : torch.device or str, optional
        Device to load the model on (default: cuda if available, else cpu)
        
    Returns:
    --------
    model : torch.nn.Module
        Loaded pretrained model in eval mode
    config : dict
        Model configuration dictionary containing:
            - url: Download URL
            - path: Local path
            - resolution: Image resolution
            - channels: Number of channels
            - conditional: Whether the model is conditional
            - architecture: Model architecture type
        
    Examples:
    ---------
    >>> from denoisers.edm_denoiser import load_pretrained_edm
    >>> 
    >>> # Load unconditional model (VE parameterization)
    >>> model, config = load_pretrained_edm('cifar10-uncond-ve')
    >>> print(f"Resolution: {config['resolution']}")
    >>> print(f"Conditional: {config['conditional']}")
    >>> 
    >>> # Load conditional model (VP parameterization)
    >>> model, config = load_pretrained_edm('cifar10-cond-vp')
    >>> 
    >>> # Use with specific device
    >>> model, config = load_pretrained_edm('cifar10-uncond', device='cuda:0')
    
    Raises:
    -------
    ValueError
        If the model_name is not recognized
        
    Notes:
    ------
    - VP (Variance Preserving) and VE (Variance Exploding) refer to different
      noise schedules and model architectures
    - For most use cases, the VE models (default) work well
    - Requires ~200-300MB download on first use
    """
    if model_name not in EDM_PRETRAINED_MODELS:
        available = list(set(EDM_PRETRAINED_MODELS.keys()) - {'cifar10-uncond', 'cifar10-cond'})
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Available models: {available}"
        )
    
    config = EDM_PRETRAINED_MODELS[model_name].copy()
    model = load_edm_model(config['path'], config['url'], device)
    
    return model, config

