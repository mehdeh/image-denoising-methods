"""
Model utilities for downloading and managing pretrained models.

This module provides utilities for downloading pretrained models from URLs
and managing model checkpoints.
"""

import os
import urllib.request
from tqdm import tqdm
import torch


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    
    def update_to(self, blocks=1, block_size=1, total_size=None):
        """Update progress bar for urllib.request.urlretrieve."""
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_file(url: str, destination: str) -> None:
    """
    Download a file from URL with progress bar.
    
    Parameters:
    -----------
    url : str
        URL to download from
    destination : str
        Local path to save the downloaded file
        
    Examples:
    ---------
    >>> from utils.model_utils import download_file
    >>> 
    >>> url = "https://example.com/model.pkl"
    >>> dest = "./models/model.pkl"
    >>> download_file(url, dest)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"Downloading from {url}")
    print(f"Saving to {destination}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Download") as progress_bar:
        urllib.request.urlretrieve(
            url,
            destination,
            reporthook=progress_bar.update_to
        )
    
    print(f"✓ Download completed: {destination}")


def ensure_model_downloaded(model_path: str, url: str) -> bool:
    """
    Ensure a model file exists, downloading if necessary.
    
    Parameters:
    -----------
    model_path : str
        Local path where model should exist
    url : str
        URL to download from if model doesn't exist
        
    Returns:
    --------
    success : bool
        True if model exists or was successfully downloaded
        
    Examples:
    ---------
    >>> from utils.model_utils import ensure_model_downloaded
    >>> 
    >>> model_path = "./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl"
    >>> url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl"
    >>> 
    >>> if ensure_model_downloaded(model_path, url):
    >>>     print("Model ready to use")
    """
    if os.path.exists(model_path):
        print(f"✓ Model found at: {model_path}")
        return True
    
    print(f"Model not found at: {model_path}")
    
    try:
        download_file(url, model_path)
        return True
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False


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
    
    Examples:
    ---------
    >>> from utils.model_utils import load_edm_model
    >>> 
    >>> model, config = load_edm_model('cuda')
    >>> if model is not None:
    ...     print(f"Model loaded: {config['architecture']}")
    """
    from edm_denoiser import load_pretrained_edm
    
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

